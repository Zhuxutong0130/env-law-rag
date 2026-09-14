"""law.index 与 chunk_meta.jsonl 的构建模块。

管线位置：chunks_all.jsonl → [本模块] → models/law.index + chunk_meta.jsonl

显式映射（方案 ❷）：chunk_meta.jsonl 每行明确写 "id" 字段（行号即 id），
faiss 检索返回的 ids 直接对应 chunk_meta[id]，不依赖"行号隐式对齐"——
即使后续 chunk_meta 行顺序被打乱，或 chunks_all 增删导致行号错位，
通过 id 字段仍能稳定定位，避免了方案 ❸（隐式行号映射）的对齐隐患。
"""
import json
import os
from pathlib import Path

import faiss
import numpy as np
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBED_DIM = 512
BATCH_SIZE = 64

# Windows 不支持 HF 默认的符号链接缓存策略，sentence_transformers 6.x 走镜像时
# 不会建 snapshots 软链，导致 FileNotFoundError。这里设镜像 endpoint + 用 snapshot_download
# 强制完整下载到本地目录后传路径给 SentenceTransformer，绕过该 bug。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """加载 bge-small-zh 向量编码器。

    使用 snapshot_download 强制完整下载到本地目录后传路径给 SentenceTransformer，
    绕过 Windows 不支持符号链接导致的 snapshots 软链未建 bug。
    抽出为独立函数，retriever.py 复用同一加载逻辑，避免重复实现。
    """
    local_dir = snapshot_download(repo_id=model_name)
    return SentenceTransformer(local_dir)


def load_chunks(path: Path) -> list[dict]:
    """读 chunks_all.jsonl，返回所有 chunk 的 list[dict]。

    每行一个 JSON 对象，字段：law_name / text / part / char_len / chapter / section / article。
    行号即后续的 id（0-based）。
    """
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_texts(chunks: list[dict]) -> list[str]:
    """决定每个 chunk 用什么文本去 encode。

    拼接格式：f"{law_name} {article}：{text}"

    理由：
    1. 孤立文本「处以五万元以上五十万元以下的罚款」单独编码时，
       与其他 40 条处罚条款向量会非常接近，用户问"罚款多少钱"时无法区分。
    2. 把 law_name + article 作为前缀拼进被编码文本，向量里就带上了
       "这是哪部法哪一条"的信号，召回时 faiss 命中能直接关联到具体条款。
    3. 前缀放前面、正文放后面：bge 是 transformer，前缀相当于"标题"，
       attention 机制能让正文每个 token 在编码时带着这层上下文。
    4. 不拼 chapter/section：它们语义较泛（"第七章 法律责任"很多法都有），
       拼进去反而稀释 article 的区分度；保留在 chunk_meta 作检索后展示即可。
    """
    texts = []
    for c in chunks:
        prefix = f"{c['law_name']} {c['article']}：" if c.get("article") else f"{c['law_name']}："
        texts.append(f"{prefix}{c['text']}")
    return texts


def build_index(texts: list[str]) -> tuple[faiss.Index, "np.ndarray"]:
    """encode → 归一化 → 建 IndexFlatIP。

    使用 model.encode 时直接 normalize_embeddings=True 出归一化向量，
    省一次 faiss.normalize_L2 调用；batch_size=64 平衡内存与速度；
    show_progress_bar=True 让批量任务进度可见。

    Returns:
        (index, embeddings): index 是已 add 完毕的 IndexFlatIP；
        embeddings 是 (N, 512) 的 float32 归一化矩阵，验收用。
    """
    model = load_model()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,   # 直接出归一化向量，省一次 faiss.normalize_L2
        show_progress_bar=True,      # 进度条，批量任务必备
    ).astype("float32")

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    return index, embeddings


def save_all(index: faiss.Index, chunks: list[dict]) -> tuple[Path, Path]:
    """law.index + chunk_meta.jsonl 落盘。

    meta 每行格式：{"id": i, **chunk 原字段}
    - id 字段是显式映射锚点：faiss 检索返回的 ids 直接对应 meta[id]，
      不靠"行号隐式对齐"，避免方案 ❸ 的对齐隐患。
    - 原 chunk 字段（law_name / text / part / char_len / chapter / section / article）
      原样保留，检索后可直接展示给用户。

    Returns:
        (index_path, meta_path): 两个落盘路径，便于主流程自检。
    """
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    processed_dir = project_root / "data" / "processed"
    models_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)

    index_path = models_dir / "law.index"
    meta_path = processed_dir / "chunk_meta.jsonl"

    faiss.write_index(index, str(index_path))

    with open(meta_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            meta = {"id": i, **chunk}
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    return index_path, meta_path


if __name__ == "__main__":
    import random

    project_root = Path(__file__).parent.parent
    chunks_path = project_root / "data" / "processed" / "chunks_all.jsonl"

    # 主流程：load → build_texts → build_index → save_all
    chunks = load_chunks(chunks_path)
    print(f"加载 chunks: {len(chunks)}")

    texts = build_texts(chunks)
    print(f"构造待编码文本: {len(texts)}")
    print(f"  示例: {texts[0][:80]}...")

    index, embeddings = build_index(texts)

    saved = save_all(index, chunks)
    index_path, meta_path = saved

    # 自检 1：三个数
    print("\n" + "=" * 60)
    print("自检 1：三个数")
    print("=" * 60)
    print(f"  index.ntotal   = {index.ntotal}（期望 344）{'✓' if index.ntotal == 344 else '✗'}")
    print(f"  向量维度       = {index.d}（期望 512）{'✓' if index.d == EMBED_DIM else '✗'}")
    file_size_kb = index_path.stat().st_size / 1024
    print(f"  索引文件大小   = {file_size_kb:.1f} KB（{index_path.name}）")

    # 自检 2：一致性——随机抽 3 个 id，验证 chunk_meta 第 id 行的 article
    # 与原 chunks_all.jsonl 第 id 行的 article 一致
    print("\n" + "=" * 60)
    print("自检 2：一致性（chunk_meta vs chunks_all）")
    print("=" * 60)
    random.seed(42)
    sample_ids = random.sample(range(len(chunks)), 3)
    sample_ids.sort()

    meta_lines = meta_path.read_text(encoding="utf-8").splitlines()
    chunk_lines = chunks_path.read_text(encoding="utf-8").splitlines()

    all_ok = True
    for cid in sample_ids:
        meta_article = json.loads(meta_lines[cid])["article"]
        chunk_article = json.loads(chunk_lines[cid])["article"]
        ok = meta_article == chunk_article
        all_ok = all_ok and ok
        print(f"  id={cid}: meta.article={meta_article!r}  chunks.article={chunk_article!r}  {'✓' if ok else '✗'}")
    print(f"  一致性自检: {'✓ 通过' if all_ok else '✗ 不一致'}")
