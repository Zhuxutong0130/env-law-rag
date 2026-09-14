"""两级检索模块：向量召回 + cross-encoder 重排。

管线位置：models/law.index + chunk_meta.jsonl + query
          → [本模块] → 排序后的 top-k chunk 列表（带双阶段分数）
"""
import json
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from embedding import load_model, EMBED_DIM

# 国内访问 HF + Windows 软链不支持 + 新版 xet 协议 401，三重坑的统一兜底：
# - HF_ENDPOINT 走镜像（embedding.py 已 setdefault，这里复用）
# - HF_HUB_DISABLE_XET 禁用 xet 协议退回传统 HTTP（reranker 大文件走 xet 会 401）
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
RERANKER_NAME = "BAAI/bge-reranker-base"
RECALL_K = 20        # 召回宽口径
FINAL_K = 3          # 最终给生成的窄口径


def load_assets(index_path: Path, meta_path: Path) -> tuple[faiss.Index, list[dict]]:
    """加载索引 + 元数据。

    fail-fast 原则：校验 index.ntotal == len(meta)，不一致直接 raise。
    索引和元数据一旦不同步（如索引重建了 meta 没重建），宁可启动就炸，
    也不要带着错误对齐往下跑——否则检索结果会指向错误 chunk，
    且这种错误极难在下游发现（看着像正常结果，其实是张冠李戴）。
    这是阶段 1 "隐式对齐"讨论的延续：用 fail-fast 把对齐隐患挡在启动阶段。
    """
    index = faiss.read_index(str(index_path))

    meta = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                meta.append(json.loads(line))

    if index.ntotal != len(meta):
        raise RuntimeError(
            f"索引与元数据不同步: index.ntotal={index.ntotal} 但 len(meta)={len(meta)}。"
            f"请确认两者由同一次 embedding.py 运行产出。"
        )
    if index.d != EMBED_DIM:
        raise RuntimeError(
            f"索引维度 {index.d} 与代码常量 EMBED_DIM={EMBED_DIM} 不一致。"
        )
    return index, meta


def recall(query: str, index, meta, model, top_k: int = RECALL_K) -> list[dict]:
    """第一级：向量召回。

    bge 检索场景 query 需加 instruction 前缀（不对称检索范式），
    文档侧在 embedding.py build_texts 拼接时已带 law_name+article 前缀，无需再加。
    encode 时 normalize_embeddings=True，归一化后 IndexFlatIP 内积 = 余弦相似度。

    Returns:
        list[dict]: top_k 个 {**chunk, "recall_score": s}，按 recall_score 降序。
    """
    q_text = QUERY_INSTRUCTION + query
    q_vec = model.encode(
        [q_text], normalize_embeddings=True, show_progress_bar=False
    ).astype("float32")

    scores, ids = index.search(q_vec, top_k)  # (1, top_k), (1, top_k)
    results = []
    for sc, idx in zip(scores[0], ids[0]):
        if idx < 0:  # faiss 在 top_k > ntotal 时会返回 -1
            continue
        chunk = dict(meta[idx])
        chunk["recall_score"] = float(sc)
        results.append(chunk)
    return results


def rerank(query: str, candidates: list[dict], reranker, top_k: int = FINAL_K) -> list[dict]:
    """第二级：cross-encoder 重排。

    cross-encoder 把 (query, doc) 作为一对输入，query 和 doc 在 transformer 内部
    全交叉注意力，能捕捉召回阶段双塔结构（query/doc 各自编码再点积）漏掉的细粒度
    语义关系——这是它精度高于纯向量的根本原因。代价是 O(N) 的 transformer 前向，
    所以只在召回阶段拿到的 20 条候选里跑，不直接跑全库。

    Returns:
        list[dict]: top_k 个 {**chunk, "recall_score", "rerank_score", "rank"}，
        按 rerank_score 降序，rank 从 1 起。
    """
    # 重排输入文本与 embedding 阶段保持一致：law_name + article + text
    pairs = []
    for c in candidates:
        prefix = f"{c['law_name']} {c['article']}：" if c.get("article") else f"{c['law_name']}："
        pairs.append([query, f"{prefix}{c['text']}"])

    rerank_scores = reranker.predict(pairs, show_progress_bar=False)

    for c, rs in zip(candidates, rerank_scores):
        c["rerank_score"] = float(rs)

    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
    for i, c in enumerate(ranked, 1):
        c["rank"] = i
    return ranked


def search(query: str, index, meta, model, reranker) -> list[dict]:
    """对外主入口：recall → rerank，一步到位。"""
    return rerank(query, recall(query, index, meta, model), reranker)


def load_reranker(model_name: str = RERANKER_NAME) -> CrossEncoder:
    """加载 cross-encoder 重排模型。与 load_model 同样走 snapshot_download 绕过 Windows bug。"""
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id=model_name)
    return CrossEncoder(local_dir)


def _is_hit(results: list[dict], expect_law: str) -> bool:
    """命中判定：top-3 里存在 expect_law 是 law_name 子串的 chunk。"""
    for c in results:
        if expect_law in c["law_name"]:
            return True
    return False


def _correct_rank(results: list[dict], expect_law: str) -> int:
    """正确 chunk 在结果列表中的 rank（1-based）；不在则返回 len(results)+1。
    rank 越小越好：rerank 应该把正确的 chunk 推到更靠前的位置。"""
    for i, c in enumerate(results, 1):
        if expect_law in c["law_name"]:
            return i
    return len(results) + 1


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    index_path = project_root / "models" / "law.index"
    meta_path = project_root / "data" / "processed" / "chunk_meta.jsonl"
    queries_path = project_root / "eval" / "test_queries.jsonl"

    # 加载资产 + 模型（一次性）
    index, meta = load_assets(index_path, meta_path)
    print(f"加载完成: index.ntotal={index.ntotal}, len(meta)={len(meta)}")
    model = load_model()
    reranker = load_reranker()

    # 读评测集
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    print(f"评测集: {len(queries)} 个 query\n")

    # 两个管线对比：纯召回 vs 召回+重排
    # 主指标：top-3 命中率（用户要求的规格）
    # 辅指标：正确 chunk 在 top-3 内的 rank（rerank 价值的敏感指标——
    #         召回天花板效应下，rerank 的价值在"把正确的排到更靠前"而非"召回回更多"）
    pure_hits_3, rerank_hits_3 = 0, 0
    pure_ranks, rerank_ranks = [], []
    top1_changed = 0
    print("=" * 90)
    print(f"{'qid':<4} {'query':<24} {'expect':<14} {'top-3 纯召回':<12} {'top-3 重排':<12} {'rank 纯→重':<12} {'top-1 变':<8}")
    print("=" * 90)

    for q in queries:
        # 纯召回 top-3（直接截断 recall 的前 3，不经过 rerank）
        recall_results = recall(q["query"], index, meta, model)
        pure_top3 = recall_results[:FINAL_K]
        pure_ok3 = _is_hit(pure_top3, q["expect_law"])
        pure_hits_3 += int(pure_ok3)
        pure_rank = _correct_rank(pure_top3, q["expect_law"])
        pure_ranks.append(pure_rank)

        # 召回 + 重排
        reranked = rerank(q["query"], recall_results, reranker)
        rerank_ok3 = _is_hit(reranked, q["expect_law"])
        rerank_hits_3 += int(rerank_ok3)
        rerank_rank = _correct_rank(reranked, q["expect_law"])
        rerank_ranks.append(rerank_rank)

        # rerank 是否改变了 top-1 的 chunk（rerank 起作用的间接证据）
        changed = pure_top3[0]["id"] != reranked[0]["id"]
        top1_changed += int(changed)

        arrow = f"{pure_rank}→{rerank_rank}"
        print(f"{q['qid']:<4} {q['query'][:22]:<24} {q['expect_law']:<14} "
              f"{'✓' if pure_ok3 else '✗':<12} {'✓' if rerank_ok3 else '✗':<12} "
              f"{arrow:<12} {'是' if changed else '否':<8}")

    print("=" * 90)
    p3 = pure_hits_3 / len(queries) * 100
    r3 = rerank_hits_3 / len(queries) * 100
    p_mean = sum(pure_ranks) / len(queries)
    r_mean = sum(rerank_ranks) / len(queries)
    print(f"\n指标                       纯召回          召回+重排        变化")
    print(f"{'-'*60}")
    print(f"top-3 命中率 (召回规格)    {pure_hits_3}/{len(queries)} ({p3:.0f}%)      {rerank_hits_3}/{len(queries)} ({r3:.0f}%)       +{rerank_hits_3 - pure_hits_3}")
    print(f"正确 chunk 平均 rank        {p_mean:.2f}           {r_mean:.2f}           {r_mean - p_mean:+.2f}")
    print(f"top-1 被重排改变的 query    -               {top1_changed}/{len(queries)}        -")
