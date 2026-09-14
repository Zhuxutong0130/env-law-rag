"""RAG 管线封装：重资源只加载一次 + LCEL 链组装。

管线位置：query → [本模块] → {"answer", "cited_chunks", "refused", "retrieved"}

设计要点：
- RAGPipeline.__init__ 把 index / meta / 编码模型 / 重排模型 / API 客户端
  一次性加载进实例属性。Gradio 每次请求都会调 answer()，若在 answer 里
  加载资源，每个问题都要重读 ~1.1GB 的 reranker——封装的意义就是
  "加载一次，服务多次"。app.py 和评测脚本都只 import 本模块，不重复装配。
- 链形态（LCEL）：
    RunnablePassthrough()          # query 原样进
    → RunnableLambda(_retrieve)    # → (query, top3_chunks) 元组
    → RunnableLambda(_generate)    # 元组消费 → 结果 dict
  用元组传中间结果比塞 dict 直白：检索的输出签名就是生成的输入签名。
"""
import sys
import time
from pathlib import Path

_SRC_DIR = str(Path(__file__).parent)
if _SRC_DIR not in sys.path:
    # 双模式导入兼容：被 app.py 以 from src.pipeline import 导入时，sys.path
    # 只有项目根；src 内部模块用的扁平导入（from embedding import ...）找不到
    # 兄弟模块。把 src 目录补进 sys.path 后，直接运行和包导入两种方式都能跑。
    sys.path.insert(0, _SRC_DIR)

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from embedding import load_model
from retriever import load_assets, load_reranker, recall, rerank
from generator import generate_answer, get_client

PROJECT_ROOT = Path(_SRC_DIR).parent


class RAGPipeline:
    """两级检索 + 生成的完整 RAG 管线。重资源只加载一次。"""

    def __init__(self):
        t0 = time.time()
        self.index, self.meta = load_assets(
            PROJECT_ROOT / "models" / "law.index",
            PROJECT_ROOT / "data" / "processed" / "chunk_meta.jsonl",
        )
        self.embed_model = load_model()
        self.reranker = load_reranker()
        self.client = get_client()   # 启动即校验 API key，fail-fast
        self.chain = self._build_chain()
        print(f"[pipeline] 资源加载完成（{time.time() - t0:.1f}s）: "
              f"ntotal={self.index.ntotal}, LCEL 链就绪")

    # ---- 链的两个节点 ----

    def _retrieve(self, query: str) -> tuple[str, list[dict]]:
        """检索节点：召回 20 → 重排取 3。返回 (query, chunks) 元组。"""
        candidates = recall(query, self.index, self.meta, self.embed_model)
        top3 = rerank(query, candidates, self.reranker)
        return query, top3

    def _generate(self, pair: tuple[str, list[dict]]) -> dict:
        """生成节点：消费 (query, chunks) 元组，调 DeepSeek 生成答案。"""
        query, chunks = pair
        out = generate_answer(query, chunks, self.client)
        # 附带完整检索结果（供 UI 展示三路来源与分数），cited_chunks 是其中被引用的子集
        out["retrieved"] = [
            {k: c[k] for k in ("id", "law_name", "article", "text", "rerank_score")}
            for c in chunks
        ]
        return out

    def _build_chain(self):
        """LCEL 组装：Passthrough → 检索 → 生成。"""
        return (
            RunnablePassthrough()
            | RunnableLambda(self._retrieve)
            | RunnableLambda(self._generate)
        )

    def answer(self, query: str) -> dict:
        """对外唯一入口：一行 invoke。返回 {"answer", "cited_chunks",
        "refused", "retrieved"}。"""
        return self.chain.invoke(query)


if __name__ == "__main__":
    # 命令行自测：2 个库内问题 + 1 个库外问题，验证链路与拒答
    rag = RAGPipeline()
    cases = [
        ("工厂半夜施工扰民怎么办", False),
        ("燃煤电厂的废气要怎么处理", False),
        ("个人所得税怎么退税", True),
    ]
    for query, expect_refused in cases:
        t0 = time.time()
        out = rag.answer(query)
        dt = time.time() - t0
        status = "✓" if out["refused"] == expect_refused else "✗"
        cited = ", ".join(f"《{c['law_name']}》{c['article']}"
                          for c in out["cited_chunks"]) or "无"
        print("=" * 70)
        print(f"[{status}] {query}  ({dt:.1f}s)  refused={out['refused']}")
        print(f"引用: {cited}")
        print(f"回答: {out['answer'][:160]}{'...' if len(out['answer']) > 160 else ''}")
