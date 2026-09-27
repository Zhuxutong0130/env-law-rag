r"""MCP Server：把 env-law-rag 的两级检索能力暴露为 MCP tool。

管线位置：models/law.index + chunk_meta.jsonl
          → [retriever.py 两级检索]
          → [本模块] → MCP 客户端（UUMit / Claude / 任意支持 MCP 的 Agent）

工具清单：
  search_law(query, top_k=3)   两级检索：向量召回 + cross-encoder 重排，返回条款片段
  pipeline_info()              返回索引规模与模型信息（自检/演示用）

设计要点：
- fail-fast：资产加载失败直接终止进程，不带着坏状态对外提供服务
  （与 retriever.load_assets 的对齐校验同一哲学）
- stdout 是 MCP 协议通道，一切日志只准走 stderr
- 模型懒加载：启动即完成 MCP 握手，两个模型推迟到首次工具调用时加载
  （CPU 上 bge-small + bge-reranker-base 约需 1-2 分钟，若在启动时同步加载，
  客户端会因握手超时直接掐掉进程——实测踩坑）
- 离线优先：HF_HUB_OFFLINE=1 时 snapshot_download 直接命中本地缓存
  （HF_HOME 已指向 D:\AI_project\hf_cache）
"""
import json
import os
import sys
from pathlib import Path

# stdout 是协议通道，日志一律走 stderr
def log(msg: str) -> None:
    print(f"[mcp-law] {msg}", file=sys.stderr, flush=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 离线优先：缓存命中就不联网（与 Docker 容器内的运行策略一致）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# HF 缓存兜底：本机约定缓存统一放 <仓库上两级>/hf_cache（即 D:\AI_project\hf_cache，
# 与 Docker 卷挂载同源）。若用户已显式设置 HF_HOME 则不覆盖。
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT.parent.parent / "hf_cache"))

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP(
    "env-law-rag",
    instructions="环境法规智能检索：输入自然语言问题，返回最相关的法规条款片段（带重排分数与条款定位）。",
)

# 懒加载的全局资产（index / meta / model / reranker）
_ASSETS: dict = {}
_LOADED = False


def _ensure_loaded() -> None:
    """首次工具调用时加载检索管线全部资产；失败 fail-fast 抛错给调用方。"""
    global _LOADED
    if _LOADED:
        return
    from retriever import load_assets, load_model, load_reranker

    index_path = PROJECT_ROOT / "models" / "law.index"
    meta_path = PROJECT_ROOT / "data" / "processed" / "chunk_meta.jsonl"

    if not index_path.exists():
        raise FileNotFoundError(f"索引不存在: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"元数据不存在: {meta_path}")

    log("加载 FAISS 索引与元数据 ...")
    index, meta = load_assets(index_path, meta_path)
    log(f"索引规模: {index.ntotal} 条 chunk")

    log("加载向量模型 bge-small-zh-v1.5 ...")
    model = load_model()
    log("加载重排模型 bge-reranker-base ...")
    reranker = load_reranker()

    _ASSETS.update(index=index, meta=meta, model=model, reranker=reranker)
    _LOADED = True
    log("检索管线就绪")


@mcp.tool()
def search_law(query: str, top_k: int = 3) -> str:
    """检索最相关的环境法规条款。

    输入自然语言问题（如"排污许可证有效期多久"），返回经过两级检索
    （向量召回 + cross-encoder 重排）的 top_k 个条款片段，按相关性降序。

    Args:
        query: 自然语言检索问题
        top_k: 返回条数，1-10，默认 3
    """
    from retriever import search

    _ensure_loaded()
    top_k = max(1, min(int(top_k), 10))
    results = search(query, _ASSETS["index"], _ASSETS["meta"],
                     _ASSETS["model"], _ASSETS["reranker"])

    output = []
    for c in results[:top_k]:
        output.append({
            "rank": c.get("rank"),
            "law_name": c.get("law_name"),
            "article": c.get("article"),
            "text": c.get("text"),
            "recall_score": round(c.get("recall_score", 0.0), 4),
            "rerank_score": round(c.get("rerank_score", 0.0), 4),
        })
    return json.dumps({"query": query, "results": output}, ensure_ascii=False, indent=2)


@mcp.tool()
def pipeline_info() -> str:
    """返回检索管线自检信息：索引规模、收录法规数、模型版本。"""
    _ensure_loaded()
    laws = {c.get("law_name") for c in _ASSETS["meta"]}
    return json.dumps({
        "service": "env-law-rag",
        "chunks": _ASSETS["index"].ntotal,
        "law_count": len(laws),
        "law_names": sorted(laws),
        "embed_model": "BAAI/bge-small-zh-v1.5",
        "rerank_model": "BAAI/bge-reranker-base",
        "pipeline": "faiss 向量召回(top-20) → cross-encoder 重排(top-k)",
    }, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    log("MCP Server 启动（stdio 模式），模型将在首次调用时懒加载")
    mcp.run()
