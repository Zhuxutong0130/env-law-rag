r"""双 Agent revise 闭环的真实调用测试。

构造一份带假引用的答案草稿（虚构第九百九十九条 + 编造"吊销许可证罚500万"），
验证：校验 Agent 能抓出幻觉 → 反馈注入重新生成 → 二次校验通过。

运行（需 DEEPSEEK_API_KEY 与本地模型缓存）：
    HF_HUB_OFFLINE=1 HF_HOME=<缓存目录> python tests/test_agent_flow.py
"""

import json
import sys
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import retriever
import agent_flow
from agent_flow import _is_fixable_refuse, _regenerate_with_feedback, verify_answer
from generator import get_client

ROOT = Path(__file__).resolve().parent.parent

HALLUCINATED_DRAFT = (
    "企业超标排污将被吊销排污许可证并处罚款500万元。"
    "（依据：《中华人民共和国大气污染防治法》第九百九十九条）"
)


def main() -> None:
    index, meta = retriever.load_assets(
        ROOT / "models" / "law.index",
        ROOT / "data" / "processed" / "chunk_meta.jsonl")
    model = retriever.load_model()
    reranker = retriever.load_reranker()
    query = "企业超标排污会有什么法律后果"
    chunks = retriever.search(query, index, meta, model, reranker)
    client = get_client()

    v = verify_answer(query, HALLUCINATED_DRAFT, chunks, client)
    print("=== 校验 Agent 对假引用答案的裁决 ===")
    print(json.dumps(v, ensure_ascii=False))
    assert v["verdict"] in ("revise", "refuse"), "假引用未被识破，校验 Agent 失效"

    fixed = _regenerate_with_feedback(
        query, chunks, HALLUCINATED_DRAFT, v["issues"], client)
    print("=== 修改后重答 ===")
    print(f"refused: {fixed['refused']}")
    print(fixed["answer"][:300])

    v2 = verify_answer(query, fixed["answer"], chunks, client)
    print("=== 二次校验 ===")
    print(json.dumps(v2, ensure_ascii=False))
    assert v2["verdict"] == "pass", "重答后仍未通过校验"
    print("REVISE LOOP TEST PASSED")


def test_refuse_downgrade_logic() -> None:
    """refuse 二分处置的逻辑测试（mock，零 API 成本）。

    虚构引用类 refuse -> 降级 revise 重答；越界作答类 refuse -> 终态。
    """
    fake_refuse = {"verdict": "refuse",
                   "issues": ["虚构引用《大气污染防治法》第九百九十九条"]}
    scope_refuse = {"verdict": "refuse",
                    "issues": ["问题超出给定条款能回答的范围，属于越界作答"]}
    assert _is_fixable_refuse(fake_refuse["issues"]) is True
    assert _is_fixable_refuse(scope_refuse["issues"]) is False

    chunks = [{"law_name": "大气污染防治法", "article": "第123条", "text": "..."}]
    with mock.patch.object(agent_flow, "generate_answer") as mg, \
         mock.patch.object(agent_flow, "verify_answer") as mv, \
         mock.patch.object(agent_flow, "_regenerate_with_feedback") as mr:
        # 路径一：虚构引用 refuse → 降级 revise → 重答 → pass
        mg.return_value = {"answer": "草稿", "cited_chunks": [], "refused": False}
        mv.side_effect = [fake_refuse, {"verdict": "pass", "issues": []}]
        mr.return_value = {"answer": "修正", "cited_chunks": [1], "refused": False}
        out = agent_flow.run_dual_agent("q", chunks, client="mock")
        assert out["revised"] and out["verify"]["verdict"] == "pass" and mr.called

        # 路径二：越界 refuse → 终态，不重答
        mg.return_value = {"answer": "强行作答", "cited_chunks": [], "refused": False}
        mv.return_value = scope_refuse
        out = agent_flow.run_dual_agent("q", chunks, client="mock")
        assert out["refused"] and not out["revised"] and not mr.called
    print("REFUSE DOWNGRADE LOGIC TEST PASSED")


if __name__ == "__main__":
    test_refuse_downgrade_logic()
    main()
