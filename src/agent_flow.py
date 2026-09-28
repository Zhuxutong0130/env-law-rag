r"""双 Agent 协同流水线：检索 Agent → 生成 → 引用校验 Agent。

系统的第 2 次能力跃迁（单 Agent 工具调用 → 双 Agent 协同）：

  1. 检索 Agent：retriever.search 的职责视角（向量召回 + 重排 + 质量门卫）
  2. 生成器：generator.generate_answer（原有逻辑，含两层拒答）
  3. 校验 Agent（本模块新增）：LLM 作为裁判，对草稿答案逐项核查——
       a) 引用真实性：答案里的《法律名》+条款号能否对回检索结果
       b) 内容忠实性：答案表述是否忠于被引条款原文（反幻觉）
       c) 越界作答：问题超出资料范围却强行回答
     裁决三态：pass / revise（附修改意见）/ refuse

编排规则（run_dual_agent）：
  生成 → 校验 →
    pass   → 直接返回
    refuse → 返回拒答
    revise → 把校验意见注入对话，重新生成一次（预算 1 次，控成本）→ 再校验

设计要点：
- 返回结构兼容旧路径 generate_answer（answer / cited_chunks / refused），
  另加 verify 字段（校验 Agent 的原始裁决）供评测与展示
- 校验 Agent 自身故障（网络/解析失败）采取 fail-open：放行答案并记录日志。
  理由：校验器是增强层不是门卫，它坏了不应该把整个服务打死——
  旧路径的构造性校验（extract_cited_chunks）仍然兜底引用真实性
- 修改预算默认 1 次：每多一轮 revise 翻一倍 API 成本，收益递减
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generator import (  # noqa: E402
    REFUSAL_MARKERS,
    SYSTEM_PROMPT,
    build_user_message,
    extract_cited_chunks,
    generate_answer,
    get_client,
)

VERIFIER_SYSTEM_PROMPT = """你是环境法规问答系统的引用校验裁判。给你：用户问题、检索到的法规条款（编号）、以及待核查的答案草稿。

逐项核查：
1. 引用真实性：答案中每个《法律名》第X条引用，是否能在给定条款中找到对应条目
2. 内容忠实性：答案的实质内容是否忠于被引条款原文，有无编造条款号、编造数字、夸大或歪曲原文
3. 越界作答：问题是否超出给定条款能回答的范围，答案却强行作答

只输出一个 JSON 对象，不要输出任何其他文字：
{"verdict": "pass|revise|refuse", "issues": ["问题描述", ...]}

裁决标准：
- pass：引用全部属实，内容忠实
- revise：引用属实但表述有偏差、遗漏关键限定条件或结论不完整（issues 里写清改什么）
- refuse：存在虚构引用、编造内容，或问题明显超出资料范围却作答
"""


def _fmt_chunks(chunks: list[dict], max_chars: int = 220) -> str:
    """把检索条款编号列出，正文截断（校验裁判只需要条款号与要点）。"""
    lines = []
    for i, c in enumerate(chunks, 1):
        text = (c.get("text") or "").replace("\n", " ")
        lines.append(f"[{i}] {c.get('law_name')} {c.get('article') or ''}: {text[:max_chars]}")
    return "\n".join(lines)


def build_verify_user_message(query: str, answer: str, chunks: list[dict]) -> str:
    return (
        f"【用户问题】\n{query}\n\n"
        f"【检索到的条款】\n{_fmt_chunks(chunks)}\n\n"
        f"【待核查答案】\n{answer}\n\n"
        "请按规定格式输出 JSON 裁决。"
    )


def _parse_verdict(raw: str) -> dict:
    """解析裁判输出；容忍 ```json 围栏。解析失败按 fail-open 处理。"""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text.strip())
    verdict = data.get("verdict")
    if verdict not in ("pass", "revise", "refuse"):
        raise ValueError(f"非法 verdict: {verdict!r}")
    return {"verdict": verdict, "issues": list(data.get("issues") or [])}


def verify_answer(query: str, answer: str, chunks: list[dict],
                  client=None) -> dict:
    """校验 Agent：LLM 逐项核查答案，返回 {"verdict", "issues"}。"""
    client = client or get_client()
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": build_verify_user_message(query, answer, chunks)},
        ],
        temperature=0.0,          # 裁判要确定性，比生成端更严格
        max_tokens=512,
    )
    try:
        return _parse_verdict(resp.choices[0].message.content)
    except (json.JSONDecodeError, ValueError, IndexError) as exc:
        print(f"[校验Agent] 输出解析失败（fail-open 放行）: {exc}")
        return {"verdict": "pass", "issues": [f"verifier_parse_error: {exc}"]}


def _regenerate_with_feedback(query: str, chunks: list[dict], draft: str,
                              issues: list[str], client) -> dict:
    """带校验意见的重新生成（第二轮对话：草稿 + 修改要求）。"""
    feedback = "\n".join(f"- {s}" for s in issues)
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(query, chunks)},
            {"role": "assistant", "content": draft},
            {"role": "user", "content":
                f"你的回答存在以下问题：\n{feedback}\n"
                "请针对以上问题修改并重新回答。保持引用格式《法律名》第X条，"
                "引用必须来自给定条款；若条款不足以回答，请明确说明并拒答。"},
        ],
        temperature=0.1,
        max_tokens=1024,
    )
    answer = resp.choices[0].message.content.strip()
    refused = any(marker in answer for marker in REFUSAL_MARKERS)
    cited = [] if refused else extract_cited_chunks(answer, chunks)
    return {"answer": answer, "cited_chunks": cited, "refused": refused}


def _is_fixable_refuse(issues: list[str]) -> bool:
    """refuse 降级判断：虚构引用/编造内容可修（降级 revise），越界作答不可修。

    理由：虚构引用时正确证据就在检索结果里，重答一次大概率修复；
    越界作答是知识库本身的边界，重答只会再编一次。
    """
    text = "".join(issues)
    return not any(kw in text for kw in ("越界", "超出", "超纲", "范围外"))


def run_dual_agent(query: str, chunks: list[dict], client=None,
                   max_revisions: int = 1) -> dict:
    """双 Agent 编排主入口。

    返回结构与 generate_answer 兼容，另加：
      verify:  最终一轮校验裁决 {"verdict", "issues"}
      revised: 是否触发过修改重答

    refuse 二分处置（v2 迭代）：
      虚构引用类 refuse → 降级为 revise（证据在手，重答可修）
      越界作答类 refuse  → 终态拒答（重答无意义）
    """
    client = client or get_client()

    # 检索 Agent 的质量门卫已内嵌在 generate_answer 第一层拒答里
    result = generate_answer(query, chunks, client)
    if result["refused"]:
        result["verify"] = {"verdict": "refuse", "issues": ["生成端已拒答"]}
        result["revised"] = False
        return result

    revised = False
    for _ in range(max_revisions + 1):
        verdict = verify_answer(query, result["answer"], chunks, client)
        if verdict["verdict"] == "pass":
            break
        if verdict["verdict"] == "refuse":
            if _is_fixable_refuse(verdict["issues"]) and not revised:
                # 虚构引用类 refuse：证据就在 chunks 里，降级 revise 重答
                verdict = {"verdict": "revise",
                           "issues": verdict["issues"] + ["（降级）引用造假可修，重答"]}
                result = _regenerate_with_feedback(
                    query, chunks, result["answer"], verdict["issues"], client)
                revised = True
                if result["refused"]:
                    verdict = {"verdict": "refuse", "issues": ["重答后生成端拒答"]}
                    break
                continue
            # 越界类 refuse 或降级重答后再次 refuse：终态
            result.update(
                answer="问题超出知识库范围，无法回答。",
                cited_chunks=[], refused=True,
            )
            break
        # revise：还有预算就重答一轮，否则带着 issues 放行
        if revised is False and max_revisions > 0:
            result = _regenerate_with_feedback(
                query, chunks, result["answer"], verdict["issues"], client)
            revised = True
            if result["refused"]:
                verdict = {"verdict": "refuse", "issues": ["重答后生成端拒答"]}
                break
        else:
            break

    result["verify"] = verdict
    result["revised"] = revised
    return result


if __name__ == "__main__":
    # 命令行自测：python agent_flow.py "问题"
    import retriever

    query = sys.argv[1] if len(sys.argv) > 1 else "排污许可证有效期是多久"
    index, meta = retriever.load_assets(
        PROJECT_ROOT / "models" / "law.index",
        PROJECT_ROOT / "data" / "processed" / "chunk_meta.jsonl")
    model = retriever.load_model()
    reranker = retriever.load_reranker()
    chunks = retriever.search(query, index, meta, model, reranker)
    out = run_dual_agent(query, chunks)
    print(json.dumps(out, ensure_ascii=False, indent=2)[:2000])
