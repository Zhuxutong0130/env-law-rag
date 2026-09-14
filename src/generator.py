"""生成模块：检索结果 + 用户问题 → 带条款引用的答案（或拒答）。

管线位置：retriever 的 top-k chunks + query → [本模块] → answer
生成器可替换：本课用 DeepSeek API，阶段 4 换本地微调模型。
"""
import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

from embedding import load_model
from retriever import load_assets, load_reranker, search, FINAL_K

SYSTEM_PROMPT = """你是环境法规咨询助手，仅依据用户提供的法规条款回答问题。
规则：
1. 只使用参考资料中的条款作答，禁止使用任何条款之外的知识，禁止推测。
2. 每个结论后注明依据，格式：（依据：《中华人民共和国大气污染防治法》第四十八条）
   ——法律名必须与参考资料中出现的法律名完全一致。
3. 判断拒答的标准是条款内容与问题的实质相关性：
   - 若资料中的条款正是针对问题所涉行为或情形的规定（哪怕只覆盖罚则、义务的一面），
     必须据此作答并注明依据，不得拒绝；
   - 问题常用口语描述行为，条款用法律术语规定同一行为（如"工厂半夜施工扰民"
     对应"夜间在噪声敏感建筑物集中区域进行产生噪声的建筑施工作业"），
     术语表述不完全一致不代表无关，视为实质相关，必须作答；
   - 若条款只是与问题共享某些字面词语（如"机动车""低碳"）而内容并非针对问题所涉行为，
     视为不足以回答，只输出一句话：问题超出知识库范围，无法回答。
   此时禁止编造条款、禁止给出任何条款之外的建议。
4. 引用条款原文时一字不改。
"""

# 拒答判定阈值（rerank top-1 分数低于它直接拒答，不花 API 钱）。
# 【实验依据】18 个 query 的 rerank top-1 分数分布（python generator.py experiment）：
#   库内 10 个: min=0.477(环评) max=0.994 mean=0.764
#   库外  8 个: 无关类 0.006~0.027；语义邻居类 0.19~0.783（野保0.19/垃圾渗滤液0.35/
#               电梯0.42/驾照0.51/碳配额0.78——被"低碳""机动车驾驶"等关键词带偏的假阳性）
# 两分布部分重叠，任何单一阈值都无法完美切分。取 0.45 的理由：
#   1. 0 误伤：库内最低分 0.477 > 0.45，10 个库内问题全部放行（margin 0.027 偏薄，
#      若日后库内 query 被误拒，优先回查此值）；
#   2. 低分垃圾全挡：6/8 库外问题（0.006~0.417）在 API 调用前拦截，零成本；
#   3. 语义邻居（碳配额 0.783、驾照 0.515）分数门卫无法分离——条款里"低碳出行"
#      "机动车排放检验"与问题字面相近但内容无关，这类硬案例交给第二层 LLM 拒答
#      （系统提示明确"参考资料不足以回答→只输出拒答话术"）。
#   这正是两层拒答设计的意义：分数门卫管明显垃圾（便宜），LLM 管语义陷阱（可靠）。
REJECT_THRESHOLD = 0.45

# 灰色带宽度：top1 分数落在 [REJECT_THRESHOLD, REJECT_THRESHOLD+0.10] 时打边界日志。
# 灰色带不改变拒答行为，只做可观测性——分数贴线的 query 是阈值是否合理的哨兵样本。
GRAY_BAND_WIDTH = 0.10

# 调用后拒答特征：LLM 按系统提示的固定话术拒答时会出现这些短语
REFUSAL_MARKERS = ["超出知识库范围", "无法回答", "无法依据"]

# 引用格式：（依据：《XXX》第XX条）
CITATION_RE = re.compile(r"《(.+?)》第(.+?)条")


def _read_api_key() -> str | None:
    """读 API key：先进程环境变量，Windows 下再兜底读用户级注册表。

    为什么有第二层：Windows 用户级环境变量只有"设置之后新开的进程"才继承，
    老终端窗口里的 os.environ 看不到它。注册表（HKCU\\Environment）是
    操作系统的用户配置存储，从那里读 key 依然不算硬编码——代码全文
    grep 不到密钥明文，key 始终只存在 OS 配置里。读到的值回填进程环境，
    后续调用不用重复读注册表。
    """
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                val, _ = winreg.QueryValueEx(k, "DEEPSEEK_API_KEY")
                if val:
                    os.environ["DEEPSEEK_API_KEY"] = val
                return val or None
        except OSError:
            return None
    return None


def get_client() -> OpenAI:
    """创建 DeepSeek 客户端（OpenAI 兼容协议）。

    API key 只从环境变量/用户级注册表读，绝不硬编码——代码里 grep 不到密钥明文。
    """
    api_key = _read_api_key()
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未设置。请先在环境变量中配置，不要硬编码到代码里。"
        )
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def format_context(chunks: list[dict]) -> str:
    """把 top-3 chunks 格式化成 Prompt 里的'参考资料'段落。

    设计要点：
    - 每条前加 [1] [2] [3] 编号，让模型引用时可以指代
    - 保留 law_name / article，引用出处信息在检索结果里现成有
    """
    lines = []
    for i, c in enumerate(chunks, 1):
        article = c.get("article") or "（未编号条款）"
        lines.append(f"[{i}] 《{c['law_name']}》{article}：{c['text']}")
    return "\n".join(lines)


def build_user_message(query: str, chunks: list[dict]) -> str:
    """组装用户消息：参考资料 + 问题，用分隔线隔开，防止问题混进条款。"""
    return f"参考资料：\n{format_context(chunks)}\n\n---\n\n问题：{query}"


def _norm_article(a: str) -> str:
    """条款号归一化：去掉"第"前缀和"条"后缀。

    正则从答案里捕获的是"八十三"，meta 里存的是"第八十三条"，
    归一化后都变成"八十三"才能对上。
    """
    a = a.strip()
    if a.startswith("第"):
        a = a[1:]
    if a.endswith("条"):
        a = a[:-1]
    return a


def extract_cited_chunks(answer: str, chunks: list[dict]) -> list[dict]:
    """从答案里解析引用，只保留确实在检索结果里的条款。

    验收标准"引用的条款确实在检索结果里"由此构造性保证：
    答案里的《法律名》+ 条款号必须能对回 chunks 里的某一条才算 cited。
    法律名做双向子串匹配——meta 存全称"中华人民共和国大气污染防治法"，
    模型可能引用简称"大气污染防治法"，两者都能对上。
    """
    key_map = {}
    for c in chunks:
        key_map[(c["law_name"], c.get("article"))] = c

    cited, seen = [], set()
    for law, art in CITATION_RE.findall(answer):
        art = _norm_article(art)
        for (full_law, full_art), chunk in key_map.items():
            if full_art is None or _norm_article(full_art) != art:
                continue
            # 双向子串：简称对全称 / 全称对简称 都算命中
            if law in full_law or full_law in law:
                if chunk["id"] not in seen:
                    seen.add(chunk["id"])
                    cited.append(chunk)
                break
    return cited


def generate_answer(query: str, chunks: list[dict], client: OpenAI | None = None) -> dict:
    """主入口。返回 {"answer": str, "cited_chunks": [...], "refused": bool}。

    拒答判定两层：
    1. 调用前：rerank 最高分 < REJECT_THRESHOLD → 直接拒，不花 API 钱
    2. 调用后：LLM 回答中带拒答特征 → 标记 refused=True
    """
    # 第一层：检索质量门卫（省 API 钱）
    top1_score = chunks[0].get("rerank_score", 0.0) if chunks else 0.0
    if not chunks or top1_score < REJECT_THRESHOLD:
        return {
            "answer": "问题超出知识库范围，无法回答。",
            "cited_chunks": [],
            "refused": True,
        }
    # 灰色带监控：分数贴着阈值的 query 检索质量存疑，作答/拒答靠 LLM 二次判断。
    # 长期统计这条日志，如果灰色带误拒/误答频发，就该回查 REJECT_THRESHOLD 取值。
    if top1_score <= REJECT_THRESHOLD + GRAY_BAND_WIDTH:
        print(f"[边界日志] top1 rerank={top1_score:.4f} 落在灰色带 "
              f"[{REJECT_THRESHOLD}, {REJECT_THRESHOLD + GRAY_BAND_WIDTH:.2f}]，"
              f"query={query!r}，交由 LLM 二次判断")

    # 调 DeepSeek API
    client = client or get_client()
    resp = client.chat.completions.create(
        model="deepseek-chat",        # V3 系列，便宜够用
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(query, chunks)},
        ],
        temperature=0.1,              # 法规问答要确定性，温度调低
        max_tokens=1024,
    )
    answer = resp.choices[0].message.content.strip()

    # 第二层：LLM 自己拒答了（条款不足以回答）
    refused = any(marker in answer for marker in REFUSAL_MARKERS)
    cited = [] if refused else extract_cited_chunks(answer, chunks)
    return {"answer": answer, "cited_chunks": cited, "refused": refused}


# ---------------- 自测 / 实验 ----------------

# 库外问题：4 个邻近主题（环保但库内没这部法）+ 4 个完全无关
OUT_OF_KB_QUERIES = [
    "垃圾填埋场渗滤液污染土壤怎么治理",     # 邻近：土壤污染防治法，不在库内
    "碳排放配额可以在市场上买卖吗",         # 邻近：碳交易，不在库内
    "野生动物能人工养殖吗",                 # 邻近：野保法，不在库内
    "危险化学品仓库离居民区要多远",         # 邻近：安全生产条例，不在库内
    "个人所得税怎么退税",                   # 无关
    "驾驶证扣12分怎么办",                   # 无关
    "公司注册资本最低要多少",               # 无关
    "小区电梯困人归谁负责",                 # 无关
]


def _load_everything():
    project_root = Path(__file__).parent.parent
    index, meta = load_assets(
        project_root / "models" / "law.index",
        project_root / "data" / "processed" / "chunk_meta.jsonl",
    )
    return index, meta, load_model(), load_reranker()


def _run_experiment():
    """阈值实验：库内问题 vs 库外问题的 rerank top-1 分数分布。"""
    project_root = Path(__file__).parent.parent
    index, meta, model, reranker = _load_everything()

    in_kb = []
    with open(project_root / "eval" / "test_queries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                in_kb.append(json.loads(line)["query"])

    def top1_score(query: str) -> float:
        results = search(query, index, meta, model, reranker)
        return results[0]["rerank_score"], results[0]

    print("=" * 76)
    print("实验：rerank top-1 分数分布（库内 vs 库外）")
    print("=" * 76)

    in_scores, out_scores = [], []
    print(f"\n{'query':<30} {'top-1 分数':>10}  命中的条款")
    print("-" * 76)
    for q in in_kb:
        s, top = top1_score(q)
        in_scores.append(s)
        print(f"{q[:28]:<30} {s:>10.4f}  《{top['law_name']}》{top.get('article')}")
    for q in OUT_OF_KB_QUERIES:
        s, top = top1_score(q)
        out_scores.append(s)
        print(f"{q[:28]:<30} {s:>10.4f}  《{top['law_name']}》{top.get('article')}")

    in_sorted = sorted(in_scores, reverse=True)
    out_sorted = sorted(out_scores, reverse=True)
    print("\n库内分数（降序）:", [f"{s:.3f}" for s in in_sorted])
    print("库外分数（降序）:", [f"{s:.3f}" for s in out_sorted])
    print(f"\n库内  min={min(in_scores):.4f}  max={max(in_scores):.4f}  mean={sum(in_scores)/len(in_scores):.4f}")
    print(f"库外  min={min(out_scores):.4f}  max={max(out_scores):.4f}  mean={sum(out_scores)/len(out_scores):.4f}")

    if min(in_scores) > max(out_scores):
        gap_low, gap_high = max(out_scores), min(in_scores)
        print(f"\n两分布完全分离！安全带 = ({gap_low:.4f}, {gap_high:.4f})")
        print(f"建议阈值取中点: {(gap_low + gap_high) / 2:.4f}")
    else:
        print("\n两分布有重叠，取误伤率最低的切点（偏保守，宁可错拒不可错答）")


def _run_selftest():
    """自测：3 个库内问题（正常回答+引用）+ 2 个库外问题（拒答）。"""
    index, meta, model, reranker = _load_everything()
    client = get_client()

    in_kb_cases = [
        "工厂半夜施工扰民怎么办",
        "往河里倒工业废水有什么后果",
        "燃煤电厂的废气要怎么处理",
    ]
    out_kb_cases = [
        "个人所得税怎么退税？",
        "驾驶证扣12分怎么办？",
    ]

    for i, query in enumerate(in_kb_cases + out_kb_cases, 1):
        tag = "库内" if i <= len(in_kb_cases) else "库外"
        results = search(query, index, meta, model, reranker)
        out = generate_answer(query, results, client)

        print("=" * 76)
        print(f"自测 {i} [{tag}] {query}")
        print(f"top-1 rerank 分数 = {results[0]['rerank_score']:.4f}（阈值 {REJECT_THRESHOLD}）")
        print(f"refused = {out['refused']}")
        print(f"answer  = {out['answer']}")
        cited_str = ", ".join(f"《{c['law_name']}》{c['article']}" for c in out["cited_chunks"])
        print(f"cited_chunks = [{cited_str}]")
        if not out["refused"]:
            # 验证引用确实在检索结果里（注意 all([]) 是空真值，必须先判非空）
            in_results = {c["id"] for c in results}
            all_valid = bool(out["cited_chunks"]) and all(
                c["id"] in in_results for c in out["cited_chunks"]
            )
            print(f"引用非空且均在检索结果里: {'✓' if all_valid else '✗'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "experiment":
        _run_experiment()
    else:
        _run_selftest()
