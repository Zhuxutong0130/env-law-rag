"""微调数据集构造：chunk_meta.jsonl → DeepSeek 蒸馏 → train.jsonl + val.jsonl。

管线位置：data/processed/chunk_meta.jsonl → [本模块] → data/finetune/{train,val}.jsonl
蒸馏策略：让 DeepSeek 看着单个真实条款出一道市民口吻的题并按推理期同款规范作答，
output 全部基于真实条款（无幻觉），instruction/input 格式与 pipeline 推理调用一致
（直接复用 generator.SYSTEM_PROMPT 与 build_user_message），保证训练-推理同分布。

用法（省钱的正确姿势：先 pilot 人肉验收，再放量）：
    python src/build_dataset.py pilot       # 只蒸馏 5 条，打印全文人工检查，不写盘
    python src/build_dataset.py build       # 全量蒸馏，断点续跑（_checkpoint.jsonl）
    python src/build_dataset.py finalize    # 从 checkpoint 切分落盘 + 统计（不调 API）
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

_SRC_DIR = Path(__file__).parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from openai import APIError, APITimeoutError, RateLimitError

from generator import (
    SYSTEM_PROMPT,
    build_user_message,
    get_client,
    _norm_article,
)

PROJECT_ROOT = _SRC_DIR.parent
META_PATH = PROJECT_ROOT / "data" / "processed" / "chunk_meta.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "finetune"
CHECKPOINT_PATH = OUT_DIR / "_checkpoint.jsonl"

VAL_RATIO = 0.10          # 每部法尾部 10% 进 val
MAX_RETRY = 3             # 单条最大尝试次数（含首次）
SLEEP_BETWEEN = 0.3       # 调用间隔，基础限速
RANDOM_SEED = 42

# 拒答样本：库外问题。两类都要有——
# (a) 完全无关（第一层分数门卫就会拦）；
# (b) 语义邻居（能过门卫，必须靠模型看着无关条款拒，对应推理期第二层 LLM 拒答）。
# output 固定拒答话术，与 generator.REFUSAL 逻辑一致；input 仍带真实检索到的 top-3
# 无关条款，让被蒸馏模型学到"检索回来的条款不相关时必须拒"。
REFUSAL_QUERIES = [
    "个人所得税怎么退税",
    "驾驶证扣12分怎么办",
    "公司注册资本最低要多少",
    "小区电梯困人归谁负责",
    "劳动合同到期公司不续签有补偿吗",
    "网购商品七天无理由退货的运费谁出",
    "垃圾填埋场渗滤液污染土壤怎么治理",     # 语义邻居：环保邻近主题，库里无土壤法
    "碳排放配额可以在市场上买卖吗",          # 语义邻居：命中"低碳出行"假阳性
    "野生动物能人工养殖吗",                  # 语义邻居：命中环保法定义条款假阳性
    "危险化学品仓库离居民区要多远",          # 语义邻居：安全生产，库里无相关法
]
N_REFUSAL_VAL = 2       # 10 条拒答：8 train + 2 val
REFUSAL_OUTPUT = "问题超出知识库范围，无法回答。"

DISTILL_PROMPT_TMPL = """你在为一个环境法规问答模型构造训练数据。下面给你一条真实法规条款，请完成两件事：

【条款出处】《{law}》{article}
【条款原文】{text}

要求：
1. question：模拟普通市民真实咨询口吻出一个问题（口语化、具体，不要照搬条款原文措辞），
   问题必须能仅依据该条款作答。题型在以下类型中选一个：违法后果/罚款多少、谁来管/向谁投诉、
   应当怎么做/义务是什么、概念界定。不同条款尽量换题型。
2. answer：只依据条款原文作答，禁止补充条款之外的任何信息；语言通顺，必要时分点；
   结尾必须另起引用，且引用格式严格为：
   （依据：《{law}》{article}）
   法律名与条款号必须与上面给出的完全一致。
3. 若条款是某长条文的片段（内容不完整），question 只问该片段覆盖的内容，不得要求片段外的信息。

只输出一个 JSON 对象，不要输出任何其他内容：
{{"question": "...", "answer": "..."}}"""


# ---------------- 基础工具 ----------------

def norm_law_name(name: str) -> str:
    """meta 里节选文件名带了后缀（环境保护法_节选），引用法规时去掉。"""
    return name.replace("_节选", "")


def load_chunks() -> list[dict]:
    """读 chunk_meta.jsonl。逐行读 + utf-8，行内字段以 chunk id 为稳定主键。"""
    chunks = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def split_by_law(chunks: list[dict], val_ratio: float = VAL_RATIO):
    """按法规切分：每部法【尾部】val_ratio 的 chunk 进 val，其余进 train。

    为什么不是随机切分（验收要求说清理由）：
    1. chunk 在文件里按条款号顺序排列，同一章的条款主题聚类（如"法律责任"章全是罚则）。
       随机切分会把同章相邻条款同时撒进 train/val，val 考的是模型"背过的主题"，
       指标虚高——泄漏的是主题分布而不只是单条文本。
    2. 取每部法尾部条款，模拟"法规后续修订新增条款"这一真实泛化场景，
       val 考的是对同分布但未见过的条款的迁移能力，更诚实。
    3. 分层（四部法都切）保证最小的法（环保法仅 20 条）在 val 里也有样本，
       不会被随机切分"恰好切没"。
    """
    by_law = {}
    for c in chunks:
        by_law.setdefault(c["law_name"], []).append(c)

    train, val = [], []
    for law, group in by_law.items():           # 文件内本就按 id/条款顺序
        group = sorted(group, key=lambda x: x["id"])
        n_val = max(1, round(len(group) * val_ratio))
        train.extend(group[:-n_val])
        val.extend(group[-n_val:])
    return train, val


# ---------------- 蒸馏单条 ----------------

def _validate_output(answer: str, law: str, article: str) -> bool:
    """机械校验：引用存在、法律名与条款号都对得上当前 chunk。

    不只抽查 5 条——全量校验，从源头保证"output 引用确实对应该 chunk"。
    """
    import re
    m = re.findall(r"《(.+?)》第(.+?)条", answer)
    if not m:
        return False
    for cited_law, cited_art in m:
        # 法律名双向子串兼容（全称/简称），条款号归一化后精确匹配
        art_ok = _norm_article(cited_art) == _norm_article(article)
        law_ok = law in cited_law or cited_law in law
        if art_ok and law_ok:
            return True
    return False


def build_qa_pair(client, chunk: dict) -> dict | None:
    """单个 chunk → 一条训练 record；API 偶发失败返回 None（不中断批量）。

    失败处理：限流(429)/超时/服务端错误/JSON 解析失败/引用校验失败
    都走指数退避重试（2^n 秒），MAX_RETRY 次后放弃。
    """
    law = norm_law_name(chunk["law_name"])
    article = chunk["article"] or "（未编号条款）"
    prompt = DISTILL_PROMPT_TMPL.format(law=law, article=article, text=chunk["text"])

    last_err = None
    for attempt in range(MAX_RETRY):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,            # 出题要有多样性，比推理期略高
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            question, answer = data["question"].strip(), data["answer"].strip()
            if not question or not answer:
                raise ValueError("question/answer 为空")
            if not _validate_output(answer, law, article):
                raise ValueError(f"引用校验失败: {answer[-60:]}")

            # input 与推理期完全同构：参考资料 + 分隔线 + 问题
            ctx_chunk = dict(chunk)
            ctx_chunk["law_name"] = law
            return {
                "instruction": SYSTEM_PROMPT,
                "input": build_user_message(question, [ctx_chunk]),
                "output": answer,
                "meta": {
                    "chunk_id": chunk["id"], "law_name": law,
                    "article": article, "kind": "answer",
                },
            }
        except (RateLimitError, APITimeoutError, APIError,
                json.JSONDecodeError, ValueError, ConnectionError) as e:
            last_err = e
            wait = 2 ** attempt
            print(f"    重试 {attempt + 1}/{MAX_RETRY}（{type(e).__name__}: {str(e)[:80]}），{wait}s 后")
            time.sleep(wait)

    print(f"  ✗ chunk {chunk['id']} 最终失败: {type(last_err).__name__}: {str(last_err)[:100]}")
    return None


# ---------------- 拒答样本（零 API 调用） ----------------

def build_refusal_records(n_val: int = N_REFUSAL_VAL):
    """库外问题 + 真实检索回来的无关 top-3 → 固定拒答话术。

    复用本地检索栈（不调 LLM API），让拒答训练样本与推理期第二层
    "看着无关条款也要拒"的输入完全同构。
    """
    from embedding import load_model
    from retriever import load_assets, load_reranker, recall, rerank

    index, meta = load_assets(
        PROJECT_ROOT / "models" / "law.index",
        PROJECT_ROOT / "data" / "processed" / "chunk_meta.jsonl",
    )
    model, reranker = load_model(), load_reranker()

    records = []
    for q in REFUSAL_QUERIES:
        top3 = rerank(q, recall(q, index, meta, model), reranker)
        for c in top3:
            c["law_name"] = norm_law_name(c["law_name"])
        records.append({
            "instruction": SYSTEM_PROMPT,
            "input": build_user_message(q, top3),
            "output": REFUSAL_OUTPUT,
            "meta": {"chunk_id": None, "law_name": None, "article": None,
                     "kind": "refusal", "query": q,
                     "top1_score": top3[0]["rerank_score"]},
        })

    # 固定前 n_val 条进 val（含两类：前几条完全无关 + 后几条语义邻居由 build 时保证顺序）
    random.Random(RANDOM_SEED).shuffle(records)
    return records[n_val:], records[:n_val]


# ---------------- 断点续跑 ----------------

def load_checkpoint() -> dict[int, dict]:
    done = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done[rec["meta"]["chunk_id"]] = rec
    return done


def append_checkpoint(rec: dict) -> None:
    with open(CHECKPOINT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------- 三个子命令 ----------------

def cmd_pilot(args):
    """5 条小样本人肉验收：每部法中部 1 条 + 全局最短 1 条（最容易翻车的片段）。"""
    chunks = load_chunks()
    by_law = {}
    for c in chunks:
        by_law.setdefault(c["law_name"], []).append(c)
    picked = [sorted(v, key=lambda x: x["id"])[len(v) // 2] for v in by_law.values()]
    picked.append(min(chunks, key=lambda x: x["char_len"]))

    client = get_client()
    ok = 0
    for i, c in enumerate(picked, 1):
        print("=" * 78)
        print(f"pilot {i}/5  chunk_id={c['id']} char_len={c['char_len']} "
              f"《{norm_law_name(c['law_name'])}》{c['article']}")
        rec = build_qa_pair(client, c)
        if rec is None:
            print("  → 失败")
            continue
        ok += 1
        print(f"【input】\n{rec['input']}")
        print(f"【output】\n{rec['output']}")
    print("=" * 78)
    print(f"pilot 完成：{ok}/5 成功。请人肉检查 question 口语化、answer 不超纲、"
          f"引用格式严格为（依据：《XX法》第X条）。合格后执行: python src/build_dataset.py build")


def cmd_build(args):
    """全量蒸馏，按 chunk_id 断点续跑；失败计数并最后报告。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    done = load_checkpoint()
    todo = [c for c in chunks if c["id"] not in done]
    print(f"全量蒸馏：共 {len(chunks)} 条，已完成 {len(done)}，本次待跑 {len(todo)}")

    client = get_client()
    success, failed = 0, 0
    failed_ids = []
    t0 = time.time()
    for i, c in enumerate(todo, 1):
        rec = build_qa_pair(client, c)
        if rec:
            append_checkpoint(rec)
            success += 1
        else:
            failed += 1
            failed_ids.append(c["id"])
        time.sleep(SLEEP_BETWEEN)
        if i % 20 == 0 or i == len(todo):
            rate = i / (time.time() - t0)
            print(f"  进度 {i}/{len(todo)}  本次成功 {success} 失败 {failed}  "
                  f"({rate:.1f} 条/s，累计 checkpoint {len(done) + success})")

    print("=" * 70)
    print(f"build 报告：本次成功 {success}，失败 {failed}，失败 ids={failed_ids}")
    print(f"checkpoint 总计 {len(load_checkpoint())}/{len(chunks)}。"
          f"失败可直接重跑 build（断点续跑只补缺失），最后执行 finalize。")


def cmd_finalize(args):
    """从 checkpoint 组装 train/val（追加拒答样本），落盘并打印统计。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    train_chunks, val_chunks = split_by_law(chunks)
    train_ids, val_ids = {c["id"] for c in train_chunks}, {c["id"] for c in val_chunks}

    done = load_checkpoint()
    train_recs = [done[i] for i in sorted(train_ids) if i in done]
    val_recs = [done[i] for i in sorted(val_ids) if i in done]
    missing = sorted((train_ids | val_ids) - set(done.keys()))

    refusal_train, refusal_val = build_refusal_records()
    train_recs.extend(refusal_train)
    val_recs.extend(refusal_val)

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(train_recs)
    rng.shuffle(val_recs)

    train_path, val_path = OUT_DIR / "train.jsonl", OUT_DIR / "val.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _print_stats(train_recs, val_recs, missing, refusal_train, refusal_val,
                 train_path, val_path)
    _spot_check(val_recs)


def _print_stats(train_recs, val_recs, missing, refusal_train, refusal_val,
                 train_path, val_path):
    print("=" * 70)
    print(f"train.jsonl: {len(train_recs)} 条  ({train_path})")
    print(f"val.jsonl:   {len(val_recs)} 条  ({val_path})")
    if missing:
        print(f"⚠️ 缺失 {len(missing)} 条（API 失败未补跑）: {missing[:20]}")
    else:
        print("条款样本无缺失（344/344）。")

    for name, recs in [("train", train_recs), ("val", val_recs)]:
        n_ref = sum(1 for r in recs if r["meta"]["kind"] == "refusal")
        laws = {}
        lens = []
        for r in recs:
            if r["meta"]["law_name"]:
                laws[r["meta"]["law_name"]] = laws.get(r["meta"]["law_name"], 0) + 1
            lens.append(len(r["output"]))
        print(f"\n[{name}] 拒答样本 {n_ref} 条；平均 output 长度 "
              f"{sum(lens) / len(lens):.0f} 字；法规分布:")
        for law, n in sorted(laws.items(), key=lambda x: -x[1]):
            print(f"    {law}: {n}")


def _spot_check(recs: list[dict], n: int = 5):
    """从 val 里抽 5 条条款样本（不抽拒答），打印 output 供核对。"""
    import re
    answers = [r for r in recs if r["meta"]["kind"] == "answer"]
    picks = random.Random(RANDOM_SEED).sample(answers, min(n, len(answers)))
    print("\n" + "=" * 70)
    print(f"抽查 {len(picks)} 条 output（引用格式 + 条款对应）：")
    for r in picks:
        m = re.findall(r"《(.+?)》第(.+?)条", r["output"])
        cited = "；".join(f"《{a}》第{b}条" for a, b in m)
        meta_ok = any(
            r["meta"]["law_name"] in a or a in r["meta"]["law_name"]
            for a, _ in m
        ) and any(_norm_article(b) == _norm_article(r["meta"]["article"]) for _, b in m)
        print(f"  chunk_id={r['meta']['chunk_id']:>3} 期望《{r['meta']['law_name']}》"
              f"{r['meta']['article']} → 引用: {cited}  {'✓' if meta_ok else '✗'}")
        print(f"    {r['output'][:100]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="微调数据集蒸馏管线")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pilot")
    sub.add_parser("build")
    sub.add_parser("finalize")
    args = parser.parse_args()

    {"pilot": cmd_pilot, "build": cmd_build, "finalize": cmd_finalize}[args.cmd](args)
