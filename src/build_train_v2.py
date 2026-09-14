"""构建 v2 训练集（短板① C 臂漏拒的修复，第 2 课作业二）。

v1 的两个结构缺陷（定位结论，见 eval/评测报告.md）：
  A. 拒答负例仅 8 条，且全部是"词汇零重叠"型全超纲题——半相关负例为零，
     模型学不到"上下文看似相关但所问不在库内→仍拒"的内容级规则；
  B. 答案正例 310 条全部是"单条款输入"，拒答负例 8 条全部是"3 条款输入"，
     推理期却是 3 条款输入的作答/拒答双模式——3 条款格式在训练里只与拒答绑定。

v2 = 310 条 v1 原样 + 24 条新型难负例（3 条款真实检索干扰 + 拒答）
              + 80 条 3 条款重渲染正例（gold + 同法邻条 + 异法噪声，答案不变）。
上采样由 4 降为 2（train_lora.py 的 REFUSAL_UPSAMPLE 环境变量）：
负例翻 4 倍后 ×4 会让有效拒答占比冲到 ~26%，有过度拒答风险，×2 约 13%。

防泄漏纪律：24 条难负例全部为新编问题，不与考卷 testset_v1 的 Q41-Q50 同题/同问点；
每条配 must-have-absent 关键词表并在全语料上核验"所问确实不可答"。

用法：python src/build_train_v2.py    # 产出 data/finetune/train_v2.jsonl
"""
import json
import random
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(_SRC_DIR))

from generator import SYSTEM_PROMPT, build_user_message  # noqa: E402
from build_dataset import REFUSAL_OUTPUT, norm_law_name  # noqa: E402

PROJECT_ROOT = _SRC_DIR.parent
TRAIN_V1 = PROJECT_ROOT / "data" / "finetune" / "train.jsonl"
CHUNKS = PROJECT_ROOT / "data" / "processed" / "chunk_meta.jsonl"
OUT = PROJECT_ROOT / "data" / "finetune" / "train_v2.jsonl"

# ---- 24 条新编难负例：(问题, 类型, 所问关键词——这些词若带答案语义地出现在语料里则该题不可用作负例) ----
HARD_NEGATIVES = [
    # 全超纲（8 条，环境词汇零重叠）
    ("小区物业费涨价需要多少比例的业主同意才合规？", "全超纲", ["物业费", "业主同意"]),
    ("银行房贷我想提前还清，会不会收违约金？", "全超纲", ["违约金", "提前还"]),
    ("我的商标注册申请被驳回了，复审流程怎么走？", "全超纲", ["商标", "复审"]),
    ("快递包裹丢失了，按什么标准赔偿？", "全超纲", ["快递", "赔偿标准"]),
    ("公司一直不给缴社保，我该去哪个部门投诉？", "全超纲", ["社保", "投诉"]),
    ("高铁上遇到霸座的，铁路部门会怎么处罚？", "全超纲", ["霸座", "铁路"]),
    ("网约车司机绕路多收费，怎么投诉退款？", "全超纲", ["网约车", "绕路"]),
    ("租房没到期房东非要赶我走，我能要求什么赔偿？", "全超纲", ["房东", "赶"]),
    # 半相关·环境话题但所问不在库内（10 条，表面词汇与库内条款重叠）
    ("企业环保信用评价被评为红色等级，会有什么后果？", "半相关", ["信用评价", "红色等级"]),
    ("排污权可以拿去银行抵押贷款吗？", "半相关", ["抵押", "贷款"]),
    ("环境侵权打官司的诉讼时效是几年？", "半相关", ["诉讼时效"]),
    ("办环评报告表大概要交多少审批费用？", "半相关", ["审批费", "费用"]),
    ("举报企业偷排有奖励金吗，按什么标准发？", "半相关", ["奖励"]),
    ("污染环境罪最重能判几年有期徒刑？", "半相关", ["有期徒刑", "判"]),
    ("环保罚款钱不够，能申请分期缴纳吗？", "半相关", ["分期缴纳"]),
    ("我的鱼塘被上游工厂毒死了鱼，打官司请律师的费用谁出？", "半相关", ["律师"]),
    ("我家附近KTV半夜音响太大，我能自己进店把他们的电闸拉了吗？", "半相关", ["电闸", "断电"]),
    ("机动车年检时尾气检测不合格，已交的检测费能退吗？", "半相关", ["检测费"]),
    # 半相关·跨部门法混合（6 条，环境词开头但落点在别的法律）
    ("邻居装修电钻吵得不行，我能报警让警察把他拘留吗？", "半相关", ["拘留"]),
    ("工厂被责令停产整治了，停产期间的员工工资谁来发？", "半相关", ["工资"]),
    ("养猪场粪污直排被处罚了，还能申请养殖补贴吗？", "半相关", ["补贴"]),
    ("光污染扰民去法院起诉，立案需要什么材料？", "半相关", ["立案", "光污染"]),
    ("小作坊无证生产食品还偷排污水，会吊销食品经营许可证吗？", "半相关", ["食品经营许可证", "吊销"]),
    ("排污不达标的企业，环保局能直接冻结它的银行账户吗？", "半相关", ["冻结", "银行账户"]),
]

N_RECTX = 80  # 3 条款重渲染正例条数


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main() -> None:
    rng = random.Random(42)
    v1 = load_jsonl(TRAIN_V1)
    chunks = load_jsonl(CHUNKS)
    chunk_by_id = {c["id"]: c for c in chunks}
    corpus = [(c["law_name"], c.get("article") or "", c["text"]) for c in chunks]

    # ---- 0) 所问关键词核验：负例的"问点"不得在语料中有可答案（语义级人工复核，机器只给线索）----
    print("== 难负例问点核验（关键词在语料中的出现位置，仅作人工复核线索）==")
    for q, tag, kws in HARD_NEGATIVES:
        hits = []
        for kw in kws:
            for law, art, text in corpus:
                if kw in text:
                    hits.append(f"{law[-6:]}§{art}:{kw}")
        flag = "⚠ 请人工确认非答案" if hits else "OK"
        print(f"  [{flag}] {q[:30]} -> {hits[:4] if hits else '无'}")

    # ---- 1) 难负例：真实检索 top-3 干扰上下文 + 拒答 ----
    from embedding import load_model
    from retriever import load_assets, load_reranker, recall, rerank

    index, meta = load_assets(
        PROJECT_ROOT / "models" / "law.index",
        CHUNKS,
    )
    model, reranker = load_model(), load_reranker()

    neg_records = []
    print("\n== 难负例检索干扰上下文预览 ==")
    for q, tag, _ in HARD_NEGATIVES:
        top3 = rerank(q, recall(q, index, meta, model), reranker)
        for c in top3:
            c["law_name"] = norm_law_name(c["law_name"])
        neg_records.append({
            "instruction": SYSTEM_PROMPT,
            "input": build_user_message(q, top3),
            "output": REFUSAL_OUTPUT,
            "meta": {"chunk_id": None, "law_name": None, "article": None,
                     "kind": "refusal", "query": q, "source": f"v2_{tag}",
                     "top1_score": top3[0]["rerank_score"]},
        })
        ctx = " / ".join(f"{c['law_name'][-6:]}§{c['article']}({c['rerank_score']:.2f})" for c in top3)
        print(f"  {q[:26]:<28} top3 = {ctx}")

    # ---- 2) 3 条款重渲染正例：gold + 同法邻条 + 异法噪声，随机位置，答案不变 ----
    answers = [r for r in v1 if r["meta"]["kind"] != "refusal"]
    picked = rng.sample(answers, N_RECTX)
    rectx = []
    for r in picked:
        gold = chunk_by_id[r["meta"]["chunk_id"]]
        # v1 答案样本 meta 无 query 字段，从 input 尾部取回（格式固定为 build_user_message）
        query = r["input"].rsplit("问题：", 1)[1].strip()
        same_law = [c for c in chunks if c["id"] != gold["id"]
                    and norm_law_name(c["law_name"]) == norm_law_name(gold["law_name"])]
        other_law = [c for c in chunks if norm_law_name(c["law_name"]) != norm_law_name(gold["law_name"])]
        d1 = rng.choice(same_law) if same_law and rng.random() < 0.7 else rng.choice(other_law)
        d2 = rng.choice(other_law)
        ctx = [dict(gold), d1, d2]
        rng.shuffle(ctx)
        for c in ctx:
            c["law_name"] = norm_law_name(c["law_name"])
        rectx.append({
            "instruction": r["instruction"],
            "input": build_user_message(query, ctx),
            "output": r["output"],
            "meta": {**r["meta"], "source": "v2_rectx3"},
        })

    out = v1 + neg_records + rectx
    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ref = sum(1 for r in out if r["meta"]["kind"] == "refusal")
    print(f"\n== train_v2.jsonl 组成 ==")
    print(f"  v1 原样 {len(v1)}（其中答案 {len(answers)} + 拒答 {len(v1) - len(answers)}）")
    print(f"  + 新难负例 {len(neg_records)}（全超纲 8 / 半相关 16）")
    print(f"  + 3 条款重渲染正例 {len(rectx)}")
    print(f"  = 共 {len(out)} 条（拒答原始 {n_ref} 条；×2 上采样后有效占比 "
          f"{n_ref * 2 / (len(answers) + len(rectx) + n_ref * 2) * 100:.1f}%）")
    print(f"  已写入 {OUT}")


if __name__ == "__main__":
    main()
