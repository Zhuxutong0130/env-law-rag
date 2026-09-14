"""阶段 5 考卷构造：chunk_meta.jsonl → eval/testset_v1.jsonl（50 题）

流程（与 build_dataset.py 同构：先小样、再放量、断点续跑、人工审完才冻结）：
    python eval/build_testset.py plan       # 程序化分层抽样 40 单元（seed 固定）→ _plan.json
    python eval/build_testset.py pilot      # 起草 3 条（易/中/难各一）人肉验 prompt
    python eval/build_testset.py draft      # 全量 40 条起草（断点续跑 _drafts.jsonl）
    python eval/build_testset.py review     # 打印 40 条草稿+条款原文，供逐条审题
    python eval/build_testset.py finalize   # 套用审题表 + 10 拒答 + dedup → testset_v1.jsonl

设计要点：
- 题型配额 义务9/罚则9/职责8/概念8/综合6；综合题为「行为条款+其罚则」配对，一题须命中
  ≥2 条；拒答 10 题全部手写（见 REFUSAL_CASES）。
- 难度配额 易12/中18/难10（6 个综合题天然为“难”）。
- 法规配额受真实定义条款分布约束：噪声14/大气13/水11/环保2（大气法在本 chunk
  集内无“本法所称”定义条，概念题让给噪声法，不拿施行日期条硬凑）。
- GT 冻结纪律：finalize 后 testset_v1.jsonl 不许改；任何迭代重测都用同一份 v1。
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SRC_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generator import get_client, _norm_article  # noqa: E402
from build_dataset import norm_law_name           # noqa: E402

META_PATH = PROJECT_ROOT / "data" / "processed" / "chunk_meta.jsonl"
TRAIN_PATH = PROJECT_ROOT / "data" / "finetune" / "train.jsonl"
OUT_DIR = _SRC_DIR
PLAN_PATH = OUT_DIR / "_plan.json"
DRAFT_PATH = OUT_DIR / "_drafts.jsonl"
TESTSET_PATH = OUT_DIR / "testset_v1.jsonl"

SEED = 20260912
MAX_RETRY = 3
DEDUP_THRESHOLD = 0.50      # eval 题对 train 题的字符 3-gram 包含率超过此值即判泄漏

# ---- 配额 ----
TYPE_ORDER = ["义务", "罚则", "职责", "概念", "综合", "拒答"]
# 四类单条款题的（法规 → 条数）；“综合”走配对，单独配置
CELL_QUOTA = {
    ("义务", "噪声"): 2, ("义务", "大气"): 4, ("义务", "水"): 3,
    ("罚则", "噪声"): 2, ("罚则", "大气"): 4, ("罚则", "水"): 3,
    ("职责", "噪声"): 2, ("职责", "大气"): 3, ("职责", "水"): 2, ("职责", "环保"): 1,
    ("概念", "噪声"): 6, ("概念", "水"): 1, ("概念", "环保"): 1,   # 真定义条全集就 8 条
}
MULTI_PAIR_N = {"噪声": 2, "大气": 2, "水": 2}     # 综合 6
LAW_ORDER = ["噪声", "大气", "水", "环保"]
# 每类题型内部的难度序列（合计 易12 中18 难4；综合 6 全难 → 难合计10）
DIFF_SEQUENCE = {
    "义务": ["易", "易", "易", "中", "中", "中", "中", "难", "难"],
    "罚则": ["易", "易", "易", "中", "中", "中", "中", "难", "难"],
    "职责": ["易", "易", "易", "中", "中", "中", "中", "中"],
    "概念": ["易", "易", "易", "中", "中", "中", "中", "中"],
    "综合": ["难"] * 6,
}

TYPE_GUIDE = {
    "义务": "考“谁应当怎么做/禁止做什么”，市民想知道合法行为边界",
    "罚则": "考违法后果与罚款，必须把条款里的处罚梯度/金额/倍数问全",
    "职责": "考“归谁管/向谁投诉举报/哪个部门履行什么监管职责”",
    "概念": "考法条里的定义与适用范围（“本法所称X是指…”），让被问者解释术语边界",
}
DIFF_GUIDE = {
    "易": "措辞可以贴近法条原文的书面表达（允许共用条款里的规范词，如“向水体排放油类”）",
    "中": "必须彻底口语化：用市民生活场景打比方，问题中不得出现条款原文的关键短语和法言法语",
    "难": "口语化 + 复合情景：一个具体生活事件里同时问到条款中的多个要点（如两种情形、两个梯度、行为+程序），"
          "需要组织多处信息才能答全",
}


# ================= 抽样 =================

def load_chunks() -> list[dict]:
    return [json.loads(l) for l in open(META_PATH, encoding="utf-8") if l.strip()]


def law_key(name: str) -> str:
    n = norm_law_name(name)
    for k, v in (("噪声", "噪声"), ("大气", "大气"), ("水污染", "水"), ("环境", "环保")):
        if k in n:
            return v
    raise ValueError(n)


def classify(c: dict) -> str:
    """条款题型启发式分桶（抽样用，出题后还会人工逐条复核）。"""
    ch = (c.get("chapter") or "").replace(" ", "").replace("　", "")
    t = c["text"]
    if "法律责任" in ch:
        return "罚则"
    # 只认真定义条；“附则”里的施行日期条不算概念（曾误收 §129/§103）
    if ("是指" in t) or ("本法所称" in t) or ("下列用语" in t):
        return "概念"
    # 职责锚点必须是真正的监管职权词，避免“会同市场监督管理部门责令召回”这类企业义务条误入
    gov_anchor = ("监督管理职责", "统一监督管理", "有权", "监督检查", "现场检查",
                  "行政强制", "查封", "举报", "投诉")
    if "主管部门" in t and any(k in t for k in gov_anchor):
        return "职责"
    if any(k in t for k in ("应当", "不得", "必须")):
        return "义务"
    return "其他"


def _trigrams(s: str) -> set[str]:
    s = re.sub(r"\s+", "", s)
    return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else set()


def pick_pairs(chunks: list[dict], used: set[int]) -> list[list[int]]:
    """综合题配对：罚则条 × 同法非罚则条，按字符 trigram 重合挑行为最对应的对。

    确定性算法（无随机）：每部法枚举（罚则条，义务/职责条）候选，按 Jaccard 排序，
    贪心选取 4 个 chunk id 互不重复的 top-N 对。配对质量在 plan/review 阶段人眼复核。
    """
    by_law = {}
    for c in chunks:
        by_law.setdefault(law_key(c["law_name"]), []).append(c)
    pairs = []
    for law in ["噪声", "大气", "水"]:
        cs = by_law[law]
        penalties = [c for c in cs if classify(c) == "罚则" and c["id"] not in used]
        others = [c for c in cs if classify(c) in ("义务", "职责") and c["id"] not in used]
        cand = []
        for p in penalties:
            gp = _trigrams(p["text"])
            for o in others:
                inter = len(gp & _trigrams(o["text"]))
                if inter >= 4:                       # 至少共享 4 个三gram，保证是同一行为域
                    cand.append((inter, p["id"], o["id"]))
        cand.sort(key=lambda x: -x[0])
        picked, local_used = [], set()
        for _, pid, oid in cand:
            if len(picked) >= MULTI_PAIR_N[law]:
                break
            if pid in local_used or oid in local_used:
                continue
            picked.append([oid, pid])               # 顺序：行为条款在前，罚则在后
            local_used.update((pid, oid))
            used.update((pid, oid))
        assert len(picked) == MULTI_PAIR_N[law], f"{law} 配对只找到 {len(picked)} 对"
        pairs.extend([[law, pk] for pk in picked])
    return pairs


def build_plan() -> list[dict]:
    chunks = load_chunks()
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for c in chunks:
        t = classify(c)
        if t in ("义务", "罚则", "职责", "概念"):
            by_cell.setdefault((t, law_key(c["law_name"])), []).append(c)

    used: set[int] = set()
    units = []

    # 先占综合题配对的 chunk，避免单条款题抽到同一条
    raw_pairs = pick_pairs(chunks, used)

    def draw(qtype: str):
        cell = [c for c in by_cell[(qtype, law)] if c["id"] not in used] \
            if (qtype, law) in by_cell else []
        rng = random.Random(f"{SEED}:{qtype}:{law}")
        rng.shuffle(cell)
        return cell

    uid = 1
    for qtype in ["义务", "罚则", "职责", "概念"]:
        type_units = []
        for law in LAW_ORDER:
            n = CELL_QUOTA.get((qtype, law), 0)
            if not n:
                continue
            pool = draw(qtype)
            assert len(pool) >= n, f"{qtype}×{law} 池仅 {len(pool)}，需 {n}"
            for c in pool[:n]:
                used.add(c["id"])
                type_units.append({
                    "unit": f"U{uid:02d}", "type": qtype,
                    "chunk_ids": [c["id"]],
                    "laws": [norm_law_name(c["law_name"])],
                    "articles": [c["article"]],
                })
                uid += 1
        # 难度在题型内部打散后按固定序列落位（不集中在某部法）
        rng = random.Random(f"{SEED}:diff:{qtype}")
        rng.shuffle(type_units)
        for u, d in zip(type_units, DIFF_SEQUENCE[qtype]):
            u["difficulty"] = d
        units.extend(sorted(type_units, key=lambda u: u["unit"]))

    for law, pair in raw_pairs:
        cs = [next(c for c in chunks if c["id"] == cid) for cid in pair]
        units.append({
            "unit": f"U{uid:02d}", "type": "综合", "difficulty": "难",
            "chunk_ids": pair,
            "laws": [norm_law_name(c["law_name"]) for c in cs],
            "articles": [c["article"] for c in cs],
        })
        uid += 1

    assert len(units) == 40
    return units


def cmd_plan(args):
    units = build_plan()
    PLAN_PATH.write_text(json.dumps(units, ensure_ascii=False, indent=2), encoding="utf-8")
    chunks = {c["id"]: c for c in load_chunks()}
    val_ids = set()
    for line in open(PROJECT_ROOT / "data" / "finetune" / "val.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["meta"].get("chunk_id") is not None:
            val_ids.add(r["meta"]["chunk_id"])
    print(f"plan 已写：{PLAN_PATH}（40 单元）")
    for u in units:
        split = "val" if u["chunk_ids"][0] in val_ids else "train-split"
        arts = " + ".join(f"{l[-5:]}{a}" for l, a in zip(u["laws"], u["articles"]))
        print(f"  {u['unit']} {u['type']} {u['difficulty']} chunks={u['chunk_ids']} "
              f"[{split}]  {arts}")


# ================= 出题 =================

DRAFT_TMPL = """你在为环境法规问答评测出一道考题。下面给你真实条款原文（一题可能给两条）。

【题型】{type}：{type_guide}
【难度】{difficulty}：{diff_guide}
{chunks_block}

要求：
1. question：站在普通市民角度提问，自然、具体；难度要求必须严格遵守。
2. answer：只依据上面给出的条款原文作答，禁止补充任何条款之外的信息（不知道实施细则、
   电话号码、实际办事地点，一律不许编）；语言通顺，必要时分点；罚则题必须覆盖全部罚款梯度。
3. answer 结尾另起一行写引用，引用格式严格为：
{cite_lines}
   法律名与条款号必须与上面给出的完全一致；给了两条就必须引用两条。
{multi_extra}
只输出一个 JSON 对象，不要输出任何其他内容：
{{"question": "...", "answer": "..."}}"""


def _chunks_block(unit_chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(unit_chunks, 1):
        lines.append(f"【条款{i}出处】《{norm_law_name(c['law_name'])}》{c['article']}\n"
                     f"【条款{i}原文】{c['text']}")
    return "\n".join(lines)


def draft_one(client, unit: dict, chunk_by_id: dict) -> dict | None:
    cs = [chunk_by_id[cid] for cid in unit["chunk_ids"]]
    cite_lines = "\n".join(f"   （依据：《{norm_law_name(c['law_name'])}》{c['article']}）" for c in cs)
    multi_extra = ("4. 这是【多条款综合题】：question 必须设计成只有同时用上两条（如“这种行为允许吗？"
                   "被抓到会怎样？”= 行为规则 + 罚则）才能答完整。"
                   if unit["type"] == "综合" else "")
    prompt = DRAFT_TMPL.format(
        type=unit["type"], type_guide=TYPE_GUIDE.get(unit["type"], "综合两条以上条款作答"),
        difficulty=unit["difficulty"], diff_guide=DIFF_GUIDE[unit["difficulty"]],
        chunks_block=_chunks_block(cs), cite_lines=cite_lines, multi_extra=multi_extra,
    )
    last_err = None
    for attempt in range(MAX_RETRY):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4, max_tokens=1024,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            q, a = data["question"].strip(), data["answer"].strip()
            if len(q) < 8 or len(a) < 15:
                raise ValueError("question/answer 过短")
            # 防 U30 类空答案：剥掉引用行后正文必须还有实质内容
            body = re.sub(r"（依据：《.+?》第[^）]+条）", "", a).strip()
            if len(body) < 15:
                raise ValueError("答案正文为空（仅引用）")
            # 逐条引用机械校验（法律名双向子串 + 条号归一化）
            cited = re.findall(r"《(.+?)》第(.+?)条", a)
            for c in cs:
                law = norm_law_name(c["law_name"])
                ok = any((law in cl or cl in law) and _norm_article(ca) == _norm_article(c["article"])
                         for cl, ca in cited)
                if not ok:
                    raise ValueError(f"缺少/错误引用 《{law}》{c['article']}")
            return {"unit": unit["unit"], "question": q, "gold_answer": a}
        except Exception as e:  # noqa: BLE001 — 各类 API/解析错统一退避
            last_err = e
            wait = 2 ** attempt
            print(f"    {unit['unit']} 重试 {attempt + 1}/{MAX_RETRY}（{type(e).__name__}: {str(e)[:80]}），{wait}s")
            time.sleep(wait)
    print(f"  ✗ {unit['unit']} 最终失败: {type(last_err).__name__}: {str(last_err)[:100]}")
    return None


def _load_plan() -> list[dict]:
    assert PLAN_PATH.exists(), "先执行 plan"
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _load_drafts() -> dict[str, dict]:
    out = {}
    if DRAFT_PATH.exists():
        for line in open(DRAFT_PATH, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                out[d["unit"]] = d
    return out


def _append_draft(d: dict) -> None:
    with open(DRAFT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")


def cmd_pilot(args):
    units = _load_plan()
    chunk_by_id = {c["id"]: c for c in load_chunks()}
    picks = [next(u for u in units if u["difficulty"] == d and u["type"] != "综合")
             for d in ("易", "中", "难")]
    client = get_client()
    for u in picks:
        d = draft_one(client, u, chunk_by_id)
        print("=" * 78)
        print(f"{u['unit']} {u['type']}/{u['difficulty']} chunks={u['chunk_ids']}")
        if d:
            cs = [chunk_by_id[cid] for cid in u["chunk_ids"]]
            print(_chunks_block(cs)[:500])
            print("Q:", d["question"])
            print("A:", d["gold_answer"])


def cmd_draft(args):
    units = _load_plan()
    chunk_by_id = {c["id"]: c for c in load_chunks()}
    done = _load_drafts()
    todo = [u for u in units if u["unit"] not in done]
    print(f"起草：共 40，已完成 {len(done)}，本次待跑 {len(todo)}")
    client = get_client()
    ok = fail = 0
    t0 = time.time()
    for i, u in enumerate(todo, 1):
        d = draft_one(client, u, chunk_by_id)
        if d:
            _append_draft(d)
            ok += 1
        else:
            fail += 1
        time.sleep(0.3)
        if i % 10 == 0 or i == len(todo):
            print(f"  进度 {i}/{len(todo)} 成功 {ok} 失败 {fail}（{i/(time.time()-t0):.1f} 条/s）")
    print(f"draft 报告：本次成功 {ok}，失败 {fail}。失败可重跑（断点续跑），之后执行 review。")


def cmd_review(args):
    """打印审题工作表：plan 元信息 + 条款原文 + 草稿。审题结论写进 REVIEW 表。"""
    units = _load_plan()
    drafts = _load_drafts()
    chunk_by_id = {c["id"]: c for c in load_chunks()}
    for u in units:
        d = drafts.get(u["unit"])
        print("=" * 78)
        print(f"{u['unit']} 【{u['type']}/{'综合' if u['type']=='综合' else u['difficulty']}】 "
              f"chunks={u['chunk_ids']}")
        for cid in u["chunk_ids"]:
            c = chunk_by_id[cid]
            print(f"  条款《{norm_law_name(c['law_name'])}》{c['article']}：{c['text']}")
        if d is None:
            print("  （尚未起草）")
            continue
        print(f"  Q: {d['question']}")
        print(f"  A: {d['gold_answer']}")


# ================= 人工审题表（逐条审核结论，finalize 强制每条必须有） =================
# verdict: "pass" 直接采用草稿；"edit" 用下面给出的字段覆盖草稿；note 必填（审题痕迹）。
REVIEW: dict[str, dict] = {
    # 人工审题痕迹（2026-09-12 逐条对照条款原文审核）：pass 的只记 note；
    # U30 草稿正文为空被判废，用 gold_answer 手工覆盖。
    "U01": {"note": "审核通过：问题口语化且覆盖建设单位/专业运营单位两主体，答案两阶段义务均忠实噪声法§68，无条款外信息。"},
    "U02": {"note": "审核通过：易题直给，禁高音喇叭广告+其他噪声防污两要点齐全，答案忠实§63。"},
    "U03": {"note": "审核通过：情景化考淘汰设备“不得转让”，答案覆盖产/进/销/用停止义务、工艺采用者、禁止转让三层，忠实大气法§27。"},
    "U04": {"note": "审核通过：政府政策义务条，答案严格复述§32能源结构/煤炭清洁利用等方向，未添加具体指标。"},
    "U05": {"note": "审核通过：易题落在该条唯一硬性句“煤层气排放应符合标准规范”，鼓励性政策未被误写成强制义务。"},
    "U06": {"note": "审核通过：难题五情形（内河/江海直达/远洋船、新建/已建码头、岸电）区分准确，无超纲。"},
    "U07": {"note": "审核通过：排污口设置双规则（环保一般规定+江河湖泊遵守水行政规定）齐全；“不能随意排污、须依规设口”为条文合理转述，未越界。"},
    "U08": {"note": "审核通过：易题，水质检测/不达标处置与报告/通报/供水单位负责四动作逐句忠实水法§71。"},
    "U09": {"note": "审核通过：政府预案职责与供水单位应急方案、事故处置义务分层清楚，忠实水法§79。"},
    "U10": {"note": "审核通过：官员渎职处分条，问题主动诱问“罚款金额倍数”，答案正确回应“仅处分、无罚款”，是有效的防编造考点。"},
    "U11": {"note": "审核通过：罚款梯度（1万-10万、拒不改正可暂停施工）与两类违法施工行为齐全，忠实噪声法§77。"},
    "U12": {"note": "审核通过：两档违法（超标生产/弄虚作假出厂）处罚拆分准确，货值1-3倍、没收销毁、停产、停止车型均忠实大气法§109。"},
    "U13": {"note": "审核通过：易题直引大气法§127刑事责任条，问答对应。"},
    "U14": {"note": "审核通过：答案完整罗列六项行为+2万-20万罚款+拒不改正责令停产整治，篇幅虽长但与§108原文一一对应、零添加。"},
    "U15": {"note": "审核通过：按日连续处罚四种适用情形与“责令改正次日起按原处罚数额”起算规则准确，忠实大气法§123。"},
    "U16": {"note": "审核通过：监管部门/人员渎职处分条，不批许可、接举报不查处等情形齐全，忠实水法§80。"},
    "U17": {"note": "审核通过：§100为法律责任章内损害赔偿纠纷证据条，答案仅答“可委托/应接受/如实提供数据”三要点，未外扩实体赔偿规则。"},
    "U18": {"note": "审核通过：复查→继续违法或拒检→依环保法按日连续处罚链条准确，答案点破按天累加且未编造倍率（条文本身无倍率）。"},
    "U19": {"note": "审核通过：生态环境部门统一监管与住建/公安/交通等分管结构清楚，正确纠正“噪声都归环保局”的误区，忠实噪声法§8。"},
    "U20": {"note": "审核通过：现场检查权、被检查者配合义务、商业秘密保密、两人两证程序全覆盖，忠实噪声法§29。"},
    "U21": {"note": "审核通过：易题问监管主体，生态环境会同四部门监督检查+不合格不得使用，忠实大气法§56。"},
    "U22": {"note": "审核通过：市场监管会同生态环境、覆盖生产进口销售使用四环节、不符标准不得产销使用，忠实大气法§40。"},
    "U23": {"note": "审核通过：四种监督检查手段、被检查者配合义务、商业秘密保护齐全，忠实大气法§29。"},
    "U24": {"note": "审核通过并改写：初稿问法与 train 同 chunk 蒸馏题 3-gram 重叠 0.527 被 dedup 闸拦截，"
                    "query 改用“污水集中处理设施”表述与三问角度重写，gold_answer 补全排放标准首句，内容仍忠实水法§50。",
            "query": "城镇污水集中处理设施把处理完的水往外排，法律对这种出水有什么硬性要求？"
                     "平时到底是谁对出水水质负直接责任？政府部门对设施排的水要不要查、查哪些东西？",
            "gold_answer": "向城镇污水集中处理设施排放水污染物，应当符合国家或者地方规定的水污染物排放标准。"
                           "城镇污水集中处理设施的运营单位，应当对城镇污水集中处理设施的出水水质负责。"
                           "环境保护主管部门应当对城镇污水集中处理设施的出水水质和水量进行监督检查。\n"
                           "（依据：《中华人民共和国水污染防治法》第五十条）"},
    "U25": {"note": "审核通过：公众保护义务与检举权、政府表彰奖励齐全；条文未点名具体受理部门，答案未编造，忠实水法§11。"},
    "U26": {"note": "审核通过：易题直给，中央/地方两级环保部门统一监管，忠实环保法§10。"},
    "U27": {"note": "审核通过：建筑施工噪声“施工过程+干扰生活”两要件拆解准确，反面排除属定义的逻辑推演，无外部知识。"},
    "U28": {"note": "审核通过：五类交通运输工具、运行时产生、干扰周围环境三要件齐全，忠实噪声法§44。"},
    "U29": {"note": "审核通过：噪声与噪声污染双定义、污染两情形（超标或未依法防控+干扰他人）准确，忠实噪声法§2。"},
    "U30": {"note": "审核判废并人工重写：DeepSeek 草稿正文为空、只剩引用行（机械校验已补防线防复发）；保留原问题，"
                    "手工补写工业噪声定义答案，补写后“工业生产活动+干扰周围生活环境”两要件齐全，忠实噪声法§34。",
            "gold_answer": "本法所称工业噪声，是指在工业生产活动中产生的干扰周围生活环境的声音。"
                           "认定工业噪声要同时满足两点：一是声音产生于工业生产活动；二是该声音干扰了周围生活环境。"
                           "因此并不是工厂里的所有声音都算工业噪声，与工业生产活动无关的声音不在该定义范围内。\n"
                           "（依据：《中华人民共和国噪声污染防治法》第三十四条）"},
    "U31": {"note": "审核通过：社会生活噪声“人为活动+排除三类噪声”的定义转述准确，装修/广场舞归类与§82口径一致。"},
    "U32": {"note": "审核通过：夜间时段（22点至次日6点、地方可另定、时长8小时）、噪声敏感建筑物、交通干线三问均落在噪声法§88原文内。"},
    "U33": {"note": "审核通过：水污染/水污染物/有毒污染物三定义准确并附区分小结，未越界答污泥、渔业水体等未问定义，忠实水法§102。"},
    "U34": {"note": "审核通过：环境定义的天然/人工两属性与“含城市和乡村”讲清，回应了“只有深山老林才算”的误区，忠实环保法§2。"},
    "U35": {"note": "审核通过：综合题真需双条——§64行为规则（高音喇叭禁令+广场活动义务）配§82罚则，"
                    "个人200-1000元/单位2000-2万元两档罚款齐全，两条引用齐。"},
    "U36": {"note": "审核通过：综合题真需双条——自行监测/保存记录/公开+重点单位自动监测联网义务，与§76两项罚则一一对应，"
                    "2万-20万、拒不改正限产停产齐全。"},
    "U37": {"note": "审核通过：综合题真需双条——餐饮油烟三项禁令对应大气法§118三档处罚（5千-5万停业整治/1万-10万并关闭/500-2万没收工具），梯度完整。"},
    "U38": {"note": "审核通过：综合题真需双条——施工单位与建设单位义务分列；处罚严格限于围挡防尘、土方遮盖与裸露地面两类，"
                    "未对“缺防尘方案/未公示”编造罚款，分寸准确。"},
    "U39": {"note": "审核通过：综合题真需双条——过驳作业须批义务配水法§90罚则，1万-10万、造成污染2万-20万、逾期代治理费用船舶承担三档齐全。"},
    "U40": {"note": "审核通过：综合题真需双条——双层罐/防渗监测与禁止无防渗沟渠坑塘存废水，对应水法§85第（七）（八）项；"
                    "该 chunk 本身无罚款金额，答案只写“处以罚款”未编造数字，忠实给定文本边界。"},
}


# ================= 拒答 10 题（手写，不出题模型代劳） =================
# 措辞刻意重写、与 build_dataset 的 10 条训练拒答题零重复（dedup 把关）。
REFUSAL_CASES = [
    {"query": "今年的年终奖发了六万元，个人所得税该走什么流程申报退税？",
     "category": "完全无关", "difficulty": "易", "expected": "refuse", "gold_articles": [],
     "gold_answer": "问题超出知识库范围，无法回答。",
     "notes": "人工命题：个税主题，四部环境法均无相关条款，第一层门卫即应拦截；与训练题“个人所得税怎么退税”措辞已重写。"},
    {"query": "驾照在一个记分周期里十二分被扣光了，是不是必须重新考科目一？",
     "category": "完全无关", "difficulty": "易", "expected": "refuse", "gold_articles": [],
     "gold_answer": "问题超出知识库范围，无法回答。",
     "notes": "人工命题：交通管理主题，与训练题“驾驶证扣12分怎么办”句式完全不同，考查门卫对无关主题的稳定性。"},
    {"query": "劳动合同还没到期公司就把我裁掉了，按法律我能拿到几个月工资的赔偿？",
     "category": "完全无关", "difficulty": "易", "expected": "refuse", "gold_articles": [],
     "gold_answer": "问题超出知识库范围，无法回答。",
     "notes": "人工命题：劳动法主题；训练题为“合同到期不续签”，本题改为“未到期被裁+赔偿月数”，情景与问法均不同。"},
    {"query": "网上买的台式电脑到家就不想要了，走七天无理由退货时寄回去的运费该由谁承担？",
     "category": "完全无关", "difficulty": "易", "expected": "refuse", "gold_articles": [],
     "gold_answer": "问题超出知识库范围，无法回答。",
     "notes": "人工命题：消保法主题；把训练题中的“商品”具体化为“台式电脑”，并改变句式，验证 dedup 与门卫。"},
    {"query": "企业年底用不完的碳排放配额，能不能挂到交易平台上卖给别的企业？",
     "category": "语义邻居", "difficulty": "难", "expected": "refuse", "gold_articles": [],
     "gold_answer": "问题超出知识库范围，无法回答。",
     "notes": "人工命题：语义邻居陷阱（大气法有“低碳/排放”词但无碳排放权交易制度）；训练题问“可以在市场上买卖吗”，本题改为“年底富余+挂平台”，必须靠语义判别而非背题。"},
    {"query": "垃圾填埋场积的渗滤液流出来把旁边农田土壤泡坏了，这块地按规定该怎么修复？",
     "category": "语义邻居", "difficulty": "难", "expected": "refuse", "gold_articles": [],
     "gold_answer": "问题超出知识库范围，无法回答。",
     "notes": "人工命题：语义邻居陷阱（水法§40 只管防渗漏/地下水监测，无土壤修复内容）；训练题问“污染土壤怎么治理”，本题加“农田+泡坏+修复”口语情景。"},
    {"query": "山里抓的果子狸想办手续搞人工繁育，法律允许个人养殖这类野生动物吗？",
     "category": "语义邻居", "difficulty": "难", "expected": "refuse", "gold_articles": [],
     "gold_answer": "问题超出知识库范围，无法回答。",
     "notes": "人工命题：语义邻居陷阱（环保法定义条款可能假阳性，野生动物保护法不在库）；训练题为泛问“能人工养殖吗”，本题具体到“果子狸+办手续”。"},
    {"query": "存放油漆稀料这类易燃易爆危化品的仓库，跟旁边居民住宅楼至少要隔多少米才合规？",
     "category": "语义邻居", "difficulty": "难", "expected": "refuse", "gold_articles": [],
     "gold_answer": "问题超出知识库范围，无法回答。",
     "notes": "人工命题：语义邻居陷阱（安全生产/危化品距离标准不在四部法内）；训练题问“离居民区要多远”，本题加“油漆稀料+住宅楼+多少米”，易被大气法仓储条款误命中。"},
    {"query": "楼上邻居半夜十二点拿电钻装修把孩子吓醒了，我该拨打什么号码举报他？",
     "category": "已知误拒", "difficulty": "难",
     "expected": "partial",
     "gold_articles": [{"law": "中华人民共和国噪声污染防治法", "article": "第八十二条"}],
     "gold_answer": "可答部分：已交付使用的住宅楼进行室内装修，应当限制作业时间并采取降噪措施，"
                    "拒不改正的可处罚（《噪声法》第八十二条）。必须拒答部分：具体举报电话号码库内无依据，"
                    "应说明无法提供号码、建议向地方政府指定的部门反映。",
     "notes": "人工命题：阶段3 Gradio 演示真实误拒案例（notes/演示4 截图留档）。expected=partial——"
              "装修时段规定可答、电话号码库外应拒，专测“半拒半答”的边界处理。"},
    {"query": "按噪声污染防治法被罚款后拖着一直不缴，会不会产生滞纳金，按什么比例算？",
     "category": "超纲追问", "difficulty": "难", "expected": "refuse", "gold_articles": [],
     "gold_answer": "问题超出知识库范围，无法回答。",
     "notes": "人工命题：库内话题（噪声罚款）的库外细节（逾期不缴的滞纳金/利息），噪声法无此条款，"
              "考查模型能否抵抗“话题很像库内”的诱惑而不编造比例数字。"},
]


# ================= dedup / 落盘 =================

def train_questions() -> list[str]:
    qs = []
    for line in open(TRAIN_PATH, encoding="utf-8"):
        r = json.loads(line)
        m = re.search(r"问题[：:](.*)$", r["input"], flags=re.S)
        qs.append(m.group(1).strip() if m else r["input"])
    return qs


def dedup_report(eval_qs: list[tuple[str, str]], train_qs: list[str]) -> bool:
    """eval 每题 vs train 全量：字符 3-gram 包含率（共享数/eval 侧 gram 数）。"""
    train_grams = [(q, _trigrams(q)) for q in train_qs]
    scored = []
    for eid, eq in eval_qs:
        eg = _trigrams(eq)
        for tq, tg in train_grams:
            if not eg:
                continue
            score = len(eg & tg) / len(eg)
            scored.append((score, eid, eq, tq))
    scored.sort(key=lambda x: -x[0])
    print("dedup：eval×train 3-gram 包含率 top5（阈值 %.2f）" % DEDUP_THRESHOLD)
    for score, eid, eq, tq in scored[:5]:
        print(f"  {score:.3f}  {eid} 《{eq[:24]}…》 ↔ train《{tq[:24]}…》")
    bad = [x for x in scored if x[0] >= DEDUP_THRESHOLD]
    print(f"超阈值对数：{len(bad)}（验收要求 0）")
    return not bad


def cmd_finalize(args):
    units = _load_plan()
    drafts = _load_drafts()
    chunk_by_id = {c["id"]: c for c in load_chunks()}

    # ---- fail-fast：40 条全部起草 + 全部有人工审题结论 ----
    missing_draft = [u["unit"] for u in units if u["unit"] not in drafts]
    missing_review = [u["unit"] for u in units if not REVIEW.get(u["unit"], {}).get("note")]
    assert not missing_draft, f"尚未起草：{missing_draft}"
    assert not missing_review, f"REVIEW 表缺审题结论：{missing_review}"

    records = []
    qno = 1
    for u in units:
        d = drafts[u["unit"]]
        rv = REVIEW[u["unit"]]
        query = rv.get("query", d["question"])
        gold_answer = rv.get("gold_answer", d["gold_answer"])
        gold_articles = rv.get("gold_articles", [
            {"law": norm_law_name(chunk_by_id[cid]["law_name"]),
             "article": chunk_by_id[cid]["article"]}
            for cid in u["chunk_ids"]
        ])
        # 引用与 gold_articles 再校验一次
        cited = re.findall(r"《(.+?)》第(.+?)条", gold_answer)
        for g in gold_articles:
            assert any((g["law"] in cl or cl in g["law"])
                       and _norm_article(ca) == _norm_article(g["article"])
                       for cl, ca in cited), f"{u['unit']} gold_answer 引用缺 {g}"
        records.append({
            "id": f"Q{qno:02d}",
            "query": query,
            "type": u["type"],
            "difficulty": u["difficulty"],
            "gold_articles": gold_articles,
            "gold_answer": gold_answer,
            "expected": "answer",
            "notes": rv["note"],
        })
        qno += 1

    # ---- 拒答 10 题 ----
    for rc in REFUSAL_CASES:
        records.append({
            "id": f"Q{qno:02d}",
            "query": rc["query"],
            "type": "拒答",
            "difficulty": rc["difficulty"],
            "gold_articles": rc["gold_articles"],
            "gold_answer": rc["gold_answer"],
            "expected": rc["expected"],
            "notes": rc["notes"],
        })
        qno += 1
    assert len(records) == 50

    # ---- dedup 闸门 ----
    ok = dedup_report([(r["id"], r["query"]) for r in records], train_questions())
    assert ok,("存在与 train.jsonl 3-gram 重叠超阈值的题，请重写对应 query 后再 finalize")

    TESTSET_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8")

    # ---- 分层统计 ----
    from collections import Counter
    print("\n" + "=" * 78)
    print(f"已冻结：{TESTSET_PATH}（{len(records)} 题，GT 起不许改）")
    type_diff = Counter((r["type"], r["difficulty"]) for r in records)
    print("\n题型 × 难度：")
    print(f"{'题型':<6}{'易':>4}{'中':>4}{'难':>4}{'合计':>5}")
    for t in TYPE_ORDER:
        row = [type_diff.get((t, d), 0) for d in ("易", "中", "难")]
        print(f"{t:<7}{row[0]:>3}{row[1]:>4}{row[2]:>4}{sum(row):>5}")
    print(f"{'合计':<7}"
          f"{sum(type_diff.get((t,'易'),0) for t in TYPE_ORDER):>3}"
          f"{sum(type_diff.get((t,'中'),0) for t in TYPE_ORDER):>4}"
          f"{sum(type_diff.get((t,'难'),0) for t in TYPE_ORDER):>4}{len(records):>5}")

    law_c = Counter()
    for r in records:
        if r["gold_articles"]:
            for g in r["gold_articles"]:
                law_c[g["law"].replace("中华人民共和国", "")] += 1
        else:
            law_c["(拒答无条款)"] += 1
    print("\n法规分布（按 gold_articles 计数）：", dict(law_c))
    print("expected 分布：", dict(Counter(r["expected"] for r in records)))
    print("拒答分类：", dict(Counter(rc["category"] for rc in REFUSAL_CASES)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="阶段5 评测考卷构造")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    sub.add_parser("pilot")
    sub.add_parser("draft")
    sub.add_parser("review")
    sub.add_parser("finalize")
    args = parser.parse_args()
    {"plan": cmd_plan, "pilot": cmd_pilot, "draft": cmd_draft,
     "review": cmd_review, "finalize": cmd_finalize}[args.cmd](args)
