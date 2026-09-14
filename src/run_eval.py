"""阶段 5：三臂消融评测 + RAGAS 裁判。

用法：
  python src/run_eval.py --arm api  --limit 5      # 试水（先生成、后裁判）
  python src/run_eval.py --arm rag
  python src/run_eval.py --arm lora
  python src/run_eval.py --arm rag --judge-only    # 只补跑 RAGAS 裁判
  python src/run_eval.py --summary                 # 读三臂结果打印总表

三臂（变量控制：同一冻结考卷 testset_v1.jsonl）：
  A api ：无检索，DeepSeek 裸答。同一套 SYSTEM_PROMPT，但用户消息里没有“参考资料”。
  B rag ：两级检索（召回20→重排3）+ DeepSeek，含分数门卫（现网管线，等价 RAGPipeline）。
  C lora：同样两级检索喂给本地 LoRA（DeepSeek-R1-Distill-1.5B + adapter），
          不走分数门卫——拒答完全由微调模型自己决定，否则与 B 的门卫行为不可区分，
          就测不到“×4 上采样教会它拒答”的效果。

生成参数三臂尽量一致：贪心解码（API 侧 temperature=0.1 近贪心），上限 1024 token。

指标：
  hit@3 / full_hit ：top-3 检索与 gold_articles 有交集 / gold 全部命中（仅 B/C，仅答题）
  引用正确率       ：答案里解析出的《法》第x条 ⊆ gold_articles（非空才算对），
                     幻觉引用（引了 gold 外条款）另外逐条计数
  拒答准确率       ：refuse 题看漏拒、answer 题看误拒，分开统计；partial 题(Q49)标 manual
  RAGAS            ：faithfulness / answer_relevancy / context_precision
                     —— 只对 expected=answer 的 40 题跑（拒答题没有可评忠实度的内容）；
                     A 臂无检索，faithfulness 以 gold 条款原文为参照语境（测裸答幻觉）；
                     context_precision 是纯检索侧指标，B/C 检索栈与输入完全相同，
                     值必然逐题相等，只在 B 臂花钱跑、C 臂记“-”。
"""
import argparse
import json
import math
import os
import sys
import time
import types
from pathlib import Path

_SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(_SRC_DIR))

# 必须在 import 任何项目模块（generator→embedding→huggingface_hub）之前设置：
# huggingface_hub 的离线常量在 import 时固化，晚于它再 setdefault 不生效。
# 三臂所需 HF 资产（base/bge/reranker/adapter）本机都有完整缓存，评测全程离线；
# DeepSeek 走 openai 客户端，与 HF 无关。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = _SRC_DIR.parent
TESTSET_PATH = PROJECT_ROOT / "eval" / "testset_v1.jsonl"
META_PATH = PROJECT_ROOT / "data" / "processed" / "chunk_meta.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval"

# 消融实验统一解码规格（API 臂）。LoRA 臂 do_sample=False 对齐“贪心”。
GEN_TEMPERATURE = 0.1
GEN_MAX_TOKENS = 1024
LORA_MAX_NEW_TOKENS = 1024

ARMS = ("api", "rag", "lora")
ARM_LABEL = {"api": "A api", "rag": "B rag", "lora": "C lora"}

from generator import (  # noqa: E402
    CITATION_RE, REFUSAL_MARKERS, SYSTEM_PROMPT, _norm_article,
    build_user_message, generate_answer, get_client, _read_api_key,
)
from build_dataset import norm_law_name  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402（ragas 的硬依赖，顶层导入无妨）


# ================= 数据装载 =================

def load_testset() -> list[dict]:
    return [json.loads(l) for l in open(TESTSET_PATH, encoding="utf-8") if l.strip()]


def load_chunks() -> list[dict]:
    return [json.loads(l) for l in open(META_PATH, encoding="utf-8") if l.strip()]


def gold_article_index(chunks: list[dict]) -> dict[tuple, list[int]]:
    """(归一化法名, 归一化条号) → 该条款的全部 chunk id（长条文可能切成多 chunk）。"""
    idx: dict[tuple, list[int]] = {}
    for c in chunks:
        if not c.get("article"):
            continue
        key = (norm_law_name(c["law_name"]), _norm_article(c["article"]))
        idx.setdefault(key, []).append(c["id"])
    return idx


def gold_chunk_ids(rec: dict, art_idx: dict[tuple, list[int]]) -> tuple[set, list]:
    """返回 (gold 覆盖到的 chunk id 集合, 未解析到 chunk 的 gold 条款列表)。"""
    ids, missing = set(), []
    for g in rec["gold_articles"]:
        key = (norm_law_name(g["law"]), _norm_article(g["article"]))
        cids = art_idx.get(key)
        if cids:
            ids.update(cids)
        else:
            missing.append(g)
    return ids, missing


def parse_citations(text: str) -> list[tuple[str, str]]:
    """从答案文本解析全部引用（不要求在检索结果里——A 臂没有检索结果）。

    返回去重后的 [(原始法名, 归一化条号), ...]，保持出现顺序。
    """
    out, seen = [], set()
    for law, art in CITATION_RE.findall(text):
        key = (law, _norm_article(art))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def citation_match(law: str, art_norm: str, gold: list[dict]) -> bool:
    """引用 ↔ gold 条款匹配：法名双向子串 + 条号归一化相等。"""
    return any(
        (law in g["law"] or g["law"] in law) and _norm_article(g["article"]) == art_norm
        for g in gold
    )


def is_refusal_text(text: str) -> bool:
    return any(marker in text for marker in REFUSAL_MARKERS)


# ================= 三个实验臂 =================

class ApiArm:
    """A 臂：无检索裸答。"""

    name = "api"

    def __init__(self):
        self.client = get_client()

    def answer(self, query: str) -> dict:
        resp = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"问题：{query}"},
            ],
            temperature=GEN_TEMPERATURE,
            max_tokens=GEN_MAX_TOKENS,
        )
        text = resp.choices[0].message.content.strip()
        return {"answer": text, "refused": is_refusal_text(text), "retrieved": None}


class RagArm:
    """B 臂：现网 RAG 管线（分数门卫 + DeepSeek 两层拒答）。"""

    name = "rag"

    def __init__(self):
        from embedding import load_model
        from retriever import load_assets, load_reranker, search
        self._search = search
        self.index, self.meta = load_assets(
            PROJECT_ROOT / "models" / "law.index", META_PATH)
        self.embed_model = load_model()
        self.reranker = load_reranker()
        self.client = get_client()

    def answer(self, query: str) -> dict:
        top3 = self._search(query, self.index, self.meta,
                            self.embed_model, self.reranker)
        out = generate_answer(query, top3, self.client)
        retrieved = [{"id": c["id"], "law_name": c["law_name"],
                      "article": c.get("article"),
                      "rerank_score": round(c["rerank_score"], 4),
                      "text": c["text"]} for c in top3]
        return {"answer": out["answer"], "refused": out["refused"],
                "retrieved": retrieved}


class LoraArm:
    """C 臂：同一套检索喂本地 LoRA；无分数门卫，拒答由模型自行决定。"""

    name = "lora"

    THINK_CLOSE_ID = 151649  # </think>

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        from embedding import load_model
        from retriever import load_assets, load_reranker, search
        from train_lora import ADAPTER_OUT, MODEL_NAME, encode_text

        self._torch = torch
        self._encode_text = encode_text
        self._search = search

        self.index, self.meta = load_assets(
            PROJECT_ROOT / "models" / "law.index", META_PATH)
        self.embed_model = load_model()
        self.reranker = load_reranker()

        print(f"[lora] 加载 base {MODEL_NAME} + adapter {ADAPTER_OUT} ...")
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16, device_map="cuda").eval()
        self.model = PeftModel.from_pretrained(base, str(ADAPTER_OUT)).eval()

    def answer(self, query: str) -> dict:
        top3 = self._search(query, self.index, self.meta,
                            self.embed_model, self.reranker)
        # 与 test_lora.py 一致：喂给模型前把 _节选 后缀清掉（提示词内法名必须干净）
        for c in top3:
            c["law_name"] = norm_law_name(c["law_name"])

        prompt_text = self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": build_user_message(query, top3)}],
            tokenize=False, add_generation_prompt=True)
        ids = self._encode_text(self.tok, prompt_text)
        torch = self._torch
        with torch.no_grad():
            y = self.model.generate(
                torch.tensor([ids], device=self.model.device),
                max_new_tokens=LORA_MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=self.tok.eos_token_id)
        new_ids = y[0][len(ids):].tolist()
        # 训练时立即闭合 think 块；保险起见仍按 </think> 切，只取正式作答
        if self.THINK_CLOSE_ID in new_ids:
            i = new_ids.index(self.THINK_CLOSE_ID)
            new_ids = new_ids[i + 1:]
        text = self.tok.decode(new_ids, skip_special_tokens=True).strip()

        retrieved = [{"id": c["id"], "law_name": c["law_name"],
                      "article": c.get("article"),
                      "rerank_score": round(c["rerank_score"], 4),
                      "text": c["text"]} for c in top3]
        return {"answer": text, "refused": is_refusal_text(text),
                "retrieved": retrieved}


def build_arm(name: str):
    return {"api": ApiArm, "rag": RagArm, "lora": LoraArm}[name]()


# ================= 逐题自动指标 =================

def score_record(rec: dict, gold_ids: set, art_idx: dict) -> dict:
    """在生成结果上算自动指标，返回要并入结果文件的字段。"""
    expected = rec["expected"]
    answer = rec["answer"]
    retrieved = rec["retrieved"]
    is_manual = expected == "partial"

    # ---- 检索命中（B/C；拒答题无 gold，不参与）----
    hit = full_hit = None
    if retrieved is not None and rec["gold_articles"]:
        got = {c["id"] for c in retrieved}
        hit = bool(got & gold_ids)
        full_hit = gold_ids.issubset(got)

    # ---- 引用解析（A 臂引的是参数记忆里的条款，可能全是幻觉）----
    cited = parse_citations(answer) if not rec["refused"] else []
    hallucinated = [[law, art] for law, art in cited
                    if not citation_match(law, art, rec["gold_articles"])]
    if is_manual or rec["refused"] or expected == "refuse":
        cite_correct = None  # 拒答不该有引用；partial 人工评
    else:
        cite_correct = bool(cited) and not hallucinated

    # ---- 拒答判定 ----
    if expected == "refuse":
        refuse_correct = rec["refused"]
    elif expected == "answer":
        refuse_correct = not rec["refused"]
    else:
        refuse_correct = None  # partial → manual

    return {
        "hit": hit, "full_hit": full_hit,
        "cited": [[l, a] for l, a in cited],
        "hallucinated_cites": hallucinated,
        "cite_correct": cite_correct,
        "refuse_correct": refuse_correct,
        "manual": is_manual,
    }


# ================= RAGAS 裁判（延迟导入 + vertexai shim）=================

def _install_ragas_shim() -> None:
    """ragas 0.4.3 顶层强导 `langchain_community.chat_models.vertexai.ChatVertexAI`，
    而新版 langchain-community(0.4.x) 已把该子模块拆走。ragas 只把它放进 isinstance
    类型表白名单，从不实例化——注入同名占位模块即可，裁判走 ChatOpenAI 不受影响。"""
    import importlib.util
    try:
        if importlib.util.find_spec("langchain_community.chat_models.vertexai"):
            return
    except ModuleNotFoundError:
        pass
    mod = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # 仅类型令牌
        pass

    mod.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = mod


class _BGEEmbeddings(Embeddings):
    """把本地 bge-small-zh SentenceTransformer 适配成 LangChain Embeddings。

    answer_relevancy 需要 embedder（把“由答案反推的问题”与原问题编码后算余弦）。
    DeepSeek 不提供 embeddings 接口，复用项目本地的 bge，评测全程不把问题文本
    发给任何云端 embedding 服务。首次用时才加载，api 臂也只多花 ~100MB 显存/内存。
    """
    _model = None

    def __init__(self):
        if _BGEEmbeddings._model is None:
            from embedding import load_model
            _BGEEmbeddings._model = load_model()
        self.m = _BGEEmbeddings._model

    def embed_documents(self, texts):
        return self.m.encode(texts, normalize_embeddings=True,
                             show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.m.encode([text], normalize_embeddings=True,
                             show_progress_bar=False)[0].tolist()


def judge_contexts(arm: str, gen_rec: dict, gold_texts: list[str]) -> list[str]:
    """裁判用 contexts：B/C 用真实检索 top-3 文本；A 臂用 gold 条款原文
    （无检索可评时，faithfulness 退化为“裸答是否忠于标准答案条款”）。"""
    if arm == "api":
        return gold_texts
    return [c["text"] for c in gen_rec["retrieved"]]


def run_ragas(arm: str, gen_records: list[dict], testset: list[dict],
              chunks: list[dict]) -> None:
    """只对 expected=answer 的题跑 RAGAS，分数原地写回 gen_records["metrics"]。

    排除逻辑：9 道 refuse 题的“无法回答”无法评忠实度，喂进去只会产生噪声分；
    1 道 partial(Q49) 走人工，也不进裁判。expected=answer 但被误拒的题保留——
     canned 拒答话术拿低分正是“误答代价”的量化信号。
    """
    _install_ragas_shim()
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    import warnings
    warnings.simplefilter("ignore", DeprecationWarning)  # ragas 0.4 的 collections/wrapper 弃用提示
    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy, context_precision, faithfulness,
    )
    from ragas.run_config import RunConfig

    by_id = {r["id"]: r for r in testset}
    chunk_by_id = {c["id"]: c for c in chunks}

    need = [r for r in gen_records
            if by_id[r["id"]]["expected"] == "answer"
            and not _metrics_complete(arm, r)]
    if not need:
        print("[ragas] 裁判分已齐全，跳过")
        return

    api_key = _read_api_key()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 未设置，裁判 LLM 无法初始化")
    # bypass_n=True：DeepSeek 只支持 n=1，answer_relevancy 要 n=3 个反推问题时
    # 由 ragas 改发 3 次独立 n=1 请求（而不是一次 n=3，后者直接 400）
    judge = LangchainLLMWrapper(ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=api_key,
        timeout=120,
    ), bypass_n=True)
    # 经验：metric 级别显式绑定，防止 ragas 回退到默认 OpenAI embeddings
    emb = LangchainEmbeddingsWrapper(_BGEEmbeddings())

    rows = []
    for r in need:
        gold = by_id[r["id"]]
        gold_texts = [chunk_by_id[cid]["text"]
                      for cid in _gold_ids_one(gold, chunks)]
        ctx = judge_contexts(arm, r, gold_texts)
        rows.append({
            "question": r["query"],
            "answer": r["answer"],
            "contexts": ctx,
            "ground_truth": gold["gold_answer"],
        })

    all_metrics = {"faithfulness": faithfulness,
                   "answer_relevancy": answer_relevancy,
                   "context_precision": context_precision}
    required = ["faithfulness", "answer_relevancy"]
    if arm in ("rag", "lora"):
        required.append("context_precision")
    # 只补缺失指标：--judge-only 再次进入时，已有分数不重算、不重复花钱
    missing = [k for k in required
               if any((r.get("metrics") or {}).get(k) is None for r in need)]
    metrics = [all_metrics[k] for k in missing]
    for m in metrics:
        m.llm = judge
    answer_relevancy.embeddings = emb

    print(f"[ragas] {arm} 臂裁判 {len(rows)} 题，指标："
          f"{[m.name for m in metrics]}（DeepSeek 裁判 + 本地 bge 编码）")
    result = evaluate(
        Dataset.from_list(rows),
        metrics=metrics,
        llm=judge,
        embeddings=emb,
        run_config=RunConfig(timeout=120, max_retries=3, max_workers=4),
        raise_exceptions=False,
    )
    df = result.to_pandas()

    for r, row in zip(need, df.itertuples(index=False)):
        d = r.setdefault("metrics", {})
        for m in metrics:
            val = getattr(row, m.name, math.nan)
            d[m.name] = None if (val is None or (isinstance(val, float) and math.isnan(val))) \
                else round(float(val), 4)
    print(f"[ragas] 完成：{len(need)} 题已写回 metrics")


def _metrics_complete(arm: str, rec: dict) -> bool:
    """裁判分齐全判定：键在且值非 None（None 表示上轮该指标裁判调用失败，应重试）。"""
    m = rec.get("metrics") or {}
    keys = ("faithfulness", "answer_relevancy") + \
        (("context_precision",) if arm in ("rag", "lora") else ())
    return all(m.get(k) is not None for k in keys)


def _gold_ids_one(gold: dict, chunks: list[dict]) -> list[int]:
    out = []
    for g in gold["gold_articles"]:
        for c in chunks:
            if c.get("article") and norm_law_name(c["law_name"]) == norm_law_name(g["law"]) \
                    and _norm_article(c["article"]) == _norm_article(g["article"]):
                out.append(c["id"])
    return out


# ================= 逐臂运行 / 断点续跑 =================

def results_path(arm: str) -> Path:
    return RESULTS_DIR / f"results_{arm}.jsonl"


def load_results(arm: str) -> dict[str, dict]:
    p = results_path(arm)
    if not p.exists():
        return {}
    return {r["id"]: r for r in
            (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())}


def save_results(arm: str, records: list[dict]) -> None:
    records = sorted(records, key=lambda r: int(r["id"][1:]))
    with open(results_path(arm), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_arm(arm: str, limit: int | None, do_judge: bool, judge_only: bool) -> None:
    testset = load_testset()
    if limit:
        testset = testset[:limit]
    chunks = load_chunks()
    art_idx = gold_article_index(chunks)

    results = load_results(arm)
    todo = [r for r in testset if r["id"] not in results]

    if not judge_only and todo:
        engine = build_arm(arm)
        print(f"[{arm}] 待生成 {len(todo)} 题（已有 {len(results)} 题断点）")
        for i, q in enumerate(todo, 1):
            t0 = time.time()
            out = engine.answer(q["query"])
            gold_ids, missing = gold_chunk_ids(q, art_idx)
            if missing:
                print(f"  ⚠ {q['id']} gold 条款未解析到 chunk：{missing}")
            rec = {
                "id": q["id"], "query": q["query"], "type": q["type"],
                "difficulty": q["difficulty"], "expected": q["expected"],
                "gold_articles": q["gold_articles"],
                "answer": out["answer"], "refused": out["refused"],
                "retrieved": out["retrieved"],
            }
            rec.update(score_record(rec, gold_ids, art_idx))
            rec["metrics"] = {}
            results[q["id"]] = rec
            tag = "拒" if rec["refused"] else "答"
            hit = f" hit={int(rec['hit'])}" if rec["hit"] is not None else ""
            print(f"  {i}/{len(todo)} {q['id']} [{tag}]{hit} "
                  f"({time.time() - t0:.1f}s) {q['query'][:28]}")
            save_results(arm, list(results.values()))  # 每题落盘，崩了不丢
    else:
        print(f"[{arm}] 生成阶段跳过（judge_only={judge_only}，待生成={len(todo)}）")

    if do_judge:
        run_ragas(arm, [results[r["id"]] for r in testset if r["id"] in results],
                  load_testset(), chunks)
        save_results(arm, list(results.values()))

    print(f"[{arm}] 结果：{results_path(arm)}（{len(results)} 题）")


# ================= 总表 =================

def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _fmt(x, pct=True, dash="-"):
    if x is None:
        return dash
    return f"{x*100:.1f}%" if pct else f"{x:.3f}"


def cmd_summary() -> None:
    all_results= {}
    for arm in ARMS:
        p = results_path(arm)
        if not p.exists():
            print(f"缺少 {p.name}，先跑 --arm {arm}")
            return
        all_results[arm] = load_results(arm)

    print("=" * 92)
    print(f"{'臂':<8}{'条款命中@3':>11}{'full_hit':>9}{'引用正确率':>11}"
          f"{'拒答召回':>9}{'faithful':>9}{'relevancy':>10}{'ctx_prec':>10}")
    print("=" * 92)

    summary_rows = {}
    for arm in ARMS:
        recs = list(all_results[arm].values())
        answer_recs = [r for r in recs if r["expected"] == "answer"]
        refuse_recs = [r for r in recs if r["expected"] == "refuse"]

        hit3 = _mean([r["hit"] for r in answer_recs]) if arm != "api" else None
        full = _mean([r["full_hit"] for r in answer_recs]) if arm != "api" else None
        cite = _mean([r["cite_correct"] for r in answer_recs])
        refuse_recall = _mean([r["refuse_correct"] for r in refuse_recs])
        faith = _mean([(r.get("metrics") or {}).get("faithfulness") for r in answer_recs])
        rel = _mean([(r.get("metrics") or {}).get("answer_relevancy") for r in answer_recs])
        ctxp = _mean([(r.get("metrics") or {}).get("context_precision")
                      for r in answer_recs]) if arm != "api" else None
        summary_rows[arm] = dict(hit3=hit3, full=full, cite=cite,
                                 refuse_recall=refuse_recall,
                                 faith=faith, rel=rel, ctxp=ctxp,
                                 n_ans=len(answer_recs), n_ref=len(refuse_recs))
        print(f"{ARM_LABEL[arm]:<8}{_fmt(hit3):>11}{_fmt(full):>9}{_fmt(cite):>11}"
              f"{_fmt(refuse_recall):>9}{_fmt(faith, pct=False):>9}"
              f"{_fmt(rel, pct=False):>10}{_fmt(ctxp, pct=False):>10}")
    print("=" * 92)

    # ---- 拒答/误拒行为分解 + 幻觉引用计数 ----
    print("\n拒答行为分解（49 题自动判定 + Q49 partial 人工）：")
    for arm in ARMS:
        recs = list(all_results[arm].values())
        leak = [r["id"] for r in recs if r["expected"] == "refuse" and not r["refused"]]
        wrong_refuse = [r["id"] for r in recs
                        if r["expected"] == "answer" and r["refused"]]
        n_hall = sum(len(r["hallucinated_cites"]) for r in recs)
        p = next(r for r in recs if r["expected"] == "partial")
        print(f"  {ARM_LABEL[arm]:<8} 漏拒(该拒却答) {len(leak)}/9 {leak}")
        print(f"  {'':<8} 误拒(该答却拒) {len(wrong_refuse)}/40 {wrong_refuse}")
        print(f"  {'':<8} 幻觉引用总条数 {n_hall}")
        print(f"  {'':<8} Q49(partial) refused={p['refused']} → manual，"
              f"回答前80字：{p['answer'][:80]}")

    # ---- 难度分层（验证“难题拉开差距”的预判）----
    print("\nfaithfulness 难度分层（answer 题）：")
    print(f"{'臂':<8}{'易':>8}{'中':>8}{'难':>8}")
    for arm in ARMS:
        cells = []
        for d in ("易", "中", "难"):
            xs = [(r.get("metrics") or {}).get("faithfulness")
                  for r in all_results[arm].values()
                  if r["expected"] == "answer" and r["difficulty"] == d]
            cells.append(_fmt(_mean(xs), pct=False))
        print(f"{ARM_LABEL[arm]:<8}{cells[0]:>8}{cells[1]:>8}{cells[2]:>8}")

    print("\n注：A 臂无检索，hit@3/ctx_precision 记 -；A 的 faithfulness 以 gold 条款原文为"
          "参照语境；C 臂检索栈与 B 完全相同，ctx_precision 应与 B 逐题相等（互为校验）。")


# ================= CLI =================

def main():
    ap = argparse.ArgumentParser(description="阶段5 三臂评测 + RAGAS 裁判")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--limit", type=int, default=None, help="只跑考卷前 N 题（试水用）")
    ap.add_argument("--judge-only", action="store_true", help="跳过生成，只补 RAGAS 分")
    ap.add_argument("--no-judge", action="store_true", help="只生成，不跑裁判")
    ap.add_argument("--summary", action="store_true", help="打印三臂总表")
    args = ap.parse_args()

    if args.summary:
        cmd_summary()
        return
    if not args.arm:
        ap.error("请指定 --arm {api,rag,lora} 或 --summary")
    run_arm(args.arm, args.limit, do_judge=not args.no_judge,
            judge_only=args.judge_only)


if __name__ == "__main__":
    main()
