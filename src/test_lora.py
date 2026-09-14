"""LoRA 微调验收：同题 / 同 prompt / 同解码参数，base vs LoRA 并排对比。

用法：python src/test_lora.py
前提：已执行 python src/train_lora.py，adapter 在 models/lora_adapter。

三道题：
  1,2 库内题（取自 val.jsonl 的 held-out 条款，训练没见过）：
      chunk 211 大气法§122 大气污染事故罚款、chunk 81 噪声法§82 广场舞扰民
  3  语义邻居拒答陷阱（top1 rerank 0.783，检索回来的条款字面相关但答不了）：
      "碳排放配额可以买卖吗"

看点：
  - base（R1-Distill-1.5B）会先长篇 <think>，库内题没法规口吻、拒答题一本正经胡说；
  - LoRA 立即闭合 think 块直答，带"（依据：《XX法》第X条）"，拒答题学会拒答。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

_SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(_SRC_DIR))

from train_lora import MODEL_NAME, ADAPTER_OUT, PROJECT_ROOT, encode_text
from generator import SYSTEM_PROMPT, build_user_message

VAL_PATH = PROJECT_ROOT / "data" / "finetune" / "val.jsonl"
VAL_CHUNK_IDS = [211, 81]                       # 库内题：大气法§122、噪声法§82
REFUSAL_QUESTION = "碳排放配额可以买卖吗"       # 最难的语义邻居陷阱

THINK_CLOSE_ID = 151649                         # </think>
MAX_NEW_TOKENS = 512


def build_test_cases() -> list[dict]:
    """组装三道题。库内题直接复用 val 记录（与训练同构的完整 input）；
    拒答题走真实检索栈取 top-3 无关条款，复刻推理期输入。"""
    cases = []
    for line in open(VAL_PATH, encoding="utf-8"):
        rec = __import__("json").loads(line)
        if rec["meta"].get("kind") == "answer" and rec["meta"]["chunk_id"] in VAL_CHUNK_IDS:
            q = rec["input"].split("问题：", 1)[1]
            cases.append({
                "tag": "库内",
                "question": q,
                "instruction": rec["instruction"],
                "input": rec["input"],
                "expect": f"《{rec['meta']['law_name']}》{rec['meta']['article']}",
                "gold": rec["output"],
            })
    assert len(cases) == 2, f"库内题取到 {len(cases)} 条，应为 2"

    # 拒答题：本地检索栈（不调 LLM）
    from build_dataset import norm_law_name
    from embedding import load_model
    from retriever import load_assets, load_reranker, recall, rerank

    index, meta = load_assets(
        PROJECT_ROOT / "models" / "law.index",
        PROJECT_ROOT / "data" / "processed" / "chunk_meta.jsonl",
    )
    embed_model, reranker = load_model(), load_reranker()
    top3 = rerank(REFUSAL_QUESTION, recall(REFUSAL_QUESTION, index, meta, embed_model), reranker)
    for c in top3:
        c["law_name"] = norm_law_name(c["law_name"])
    cases.append({
        "tag": "拒答",
        "question": REFUSAL_QUESTION,
        "instruction": SYSTEM_PROMPT,
        "input": build_user_message(REFUSAL_QUESTION, top3),
        "expect": "应拒答（库内无碳排放权交易相关条款）",
        "gold": "问题超出知识库范围，无法回答。",
        "top1_score": top3[0]["rerank_score"],
    })
    return cases


def prompt_ids(tok, case: dict) -> list[int]:
    """必须与训练时的 prompt 构造逐 token 一致（train_lora.build_example 同款）。"""
    text = tok.apply_chat_template(
        [{"role": "system", "content": case["instruction"]},
         {"role": "user", "content": case["input"]}],
        tokenize=False, add_generation_prompt=True,
    )
    return encode_text(tok, text)


@torch.no_grad()
def generate(model, tok, ids: list[int]) -> list[int]:
    x = torch.tensor([ids], device=model.device)
    y = model.generate(
        x, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    return y[0][len(ids):].tolist()


def split_answer(tok, new_ids: list[int]) -> tuple[str, str, bool]:
    """按 </think>(151649) 切思考/作答。返回 (思考, 正式回答, 是否被截断)。"""
    truncated = len(new_ids) >= MAX_NEW_TOKENS and new_ids[-1] != tok.eos_token_id
    if THINK_CLOSE_ID in new_ids:
        i = new_ids.index(THINK_CLOSE_ID)
        think = tok.decode(new_ids[:i], skip_special_tokens=True).strip()
        answer = tok.decode(new_ids[i + 1:], skip_special_tokens=True).strip()
    else:
        think, answer = "", tok.decode(new_ids, skip_special_tokens=True).strip()
    return think, answer, truncated


def has_citation(text: str) -> bool:
    import re
    return bool(re.search(r"《.+?》第.+?条", text))


def is_refusal(text: str) -> bool:
    return ("超出知识库范围" in text) or ("无法回答" in text and len(text) < 40)


def main():
    print("组装测试题（拒答题先跑本地检索栈）...")
    cases = build_test_cases()

    print(f"加载 base 模型：{MODEL_NAME}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cuda",
    ).eval()

    prompts = [prompt_ids(tok, c) for c in cases]

    print("\n>>> 先用 base（不挂 adapter）生成 ...")
    base_out = [generate(base, tok, p) for p in prompts]

    print(">>> 挂 LoRA adapter 后再生成 ...")
    model = PeftModel.from_pretrained(base, str(ADAPTER_OUT)).eval()
    lora_out = [generate(model, tok, p) for p in prompts]

    rows = []
    for i, case in enumerate(cases):
        b_think, b_ans, b_cut = split_answer(tok, base_out[i])
        l_think, l_ans, l_cut = split_answer(tok, lora_out[i])
        rows.append(dict(case=case, b_think=b_think, b_ans=b_ans, b_cut=b_cut,
                         l_think=l_think, l_ans=l_ans, l_cut=l_cut))

    for i, r in enumerate(rows, 1):
        c = r["case"]
        print("\n" + "=" * 78)
        print(f"【题 {i}｜{c['tag']}】{c['question']}")
        print(f"期望：{c['expect']}")
        if c["tag"] == "拒答":
            print(f"（检索 top1 rerank 分数：{c['top1_score']:.3f}）")
        print("-" * 78)
        print(f"[BASE] 思考 {len(r['b_think'])} 字"
              + ("（无思考段）" if not r['b_think'] else "")
              + ("  ⚠输出截断" if r['b_cut'] else ""))
        if r["b_think"]:
            print("  思考节选：" + r["b_think"].replace("\n", " ")[:240] + ("..." if len(r["b_think"]) > 240 else ""))
        print("  回答：" + r["b_ans"].replace("\n", " ")[:400])
        print("-" * 78)
        print(f"[LoRA] 思考 {len(r['l_think'])} 字"
              + ("（立即闭合 think，直接作答）" if not r['l_think'] else "")
              + ("  ⚠输出截断" if r['l_cut'] else ""))
        if r["l_think"]:
            print("  思考节选：" + r["l_think"].replace("\n", " ")[:240])
        print("  回答：" + r["l_ans"].replace("\n", " ")[:400])
        print("-" * 78)
        print("  Gold：" + c["gold"].replace("\n", " ")[:400])

    # 汇总：肉眼差异信号
    print("\n" + "=" * 78)
    print("对比汇总（引用 / 拒答 / 是否思考）：")
    diff_count = 0
    for r in rows:
        c = r["case"]
        b_ref, l_ref = has_citation(r["b_ans"]), has_citation(r["l_ans"])
        b_refuse, l_refuse = is_refusal(r["b_ans"]), is_refusal(r["l_ans"])
        b_thinks, l_thinks = bool(r["b_think"]), bool(r["l_think"])
        if c["tag"] == "拒答":
            improved = l_refuse and not b_refuse
        else:
            improved = l_ref and (not b_ref or b_thinks != l_thinks)
        diff_count += bool(improved)
        print(f"  {c['tag']} {c['question'][:22]:<24} | "
              f"base: 引用={int(b_ref)} 拒答={int(b_refuse)} 思考={int(b_thinks)}  "
              f"→  LoRA: 引用={int(l_ref)} 拒答={int(l_refuse)} 思考={int(l_thinks)}"
              f"  {'✅差异明显' if improved else '—'}")
    print(f"\n肉眼可见风格差异：{diff_count}/3（验收线 ≥2）")


if __name__ == "__main__":
    main()
