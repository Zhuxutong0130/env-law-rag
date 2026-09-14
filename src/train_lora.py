"""LoRA 微调：data/finetune/{train,val}.jsonl → models/lora_adapter

用法：
    python src/train_lora.py            # 训练（约 60 step，5070 上 10~25 分钟）
    python src/test_lora.py            # 微调前后 3 题同题对比

两个关键设计（踩坑后定的，别乱改）：
1. 训练格式 = 推理格式。prompt 一律用 tokenizer.apply_chat_template 渲染后整体
   encode（不能直接用 apply_chat_template(tokenize=True)——transformers 5.x 会返回
   嵌套 Encoding 对象）。本模型模板的生成头强制以 "<think>\n" 开头，因此 SFT 目标
   构造为「</think>\\n\\n + 答案 + eos」：教会 adapter 立刻闭合思考块、直接作答
   （DeepSeek 官方的免思考模式）。loss 只落在这一段（含闭合标签与答案），
   system/user/生成头全部 -100。
2. bf16 普通 LoRA（非 QLoRA）：1.5B 模型 12GB 显存绰绰有余，没必要引 bitsandbytes。
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from peft import LoraConfig, get_peft_model

_SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(_SRC_DIR))

PROJECT_ROOT = _SRC_DIR.parent
# 允许实验性数据集/上采样倍率经环境变量注入（默认值 = v1 冻结配方，行为不变）
TRAIN_PATH = Path(os.environ.get("TRAIN_FILE", str(PROJECT_ROOT / "data" / "finetune" / "train.jsonl")))
VAL_PATH = PROJECT_ROOT / "data" / "finetune" / "val.jsonl"
ADAPTER_OUT = PROJECT_ROOT / "models" / "lora_adapter"
LOG_PATH = PROJECT_ROOT / "models" / "lora_train_log.json"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# 超参（课程建议值）
MAX_LEN = 1024
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LR = 1e-4
EPOCHS = 3
PER_DEVICE_BATCH = 1      # all-linear LoRA + 长序列在 12GB 上 batch 2 会 OOM，用 1
GRAD_ACCUM = 16           # 有效 batch 仍 = 1×16 = 16
REFUSAL_UPSAMPLE = int(os.environ.get("REFUSAL_UPSAMPLE", "4"))
# 拒答仅 8/318（2.5%），训练集上采样 ×4 防止被答案模式淹没；val 不动
# （v2 实验：难负例扩到 32 条后降为 ×2，防止摆向过度拒答——见 eval/评测报告.md）


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def encode_text(tok, text: str) -> list[int]:
    """模板渲染后整体 encode。

    模板里的 <｜User｜>/<think> 等特殊串会被识别为单 id（add_special_tokens=False
    只负责不再额外插 BOS——BOS 已在模板内）。
    """
    return tok.encode(text, add_special_tokens=False)


def build_example(rec: dict, tok, max_len: int = MAX_LEN) -> dict:
    """record → {input_ids, attention_mask, labels}。

    序列布局：
      <bos>system<｜User｜>input<｜Assistant｜><think>\\n   ← prompt，label=-100
      </think>\\n\\noutput<eos>                              ← 唯一计 loss 段
    """
    prompt_text = tok.apply_chat_template(
        [{"role": "system", "content": rec["instruction"]},
         {"role": "user", "content": rec["input"]}],
        tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = encode_text(tok, prompt_text)

    # 151649 = </think>：紧跟生成头的 <think>\n 立刻闭合 → 免思考直答
    answer_ids = [tok.convert_tokens_to_ids("</think>")]
    answer_ids += encode_text(tok, "\n\n" + rec["output"].strip())
    answer_ids += [tok.eos_token_id]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    # 保险截断（实测最长 859 token，正常不触发）：只从 prompt 中部（参考资料正文
    # 所在区间）删 token，保住 system 头、问题尾部与完整答案。
    if len(input_ids) > max_len:
        overflow = len(input_ids) - max_len
        cut_lo, cut_hi = 64, len(prompt_ids) - 64
        cut_hi = max(cut_hi, cut_lo)
        overflow = min(overflow, cut_hi - cut_lo)
        del input_ids[cut_lo:cut_lo + overflow]
        del labels[cut_lo:cut_lo + overflow]

    return {"input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels}


class SFTDataset(Dataset):
    def __init__(self, records, tok):
        self.examples = [build_example(r, tok) for r in records]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def make_collator(pad_id: int):
    def collate(features: list[dict]):
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, attn, labels = [], [], []
        for f in features:
            pad = maxlen - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * pad)   # 右 padding
            attn.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    return collate


def assert_mask_alignment(rec: dict, tok) -> None:
    """启动时 fail-fast 自检：被 mask 的部分必须能拼回 prompt，计 loss 的部分
    必须能拼回答案，防止 label 错位（SFT 最隐蔽的坑）。"""
    ex = build_example(rec, tok)
    ids, labels = ex["input_ids"], ex["labels"]
    masked = [t for t, l in zip(ids, labels) if l == -100]
    supervised = [l for l in labels if l != -100]
    masked_text = tok.decode(masked, skip_special_tokens=False)
    sup_text = tok.decode(supervised, skip_special_tokens=False)
    assert rec["output"].strip()[:20] in sup_text, "答案未进入 loss 段！"
    assert rec["input"][-20:] in masked_text, "用户问题被计入 loss！"
    assert rec["output"].strip()[:10] not in masked_text, "答案泄漏进 mask 段！"
    print(f"[自检] label mask 对齐 OK：mask {len(masked)} / 监督 {len(supervised)} token")
    print(f"[自检] 监督段开头: {sup_text[:60]!r}")


def main():
    print("加载 tokenizer/模型：", MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_recs = load_records(TRAIN_PATH)
    val_recs = load_records(VAL_PATH)
    _ref = [r for r in train_recs if r["meta"]["kind"] == "refusal"]
    _ans = [r for r in train_recs if r["meta"]["kind"] != "refusal"]
    train_recs = _ans + _ref * REFUSAL_UPSAMPLE
    print(f"train={len(train_recs)}（答案 {len(_ans)} + 拒答 {len(_ref)}x{REFUSAL_UPSAMPLE}"
          f"={len(_ref) * REFUSAL_UPSAMPLE}） val={len(val_recs)}（拒答不重采样）")
    assert_mask_alignment(train_recs[0], tok)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = SFTDataset(train_recs, tok)
    val_ds = SFTDataset(val_recs, tok)

    args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "models" / "_train_tmp"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=2,                    # 约 60 step × 3% ≈ 2（v5 已移除 warmup_ratio）
        max_grad_norm=1.0,                 # v5 改名（旧名 gradient_clipping）
        bf16=True,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],                       # 关掉 wandb
        dataloader_num_workers=0,           # Windows 防多进程卡死
        seed=42,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=make_collator(tok.pad_token_id),
        processing_class=tok,
    )
    trainer.train()

    ADAPTER_OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_OUT)
    tok.save_pretrained(ADAPTER_OUT)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print(f"adapter 已保存：{ADAPTER_OUT}")
    print(f"loss 日志已保存：{LOG_PATH}")
    print("最后 12 条日志：")
    for item in trainer.state.log_history[-12:]:
        print(" ", json.dumps(item, ensure_ascii=False))
    print("接下来执行：python src/test_lora.py")


if __name__ == "__main__":
    main()
