import unicodedata
import re
from pathlib import Path

def clean_text(raw: str) -> str:
    """输入法规原文，输出清洗后的文本。"""
    # ① NFKC 归一化（半角数字/字母统一，全角标点转半角）
    clean = unicodedata.normalize('NFKC', raw)
    # ② 将部分英文半角标点还原为中文全角标点
    half_to_full = str.maketrans(',.!?;:()', '，。！？；：（）')
    clean = clean.translate(half_to_full)
    # ③ 拆行 → strip → 去空行 → 拼回
    lines = clean.splitlines()
    stripped_lines = [line.strip() for line in lines]
    non_empty_lines = [line for line in stripped_lines if line]   # 去除空行
    clean = '\n'.join(non_empty_lines)
    # ④ 截取从第一条开始的部分
    idx = clean.find('第一条')
    if idx != -1:
        clean = clean[idx:]
    return clean

def count_articles(text: str) -> int:
    """统计文本中'第X条'出现的次数（X为中文数字）。"""
    pattern = r'第[零一二三四五六七八九十百千万]+条'
    return len(re.findall(pattern, text))

if __name__ == "__main__":
    # 目录设置
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    out_dir = Path(__file__).parent.parent / "data" / "processed"
    out_dir.mkdir(exist_ok=True)

    for f in raw_dir.glob("*.txt"):
        # 读取原始文本
        raw_text = f.read_text(encoding="utf-8")
        # 统计清洗前
        before_len = len(raw_text)
        before_articles = count_articles(raw_text)

        # 清洗
        cleaned = clean_text(raw_text)

        # 统计清洗后
        after_len = len(cleaned)
        after_articles = count_articles(cleaned)

        # 打印信息
        print(f"{f.name}:")
        print(f"  原始字符数: {before_len}, 清洗后字符数: {after_len}")
        print(f"  原始条款数: {before_articles}, 清洗后条款数: {after_articles}\n")

        # 写出清洗后文件
        out_path = out_dir / f"清洁_{f.name}"
        out_path.write_text(cleaned, encoding="utf-8")
        print(f"  已写出: {out_path}\n")