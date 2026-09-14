
import re
from pathlib import Path
from typing import List, Dict

ARTICLE_PATTERN = re.compile(r'第[一二三四五六七八九十百千零〇]+条')
CHAPTER_PATTERN = re.compile(r'第[一二三四五六七八九十百]+章')
SECTION_PATTERN = re.compile(r'第[一二三四五六七八九十百]+节')
MAX_LEN = 500

def split_long_article(text: str, law_name: str, max_len: int = MAX_LEN) -> list[dict]:
    """对单一条款文本进行二次切分（贪心装箱）。

    策略：按句号切句子，往当前段塞，塞不下才开新段；
    仍超长的段按分号再切一次（法律文本分号即列举项边界，是天然断点）。
    若分号子句仍超长，接受超限——保留语义完整优先于硬塞长度上限，
    不强行按逗号切散子句（破坏语义）。

    Args:
        text: 条款原始文本
        law_name: 所属法律名称
        max_len: 单段最大长度（字符数）

    Returns:
        List[Dict]: 每个元素含 law_name / text / part / char_len
    """
    if len(text) <= max_len:
        return [{"law_name": law_name, "text": text, "part": 1, "char_len": len(text)}]

    def split_keep_sep(s: str, sep: str) -> list[str]:
        """按 sep 切分，分隔符保留在前一段末尾（句子完整）。"""
        parts = re.split(f"({re.escape(sep)})", s)
        chunks = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and parts[i + 1] == sep:
                if parts[i]:
                    chunks.append(parts[i] + sep)
                i += 2
            else:
                if parts[i]:
                    chunks.append(parts[i])
                i += 1
        return chunks

    def greedy_pack(sentences: list[str]) -> list[str]:
        """贪心装箱：往当前段塞句子，塞不下就开新段。"""
        segments = []
        current = ""
        for sent in sentences:
            if current and len(current) + len(sent) > max_len:
                segments.append(current)
                current = sent
            else:
                current += sent
        if current:
            segments.append(current)
        return segments

    # 第一层：按句号切，贪心装箱
    sentences = split_keep_sep(text, "。")
    segments = greedy_pack(sentences)

    # 第二层：仍超长的段按分号再切（水法85条这种分号列举场景）
    final_segments = []
    for seg in segments:
        if len(seg) <= max_len:
            final_segments.append(seg)
        else:
            sub_sentences = split_keep_sep(seg, "；")
            final_segments.extend(greedy_pack(sub_sentences))

    return [
        {"law_name": law_name, "text": seg, "part": i + 1, "char_len": len(seg)}
        for i, seg in enumerate(final_segments)
    ]

def chunk_law(cleaned: str, law_name: str) -> list[dict]:
    """
    清洗后文本 → chunk 列表。
    输入应为已清洗的法规文本（从第一章/第一条开始）。
    遍历行时维护 current_chapter / current_section，
    遇到"第X章"/"第X节"行先 flush 上一条再更新对应变量，
    每条 chunk 都带上当时的 chapter / section（多级定位提升检索可信度）。
    """
    lines = cleaned.splitlines()
    current_chapter = None
    current_section = None
    current_article_title = None
    current_article_text = []
    chunks = []

    def flush_article():
        nonlocal current_article_title, current_article_text
        if current_article_title is not None:
            # 中文硬换行常在词中间，用空格拼接会劈开词（影响 embedding）
            # 故直接拼接，保留原词形
            full_text = "".join(current_article_text).strip()
            if not full_text:
                full_text = current_article_title  # 保底
            sub_chunks = split_long_article(full_text, law_name)
            for sub in sub_chunks:
                sub["chapter"] = current_chapter
                sub["section"] = current_section
                sub["article"] = current_article_title
                chunks.append(sub)
            current_article_title = None
            current_article_text = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # 章标题：上一条正文结束之后出现，先 flush 再更新
        if CHAPTER_PATTERN.match(stripped):
            flush_article()
            current_chapter = stripped
            current_section = None  # 进入新章，节失效
            continue

        # 节标题：同章思路，先 flush 上一条再更新 section
        if SECTION_PATTERN.match(stripped):
            flush_article()
            current_section = stripped
            continue

        # 条款检测
        match = ARTICLE_PATTERN.match(stripped)
        if match:
            flush_article()
            current_article_title = match.group(0)
            content = stripped[len(match.group(0)):].strip()
            current_article_text = [content] if content else []
        else:
            # 普通内容行
            if current_article_title is not None:
                current_article_text.append(stripped)
            else:
                # 可能为前言，忽略
                pass

    flush_article()
    return chunks

if __name__ == "__main__":
    import json

    processed_dir = Path(__file__).parent.parent / "data" / "processed"

    all_chunks = []
    print("=" * 60)
    print("逐部法规统计")
    print("=" * 60)
    for f in sorted(processed_dir.glob("清洁_*.txt")):
        law_name = f.stem.replace("清洁_", "")
        cleaned = f.read_text(encoding="utf-8")
        chunks = chunk_law(cleaned, law_name)
        lengths = [len(c["text"]) for c in chunks]
        print(f"\n{law_name}:")
        print(f"  chunk 数: {len(chunks)}")
        if lengths:
            print(f"  长度分布: 最短={min(lengths)}, 平均={sum(lengths) // len(lengths)}, 最长={max(lengths)}")
        all_chunks.extend(chunks)

    # 写入 chunks_all.jsonl
    out_path = processed_dir / "chunks_all.jsonl"
    with open(out_path, "w", encoding="utf-8") as fout:
        for c in all_chunks:
            fout.write(json.dumps(c, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(f"已写入: {out_path}")
    print(f"全部 chunk 数: {len(all_chunks)}")

    # 抽查：水法第九十条（最后一条附近，附则前容易切漏）
    print("\n" + "=" * 60)
    print("抽查：中华人民共和国水污染防治法 第九十条")
    print("=" * 60)
    water_chunks = [c for c in all_chunks if c["law_name"] == "中华人民共和国水污染防治法"]
    print(f"水法 chunk 总数: {len(water_chunks)}")
    print(f"验证: chunk 数 >= 104 ? {'✓ 通过' if len(water_chunks) >= 104 else '✗ 未达标'}")
    art90 = [c for c in water_chunks if c.get("article") == "第九十条"]
    if art90:
        for c in art90:
            print(f"\n  article: {c['article']}")
            print(f"  chapter: {c['chapter']}")
            print(f"  text: {c['text'][:120]}{'...' if len(c['text']) > 120 else ''}")
    else:
        print("  ✗ 未找到第九十条 chunk！")

    # 抽查：水法第八十五条（验证贪心装箱 + part/char_len 字段）
    print("\n" + "-" * 60)
    print("抽查：第八十五条（验证贪心装箱 + part/char_len）")
    print("-" * 60)
    art85 = [c for c in water_chunks if c.get("article") == "第八十五条"]
    for c in art85:
        over = " ⚠超限" if c["char_len"] > 500 else ""
        print(f"\n  part {c['part']}/{len(art85)} | char_len={c['char_len']}{over} | chapter={c['chapter']}")
        print(f"  text: {c['text'][:100]}{'...' if len(c['text']) > 100 else ''}")

    # 抽查：大气法第四十二条（之前末尾粘着"第二节 工业污染防治"节标题）
    print("\n" + "-" * 60)
    print("抽查：大气污染防治法 第四十二条（验证节标题已剥离）")
    print("-" * 60)
    air_chunks = [c for c in all_chunks if c["law_name"] == "中华人民共和国大气污染防治法"]
    art42 = [c for c in air_chunks if c.get("article") == "第四十二条"]
    for c in art42:
        tail = c["text"][-20:]
        polluted = "节" in tail or "工业污染防治" in tail
        print(f"\n  part {c['part']}/{len(art42)} | char_len={c['char_len']}")
        print(f"  chapter: {c['chapter']} | section: {c.get('section')}")
        print(f"  text 末尾20字: ...{tail}")
        print(f"  节标题污染: {'✗ 仍有' if polluted else '✓ 已剥离'}")

    # 抽查：所有节标题应作为 section 字段出现，而非混进正文
    print("\n" + "-" * 60)
    print("验证：大气法所有 chunk 中是否还有节标题混进正文")
    print("-" * 60)
    leaked = [c for c in air_chunks
              if SECTION_PATTERN.search(c["text"]) and not SECTION_PATTERN.match(c["text"])]
    print(f"  正文中含节标题的 chunk 数: {len(leaked)} {'✓' if not leaked else '✗ 仍有泄漏'}")
    if leaked:
        for c in leaked[:3]:
            print(f"    - {c['article']} part{c['part']}: ...{c['text'][-30:]}")
    sections_used = sorted({c["section"] for c in air_chunks if c.get("section")})
    print(f"  作为 section 元数据出现的节标题数: {len(sections_used)}")
    for s in sections_used:
        print(f"    - {s}")
