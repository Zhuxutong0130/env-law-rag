import re
raw = """
中华人民共和国环境保护法
（2014年修订）

　　第一条　为保护和改善环境，防治污染和其他公害，保障公众健康，推进生态文明建设，促进经济社会可持续发展，制定本法。

　　第二条　本法所称环境，是指影响人类生存和发展的各种天然的和经过人工改造的自然因素的总体，包括大气、水、海洋、土地、矿藏、森林、草原、湿地、野生生物、自然遗迹、人文遗迹、自然保护区、风景名胜区、城市和乡村等。


　　第三条　本法适用于中华人民共和国领域和中华人民共和国管辖的其他海域。
"""
num = 0
count = raw.count("环境")
print("环境出现的次数为：", count)

cleaned = raw.replace("\u3000", "")
cleaned = cleaned.replace("\n", "")
print("去掉全角空格和换行后的文本为：", cleaned)


def split_articles(raw) -> list[dict]:
    lines = raw.strip().splitlines()
    law = ""
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("(") and not stripped.startswith("第"):
            law = stripped
            break
    pattern = re.compile(r'(第[一二三四五六七八九十百零\d]+条)')
    matches = list(pattern.finditer(raw))
    articles = []
    for i, match in enumerate(matches):
        title = match.group(1)          # 例如 "第一条"
        start = match.start()           # 标题起始位置
        content_start = match.end()     # 内容起始位置（标题之后）
        # 结束位置：下一条标题的起始，或文本末尾
        end = matches[i+1].start() if i+1 < len(matches) else len(raw)

        # 截取内容并去除首尾空白（保留内部换行/空格）
        content = raw[content_start:end].strip()

        articles.append({
            "law": law,
            "article": title,
            "text": content
        })

    return articles

if __name__ == "__main__":
    result = split_articles(raw)
    for a in result:
        print(a["article"], "→", a["text"][:20], "...")
    longest = max(result, key=lambda x: len(x["text"]))   # 练习4
    print("最长条款：", longest["article"], "长度", len(longest["text"]))



    
