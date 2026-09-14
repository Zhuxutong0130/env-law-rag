#练习1
import json

with open("data/raw/环境保护法_节选.txt", encoding="utf-8") as f:  
    raw = f.read()
print("总字符数：", len(raw))

#练习2
from split_articles import split_articles
articles = split_articles(raw)
for a in articles:
        print(a["article"], "→", a["text"][:20], "...")

#练习3
with open("data/processed/articles.jsonl", "w", encoding="utf-8") as f:  
    for a in articles:  
        f.write(json.dumps(a, ensure_ascii=False) + "\n")
with open("data/processed/articles.jsonl", encoding="utf-8") as f:  
    for line in f:                       # 逐行读，内存友好  
        a = json.loads(line)  
        print(a["article"])

