"""Gradio 演示入口：环境法规智能问答。

app.py 放项目根目录（演示入口与 src 分离）：`python app.py` 启动后浏览器打开
http://127.0.0.1:7860 。资源路径由 src/pipeline.py 用 __file__ 定位，
不受 Gradio 工作目录影响。
"""
import gradio as gr

from src.pipeline import RAGPipeline

rag = RAGPipeline()   # 模块级加载一次——每次请求复用，不重复加载重资源

DEMO_QUERIES = [
    "工厂半夜施工扰民怎么办",
    "餐馆排烟到居民楼找谁投诉",
    "个人所得税怎么退税",
    "楼上深夜装修电钻吵醒孩子，该打什么电话举报？",
]


def chat(query: str):
    out = rag.answer(query)

    # 拒答体验：明确提示可用的知识库范围，而不是空白或报错
    if out["refused"]:
        answer_md = (
            "### ⚠️ 已拒答\n\n"
            "问题超出知识库范围，无法回答。\n\n"
            "本系统知识库目前覆盖《噪声污染防治法》《大气污染防治法》"
            "《水污染防治法》《环境保护法》，请换个环境法规相关的问题。"
        )
        sources_md = "—— 拒答不产生引用 ——"
        return answer_md, sources_md

    cited_ids = {c["id"] for c in out["cited_chunks"]}
    lines = []
    for c in out["retrieved"]:
        mark = "✅ 被引用" if c["id"] in cited_ids else "未被引用"
        lines.append(
            f"**《{c['law_name']}》{c['article']}** · rerank {c['rerank_score']:.3f} · {mark}\n\n"
            f"> {c['text']}"
        )
    sources_md = "\n\n---\n\n".join(lines)
    return out["answer"], sources_md


with gr.Blocks(title="环境法规智能问答") as demo:
    gr.Markdown(
        "# 环境法规智能问答系统\n"
        "基于 RAG（检索增强生成）· 答案附条款溯源 · 知识库外问题自动拒答"
    )

    inp = gr.Textbox(label="你的问题", placeholder="例：工厂半夜施工扰民怎么办？", lines=2)
    btn = gr.Button("提问", variant="primary")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 回答")
            out = gr.Markdown()
        with gr.Column(scale=2):
            gr.Markdown("### 引用条款（检索 top-3）")
            sources = gr.Markdown()

    gr.Examples(
        examples=[[q] for q in DEMO_QUERIES],
        inputs=[inp], outputs=[out, sources], fn=chat,
        cache_examples=False, label="演示问题（点击自动提问）",
    )

    btn.click(chat, inputs=inp, outputs=[out, sources])
    inp.submit(chat, inputs=inp, outputs=[out, sources])

demo.launch(server_name="0.0.0.0")   # 容器内必须监听 0.0.0.0，宿主机端口映射才能访问到；本地跑不受影响
