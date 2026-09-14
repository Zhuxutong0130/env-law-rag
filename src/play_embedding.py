from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

def main():
    # sentence_transformers 6.x 在镜像模式下用 hf_hub_download 不会建 snapshots 软链，
    # 导致 FileNotFoundError。改用 snapshot_download 强制完整下载到本地目录，
    # 再把本地路径传给 SentenceTransformer，绕过这个 bug。
    local_dir = snapshot_download(repo_id="BAAI/bge-small-zh-v1.5")
    model = SentenceTransformer(local_dir)   # 首次会下载
  
    sentences = [  
        "夜间建筑施工产生噪声污染的，应遵守相关规定。",     # A：噪声·施工  
        "工厂夜间施工扰民，法律如何规定？",               # B：噪声·施工（换个说法）  
        "向水体排放油类废液的，责令改正并处罚款。",        # C：水污染  
        "排放大气污染物超过标准的，应当受到处罚。",        # D：大气污染  
    ]

    # TODO 1: 用 model.encode(sentences) 得到向量矩阵，打印 shape
    #   期望输出形状 (4, 512)
    embeddings = model.encode(sentences)
    print("向量矩阵 shape:", embeddings.shape)

    # TODO 2: 计算两两余弦相似度，打印 4x4 矩阵
    #   提示：model.encode 有个参数 normalize_embeddings=True 可以直接归一化
    #   归一化后 相似度 = 两向量点积，np.dot(a, b) 或矩阵 @ 矩阵.T
    normed = model.encode(sentences, normalize_embeddings=True)
    sim_matrix = normed @ normed.T
    print("\n4x4 余弦相似度矩阵：")
    print(sim_matrix)

    # TODO 3: faiss 建索引（IndexFlatIP + L2 归一化 = 余弦相似度检索）
    import faiss

    faiss.normalize_L2(embeddings)            # 原地归一化，覆盖原矩阵
    index = faiss.IndexFlatIP(512)            # 512 维，内积索引
    index.add(embeddings.astype("float32"))   # faiss 要 float32

    # 用 sentences[1]（工厂夜间施工）作 query，同样要归一化
    QUERY_INDEX = 1
    query = model.encode([sentences[QUERY_INDEX]]).astype("float32")
    faiss.normalize_L2(query)
    # 多取一个用于过滤 query 自身（否则 top1 永远是自己 score=1.0）
    k = 2
    scores, ids = index.search(query, k=k + 1)
    mask = ids[0] != QUERY_INDEX              # 排除自身
    top_ids = ids[0][mask][:k]
    top_scores = scores[0][mask][:k]
    print(f"\nfaiss top-{k} 检索结果（query = sentences[{QUERY_INDEX}]，已排除自身）：")
    print("  ids:   ", top_ids.tolist())
    print("  scores:", top_scores.tolist())
    for rank, (idx, sc) in enumerate(zip(top_ids, top_scores), 1):
        print(f"  rank {rank}: [{idx}] {sentences[idx][:24]}...  score={sc:.4f}")

    # 验收：top1 应为 A（下标 0），score 与 sim_matrix[1][0] 一致（两条路算同一个数）
    print("\n验收：")
    print(f"  top1 id       = {top_ids[0]}（期望=0）")
    print(f"  top1 score   = {top_scores[0]:.7f}")
    print(f"  sim_matrix[1][0] = {sim_matrix[1][0]:.7f}")
    consistent = abs(top_scores[0] - sim_matrix[1][0]) < 1e-5
    print(f"  两条路一致？ {'✓ 通过' if consistent else '✗ 不一致'}")

if __name__ == "__main__":
    main()
