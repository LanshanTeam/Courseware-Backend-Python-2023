import numpy as np
from dotenv import find_dotenv, load_dotenv
from langchain_community.embeddings import QianfanEmbeddingsEndpoint

_ = load_dotenv(find_dotenv())
# 定义 Embeddings
embedding = QianfanEmbeddingsEndpoint(
    streaming=True,
    model="Embedding-V1",
    chunk_size=16,
)


query1 = "机器学习"
query2 = "强化学习"
query3 = "大语言模型"

# 通过对应的 embedding 类生成 query 的 embedding。
emb1 = embedding.embed_query(query1)
emb2 = embedding.embed_query(query2)
emb3 = embedding.embed_query(query3)

# 将返回结果转成 numpy 的格式，便于后续计算
emb1 = np.array(emb1)
emb2 = np.array(emb2)
emb3 = np.array(emb3)

print(f"{query1} 生成的为长度 {len(emb1)} 的 embedding , 其前 30 个值为： {emb1[:30]}")