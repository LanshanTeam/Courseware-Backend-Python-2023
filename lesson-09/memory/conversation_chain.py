from langchain.vectorstores import Chroma
from dotenv import load_dotenv, find_dotenv
import os

from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.embeddings import QianfanEmbeddingsEndpoint

_ = load_dotenv(find_dotenv())
llm = QianfanChatEndpoint(
    streaming=False,
    model="ERNIE-Bot",
)

embedding = QianfanEmbeddingsEndpoint(
    streaming=True,
    model="Embedding-V1",
    chunk_size=16,
)

# 向量数据库持久化路径
persist_directory = 'path'
# 加载数据库
vectordb = Chroma(
    persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    embedding_function=embedding
)

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
    return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
)
from langchain.chains import ConversationalRetrievalChain

retriever = vectordb.as_retriever()

qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)

question = "我可以学习到关于强化学习的知识吗？"
result = qa({"question": question})
print(result['answer'])
