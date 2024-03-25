import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredMarkdownLoader
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.vectorstores.chroma import Chroma

# pdf
# 加载 PDF
loaders = [
    PyMuPDFLoader("path")  # 机器学习,
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# md
folder_path = "path"
files = os.listdir(folder_path)
loaders = []
for one_file in files:
    loader = UnstructuredMarkdownLoader(os.path.join(folder_path, one_file))
    loaders.append(loader)
for loader in loaders:
    docs.extend(loader.load())

# 切分文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)

# 定义 Embeddings
embedding = QianfanEmbeddingsEndpoint(
    streaming=True,
    model="Embedding-V1",
    chunk_size=16,
)

# 定义持久化路径
persist_directory = 'path'

# 加载数据库
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)
