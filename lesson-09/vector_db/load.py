from langchain.vectorstores import Chroma
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import find_dotenv, load_dotenv
from langchain_community.embeddings import QianfanEmbeddingsEndpoint

_ = load_dotenv(find_dotenv())

# 加载 PDF
loaders_chinese = [
    PyMuPDFLoader("path")  # 南瓜书
    # 大家可以自行加入其他文件
]
docs = []
for loader in loaders_chinese:
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

persist_directory = './chroma'

vectordb = Chroma.from_documents(
    documents=split_docs[:100],  # 为了速度，只选择了前 100 个切分的 doc 进行生成。
    embedding=embedding,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)

vectordb.persist()

# 直接载入
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)
