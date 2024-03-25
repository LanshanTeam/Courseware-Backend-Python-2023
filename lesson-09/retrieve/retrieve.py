from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.vectorstores.chroma import Chroma

# 声明一个检索式问答链
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
from langchain.prompts import PromptTemplate

# Build prompt
template = """使用以下上下文片段来回答最后的问题。如果你不知道答案，只需说不知道，不要试图编造答案。答案最多使用三个句子。尽量简明扼要地回答。在回答的最后一定要说"感谢您的提问！"
{context}
问题：{question}
有用的回答："""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
# 假设已经在以下路径持久化
persist_directory = 'path'

# 直接载入
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

question = " 2025 年大语言模型效果最好的是哪个模型"
result = qa_chain({"query": question})
print(f"LLM 对问题的回答：{result['result']}")

