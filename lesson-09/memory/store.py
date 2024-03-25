# 这页你们就当作没有吧，因为运行不了

from uuid import uuid4

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.memory import ZepMemory
from langchain.retrievers import ZepRetriever
from langchain.schema import AIMessage, HumanMessage
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import OpenAI

# 设置 Zep 服务器的 API URL
ZEP_API_URL = "http://localhost:8000"

# 生成一个唯一的用户标识符
session_id = str(uuid4())

# 设置 OpenAI 密钥和 Zep API 密钥
openai_key = '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'
zep_api_key = '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'

# 创建一个用于在线搜索的工具对象
search = WikipediaAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description=(
            "useful for when you need to search online for answers. You should ask"
            " targeted questions"
        ),
    ),
]

# 设置 Zep 聊天历史
memory = ZepMemory(
    session_id=session_id,
    url=ZEP_API_URL,
    api_key=zep_api_key,
    memory_key="chat_history",
)

# 初始化 OpenAI 代理
llm = OpenAI(temperature=0, openai_api_key=openai_key)
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)

# 预加载一些消息到内存中，以演示自动摘要功能
test_history = [
    {"role": "human", "content": "Who was Octavia Butler?"},
    {
        "role": "ai",
        "content": (
            "Octavia Estelle Butler (June 22, 1947 – February 24, 2006) was an American"
            " science fiction author."
        ),
    },
    {"role": "human", "content": "Which books of hers were made into movies?"},
    {
        "role": "ai",
        "content": (
            "The most well-known adaptation of Octavia Butler's work is the FX series"
            " Kindred, based on her novel of the same name."
        ),
    },
    {"role": "human", "content": "Who were her contemporaries?"},
    {
        "role": "ai",
        "content": (
            "Octavia Butler's contemporaries included Ursula K. Le Guin, Samuel R."
            " Delany, and Joanna Russ."
        ),
    },
    {"role": "human", "content": "What awards did she win?"},
    {
        "role": "ai",
        "content": (
            "Octavia Butler won the Hugo Award, the Nebula Award, and the MacArthur"
            " Fellowship."
        ),
    },
    {
        "role": "human",
        "content": "Which other women sci-fi writers might I want to read?",
    },
    {
        "role": "ai",
        "content": "You might want to read Ursula K. Le Guin or Joanna Russ.",
    },
    {
        "role": "human",
        "content": (
            "Write a short synopsis of Butler's book, Parable of the Sower. What is it"
            " about?"
        ),
    },
    {
        "role": "ai",
        "content": (
            "Parable of the Sower is a science fiction novel by Octavia Butler,"
            " published in 1993. It follows the story of Lauren Olamina, a young woman"
            " living in a dystopian future where society has collapsed due to"
            " environmental disasters, poverty, and violence."
        ),
        "metadata": {"foo": "bar"},
    },
]

# 将预加载的消息添加到内存中
for msg in test_history:
    memory.chat_memory.add_message(
        (
            HumanMessage(content=msg["content"])
            if msg["role"] == "human"
            else AIMessage(content=msg["content"])
        ),
        metadata=msg.get("metadata", {}),
    )

ans = agent_chain.run(
    input="What is the book's relevance to the challenges facing contemporary society?",
)

print(ans)
