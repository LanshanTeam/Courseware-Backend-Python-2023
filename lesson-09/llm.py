from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv, find_dotenv
import warnings

_ = load_dotenv(find_dotenv())
# 获取环境变量 OPENAI_API_KEY
# openai.api_key = os.environ['OPENAI_API_KEY']
# openai.base_url = os.environ['OPENAI_BASE_URL']

llm = OpenAI(temperature=0)
# Notice that "chat_history" is present in the prompt template
template = """You are a nice chatbot having a conversation with a human.

Previous conversation:
{chat_history}

New human question: {question}
Response:"""
prompt = PromptTemplate.from_template(template)
# Notice that we need to align the `memory_key`
memory = ConversationBufferMemory(memory_key="chat_history")
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

print(memory)

res = conversation({"question": "hi"})

print(res)

print(memory)
