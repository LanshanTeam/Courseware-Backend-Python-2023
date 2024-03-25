# wenxin_predict.py

from langchain_community.chat_models import QianfanChatEndpoint
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
llm = QianfanChatEndpoint(
    streaming=False,
    model="ERNIE-Bot",
)
res = llm.predict("早上好")

print(res)
