from langchain_openai.chat_models import ChatOpenAI
import os
import openai
from dotenv import load_dotenv, find_dotenv
import warnings

# warnings.filterwarnings("ignore")

# 读取本地/项目的环境变量。

# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 获取环境变量 OPENAI_API_KEY
openai.api_key = os.environ['OPENAI_API_KEY']
openai.base_url = os.environ['OPENAI_BASE_URL']

chat = ChatOpenAI(temperature=0.0)

from langchain.prompts import ChatPromptTemplate

# 这里我们要求模型对给定文本进行中文翻译
template_string = """Translate the text \
that is delimited by triple backticks \
into a Chinese. \
text: ```{text}```
"""

# 接着将 Template 实例化
chat_template = ChatPromptTemplate.from_template(template_string)

# 我们首先设置变量值
text = "Today is a nice day."

# 接着调用 format_messages 将 template 转化为 message 格式

message = chat_template.format_messages(text=text)
print(message)

response = chat(message)
print(response)
