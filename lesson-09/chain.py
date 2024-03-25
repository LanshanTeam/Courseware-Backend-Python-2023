from dotenv import load_dotenv, find_dotenv
from langchain.chains import SimpleSequentialChain, LLMChain
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.prompts import ChatPromptTemplate

_ = load_dotenv(find_dotenv())
llm = QianfanChatEndpoint(
    streaming=False,
    model="ERNIE-Bot",
)

# 创建两个子链

# 提示模板 1 ：这个提示将接受产品并返回最佳名称来描述该公司
first_prompt = ChatPromptTemplate.from_template(
    "描述制造{product}的一个公司的最好的名称是什么"
)
chain_one = LLMChain(llm=llm, prompt=first_prompt)

# 提示模板 2 ：接受公司名称，然后输出该公司的长为20个单词的描述
second_prompt = ChatPromptTemplate.from_template(
    "写一个20字的描述对于下面这个\
    公司：{company_name}的"
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)

# 构建简单顺序链
# 现在我们可以组合两个LLMChain，以便我们可以在一个步骤中创建公司名称和描述
overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two], verbose=True)

# 运行简单顺序链
product = "大号床单套装"
overall_simple_chain.run(product)
