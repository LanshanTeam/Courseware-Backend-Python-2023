from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
llm = OpenAI(temperature=0)
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)
res = conversation.predict(input="Hi there!")
print("predict:\n", res)
res = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
print("predict:\n", res)
"""
> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi there!
AI:

> Finished chain.
predict:
  Hello! It's nice to meet you. I am an AI created by OpenAI. I am constantly learning and improving my abilities through machine learning algorithms. How can I assist you today?


> Entering new ConversationChain chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
Human: Hi there!
AI:  Hello! It's nice to meet you. I am an AI created by OpenAI. I am constantly learning and improving my abilities through machine learning algorithms. How can I assist you today?
Human: I'm doing well! Just having a conversation with an AI.
AI:

> Finished chain.
predict:
  That's great to hear! I am always happy to engage in conversations and learn more about human interactions. Is there anything specific you would like to talk about?
"""