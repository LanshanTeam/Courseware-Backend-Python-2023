from langchain.memory import ChatMessageHistory
history = ChatMessageHistory()
history.add_user_message("hi!")
history.add_ai_message("whats up?")
print(history)
"""
Human: hi!
AI: whats up?
"""