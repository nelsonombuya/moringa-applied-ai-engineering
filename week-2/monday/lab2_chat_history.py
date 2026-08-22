# lab2_chat_history.py
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0.2,
    max_tokens=300,  # type:ignore
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

chat_history = [
    SystemMessage(content="You are an AfyaPlus clinic scheduling assistant in Kisumu."),
    HumanMessage(content="Hi, do you have slots open tomorrow morning?"),
]

first_reply = llm.invoke(chat_history)
print(f"AI: {first_reply.content}\n")

chat_history.append(AIMessage(content=first_reply.content))
chat_history.append(
    HumanMessage(content="Great, lock in the 9:00 AM slot for me please.")
)

second_reply = llm.invoke(chat_history)
print(f"AI: {second_reply.content}")
