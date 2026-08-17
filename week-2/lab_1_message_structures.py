# lab1_message_structures.py
from os import environ

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",  # OpenRouter uses "provider/model" naming
    temperature=0.3,
    max_tokens=300,
    api_key=environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

messages = [
    SystemMessage(
        content=(
            "You are a Senior AI Engineer specialising in medical triage interfaces "
            "at AfyaPlus Health in Kenya. Provide professional, safe, structured guidance."
        )
    ),
    HumanMessage(
        content="Explain the structural difference between a deterministic chain and a dynamic agent."
    ),
]

response = llm.invoke(messages)
print(response.content)
