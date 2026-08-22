# challenge1_swahili_guardrails.py
# Core skill (from Lab 1): isolate rules in a SystemMessage, user text in a HumanMessage.
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",  # OpenRouter uses "provider/model" naming
    temperature=0.3,
    max_tokens=300,  # type:ignore
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# TODO: Build the message array the same way you did in Lab 1.
#       The behaviour you must achieve: the assistant answers clinically in English,
#       THEN always appends exactly one localized Swahili reassurance line.
#       Decide which message that rule belongs in, and why.
messages = [
    # ... your SystemMessage and HumanMessage here ...
    SystemMessage(
        content=(
            "You are a Senior AI Engineer specialising in medical triage interfaces "
            "at AfyaPlus Health in Kenya. Provide professional, safe, structured guidance."
            "Whenever you respond, you must always append exactly one localized Swahili reassurance line at the end of your response."
        )
    ),
    HumanMessage(
        content="Explain the structural difference between a deterministic chain and a dynamic agent."
    ),
]

response = llm.invoke(messages)
print(response.content)
