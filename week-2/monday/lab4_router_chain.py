# lab4_router_chain.py
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# A low temperature (0.0) removes creative variation, making classification deterministic.
router_llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0.0,
    max_tokens=1,  # type:ignore # This will return only a single token, not necessarily a single word, but it will be enough for our classification task.
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

router_prompt = ChatPromptTemplate.from_template(
    "Classify the patient message as exactly one word, either 'CLINICAL' or 'BILLING'.\n"
    "Message: {message}\nClassification:"
)

router_chain = router_prompt | router_llm | StrOutputParser()

route = (
    router_chain.invoke({"message": "I was charged twice for my last visit"})
    .upper()
    .strip()
)
print(f"Routed to: {route}")
