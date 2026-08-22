# broken_router.py
from os import environ

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Instantiate the reasoning brain
load_dotenv()
llm = ChatOpenAI(
    temperature=0.0,  # Fixed bug by setting the temperature to 0.0 for deterministic classification
    model="openai/gpt-4o-mini",
    max_tokens=5,  # type:ignore
    api_key=environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)
prompt = ChatPromptTemplate.from_template(
    "Classify as CLINICAL or BILLING: {msg}\nAnswer:"
)
chain = prompt | llm | StrOutputParser()
print(chain.invoke({"msg": "My chest hurts badly"}))
