# lab3_sequential_chain.py
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0.2,
    max_tokens=300,  # type:ignore
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

prompt = ChatPromptTemplate.from_template(
    "Draft a one-sentence administrative acknowledgement for an AfyaPlus patient about: {topic}"
)

parser = StrOutputParser()

# It basically takes the output of the previous step and feeds it into the next step, in a sequential manner.
chain = prompt | llm | parser  # the LCEL pipe

# While also using the {template} to format the input for the next step in the chain.
result = chain.invoke({"topic": "a delayed lab result"})
print(result)
