# challenge13_grounding.py
# Core skill (from Lab 11): a retriever feeding a grounded prompt in an LCEL chain.

from os import environ

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)
store = InMemoryVectorStore(embeddings)
store.add_texts(["AfyaPlus reimburses staff travel within 30 days of an approved claim."])
retriever = store.as_retriever()

# In Lab 11 your prompt answered from {context}. Here the goal is stricter:
# the model must REFUSE when the answer is not in the context.
# Write the prompt so an out-of-scope question returns a fixed refusal phrase.
prompt = ChatPromptTemplate.from_template(
    # ... your grounded, refusal-enforcing template here ...
    "Answer using only this context. If not found, say 'Information not found.'\n"
    "Context: {context}\nQuestion: {question}"
)

llm = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct",  # or "openai/gpt-oss-20b:free"
    temperature=0.0,
    api_key=environ["OPENROUTER_API_KEY"],  # type: ignore
    base_url="https://openrouter.ai/api/v1",
)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print(chain.invoke("Best tourist beaches in Mombasa?"))  # should refuse
