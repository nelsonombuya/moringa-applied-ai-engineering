# lab11_rag.py
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
vector_store = InMemoryVectorStore(embeddings)

vector_store.add_texts(
    [
        "AfyaPlus reimburses staff travel within 30 days of an approved claim.",
        "Clinic equipment purchases above $500 require director sign-off.",
    ]
)
retriever = vector_store.as_retriever()

prompt = ChatPromptTemplate.from_template(
    "Answer using only this context. If not found, say 'Information not found.'\n"
    "Context: {context}\nQuestion: {question}"
)
llm = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct",  # or "openai/gpt-oss-20b:free"
    temperature=0.0,
    api_key=environ["OPENROUTER_API_KEY"],  # type: ignore
    base_url="https://openrouter.ai/api/v1",
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("How soon is staff travel reimbursed?"))
