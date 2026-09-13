from os import environ

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()
# TODO: Step 1 - Process procurement_policy into a text string list chunk array
policy_texts = [
    "AfyaPlus reimburses staff travel within 30 days of an approved claim.",
    "Clinic equipment purchases above $500 require director sign-off.",
    "Procurement above $1,000 needs two approvals.",
]

# TODO: Step 2 - Initialize OpenAIEmbeddings and push the texts into an
# InMemoryVectorStore
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_texts(policy_texts)

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

# TODO: Step 3 - Construct the LCEL rag_chain binding the retriever, prompt, and
# ChatOpenAI model
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# TODO: Step 4 - Invoke and test the chain with the prompt:
# "What approval signature is needed for a logistics hardware purchase of $1,500?"
print(rag_chain.invoke("How many approvals are needed above $1,000?"))
