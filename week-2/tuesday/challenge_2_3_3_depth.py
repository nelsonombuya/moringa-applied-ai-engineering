# challenge_2_3_3_depth.py
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def run_deep_vector_retrieval():
    print("--- Executing Deep Vector Retrieval ---")
    manual_data_dir = Path(__file__).resolve().parent / "manual_data"
    loader = TextLoader(manual_data_dir / "operational_guidelines.txt")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=20
    ).split_documents(loader.load())
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = Chroma.from_documents(chunks, embeddings)

    # TODO: Step 1 - Configure as_retriever with search_kwargs targeting k=3
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # TODO: Step 2 - Invoke with a target query
    query = "What is the policy on appointments?"
    matches = retriever.invoke(query)

    # TODO: Step 3 - Iterate through the match collection and print each item
    for i, match in enumerate(matches):
        print(f"Match {i + 1}: {match.page_content}")


if __name__ == "__main__":
    run_deep_vector_retrieval()
