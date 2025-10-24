from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from pydantic import SecretStr


import os
from dotenv import load_dotenv
load_dotenv()


def load_pdf(file_path):
    pdfLoader = PyPDFLoader(file_path)
    docs = pdfLoader.load()

    return docs


def text_splitter(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    return all_splits


# Utilizes google embedding models to create embeddings
# and store them in Chroma vector database
def create_and_store_embeddings(document_splits):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_chroma import Chroma
    import os

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("❌ GOOGLE_API_KEY not found. Please set it in your .env file.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=SecretStr(google_api_key)
    )

    vector_store = Chroma(
        collection_name="langchain_test",
        embedding_function=embeddings,
        chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE"),
    )

    vector_store.add_documents(document_splits)

    count = vector_store._collection.count()
    print(f"✅ Successfully stored {count} document embeddings in Chroma.")

    results = vector_store.similarity_search("test", k=1)
    if results:
        print("✅ Embedding retrieval test passed.")
    else:
        print("⚠️ Retrieval test failed or no data found.")

    return vector_store


if __name__ == "__main__":
    pdf_path = "./Demo2_last.pdf"
    document = load_pdf(pdf_path)

    document_splits = text_splitter(document)
    # print(f"Number of splits: {len(document_splits)}")
    # print(document_splits[0].page_content)
    create_and_store_embeddings(document_splits)

