from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from pydantic import SecretStr
import os
from dotenv import load_dotenv
load_dotenv()


def create_store_and_access_embeddings(document_splits, add_new_docs=False):
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

    if add_new_docs:
        vector_store.add_documents(document_splits)

    stored_data = vector_store.get()
    count = len(stored_data.get("ids", []))
    print(f"✅ Chroma collection currently contains {count} document embeddings.")
    return vector_store

if __name__ == "__main__":
    pass