# === LangChain Imports ===
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import tool

from pydantic import SecretStr
import os
from dotenv import load_dotenv
load_dotenv()


# === Utility Functions ===
def get_secret(key_name: str) -> SecretStr:
    value = os.getenv(key_name)
    if not value:
        raise ValueError(f"❌ {key_name} not found. Please set it in your .env file.")
    return SecretStr(value)


def load_pdf(file_path):
    pdfLoader = PyPDFLoader(file_path)
    docs = pdfLoader.load()
    return docs


def text_splitter(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    return splitter.split_documents(docs)


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


# === Global Vector Store ===
vector_store = None


# === LangChain Tool ===
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant context from the indexed documents."""
    global vector_store
    if vector_store is None:
        raise ValueError("❌ Vector store is not initialized yet.")
    
    retrieved_docs = vector_store.similarity_search(query, k=3)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# === Setup Agent ===
tools = [retrieve_context]
system_prompt = (
    "You are a helpful assistant with access to a retrieval tool. "
    "When answering questions, use the 'retrieve_context' tool to fetch relevant data first."
)

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5,
    api_key=get_secret("GROQ_API_KEY")
)

agent = create_agent(model, tools, system_prompt=system_prompt)


# === Query Function ===
def answer_query(query: str):
    """Run the full RAG pipeline for a given user query."""
    for step in agent.stream({"messages": [{"role": "user", "content": query}]}, stream_mode="values"):
        step["messages"][-1].pretty_print()


# === Main ===
if __name__ == "__main__":
    pdf_path = "./Demo2_last.pdf"
    document = load_pdf(pdf_path)
    document_splits = text_splitter(document)

    # Initialize global vector_store
    vector_store = create_store_and_access_embeddings(document_splits, add_new_docs=False)

    # Ask your query
    query = "Tell me about the names who created presentation slides?"
    answer_query(query)
