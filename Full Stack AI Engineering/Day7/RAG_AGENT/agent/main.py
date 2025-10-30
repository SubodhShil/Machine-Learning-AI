from langchain.tools import tool
from ..document_indexing.store_vector_db import create_store_and_access_embeddings
from ..document_indexing.load_data_source import get_article_data

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model



model = init_chat_model("groq:openai/gpt-oss-120b")

import asyncio
article_url = input("Enter the article URL to load and index: ")
docs = asyncio.run(get_article_data(article_url))
vector_store = create_store_and_access_embeddings(docs)


model = 

@tool(response_format="content_and_artifact")
def retrieve_context(query: str, k: int = 3):
    retrieved_docs = vector_store.similarity_search(query, k=k)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


def agent_executor(model, ):
    tools = [retrieve_context]
    prompt = (
        """
        You are an AI assistant that helps users by retrieving relevant information from a set of documents.
        
        Your task is to retrieve the most relevant documents based on the user's query and provide additional context if necessary and guide the user accordingly.
        """
    )
    
    agent = create_agent(model, tools, system_prompt=prompt)
    return agent





if __name__ == "__main__":
    pass