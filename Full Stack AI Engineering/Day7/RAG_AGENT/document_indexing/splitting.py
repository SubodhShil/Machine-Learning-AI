from langchain_text_splitters import RecursiveCharacterTextSplitter
from .load_data_source import get_article_data


async def split_documents(url: str):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True,
    )

    docs = await get_article_data(url)
    all_splitted_docs = text_splitter.split_documents(docs)

    return all_splitted_docs

if __name__ == "__main__":
    import asyncio

    url = "https://www.bbc.com/news/articles/cvgmp428plvo"
    splitted_docs = asyncio.run(split_documents(url))
    print(f"✅ Document split into {len(splitted_docs)} chunks.")