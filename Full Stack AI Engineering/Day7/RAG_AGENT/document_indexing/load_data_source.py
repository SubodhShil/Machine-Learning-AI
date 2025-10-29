from bs4 import BeautifulSoup, SoupStrainer  # type: ignore
from langchain_community.document_loaders import WebBaseLoader
import asyncio


async def get_article_data(url: str):

    # === Using BeautifulSoup and Langchain to parse and load only <article> tags ===
    only_articles = SoupStrainer("article")

    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": only_articles},
    )
    docs = await asyncio.to_thread(loader.load)

    if not docs:
        return "❌ No article tags found."

    return docs


if __name__ == "__main__":
    print(asyncio.run(get_article_data("https://www.bbc.com/news/articles/cvgmp428plvo")))
