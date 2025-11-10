from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import time

demo_text = """
Demo text, often referred to as placeholder text, is a block of text that is used to fill a space on a website, document, or template where the actual content will eventually be placed. The most common example of demo text is “Lorem ipsum,” a pseudo-Latin text used since the 1500s to demonstrate the visual form of a document without relying on meaningful content.

Demo text, often referred to as placeholder text, is a block of text that is used to fill a space on a website, document, or template where the actual content will eventually be placed. The most common example of demo text is “Lorem ipsum,” a pseudo-Latin text used since the 1500s to demonstrate the visual form of a document without relying on meaningful content.
"""

# == Character level splitting
# splitter = CharacterTextSplitter(
#     chunk_size=100,
#     chunk_overlap=20,
#     separator=''
# )


# == Smart recursive text splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
)


# == text based loading ==
# result = splitter.split_text(demo_text)
# print(result)


# == PDF text length splitting ==
start_time = time.monotonic()
loader = PyPDFLoader('Turn_Screw.pdf')
docs = loader.load()


# docs = loader.lazy_load()
# for document in docs:
#     print(document.page_content)
# all_text = "".join([document.page_content for document in docs])

result = splitter.split_documents(docs)
print(result[5].page_content)
print(result[6].page_content)
end_time = time.monotonic()
print(f"Time taken: {end_time - start_time} seconds")
# result = splitter.split_text()
