from langchain_community.document_loaders import TextLoader
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

import time
from dotenv import load_dotenv
load_dotenv()

loader = TextLoader('test.txt', encoding='utf-8')
docs = loader.load()

print(type(docs), "\n\n\n", docs)


model = init_chat_model("groq:openai/gpt-oss-120b")
parser = StrOutputParser()
prompt = PromptTemplate(
    template='Write a comprehensive summary for the following text\n{text}',
    input_variables=['text']
)

start_time = time.monotonic()
chain = prompt | model | parser
result = chain.invoke({'text': docs[0].page_content})
end_time = time.monotonic()
print(result)
print(f"Time taken: {end_time - start_time} seconds")
