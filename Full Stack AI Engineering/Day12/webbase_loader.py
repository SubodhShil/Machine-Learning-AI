from langchain_community.document_loaders import WebBaseLoader
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

import time
from dotenv import load_dotenv
load_dotenv()

# load the web page content
url = "https://www.startech.com.bd/pc-power-pcg24f100d-monitor"
loader = WebBaseLoader(url)
docs = loader.load()
full_text = "\n".join([doc.page_content for doc in docs])
# print(docs, "\n\n", len(docs))


# LLM response
model = init_chat_model("groq:openai/gpt-oss-120b")
parser = StrOutputParser()
prompt = PromptTemplate(
    template='Answer the following question\n{question} from the following text\n{text}',
    input_variables=['text']
)


chain = prompt | model | parser
result = chain.invoke({'question': 'What is the price of PC Power PCG24F100D 24" FHD 100Hz IPS Monitor and warranty', 'text': full_text})
print(result)