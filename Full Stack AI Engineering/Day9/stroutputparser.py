import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

import time

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash', 
    api_key=os.getenv("GEMINI_API_KEY")
)


template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text: {text}",
    input_variables=["text"],
)

parser = StrOutputParser()

start_time = time.monotonic()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'black hole'})

end_time = time.monotonic()

print(result)
print(f"Time taken: {end_time - start_time} seconds")
