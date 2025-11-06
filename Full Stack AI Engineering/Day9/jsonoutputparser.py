import time
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


from dotenv import load_dotenv
load_dotenv()


model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    api_key=os.getenv("GEMINI_API_KEY")
)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age, strengths, and weaknesses of a superhero:  {name} {format_instruction}",
    input_variables=["name"],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)


start_time = time.monotonic()
chain = template | model | parser
result = chain.invoke({'name': 'Wonder woman'})
end_time = time.monotonic()

print(result)
print(f"Time taken: {end_time - start_time} seconds")
