import time
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field
import time 

from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    api_key=os.getenv("GOOGLE_API_KEY")
)


class Person(BaseModel):
    name: str = Field(description="name of the person")
    age: int = Field(description="age of the person")
    city: str = Field(description="Name of the city person belongs to")


parser = PydanticOutputParser(pydantic_object=Person)

start_time = time.monotonic()
template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n{format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)
chain = template | model | parser
result = chain.invoke({'place': 'American'})
end_time = time.monotonic()

print(result)
print(f"Time taken: {end_time - start_time} seconds")

