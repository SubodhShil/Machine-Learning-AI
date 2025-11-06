import os 
import time 
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()


class Review(BaseModel):
    key_themes:list[str] = Field(description="Write down all the key themes discussed in the review")
    summary: str = Field(description="")
    sentiment: Literal['positive', 'negative'] = Field(description="Overall sentiment of the review")
    pros: Optional[list[str]] = Field(description="List of pros mentioned in the review")


model = GoogleGenerativeAI(model='gemini-2.5-flash',api_key=os.getenv("GEMINI_API_KEY"))
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
This product was fantastic! It exceeded all my expectations and I would highly recommend it to anyone looking for quality and reliability.                                
""")

print(result)