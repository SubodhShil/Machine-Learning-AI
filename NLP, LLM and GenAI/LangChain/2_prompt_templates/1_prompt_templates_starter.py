from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint


from rich.console import Console
import os


import warnings
from datetime import datetime


# prompt template
from langchain_core.prompts import ChatPromptTemplate


# Avoid warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Initialize console and load environment variables
console = Console()
load_dotenv()


# HuggingFace LLM setup
hf_api_key = os.getenv("HF_TOKEN")
if not hf_api_key:
    raise ValueError("HF_TOKEN not found in .env file")


gemini_llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
)


# Template for the chat prompt
template = "Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skills} as a key strength. Keep it to 4 lines max."


prompt_template = ChatPromptTemplate.from_template(template)


prompt = prompt_template.invoke({
    "tone": "energetic",
    "company": "Google",
    "position": "SDE 2",
    "skills": "Software Engineering, AI"
})


result = gemini_llm.invoke(prompt)
console.print(f"[bold yellow]{result.content}[/bold yellow]")
