# LLM's
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# from langchain_community.llms import HuggingFaceHub
# from transformers import pipeline


from rich.console import Console
import os
from dotenv import load_dotenv


# import classes from langchain
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# external libraries imnport
console = Console()


# Load environment variables from .env file
load_dotenv()


messages = [
    SystemMessage("You are a history expert."),
    HumanMessage("I would like to know about the history of Asia.")
]


""" API keys """
hf_api_key = os.getenv("HF_TOKEN")


""" gemma_llm = HuggingFaceHub(
    repo_id="google/gemma-7b",
    model_kwargs={"temperature": 0.7, "max_new_tokens": 300},
    huggingfacehub_api_token=hf_api_key
)
user_prompt = "History of WW2."
ai_response = gemma_llm.invoke(messages)
console.print(f"[bold yellow]{ai_response}[bold yellow]") """


""" mistral_llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.7},
    huggingfacehub_api_token=hf_api_key
) """


gemini_llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
)


user_prompt = input("Enter your prompt: ")
ai_response = gemini_llm.invoke(user_prompt)
console.print(f"\n\n[bold yellow]{ai_response.content}[/bold yellow]")


# response_lines = ai_response.split("\n")
# Join all lines except the first one
# cleaned_response = "\n".join(response_lines[1:]).strip()

# Print only AI's response
# console.print(f"\n\n[bold green]User: {user_prompt}[/bold green]")
# console.print(f"[bold yellow]AI: {cleaned_response}[/bold yellow]")
