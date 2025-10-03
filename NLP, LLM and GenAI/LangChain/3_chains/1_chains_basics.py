from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv


from langchain_huggingface import HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


from rich.console import Console
import os


import warnings
from datetime import datetime


# Avoid warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Initialize console and load environment variables
console = Console()
load_dotenv()


# HuggingFace LLM setup
hf_api_key = os.getenv("HF_TOKEN")
if not hf_api_key:
    raise ValueError("HF_TOKEN not found in .env file")


# Configure your suitable LLM model
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You're a facts expert who knows facts about {animal}"),
    ("human", "Tell me {fact_cnt} facts about {animal}")
])


# Create the chain: prompt -> LLM -> string output
chain = prompt_template | gemini_llm | StrOutputParser()

# user input for animal and fact count
animal = console.input("[bold blue]Enter an animal you want to learn about: [/bold blue]")
fact_cnt = console.input("[bold blue]How many facts would you like to know? [/bold blue]")

# Run the chain with user inputs
try:
    result = chain.invoke({"animal": animal, "fact_cnt": fact_cnt})
    console.print(f"[bold green]Facts about {animal} (generated {datetime.now()}):[/]")
    console.print(result)
except Exception as e:
    console.print(f"[bold red]Error:[/] {str(e)}")


""" 
# Chatbot-like interface
system_message = SystemMessage(content="You're a helpful AI assistant.")

while True:
    user_prompt = console.input("[bold red]You 👨🏻‍💻:[/bold red] ")

    # If user wants to quit
    if user_prompt.lower() in ("exit", "quit", "q"):
        console.print("[bold green]Goodbye! 👋[/bold green]")
        break

    # Generate AI response
    try:
        ai_response = gemini_llm.invoke(user_prompt)
        console.print(f"\n[bold yellow]{ai_response}[/bold yellow]")
    except Exception as e:
        console.print(f"[red]Error generating AI response: {e}[/red]")
        continue 
"""
