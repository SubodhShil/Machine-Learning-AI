from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv


from langchain_huggingface import HuggingFaceEndpoint
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
llm = None


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
        ai_response = llm.invoke(user_prompt)
        console.print(f"\n[bold yellow]{ai_response}[/bold yellow]")
    except Exception as e:
        console.print(f"[red]Error generating AI response: {e}[/red]")
        continue
