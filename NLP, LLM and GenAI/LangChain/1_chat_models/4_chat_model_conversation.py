# LLM's
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek


from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
from transformers import pipeline


# pip libraries
from rich.console import Console
import os
from dotenv import load_dotenv


# to avoid warnings 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# import classes from langchain
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# external libraries imnport
console = Console()


# Load environment variables from .env file
load_dotenv()


# Deepseek
""" deepseek_llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
) """


hf_api_key = os.getenv("HF_TOKEN")

# Qwen
Qwen_llm = HuggingFaceEndpoint(
    model="Qwen/QwQ-32B",
    temperature=0.7,
    huggingfacehub_api_token=hf_api_key
)

""" Chat bot like interface in console """
chat_history = []

# initial system message
system_message = SystemMessage(content="You're a helpful AI assistant.")
chat_history.append(system_message)

while True:
    user_prompt = console.input("[bold red]You 👨🏻‍💻:[/bold red] ")

    # if user is quitting
    if user_prompt.lower() in ("exit", "quit", "q"):
        console.print("[bold green]Goodbye! 👋[/bold green]")
        break

    ai_response = Qwen_llm.invoke(user_prompt)
    console.print(f"\n\n[bold yellow]{ai_response}[/bold yellow]")


# console.print(f"[bold green] --- Message History --- [/bold green]")
# print(chat_history)
