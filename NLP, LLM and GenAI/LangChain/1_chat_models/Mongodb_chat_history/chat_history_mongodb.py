from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from pymongo import MongoClient
from rich.console import Console
import os
import warnings
from datetime import datetime

# Avoid warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize console and load environment variables
console = Console()
load_dotenv()

# MongoDB Atlas setup
CLUSTER_URI = os.getenv("CLUSTER_URI")
if not CLUSTER_URI:
    raise ValueError("CLUSTER_URI not found in .env file")

DB_NAME = "langchain_test_db"
COLLECTION_NAME = "chat_history"
SESSION_ID = "hDFDuoD0L91zNeSU17sd"

# Initialize MongoDB client
try:
    client = MongoClient(CLUSTER_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    console.print(
        "[green]MongoDB connection established successfully![/green]")
except Exception as e:
    console.print(f"[red]Error connecting to MongoDB: {e}[/red]")
    exit(1)

# HuggingFace LLM setup
hf_api_key = os.getenv("HF_TOKEN")
if not hf_api_key:
    raise ValueError("HF_TOKEN not found in .env file")


Qwen_llm = HuggingFaceEndpoint(
    model="Qwen/QwQ-32B",
    temperature=0.7,
    huggingfacehub_api_token=hf_api_key
)

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
        ai_response = Qwen_llm.invoke(user_prompt)
        console.print(f"\n[bold yellow]{ai_response}[/bold yellow]")
    except Exception as e:
        console.print(f"[red]Error generating AI response: {e}[/red]")
        continue

    # Store prompt and response in MongoDB
    chat_entry = {
        "session_id": SESSION_ID,
        "prompt": user_prompt,
        "response": ai_response,
        "timestamp": datetime.now()
    }
    try:
        collection.insert_one(chat_entry)
        console.print("[green]Your chat entry saved to MongoDB![/green]")
    except Exception as e:
        console.print(f"[red]Error saving to MongoDB: {e}[/red]")

# Display chat history from MongoDB
console.print(f"\n[bold green] --- Message History --- [/bold green]")
try:
    history = collection.find({"session_id": SESSION_ID}).sort("timestamp", 1)
    for entry in history:
        console.print(f"[cyan]You: {entry['prompt']}[/cyan]")
        console.print(f"[yellow]AI: {entry['response']}[/yellow]")
except Exception as e:
    console.print(f"[red]Error retrieving chat history: {e}[/red]")
