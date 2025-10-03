from fastapi import FastAPI, HTTPException
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatAnthropic, ChatOpenAI
from langchain_anthropic import ChatAnthropic as NewChatAnthropic
from langserve import add_routes
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from rich.console import Console
import os
import warnings
from typing import Dict, Any

# Avoid warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize console and load environment variables
console = Console()
load_dotenv()

# Configure models with error handling
try:
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
    )

    openai_llm = ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        temperature=0.7,
    )

    # Replace with new Anthropic initialization
    anthropic_llm = NewChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0.7,
    )
except Exception as e:
    console.print(f"[red]Error initializing LLM models: {str(e)}[/red]")
    raise

# Initialize FastAPI app with additional metadata
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Dictionary of LLMs
# to enable using other models can be add them here
llm_map = {
    "gemini": gemini_llm,
    "openai": openai_llm,
    "anthropic": anthropic_llm
}

# Add a unified route for jokes
@app.get("/{llm}/joke")
async def get_joke(llm: str, topic: str):
    if llm not in llm_map:
        raise HTTPException(status_code=404, detail="LLM not found")
    
    joke_prompt = ChatPromptTemplate.from_template(
        "You are a professional comedian. Create a funny and appropriate joke about {topic}. "
        "The joke should be clever and suitable for all audiences."
    )
    
    # Use the selected LLM to generate a joke
    selected_llm = llm_map[llm]
    response = (joke_prompt | selected_llm).invoke({"topic": topic})
    return {"joke": response.content}

# Add routes with error handling
try:
    add_routes(
        app,
        gemini_llm,
        path="/gemini",
        config_keys=["temperature"],
    )

    add_routes(
        app,
        openai_llm,
        path="/openai",
        config_keys=["temperature"],
    )

    add_routes(
        app,
        anthropic_llm,
        path="/anthropic",
        config_keys=["temperature"],
    )

    # Prompt template for jokes
    joke_prompt = ChatPromptTemplate.from_template(
        "You are a professional comedian. Create a funny and appropriate joke about {topic}. "
        "The joke should be clever and suitable for all audiences."
    )

    """ While generating joke form '/joke' endpoint, remember to change llm as your choice"""
    add_routes(
        app,
        joke_prompt | gemini_llm,
        path="/joke",
    )

    
except Exception as e:
    console.print(f"[red]Error adding routes: {str(e)}[/red]")
    raise

if __name__ == "__main__":
    import uvicorn
    
    # uvicorn server to run the app
    uvicorn.run(
        app, 
        host="localhost", 
        port=8000,
        log_level="info",
        reload=True
    )
