from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv
load_dotenv()


# === Utility Functions ===
def get_secret(key_name: str) -> SecretStr:
    value = os.getenv(key_name)
    if not value:
        raise ValueError(f"❌ {key_name} not found. Please set it in your .env file.")
    return SecretStr(value)



model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5,
    api_key=get_secret("GROQ_API_KEY")
)



generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent "
        ),
    ]
)
