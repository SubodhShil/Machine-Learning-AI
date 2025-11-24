from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from pydantic import SecretStr

import os
from dotenv import load_dotenv
load_dotenv()


generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            "Generate the best twitter post possible for the user's request."
            "If the user provides critique, respond with a revised version of your previous attempts"
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)


reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet"
            "Always provide detailed recommendations, including requests for length, virality, style etc."
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)


# === Utility Functions ===
def get_secret(key_name: str) -> SecretStr:
    value = os.getenv(key_name)
    if not value:
        raise ValueError(
            f"❌ {key_name} not found. Please set it in your .env file.")
    return SecretStr(value)


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5,
    api_key=get_secret("GROQ_API_KEY")
)


generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm
