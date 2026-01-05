# ======= IMPORTS =======
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage, BaseMessage

from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from pydantic import SecretStr

import sqlite3

import os
from dotenv import load_dotenv

load_dotenv()


# === Utility Functions ===
def get_secret(key_name: str) -> SecretStr:
    value = os.getenv(key_name)
    if not value:
        raise ValueError(f"❌ {key_name} not found. Please set it in your .env file.")
    return SecretStr(value)


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5,
    api_key=get_secret("GROQ_API_KEY"),
)


# defined graph state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


""" Database """
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)


# checkpointer
# checkpointer = InMemorySaver()
checkpointer = SqliteSaver(conn=conn)


# structure of the graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)


# edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# test
# CONFIG = {"configurable": {"thread_id": "thread-1"}}
# response = chatbot.invoke(
#     {"messages": [HumanMessage(content='My name is Subodh "Chandra Shil')]},
#     config=CONFIG,
# )

# print(response)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])

    return list(all_threads)
