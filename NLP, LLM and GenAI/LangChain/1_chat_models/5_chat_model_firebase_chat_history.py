from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_openai import ChatOpenAI


# LLM's
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
from transformers import pipeline


from rich.console import Console
import os
import hashlib


# to avoid warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# external libraries imnport
console = Console()
load_dotenv()


"""
Steps to replicate this example:
1. Create a Firebase account
2. Create a new Firebase project and FireStore Database
3. Retrieve the Project ID
4. Install the Google Cloud CLI on your computer
    - https://cloud.google.com/sdk/docs/install
    - Authenticate the Google Cloud CLI with your Google account
        - https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
    - Set your default project to the new Firebase project you created
5. pip install langchain-google-firestore
6. Enable the Firestore API in the Google Cloud Console:
    - https://console.cloud.google.com/apis/enableflow?apiid=firestore.googleapis.com&project=crewai-automation
"""


# Firestore credential invoke
PROJECT_ID = os.getenv('FIRESTORE_PROJECT_ID')
# print(PROJECT_ID)

CLIENT = firestore.Client(project=PROJECT_ID)
print(CLIENT)

# SESSION_ID = hashlib.sha256(str(PROJECT_ID).encode()).hexdigest()
SESSION_ID = "hDFDuoD0L91zNeSU17sd"
COLLECTION_NAME = "chat_history"

print(SESSION_ID)


doc_ref = CLIENT.collection(COLLECTION_NAME).document(SESSION_ID)

# Only create if it doesn't exist
if not doc_ref.get().exists:
    # Required field for langchain-google-firestore
    doc_ref.set({"messages": []})

chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=CLIENT
)


print("The chat history has created successfully!!")


# hf_api_key = os.getenv("HF_TOKEN")

# # Qwen
# Qwen_llm = HuggingFaceEndpoint(
#     model="Qwen/QwQ-32B",
#     temperature=0.7,
#     huggingfacehub_api_token=hf_api_key
# )


# """ Chat bot like interface in console """

# # initial system message
# system_message = SystemMessage(content="You're a helpful AI assistant.")

# while True:
#     user_prompt = console.input("[bold red]You 👨🏻‍💻:[/bold red] ")

#     # if user is quitting
#     if user_prompt.lower() in ("exit", "quit", "q"):
#         console.print("[bold green]Goodbye! 👋[/bold green]")
#         break

#     # adding user message
#     chat_history.add_user_message(user_prompt)

#     # adding AI response
#     ai_response = Qwen_llm.invoke(user_prompt)
#     chat_history.add_ai_message(ai_response)
#     console.print(f"\n\n[bold yellow]{ai_response}[/bold yellow]")


# console.print(f"\n\n[bold green] --- Message History --- [/bold green]")
# print(chat_history)
