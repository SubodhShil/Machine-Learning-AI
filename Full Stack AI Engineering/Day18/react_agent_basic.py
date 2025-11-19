import datetime
import time
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import tool
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv()


start_time = time.monotonic()


""" === With google.genai library ===  """
# client = genai.Client()
# response = client.models.generate_content(
#     model="gemini-3-pro-preview",
#     contents="How does AI work?",
#     config=types.GenerateContentConfig(
#         thinking_config=types.ThinkingConfig(thinking_level="low")
#     ),
# )
# result = response.text

llm_model = ChatGoogleGenerativeAI(
    model="gemini-3-pro-preview",
    api_key=os.getenv("GEMINI_API_KEY"),
)
# result = llm_model.invoke("Give me a fact about LLM.").content

search_tool = TavilySearch(search_depth="basic")


# == Creating a custom tool
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Get the current system time in the specified format.
    
    Args:
        format: Python datetime format string (default: "%Y-%m-%d %H:%M:%S")
    
    Returns:
        Formatted current time as a string
    """
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time


agent = create_agent(
    tools=[search_tool, get_system_time],
    model=llm_model,
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "When was Sheikh Hasina got death sentence, how many days ago it was announce from this instance"}
    ]
})

end_time = time.monotonic()
print(result)
print(f"Time taken: {end_time - start_time} seconds")
