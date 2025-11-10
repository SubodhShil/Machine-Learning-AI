import time
import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence


from dotenv import load_dotenv
load_dotenv()


model = init_chat_model("groq:openai/gpt-oss-120b")
parser = StrOutputParser()


prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic'],
)

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain = RunnableSequence(prompt, model, parser)
print(chain.invoke({'topic': 'AI'}))