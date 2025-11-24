from langgraph.graph import END, MessageGraph, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage
from chains import generation_chain, reflection_chain
from typing import List, Sequence
from pydantic import SecretStr
from dotenv import load_dotenv

import time

load_dotenv()


graph = MessageGraph()
REFLECT = "reflect"
GENERATE = "generate"


def generate_node(state):
    return generation_chain.invoke({
        "messages": state
    })


def reflect_node(state):
    return reflection_chain.invoke({
        "messages": state
    })
    return [HumanMessage(content=response.content)]


# Create the nodes
graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

# set the entry point of node
graph.set_entry_point(GENERATE)


LOOP_COUNT = 6
def should_continue(state):
    if (len(state) > LOOP_COUNT):
        return END
    return REFLECT


# right after the generation task we should go for 'should_continue'
graph.add_conditional_edges(GENERATE, should_continue)

# if 'should_continue' is only reflect then it will have only one path which is generate
graph.add_edge(REFLECT, GENERATE)


app = graph.compile()
print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

start_time = time.monotonic()
response = app.invoke(HumanMessage(content="Future of AI agents"))
end_time = time.monotonic()


print(response)
print(f"\nTotal time taken {end_time - start_time} seconds")

