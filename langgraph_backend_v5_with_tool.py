#Importing libraries
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Creating model using HuggingFaceEndpoint and ChatHuggingFace
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

# Defining state class for graph
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

#Defining chat node function for graph
def chat_node(state: ChatState) -> ChatState:
    #user query from state
    messages = state['messages']

    #send the query to the model
    response = model.invoke(messages)

    #save the response to the state
    return {'messages': [response]}

# Creating graph
graph = StateGraph(ChatState)

# Add nodes
graph.add_node("chat_node", chat_node)

# Add edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Sqlite saver and Checkpointer
conn = sqlite3.connect(database = 'langgraph_chatbot.db', check_same_thread=False)  # Ensure the database file is created
checkpointer = SqliteSaver(conn=conn)

# Compile graph into chatbot
chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)