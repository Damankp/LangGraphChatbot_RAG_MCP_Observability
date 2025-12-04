#Importing libraries
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import requests

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# ------------------------------1. Load environment variables------------------------------

load_dotenv()

# ------------------------------2. Define LLM and Model------------------------------------

# Creating model using HuggingFaceEndpoint and ChatHuggingFace

# Using HuggingFace API to access OpenSource LLM - LLAMA
# llm = HuggingFaceEndpoint(
#     # repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     repo_id="openai/gpt-oss-120b",
#     task="text-generation"
# )

# model = ChatHuggingFace(llm = llm)

# Using OpenAI LLM
llm = ChatOpenAI()

# ------------------------------3. Tools---------------------------------------------------

# Web search tool
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=7EBOK466A62RQXNY"
    r = requests.get(url)
    return r.json()


tools = [search_tool, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(tools)

# ------------------------------4. Graph State---------------------------------------------
# Defining state class for graph
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ------------------------------5. Graph Nodes----------------------------------------------

#Defining chat node function for graph
def chat_node(state: ChatState) -> ChatState:
    """LLM node that may answer or request a tool call."""

    #user query from state
    messages = state['messages']

    #send the query to the model
    response = llm_with_tools.invoke(messages)

    #save the response to the state
    return {'messages': [response]}

tool_node = ToolNode(tools)

# ------------------------------6. Checkpointer----------------------------------------------

# Sqlite saver and Checkpointer
conn = sqlite3.connect(database = 'langgraph_chatbot.db', check_same_thread=False)  # Ensure the database file is created
checkpointer = SqliteSaver(conn=conn)

# ------------------------------7. Graph Definition----------------------------------------------

# Creating graph
graph = StateGraph(ChatState)

# Add nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# Add edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

# Compile graph into chatbot
chatbot = graph.compile(checkpointer=checkpointer)

# ------------------------------8. Utility Functions----------------------------------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)