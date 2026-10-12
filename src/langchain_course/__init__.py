from typing import List
from pydantic import BaseModel, Field 


from dotenv import load_dotenv
import os

load_dotenv()

from langchain import tools
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient
from langchain_tavily import TavilySearch

tavily = TavilyClient()

@tool
def search(query: str):
    """
    Tool that searches over internet for weather information
    Args:
        query (str): The search query
    Returns:
        str: The search results
    """

    print(f"Searching for: {query}")
    return tavily.search(query)

@tool
def searchDB(query: str):
    """
    Tool that searches user data over database
    Args:
        query (str): The search query
    Returns:
        str: The search results
    """

    print(f"Searching for: {query}")
    return "Dara is a user in the system work as a software engineer in finance."


class Source(BaseModel):
    """Schema for a source used by the agent"""
    source: str = Field (description="The url of the source")

class AgentResponse(BaseModel):
    """Schema for the response from the agent with answer and sources"""
    answer: str = Field(description="The result of the agent's action")
    sources: List[Source] = Field(default_factory=list, description="The sources used to generate the answer")

model = os.getenv("OLLAMA_MODEL") 
url = os.getenv("OLLAMA_BASE_URL")
llm = ChatOllama(model=model, temperature=0.7, base_url=url)
tools = [searchDB, TavilySearch(), search]
agent = create_agent(model=llm, tools=tools, system_prompt="You are a helpful assistant.", response_format=AgentResponse)


def main() -> None:
    # message = HumanMessage(content="What's the weather like in Tokyo?")
    # result = agent.invoke({"messages": [message]})
    # print(result)
    message = HumanMessage(content="Who is dara and she want to know the weather in Tokyo?")
    result = agent.invoke({"messages": [message]})
    print(result)
    # message = HumanMessage(content="Search for 3 job postings for AI Engineer using Langgraph where the company is located in MA with main tech stack in .NET")
    # result = agent.invoke({"messages": [message]})
    # print(result)


if __name__ == "__main__":
    main()