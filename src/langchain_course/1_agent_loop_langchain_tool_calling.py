from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable

MODEL = os.environ["OLLAMA_MODEL"]  # Get the model name from environment variables
URL = os.environ["OLLAMA_BASE_URL"]  # Get the URL from environment variables
MAX_ITERATIONS = 5  # Set the maximum number of iterations for the agent loop


#   Tools for the agent to use


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in catalog."""
    # Placeholder implementation - replace with actual product price lookup logic
    print(f"Executing get_product_price for product: {product}")
    prices = {
        "laptop": 999.99,
        "mouse": 19.99,
        "keyboard": 49.99,
        "monitor": 199.99,
        "headphones": 89.99,
    }
    return prices.get(product, 0.0)  # Return 0.0 if product not found


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the discounted price.
        Available discount tiers: 
        - none: 0% discount
        - bronze: 5% discount
        - silver: 10% discount
        - gold: 15% discount
    """
    print(f"Executing apply_discount with price: {price} and discount_tier: {discount_tier}")
    discount_rates = {
        "none": 0.0,
        "bronze": 0.05,
        "silver": 0.10,
        "gold": 0.15,
    }
    discount_rate = discount_rates.get(discount_tier, 0.0)
    discounted_price = price * (1 - discount_rate)
    return discounted_price


#  Agent loop to handle user queries and tool calls
@traceable(name="Langchain Agent Loop")
def run_agent(question: str = None):
    tools = [get_product_price, apply_discount]  # List of tools available to the agent
    tool_dict = {tool.name: tool for tool in tools}  # Create a dictionary of tools for easy access


    llm = init_chat_model(model=MODEL, model_provider="ollama", base_url=URL)  # Initialize the chat model with the specified model and URL
    llm_with_tools = llm.bind_tools(tools)  # Bind the tools to the chat model 

    print(f"Question: {question}")  # Print the user's question
    print("=" * 50)  # Print a separator line

    messages = [
        SystemMessage(
            content=(
            "You are a helpful assistant that can answer questions about product prices and apply discounts. "
            "You have access to the following tools: "
            "- get_product_price(product: str) -> float: Look up the price of a product in catalog. "
            "- apply_discount(price: float, discount_tier: str) -> float: Apply a discount tier to a price and return the discounted price. "
            "Available discount tiers: none, bronze, silver, gold. "
            "Please use these tools to answer the user's question."
            "Strict Rules - you must follow these exact rules when answering the user's question: \n"
            "1. Never assume the price of a product or the discount tier. Always use the tools to get the price and apply the discount. \n"
            "2. Only call apply_discount after you have obtained the price of the product using get_product_price. Pass the exact price and discount tier to the function. \n"
            "3. Never calculate the price or discount manually. Always use the tools to get the price and apply the discount. \n"
        )),
        HumanMessage(content=question)  # Add the user's question as a human message
    ]

    for iterations in range(1, MAX_ITERATIONS + 1):  # Loop for a maximum number of iterations
        print(f"Iteration {iterations }/{MAX_ITERATIONS}")  # Print the current iteration
        ai_message = llm_with_tools.invoke(messages)  # Get the AI's response using the chat model with tools

        tool_calls = ai_message.tool_calls  # Extract any tool calls from the AI's response

        if not tool_calls:  # If there are no tool calls, return the AI's response and exit the loop
            print(f"AI Response: {ai_message.content}")  # Print the AI's response
            return ai_message.content

        # Process only 1 tool call per iteration to avoid infinite loops
        tool_call = tool_calls[0]  # Get the first tool call
        tool_name = tool_call.get("name")  # Get the name of the tool being called
        tool_args = tool_call.get("args", {})  # Get the arguments for the tool call
        tool_call_id = tool_call.get("id")  # Get the ID of the tool call

        print(f"Tool Call: {tool_name} with args: {tool_args}")  # Print the tool call details

        tool_to_use = tool_dict.get(tool_name)  # Get the tool function from the dictionary using the tool name
        if tool_to_use is None:  # If the tool is not found, print an error
            raise ValueError(f"Tool '{tool_name}' not found. Available tools: {list(tool_dict.keys())}")

        # Call the tool function with the provided arguments
        observation = tool_to_use.invoke(tool_args)  # Invoke the tool with the provided arguments
        print(f"Tool Result: {observation}")  # Print the observation returned by the tool

        # Add the tool call and observation to the messages for the next iteration
        messages.append(ai_message)  # Add the AI's message to the messages list
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))
        print(f"Tool_call_id: {tool_call_id}") # Print the tool call ID and observation

    print("Error: Max iterations reached without a final answer.")  # Print an error if the maximum iterations are reached
    return None  # Return None if the maximum iterations are reached without a final answer



if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of the laptop with a gold discount?")  # Start the agent loop when the script is executed