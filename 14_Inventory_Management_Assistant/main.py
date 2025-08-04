import chainlit as cl
from agents import Agent, Runner, function_tool
import json
import os
from dotenv import load_dotenv
from agents import AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

@function_tool
def add_to_inventory(item: str, quantity: int) -> str:
    try:
        if os.path.exists("inventory.json"):
            with open("inventory.json", "r") as file:
                inventory = json.load(file)
        else:
            inventory = {}
        inventory[item] = inventory.get(item, 0) + quantity
        with open("inventory.json", "w") as file:
            json.dump(inventory, file, indent=4)
        return f"Added {quantity} of {item} to inventory."
    except Exception as e:
        return f"Error adding to inventory: {str(e)}"

@function_tool
def get_inventory(item: str) -> str:
    try:
        if os.path.exists("inventory.json"):
            with open("inventory.json", "r") as file:
                inventory = json.load(file)
            quantity = inventory.get(item, 0)
            return f"There are {quantity} {item} in inventory."
        else:
            return f"No inventory found for {item}."
    except Exception as e:
        return f"Error reading inventory: {str(e)}"

inventory_agent = Agent(
    name="banana",
    instructions="You are an inventory management agent named banana. You can add items to the inventory and check how many of a specific item are available.",
    tools=[add_to_inventory, get_inventory]
)

@cl.on_chat_start
async def start():
    await cl.Message(
        content="# Inventory Management Assistant 🏷️\n\nWelcome! I am your inventory assistant. You can:\n- Add items to inventory\n- Check item quantity\n\n_Just type your request below!_"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    user_input = message.content
    result = Runner.run_sync(inventory_agent, user_input, run_config=config)
    await cl.Message(content=result.final_output).send()

if __name__ == "__main__":
    cl.run()
