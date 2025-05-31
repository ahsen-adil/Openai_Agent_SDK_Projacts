import os
import asyncio
from dotenv import load_dotenv
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig

# Load environment variables
load_dotenv()

# Get Gemini API Key from .env
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")

# Create OpenAI-compatible Gemini client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Create model object
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client
)

# Create config for running the agent
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Create Writer Agent
writer_agent = Agent(
    name="✍️ Writer Agent",
    instructions="""
        You are a helpful writing assistant. You can write essays, poems, stories, emails,
        and more. Be creative, clear, and helpful.
    """
)

# Welcome message with big heading
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="""
# ✍️ Writer Agent

Welcome! I can help you write **essays**, **poems**, **emails**, **stories**, and more.  
🟢 Just type what you want me to write!
"""
    ).send()

# Respond to user input
@cl.on_message
async def handle_message(message: cl.Message):
    user_input = message.content

    await cl.Message(content="⏳ Generating your response...").send()

    try:
        # Run synchronous Runner inside async using thread executor
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: Runner.run_sync(
                writer_agent,
                input=user_input,
                run_config=config
            )
        )

        # Send result to Chainlit UI
        await cl.Message(
            author=writer_agent.name,
            content=result.final_output
        ).send()

    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()
