import datetime
import asyncio
import os
import json
import re
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel
import chainlit as cl

from agents import (
    Agent,
    set_tracing_disabled,
    function_tool,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    RunConfig
)

# 🔐 Load API key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not set. Check your .env file.")

# Gemini setup
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

set_tracing_disabled(disabled=True)

# Calendar Event Schema
class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: List[str]
    location: Optional[str] = None
    description: Optional[str] = None

# Date Validation Tool
@function_tool
def validate_date(date_str: str) -> str:
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y"]
    for fmt in formats:
        try:
            return datetime.datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str

# Calendar Event Agent
calendar_agent = Agent(
    name="Calendar Event Extractor",
    instructions="""
    Extract all calendar events from a message. Return a JSON array.
    Each event must include:
    - name
    - date (use YYYY-MM-DD, validated by validate_date)
    - participants (list)
    - location (optional)
    - description (optional)
    If no event is found, return an empty array.
    Format output inside a ```json block.
    """,
    tools=[validate_date]
)

# General Conversation Agent
chat_agent = Agent(
    name="Friendly Chatbot",
    instructions="""
    You are a helpful and friendly assistant. Respond naturally to greetings,
    general questions, or small talk. Be clear, concise, and polite.
    """,
)

# Chainlit Start
@cl.on_chat_start
async def start():
    await cl.Message(content="👋 **Calendar Event Extractor Agent!**\n\nSend me an email or message containing events, and I’ll extract them for you!").send()

# Chainlit Message Handler
@cl.on_message
async def handle_message(message: cl.Message):
    user_input = message.content

    # First try: calendar event extraction
    result = await Runner.run(calendar_agent, user_input, run_config=config)
    cleaned_output = re.sub(r"^```(?:json)?\s*|\s*```$", "", result.final_output.strip())

    try:
        events = json.loads(cleaned_output)
        if isinstance(events, list) and events:
            # Valid events found 
            response = "📅 **Extracted Events:**\n"
            for event in events:
                response += f"\n**Event Name:** {event.get('name')}\n"
                response += f"**Date:** {event.get('date')}\n"
                response += f"**Participants:** {', '.join(event.get('participants', [])) or 'None'}\n"
                response += f"**Location:** {event.get('location') or 'None'}\n"
                response += f"**Description:** {event.get('description') or 'None'}\n"
            await cl.Message(content=response.strip()).send()
            return
    except Exception:
        pass

    # Fallback: general chat
    chat_result = await Runner.run(chat_agent, user_input, run_config=config)
    await cl.Message(content=chat_result.final_output.strip()).send()
