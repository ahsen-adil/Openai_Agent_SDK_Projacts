import datetime
import asyncio
import os
import json
import re
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import (
    Agent,
    set_tracing_disabled,
    function_tool,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    RunConfig
)

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

set_tracing_disabled(disabled=True)


# Output schema
class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: List[str]
    location: Optional[str] = None
    description: Optional[str] = None


# Basic Agent
calendar_extractor = Agent(
    name="Basic Calendar Event Extractor",
    instructions="""
    You are an assistant that extracts one calendar event from plain text. 
    Return a structured JSON object with:
    - name (event name)
    - date (YYYY-MM-DD)
    - participants (list of names)
    - location (if mentioned)
    - description (if available)
    """,
    output_type=CalendarEvent,
)

# Tool for validating date
@function_tool
def validate_date(date_str: str) -> str:
    """Validate and format a date string to YYYY-MM-DD format"""
    try:
        formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y"]
        for fmt in formats:
            try:
                parsed_date = datetime.datetime.strptime(date_str, fmt)
                return parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return date_str
    except Exception:
        return date_str

# Advanced Agent
advanced_calendar_extractor = Agent(
    name="Advanced Calendar Event Extractor",
    instructions="""
    You are an assistant that extracts all calendar events from a message. Return a JSON array.
    Each event must include:
    - name
    - date (use YYYY-MM-DD format using validate_date tool)
    - participants (names in a list)
    - location (if any)
    - description (if any)
    If no value is present for a field, use null or an empty list.
    
    Format the response inside a JSON code block (```json ... ```).
    """,
    tools=[validate_date],
)

# Main execution
async def main():
    simple_text = "Let's have a team meeting on 2023-05-15 with John, Sarah, and Mike."

    complex_text = """
    Hi team,

    I'm scheduling our quarterly planning session for May 20, 2023 at the main conference room.
    All department heads (Lisa, Mark, Jennifer, and David) should attend. We'll be discussing
    our Q3 objectives and reviewing Q2 performance. Please bring your department reports.

    Also, don't forget about the company picnic on 06/15/2023!
    """

    print("\n--- Basic Calendar Extractor (Gemini) ---")
    result = await Runner.run(
        calendar_extractor,
        simple_text,
        run_config=config
    )
    print("Extracted Event:", result.final_output)

    print("\n--- Advanced Calendar Extractor with Tool (Gemini) ---")
    result = await Runner.run(
        advanced_calendar_extractor,
        complex_text,
        run_config=config
    )

    print("Raw JSON Output:\n", result.final_output)

    # Clean and parse Gemini response
    cleaned_output = re.sub(r"^```(?:json)?\s*|\s*```$", "", result.final_output.strip())

    try:
        events = json.loads(cleaned_output)
        print("\nParsed Events:")
        for event in events:
            print(f"\nEvent Name: {event.get('name')}")
            print(f"Date: {event.get('date')}")
            print(f"Participants: {', '.join(event.get('participants', [])) or 'None'}")
            print(f"Location: {event.get('location') or 'None'}")
            print(f"Description: {event.get('description') or 'None'}")
    except Exception as e:
        print("Error parsing response:", e)


if __name__ == "__main__":
    asyncio.run(main())
