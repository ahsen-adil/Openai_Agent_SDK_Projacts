# agent.py

from youtube_transcript_api import YouTubeTranscriptApi
import re
import os
from dotenv import load_dotenv
import chainlit as cl
from agents import Agent, function_tool, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from openai.types.responses import ResponseTextDeltaEvent

# Load environment variables
load_dotenv()

# Get Gemini API Key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")

# Setup Gemini client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Setup model and config
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Agent instructions
instructions = (
    "You provide help with tasks related to YouTube videos. "
    "Always use the `fetch_youtube_transcript` tool to fetch the transcript of a YouTube video."
)

# Tool function
@function_tool
def fetch_youtube_transcript(url: str) -> str:
    """
    Fetch transcript from YouTube video and return formatted result.
    """
    video_id_pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    video_id_match = re.search(video_id_pattern, url)

    if not video_id_match:
        raise ValueError("Invalid YouTube URL")

    video_id = video_id_match.group(1)

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        formatted_entries = []
        for entry in transcript:
            minutes = int(entry['start'] // 60)
            seconds = int(entry['start'] % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            formatted_entries.append(f"{timestamp} {entry['text']}")
        return "\n".join(formatted_entries)
    except Exception as e:
        raise Exception(f"Error fetching transcript: {str(e)}")

# Define the agent
agent = Agent(
    name="YouTube Transcript Agent",
    instructions=instructions,
    tools=[fetch_youtube_transcript],
)

# Chainlit heading on chat start
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="# 🎬 YouTube Transcript Agent\n\nPaste a YouTube video link and ask anything like:\n- 'Summarize this video'\n- 'List key points'\n- 'What is the video about?'\n\nThe agent will automatically fetch the transcript and respond."
    ).send()

# Chainlit message handler
@cl.on_message
async def handle_message(message: cl.Message):
    input_items = [{"role": "user", "content": message.content}]
    msg = cl.Message(content="")

    result = Runner.run_streamed(agent, input_items, run_config=config)

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            msg.content += event.data.delta
            await msg.stream_token(event.data.delta)

        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                await cl.Message(content="⏳ Fetching transcript...").send()
            elif event.item.type == "tool_call_output_item":
                await cl.Message(content=f"✅ Transcript fetched:\n\n{event.item.output}").send()

    await msg.send()
