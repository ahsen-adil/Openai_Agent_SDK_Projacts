# imports
from youtube_transcript_api import YouTubeTranscriptApi
import re
from agents import Agent, function_tool, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv
import asyncio
import os

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
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Create config for running the agent
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)
# define instructions
instructions = "You provide help with tasks related to YouTube videos. and always use tool `fetch_youtube_transcript` to fetch the transcript of a YouTube video. " \

# define tool
@function_tool
def fetch_youtube_transcript(url: str) -> str:
    """
    Extract transcript with timestamps from a YouTube video URL and format it for LLM consumption
    
    Args:
        url (str): YouTube video URL
        
    Returns:
        str: Formatted transcript with timestamps, where each entry is on a new line
             in the format: "[MM:SS] Text"
    """
    # Extract video ID from URL
    video_id_pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    video_id_match = re.search(video_id_pattern, url)
    
    if not video_id_match:
        raise ValueError("Invalid YouTube URL")
    
    video_id = video_id_match.group(1)
    
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Format each entry with timestamp and text
        formatted_entries = []
        for entry in transcript:
            # Convert seconds to MM:SS format
            minutes = int(entry['start'] // 60)
            seconds = int(entry['start'] % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            
            formatted_entry = f"{timestamp} {entry['text']}"
            formatted_entries.append(formatted_entry)
        
        # Join all entries with newlines
        return "\n".join(formatted_entries)
    
    except Exception as e:
        raise Exception(f"Error fetching transcript: {str(e)}")

# define agent
agent = Agent(
    name="YouTube Transcript Agent",
    instructions=instructions,
    tools=[fetch_youtube_transcript],
    )

# test agent
async def main():
    input_items = []

    print("=== YouTube Transcript Agent ===")
    print("Type 'exit' to end the conversation")
    print("Ask me anything about YouTube videos!")

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        input_items.append({"content": user_input, "role": "user"})
        
        # Check for exit command
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nGoodbye!")
            break
            
        if not user_input:
            continue

        print("\nAgent: ", end="", flush=True)
        result = Runner.run_streamed(
            agent,
            input=input_items,
            run_config=config,
        )
        
        async for event in result.stream_events():
            # We'll ignore the raw responses event deltas
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
            elif event.type == "agent_updated_stream_event":
                continue
            elif event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    print("\n-- Fetching transcript...")
                elif event.item.type == "tool_call_output_item":
                    input_items.append({"content": f"Transcript:\n{event.item.output}", "role": "system"})
                    print("-- Transcript fetched.")
                elif event.item.type == "message_output_item":
                    input_items.append({"content": f"{event.item.raw_item}", "role": "assistant"})
                else:
                    pass  # Ignore other event types

        print("\n")  # Add a newline after each response

if __name__ == "__main__":
    asyncio.run(main())