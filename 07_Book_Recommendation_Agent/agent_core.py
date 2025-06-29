# agent_core.py
import aiohttp
from agents import Agent, function_tool, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
import os
from dotenv import load_dotenv

load_dotenv()

# Setup Gemini
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not set in .env")

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

# Tool
@function_tool
async def recommend_books(topic: str) -> str:
    url = f"https://www.googleapis.com/books/v1/volumes?q={topic}&maxResults=5"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                return f"❌ Failed to fetch books for: {topic}"
            data = await response.json()

    books = data.get("items", [])
    if not books:
        return f"❌ No books found for: {topic}"

    output = f"📚 Recommended Books for **{topic}**:\n"
    for idx, book in enumerate(books, 1):
        info = book.get("volumeInfo", {})
        title = info.get("title", "Unknown Title")
        authors = ", ".join(info.get("authors", ["Unknown Author"]))
        desc = info.get("description", "No description available.")[:200] + "..."
        link = info.get("infoLink", "#")

        output += (
            f"\n{idx}. **{title}** by *{authors}*\n"
            f"   🔹 {desc}\n"
            f"   🔗 [More Info]({link})\n"
        )

    return output.strip()

# Agent setup
instructions = (
    "You're a helpful reading assistant. "
    "When users ask for book suggestions, use the `recommend_books` tool to fetch real book data."
)

agent = Agent(
    name="Book Recommendation Agent",
    instructions=instructions,
    tools=[recommend_books]
)
