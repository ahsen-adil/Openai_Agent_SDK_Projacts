import streamlit as st
import asyncio
from dataclasses import dataclass
from typing import List
from agents import Agent, Runner, set_tracing_disabled, function_tool, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get Gemini API Key
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")

# Gemini client setup
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Gemini-compatible model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# RunConfig without model_provider
config = RunConfig(
    model=model,
    tracing_disabled=True
)

set_tracing_disabled(disabled=True)

# --- Context and Tools ---
@dataclass
class Purchase:
    id: str
    name: str
    price: float
    date: str

@dataclass
class UserContext:
    uid: str
    is_pro_user: bool

    async def fetch_purchases(self) -> List[Purchase]:
        mock_data = {
            "user123": [
                Purchase(id="p1", name="Basic Plan", price=9.99, date="2023-01-15"),
                Purchase(id="p2", name="Premium Add-on", price=4.99, date="2023-02-20")
            ],
            "user456": [],
            "user789": [
                Purchase(id="p3", name="Enterprise Plan", price=49.99, date="2024-05-01")
            ],
            "user999": [
                Purchase(id="p4", name="Pro Membership", price=19.99, date="2024-03-10"),
                Purchase(id="p5", name="Training Module", price=14.99, date="2024-03-15")
            ]
        }
        return mock_data.get(self.uid, [])

@function_tool
async def get_user_info(context: UserContext) -> str:
    user_type = "Pro" if context.is_pro_user else "Free"
    return f"🆔 User ID: `{context.uid}`\n👤 Account Type: **{user_type}**"

@function_tool
async def get_purchase_history(context: UserContext) -> str:
    purchases = await context.fetch_purchases()
    if not purchases:
        return "🛍️ No purchase history found."
    result = "🧾 **Purchase History:**\n"
    for p in purchases:
        result += f"- **{p.name}** – `${p.price}` on `{p.date}`\n"
    return result

@function_tool
async def get_personalized_greeting(context: UserContext) -> str:
    if context.is_pro_user:
        return "💎 Welcome back to our premium service! We value your continued support."
    else:
        return "👋 Welcome! Consider upgrading to our Pro plan for additional features."

# --- Agent Setup ---
user_context_agent = Agent[UserContext](
    name="User Context Agent",
    instructions="""
    You are a helpful assistant that provides personalized responses based on user context.
    Use the available tools to retrieve user information and provide tailored assistance.
    For pro users, offer more detailed information and premium suggestions and always call tool.
    """,
    tools=[get_user_info, get_purchase_history, get_personalized_greeting],
)

# --- Streamlit UI ---
st.set_page_config(page_title="User Assistant", page_icon="🤖")
st.title("🤖 Pro & Free User Track Agent")
st.markdown("Ask your AI assistant about your profile, account type, or purchases!")

# --- Manual User ID Input ---
st.subheader("🔐 Enter Your User ID")
uid_input = st.text_input("🆔 User ID:")
is_pro_input = st.checkbox("🌟 Are you a Pro user?", value=False)

query = st.text_input("💬 Ask something:", "Tell me about myself and my purchases")

# --- LLM Query Execution ---
if st.button("Ask Gemini"):
    if not uid_input.strip():
        st.warning("Please enter a valid User ID before asking.")
    else:
        with st.spinner("Gemini is thinking..."):

            async def get_response():
                context = UserContext(uid=uid_input.strip(), is_pro_user=is_pro_input)
                response = await Runner.run(user_context_agent, query, context=context, run_config=config)
                return response.final_output

            output = asyncio.run(get_response())
            st.success("✅ Gemini's Response:")
            st.markdown(output)
