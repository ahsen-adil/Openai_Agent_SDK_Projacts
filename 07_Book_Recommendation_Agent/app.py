# app.py

import streamlit as st
import asyncio
from agents import Runner
from openai.types.responses import ResponseTextDeltaEvent
from agent_core import agent, config

# === Page Config ===
st.set_page_config(page_title="📚 Book Recommendation Agent", layout="centered")

# === Header ===
st.markdown("""
    <h1 style='text-align: center;'>📚 Book Recommendation Agent</h1>
    <p style='text-align: center; font-size: 18px;'>Get personalized book suggestions powered by AI</p>
    <hr style="margin: 20px 0;">
""", unsafe_allow_html=True)

# === Input Field ===
st.markdown("### 🔍 Enter a Topic, Genre, or Author:")
user_input = st.text_input("", placeholder="e.g. artificial intelligence, self-help, J.K. Rowling")

# === Centered Button ===
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search_clicked = st.button("✨ Recommend Books", use_container_width=True)

# === Result Area ===
if search_clicked and user_input:
    input_items = [{"role": "user", "content": user_input}]
    result_placeholder = st.empty()
    loading_msg = st.empty()
    loading_msg.markdown("⏳ *Fetching recommendations... Please wait.*")

    async def run_agent():
        result = Runner.run_streamed(agent, input=input_items, run_config=config)
        output = ""

        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                output += event.data.delta
                result_placeholder.markdown(output)
            elif event.type == "run_item_stream_event":
                if event.item.type == "tool_call_output_item":
                    input_items.append({"role": "system", "content": event.item.output})
                elif event.item.type == "message_output_item":
                    input_items.append({"role": "assistant", "content": event.item.raw_item})

    asyncio.run(run_agent())
    loading_msg.empty()  # ✅ Hide loading message after done

# === Footer ===
st.markdown("""
    <hr style="margin-top: 40px;">
    <div style='text-align: center; font-size: 14px; color: gray;'>
        Made with ❤️ by Ahsan · Powered by Gemini + Google Books API
    </div>
""", unsafe_allow_html=True)
