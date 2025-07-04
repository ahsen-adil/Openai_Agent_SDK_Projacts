import asyncio
import base64
import os
import streamlit as st
from dotenv import load_dotenv

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig

# Load GEMINI API Key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY not set in .env file.")
    st.stop()

# Gemini client setup
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

# Image to base64 converter
def image_to_base64(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode("utf-8")

# Streamlit UI
st.set_page_config(page_title="🍎 Food Freshness Checker", layout="centered")
st.title("🍎 Food Freshness & Safety Agent")
st.markdown("Upload a photo of any **fruit, vegetable, or meat**, and the AI agent will tell you if it's fresh, usable, and how to store it.")

uploaded_file = st.file_uploader("📷 Upload Food Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=250)  # You can change width as you like

    if st.button("🧠 Analyze Food"):
        with st.spinner("Analyzing food image..."):

            async def analyze():
                b64_image = image_to_base64(uploaded_file)

                # Agent instructions
                instructions = """
You are a food safety and freshness detection expert. When given a food image (fruits, vegetables, or meat), analyze it carefully and respond in the following format:

Freshness: [Fresh / Slightly Spoiled / Spoiled]  
Usability: [Safe to eat / Use with caution / Do not consume]  
Storage Advice: [e.g. Store in fridge, freeze, or keep at room temperature]

Strict Rules:
- Do not give extra explanation.
- Focus on visual signs: color, texture, mold, etc.
- Be short and to the point.
- Assume the user is asking for quick food safety advice.
                """

                agent = Agent(
                    name="Food Safety Expert Agent",
                    instructions=instructions.strip(),
                )

                result = await Runner.run(
                    agent,
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "detail": "auto",
                                    "image_url": f"data:image/jpeg;base64,{b64_image}",
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": "Analyze this food and give freshness, usability, and storage advice.",
                        },
                    ],
                    run_config=config
                )

                return result.final_output

            output = asyncio.run(analyze())
            st.success("✅ AI Result:")
            st.markdown(f"<div style='font-size:22px; line-height:1.6'>{output}</div>", unsafe_allow_html=True)
