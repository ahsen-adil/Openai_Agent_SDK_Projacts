import asyncio
import base64
import os
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig

# Load API Key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY not set in .env file.")
    st.stop()

# Setup Gemini API
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

def image_to_base64(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode("utf-8")

# Inject dark theme styling
st.markdown("""
    <style>
    body, .stApp {
        background-color: #121212;
        color: #e0e0e0;
    }
    h1, h2, h3 {
        color: #FFD54F;
        text-align: center;
    }
    .stButton>button {
        background-color: #FFD54F;
        color: black;
        border-radius: 10px;
        font-weight: bold;
        padding: 0.6em 1.5em;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ffb300;
        color: white;
    }
    .pill-box {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        color: #e0e0e0;
        font-size: 17px;
        line-height: 1.6;
    }
    .stFileUploader {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 1em;
    }
    </style>
""", unsafe_allow_html=True)

# App UI
st.set_page_config(page_title="💊 Pill Identifier Agent", layout="centered")

st.markdown("<h1>💊 Pill Identifier Agent</h1>", unsafe_allow_html=True)
st.markdown("""
Upload a **photo of a pill or tablet**, and this smart agent will identify its **name, usage, dosage**, and give you **warnings** if needed.  
Helps avoid wrong medication, especially for elders. ⚠️💊
""")

uploaded_file = st.file_uploader("📷 Upload Pill Image", type=["jpg", "jpeg", "png"])

class Pill(BaseModel):
  Name: str
  Dosage: str
  Use: str
  Warnings: str

if uploaded_file:
    st.image(uploaded_file, caption="📸 Uploaded Pill", width=250)
    st.info("✅ Tip: Use a clear, single-pill image for best results.")

    if st.button("🔍 Identify Pill"):
        with st.spinner("🧠 AI is analyzing... please wait."):

            async def analyze():
                b64_image = image_to_base64(uploaded_file)

                instructions = """
You are a pharmaceutical expert. When a user uploads an image of a pill or tablet, you must:

1. Identify the pill name or type (based on common visuals).
2. Provide typical dosage information (if known).
3. List its primary use (e.g., pain relief, antibiotic).
4. Warn about common side effects and who should avoid it.

Respond in this format only:

**Name:**  
**Dosage:**  
**Use:**  
**Warnings:**

Avoid extra explanation unless critical. If the image is unclear or unidentifiable, say so politely.
"""

                agent = Agent(
                    name="Pill Identifier Expert",
                    instructions=instructions.strip(),
                    output_type=Pill,
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
                            "content": "Identify this pill and give details as per your instructions.",
                        },
                    ],
                    run_config=config
                )

                return result.final_output

            response = asyncio.run(analyze())

        st.markdown("### ✅ Pill Information")
        st.markdown(f"<div class='pill-box'>{response}</div>", unsafe_allow_html=True)

else:
    st.warning("📎 Please upload an image to continue.")
