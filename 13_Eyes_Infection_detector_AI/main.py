import asyncio
import base64
import os
import streamlit as st
from dotenv import load_dotenv

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig

# Load Gemini API Key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY not set.")
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

# Convert image to base64
def image_to_base64(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode("utf-8")

# Improved Streamlit UI
st.set_page_config(page_title="👁️ Eye Infection Detector", layout="centered")
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>👁️ Eye Infection Detector AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Upload a <strong>close-up photo of an eye</strong>.<br>The AI will detect redness, dryness, or symptoms of infection and give a quick suggestion.</p>", unsafe_allow_html=True)

st.markdown("---")

uploaded_file = st.file_uploader("📷 Upload Eye Photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="🖼️ Uploaded Eye Image", width=300)

    st.markdown("")

    analyze_btn = st.button("🔍 Analyze Eye", use_container_width=True)

    if analyze_btn:
        with st.spinner("⏳ Analyzing eye image..."):

            async def analyze_eye():
                b64_image = image_to_base64(uploaded_file)

                instructions = """
You are an eye care specialist AI that detects signs of common eye issues from close-up eye images.

On receiving an eye image:
1. Check for **redness, dryness, or infection symptoms**
2. Check for **vision warning signs** like yellowing, swelling, or cloudiness
3. Respond only in this format:

**Detected Issues:**  
**Concern Level:** Low / Moderate / High  
**Recommendation:** No Action / Monitor / Visit Doctor / Emergency Visit  
**Quick Note (1–2 lines):**

Keep it brief and medically cautious.
"""

                agent = Agent(
                    name="Eye Health Checker",
                    instructions=instructions.strip()
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
                            "content": "Please analyze this eye image for any infection or vision-related problems.",
                        },
                    ],
                    run_config=config
                )

                return result.final_output

            output = asyncio.run(analyze_eye())

            st.markdown("---")
            st.success("✅ **Eye Health Report:**")
            st.markdown(f"<div style='background-color:#f0f8ff; padding: 15px; border-radius: 10px; font-size: 16px;'>{output}</div>", unsafe_allow_html=True)

else:
    st.markdown("⬆️ Upload an eye image to begin analysis.", unsafe_allow_html=True)
