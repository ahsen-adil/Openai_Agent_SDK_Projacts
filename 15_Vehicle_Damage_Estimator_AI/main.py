import asyncio
import base64
import os
import streamlit as st
from dotenv import load_dotenv

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig

# Load API Key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("❌ GEMINI_API_KEY not set.")
    st.stop()

# Gemini Client Setup
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

# Streamlit UI
st.set_page_config(page_title="🚘 Vehicle Damage Estimator", layout="centered")
st.title("🚘 Vehicle Damage Estimator AI")
st.markdown("Upload an image of the damaged **car, bumper, door, headlight**, etc. AI will evaluate the damage and give repair cost range in PKR.")

uploaded_file = st.file_uploader("📷 Upload Damage Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Damage Image", width=250)

    if st.button("🔍 Estimate Damage & Cost"):
        with st.spinner("Analyzing vehicle damage..."):

            async def analyze():
                b64_image = image_to_base64(uploaded_file)

                instructions = """
You are an experienced automobile damage estimator in Pakistan.

When shown a photo of car damage (e.g., bumper, door, fender, headlight):
- Estimate **Severity**: Minor / Moderate / Severe
- Estimate **Repair Cost Range** in **PKR** (typical Pakistani auto workshop prices)
- Describe the **Type of Repair** (e.g., repaint, replacement, denting)
- Give a **brief explanation** of the cause and what a mechanic would typically do in Pakistan

Respond in the following format:

**Severity:**  
**Estimated Repair Cost (PKR):**  
**Type of Repair:**  
**Details:** (2-4 lines about the damage and repair process as done in Pakistani workshops)

Be realistic and consider that parts and labor cost less in Pakistan.
"""

                agent = Agent(
                    name="Vehicle Damage Estimator - PK",
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
                            "content": "Please assess this vehicle damage and give estimate for Pakistani market.",
                        },
                    ],
                    run_config=config
                )

                return result.final_output

            output = asyncio.run(analyze())
            st.success("📋 Damage Assessment:")
            st.markdown(output)
