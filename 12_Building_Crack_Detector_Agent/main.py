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

# Structured Response Format
class StructuredClass(BaseModel):
    Severity: str
    RiskLevel: str 
    Recommendation: str
    Reasoning: str

# Convert Image to Base64
def image_to_base64(uploaded_file):
    return base64.b64encode(uploaded_file.read()).decode("utf-8")

# Streamlit UI Setup
st.set_page_config(page_title="🏗️ Crack Detector Agent", layout="centered", initial_sidebar_state="collapsed")

# Custom Dark Mode Styling
st.markdown("""
    <style>
        body {
            background-color: #0f0f0f;
            color: #f0f0f0;
        }
        .stApp {
            background-color: #0f0f0f;
            color: #f0f0f0;
        }
        .stButton button {
            background-color: #1f1f1f;
            color: #ffffff;
            border: 1px solid #444;
            border-radius: 8px;
            padding: 8px 16px;
        }
        .stButton button:hover {
            background-color: #333;
        }
        .stMarkdown, .stText {
            font-size: 1.1rem;
            color: #f0f0f0;
        }
    </style>
""", unsafe_allow_html=True)

# Heading
st.markdown("## 🏗️ Building Crack Detector Agent")
st.markdown("🖼️ Upload a **clear image** of a crack in a wall, floor, or structure.\n\nAI will detect **severity**, **risk level**, and give a professional **recommendation**.")

# Upload UI
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="📸 Uploaded Image Preview", width=300)

    if st.button("🔍 Analyze Crack"):
        with st.spinner("Analyzing image with AI..."):

            async def analyze():
                b64_image = image_to_base64(uploaded_file)

                instructions = """
You are a civil engineering expert trained in structural risk assessment.

When given an image of a crack, evaluate using this structured format:
1. **Severity**: Minor / Moderate / Severe
2. **Risk Level**: Low / Medium / High
3. **Recommendation**: Ignore / Monitor / Repair / Immediate Inspection
4. **Reasoning**: Brief explanation of your evaluation based on the visible crack characteristics.

Respond ONLY in the format below:

**Severity:**  
**Risk Level:**  
**Recommendation:**  
**Reasoning:**

Be concise and accurate. Avoid unrelated information.
"""

                agent = Agent(
                    name="Crack Safety Inspector",
                    instructions=instructions.strip(),
                    output_type=StructuredClass
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
                            "content": "Please assess this structural crack and give a civil engineering analysis.",
                        },
                    ],
                    run_config=config
                )

                return result.final_output

            output = asyncio.run(analyze())

            # Display Results
            st.markdown("### ✅ AI Structural Assessment Result")
            st.markdown("---")
            st.markdown(f"""
            **🛠️ Severity:** {output.Severity}  
            **⚠️ Risk Level:** {output.RiskLevel}  
            **🧰 Recommendation:** {output.Recommendation}  
            **🧠 Reasoning:** {output.Reasoning}
            """)
