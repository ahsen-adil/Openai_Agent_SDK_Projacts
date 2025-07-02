import os
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    RunConfig,
    set_tracing_disabled
)
import chainlit as cl

# 🔐 Load API key from .env
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")

# 🔧 Setup Gemini Client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

set_tracing_disabled(disabled=True)

# 🧾 Output Schema
class Experience(BaseModel):
    company: str
    role: str
    duration: str

class ResumeInfo(BaseModel):
    name: str
    skills: List[str]
    experience: List[Experience]
    education: List[str]

# 🤖 Resume Analyzer Agent
resume_analyzer = Agent(
    name="Resume Analyzer",
    instructions="""
You are an assistant that extracts structured resume information from plain text.

Return a JSON with:
- name (Full name of the person)
- skills (List of skills/technologies)
- experience (List of experiences including company, role, and duration)
- education (List of education entries)

Use empty lists if anything is missing.
""",
    output_type=ResumeInfo
)

# 📚 Helper: Format Experience
def format_experience(experiences: List[Experience]) -> str:
    if not experiences:
        return "_No experience found._"
    return "\n".join(
        [f"- **{exp.role}** at *{exp.company}* (`{exp.duration}`)" for exp in experiences]
    )

# 🎓 Helper: Format Education
def format_education(education: List[str]) -> str:
    if not education:
        return "_No education entries found._"
    return "\n".join([f"- 🎓 {edu}" for edu in education])

# 🟢 Show heading and welcome message on first screen
@cl.on_chat_start
async def start_chat():
    await cl.Message(
        content="""
# 🤖 **Resume Analyzer Agent**

Welcome! Paste your resume text below and I’ll analyze it to extract:

- Full Name  
- Skills  
- Work Experience  
- Education  

📄 Just send me the plain text of your resume to begin!
"""
    ).send()

# 🧠 Chainlit App Logic
@cl.on_message
async def analyze_resume(message: cl.Message):
    await cl.Message(content="⏳ *Analyzing your resume...*").send()

    try:
        result = await Runner.run(
            resume_analyzer,
            message.content,
            run_config=config
        )
        data = result.final_output

        # 💡 Format the response with markdown styling
        response = f"""# 📄 **Resume Analysis Result**

---

### 👤 **Name**
**{data.name}**

---

### 🧠 **Skills**
{', '.join(data.skills) if data.skills else 'No skills found'}

---

### 💼 **Experience**
{format_experience(data.experience)}

---

### 🎓 **Education**
{format_education(data.education)}

---

✅ *Resume successfully analyzed!*
"""

        await cl.Message(content=response).send()

    except Exception as e:
        await cl.Message(content=f"❌ **Error analyzing resume:** `{str(e)}`").send()
