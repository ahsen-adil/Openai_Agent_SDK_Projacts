import asyncio
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

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")

# Setup Gemini Client
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

# ✅ Define output schema
class Experience(BaseModel):
    company: str
    role: str
    duration: str

class ResumeInfo(BaseModel):
    name: str
    skills: List[str]
    experience: List[Experience]
    education: List[str]

# 🎯 Create Resume Analyzer Agent
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

# 🧪 Sample resume-like text
resume_text = """
My name is jhon. I have experience working as a Web Developer at XedInc for 2 years and as a Frontend Intern at Technalogia for 6 months. I know HTML, CSS, JavaScript, React, Next.js, and Tailwind CSS. I graduated from Virtual University with a BS in Computer Science. I also did an AI Chatbot course with Sir Hammad.
"""

# 🚀 Main function
async def main():
    print("\n--- Resume Analyzer Output ---")
    result = await Runner.run(
        resume_analyzer,
        resume_text,
        run_config=config
    )

    resume_data = result.final_output
    print("\nExtracted Resume Data:")
    print(f"Name: {resume_data.name}")
    print(f"Skills: {', '.join(resume_data.skills)}")
    print("\nExperience:")
    for exp in resume_data.experience:
        print(f" - {exp.role} at {exp.company} for {exp.duration}")
    print("\nEducation:")
    for edu in resume_data.education:
        print(f" - {edu}")

# 🔁 Run main
if __name__ == "__main__":
    asyncio.run(main())
