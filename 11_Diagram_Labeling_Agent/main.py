import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
import base64
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not set.")

# Gemini setup
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

# Define Agent
agent = Agent(
    name="Science Assistant",
    instructions="""
You are a helpful science assistant that can label diagrams and answer questions about them and general knowledge.

When provided with a diagram image, you should identify its type and label its major components with brief explanations.

When asked questions, you should answer based on your knowledge and any previously provided information, such as diagram labels.
"""
)

# Convert image to base64
def image_to_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Chainlit app
@cl.on_chat_start
async def start():
    # Show heading in UI
    await cl.Message(
        content="# 🧠 Diagram Labeling Agent"
    ).send()

    # Ask for image upload
    files = await cl.AskFileMessage(
        content="Please upload a scientific diagram (e.g. heart, liver, or circuit).",
        accept=["image/jpeg", "image/png"],
        max_size_mb=10,
        timeout=180,
    ).send()
    if not files:
        await cl.Message(content="No file was uploaded.").send()
        return
    file = files[0]
    
    # Display uploaded image
    image_element = cl.Image(path=file.path, name="uploaded_diagram")
    await cl.Message(
        content="You uploaded this diagram:",
        elements=[image_element]
    ).send()
    
    # Convert image to base64
    b64_image = image_to_base64(file.path)
    
    # Initialize agent_history
    agent_history = [
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
            "content": "Please label this diagram and provide brief explanations for each part.",
        }
    ]
    
    # Run agent to label diagram
    try:
        response = await Runner.run(
            agent,
            agent_history,
            run_config=config
        )
        labels = response.final_output
        
        # Store labels in user_session
        cl.user_session.set("diagram_labels", labels)
        cl.user_session.set("agent_history", agent_history + [{"role": "assistant", "content": labels}])
        
        # Send labels back
        await cl.Message(content=labels).send()
    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()

@cl.on_message
async def main(msg: cl.Message):
    # Get stored agent history
    agent_history = cl.user_session.get("agent_history")
    
    # If no history exists, initialize it
    if agent_history is None:
        agent_history = []
    
    # Add new user message to history
    agent_history.append({"role": "user", "content": msg.content})
    
    # Run agent with updated history
    try:
        response = await Runner.run(
            agent,
            agent_history,
            run_config=config
        )
        
        # Add assistant's response to history
        agent_history.append({"role": "assistant", "content": response.final_output})
        
        # Update user_session
        cl.user_session.set("agent_history", agent_history)
        
        # Send response back
        await cl.Message(content=response.final_output).send()
    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()
