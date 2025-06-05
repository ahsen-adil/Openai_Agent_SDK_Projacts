from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled
from openai import AsyncOpenAI  
from dotenv import load_dotenv
import os
import requests
import gradio as gr

# Load environment variable
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in .env file.")

# OpenAI Client
client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=gemini_api_key
)

# Disable tracing
set_tracing_disabled(disabled=True)

@function_tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Convert a given amount from one currency to another.
    """
    response = requests.get(f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}")
    data = response.json()
    rate = data['rates'].get(to_currency.upper())

    if rate:
        converted = amount * rate
        return f"{amount} {from_currency.upper()} is equal to {converted:.2f} {to_currency.upper()}."
    else:
        return f"Currency {to_currency.upper()} is not supported."

# Agent setup
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client,
)

agent: Agent = Agent(
    name="Currency Converter Agent",
    instructions="You are a helpful currency converter chatbot. You convert currencies using live exchange rates.",
    model=model,
    tools=[convert_currency]
)

# Async handler
async def chatbot_ui(message, history):
    result = await Runner.run(agent, message)
    return result.final_output

# ✅ Responsive UI — Clean Layout
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("### 💱 Currency Converter Agent", elem_id="main-title")

    chatbot = gr.Chatbot(
        label="Chat with the Agent",
        height=400,
        show_copy_button=True,
        bubble_full_width=False,
        avatar_images=("👤", "🤖")
    )

    with gr.Row(equal_height=True):
        msg = gr.Textbox(
            placeholder="Type your conversion request, e.g. 'Convert 100 USD to PKR'",
            show_label=False,
            container=False,
            scale=6
        )
        send_btn = gr.Button("Send 💬", variant="primary", scale=1)

    clear_btn = gr.Button("🧹 Clear Chat", variant="secondary")

    async def respond(message, history):
        response = await chatbot_ui(message, history)
        history.append([message, response])
        return "", history

    send_btn.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: [], None, chatbot, queue=False)

demo.launch(share=True)
