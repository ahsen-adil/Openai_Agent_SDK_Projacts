import aiohttp
import os
import logging
import re
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fuzzywuzzy import fuzz
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Shopping Assistant Chatbot")

# Get absolute path for static folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Mount static files directory
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    logger.info(f"Static files mounted at {STATIC_DIR}")
else:
    logger.error(f"Static directory not found at {STATIC_DIR}")

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    logger.error("GEMINI_API_KEY not set")
    raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")

# Create OpenAI-compatible Gemini client
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Create model object
model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=client,
)

# Disable tracing
set_tracing_disabled(disabled=True)

# Pydantic models
class ShoppingResponse(BaseModel):
    response: str = Field(..., max_length=1000)

class InputSpamCheck(BaseModel):
    is_spammy: bool
    reasoning: str

class OutputQualityCheck(BaseModel):
    is_promotional: bool
    is_too_long: bool
    reasoning: str

class ChatRequest(BaseModel):
    message: str

# Function to correct typos in query
def correct_typo(query: str) -> str:
    categories = ["electronics", "jackets", "jewelry", "clothing", "laptop", "shoes"]
    for category in categories:
        if fuzz.ratio(query.lower(), category.lower()) > 80:
            logger.info(f"Corrected typo: {query} -> {category}")
            return category
    return query

# Function to truncate response
def truncate_response(text: str, max_length: int = 480) -> str:
    if len(text) > max_length:
        logger.warning(f"Truncating response from {len(text)} to {max_length} chars")
        return text[:max_length].rsplit(";", 1)[0] + "..." if ";" in text else text[:max_length] + "..."
    return text

# Function to fetch product data
async def fetch_fake_store_products(query: Optional[str] = None, product_id: Optional[int] = None, low_price: bool = False) -> str:
    async with aiohttp.ClientSession() as session:
        if product_id:
            url = f"https://fakestoreapi.com/products/{product_id}"
        else:
            url = "https://fakestoreapi.com/products"
        
        logger.info(f"Fetching products: URL={url}, Query={query}, Product_ID={product_id}, Low_Price={low_price}")
        async with session.get(url) as response:
            if response.status != 200:
                logger.error(f"Failed to fetch products: {response.status}")
                return "Error: Failed to fetch products"
            data = await response.json()
            logger.debug(f"API response: {data}")
            
            if product_id:
                if isinstance(data, dict):
                    return f"{data['title']} (${data['price']})"
                return "No product found"
            
            if query:
                query = correct_typo(query)
                # Map query to categories
                category_map = {
                    "jackets": ["men's clothing", "women's clothing"],
                    "laptop": ["electronics"],
                    "electronics": ["electronics"],
                    "shoes": ["men's clothing", "women's clothing"],
                    "clothing": ["men's clothing", "women's clothing"],
                    "jewelry": ["jewelery"]
                }
                allowed_categories = category_map.get(query.lower(), [query.lower()])
                
                # Strict category-based filtering
                filtered = [
                    f"{product['title']} (${product['price']})"
                    for product in data
                    if (
                        any(cat.lower() in product["category"].lower() for cat in allowed_categories) and
                        (
                            query.lower() == "jackets" and "jacket" in product["title"].lower() or
                            query.lower() in ["laptop", "electronics"] and "electronics" in product["category"].lower() and not any(
                                kw in product["title"].lower() for kw in ["backpack", "t-shirt", "jacket"]
                            ) or
                            query.lower() == "shoes" and False  # No shoes, force fallback
                        )
                    )
                    and (not low_price or product["price"] < 100)
                ][:3]
                logger.info(f"Filtered products: {len(filtered)} found for query '{query}'")
                
                if filtered:
                    return truncate_response("; ".join(filtered))
                else:
                    # Fallback: Relevant category
                    if query.lower() in ["laptop", "electronics", "elctronices"]:
                        electronics = [
                            f"{product['title']} (${product['price']})"
                            for product in data
                            if "electronics" in product["category"].lower() and (not low_price or product["price"] < 100)
                        ][:3]
                        if electronics:
                            return truncate_response(f"No {query} found, but check out these low-priced electronics: {'; '.join(electronics)}")
                    elif query.lower() in ["jackets", "shoes", "clothing"]:
                        clothing = [
                            f"{product['title']} (${product['price']})"
                            for product in data
                            if any(cat in product["category"].lower() for cat in ["men's clothing", "women's clothing"])
                            and (not low_price or product["price"] < 100)
                            and ("jacket" in product["title"].lower() if query.lower() == "jackets" else True)
                            and not any(kw in product["title"].lower() for kw in ["backpack", "t-shirt"])
                        ][:3]
                        if clothing:
                            return truncate_response(f"No {query} found, but check out these low-priced clothing items: {'; '.join(clothing)}")
                    # General fallback
                    cheapest = sorted(data, key=lambda x: x["price"])[:3]
                    cheapest_list = [f"{p['title']} (${p['price']})" for p in cheapest if not any(
                        kw in p["title"].lower() for kw in ["backpack", "t-shirt"]
                    )][:3]
                    return truncate_response(f"No products found for '{query}'. Try these low-priced items: {'; '.join(cheapest_list)}")
            
            return "No products found"

# Guardrail agents
input_guardrail_agent = Agent(
    name="InputSpamCheck",
    instructions="Check if the user input is spammy (e.g., repeated words) or irrelevant to shopping. Allow simple greetings like 'hello' or 'hi' and queries about products, categories, or orders.",
    output_type=InputSpamCheck,
    model=model,
)

output_guardrail_agent = Agent(
    name="OutputQualityCheck",
    instructions="Check if the output is too long (>100 words) or contains promotional phrases like 'best deal ever'.",
    output_type=OutputQualityCheck,
    model=model,
)

# Input guardrail
@input_guardrail
async def spam_input_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    if isinstance(input, list):
        input_str = " ".join([item.text for item in input if hasattr(item, "text")])
    else:
        input_str = input

    words = input_str.lower().split()
    word_count = {word: words.count(word) for word in set(words)}
    is_spammy = any(count > 4 for count in word_count.values())
    is_relevant = (
        len(words) <= 2 and any(keyword in input_str.lower() for keyword in ["hello", "hi", "hey"]) or
        any(keyword in input_str.lower() for keyword in ["product", "order", "return", "price", "buy", "shop", "jacket", "id", "show", "search"]) or
        any(word.isdigit() for word in words)
    )
    logger.info(f"Input: {input_str}, Spammy: {is_spammy}, Relevant: {is_relevant}")

    result = await Runner.run(input_guardrail_agent, input_str, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=is_spammy or (not is_relevant and result.final_output.is_spammy),
    )

# Output guardrail
@output_guardrail
async def quality_output_guardrail(
    ctx: RunContextWrapper, agent: Agent, output: ShoppingResponse
) -> GuardrailFunctionOutput:
    response_text = output.response
    word_count = len(response_text.split())
    is_too_long = word_count > 100
    promotional_phrases = ["best deal ever", "unbeatable price", "limited offer"]
    is_promotional = any(phrase in response_text.lower() for phrase in promotional_phrases)

    result = await Runner.run(output_guardrail_agent, response_text, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=is_too_long or is_promotional or result.final_output.is_promotional,
    )

# Main shopping assistant agent
shopping_agent = Agent(
    name="ShoppingAssistant",
    instructions="You are a helpful online shopping assistant. For product queries, call fetch_fake_store_products directly and format the response. Answer queries about orders or returns concisely and professionally. For simple greetings like 'hello' or 'hi', respond politely (e.g., 'Hi! How can I help with your shopping today?'). Avoid promotional language.",
    output_type=ShoppingResponse,
    input_guardrails=[spam_input_guardrail],
    output_guardrails=[quality_output_guardrail],
    model=model,
)

# Serve index.html
@app.get("/")
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        logger.info(f"Serving index.html from {index_path}")
        return FileResponse(index_path)
    logger.error(f"index.html not found at {index_path}")
    raise HTTPException(status_code=404, detail="index.html not found")

# Chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        input_text = request.message.strip()
        if not input_text:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        logger.info(f"Processing chat input: {input_text}")

        # Extract search query and check for "low price"
        low_price = "low price" in input_text.lower()
        search_match = re.search(r"(?:show|search|find)\s+(?:me\s+)?([\w\s]+)", input_text.lower(), re.IGNORECASE)
        if search_match:
            query = search_match.group(1).strip()
            if low_price:
                query = query.replace("low price", "").strip()
            logger.info(f"Extracted search query: {query}, Low_Price={low_price}")
            tool_result = await fetch_fake_store_products(query=query, low_price=low_price)
            response = ShoppingResponse(response=f"Found: {tool_result}")
            return {"response": response.response}
        
        # Handle product ID queries
        if "id" in input_text.lower():
            words = input_text.split()
            product_id = next((int(word) for word in words if word.isdigit()), None)
            if product_id:
                tool_result = await fetch_fake_store_products(product_id=product_id)
                response = ShoppingResponse(response=f"Product: {tool_result}")
                return {"response": response.response}

        # Use agent for other queries
        result = await Runner.run(shopping_agent, input_text)
        return {"response": result.final_output.response}

    except InputGuardrailTripwireTriggered:
        return {"response": "Error: Input is spammy or irrelevant!"}
    except OutputGuardrailTripwireTriggered:
        return {"response": "Error: Output is too long or promotional!"}
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Test function
async def main():
    print("\nTest 1: Search for jackets")
    try:
        input_text = "Show me jackets"
        search_match = re.search(r"(?:show|search|find)\s+(?:me\s+)?([\w\s]+)", input_text.lower())
        if search_match:
            tool_result = await fetch_fake_store_products(query=search_match.group(1).strip())
            result = ShoppingResponse(response=f"Found: {tool_result}")
            print(f"Response: {result.response}")
        else:
            result = await Runner.run(shopping_agent, input_text)
            print(f"Response: {result.final_output.response}")
    except InputGuardrailTripwireTriggered:
        print("Error: Input is spammy or irrelevant!")
    except OutputGuardrailTripwireTriggered:
        print("Error: Output is too long or promotional!")
    except Exception as e:
        print(f"Error: {str(e)}")

    print("\nTest 2: Get product ID 1")
    try:
        input_text = "Tell me about product ID 1"
        if "id" in input_text.lower():
            tool_result = await fetch_fake_store_products(product_id=1)
            result = ShoppingResponse(response=f"Product: {tool_result}")
            print(f"Response: {result.response}")
        else:
            result = await Runner.run(shopping_agent, input_text)
            print(f"Response: {result.final_output.response}")
    except InputGuardrailTripwireTriggered:
        print("Error: Input is spammy or irrelevant!")
    except OutputGuardrailTripwireTriggered:
        print("Error: Output is too long or promotional!")
    except Exception as e:
        print(f"Error: {str(e)}")

    print("\nTest 3: Spammy input")
    try:
        input_text = "hello hello hello hello"
        result = await Runner.run(shopping_agent, input_text)
        print(f"Response: {result.final_output.response}")
    except InputGuardrailTripwireTriggered:
        print("Error: Input is spammy or irrelevant!")
    except OutputGuardrailTripwireTriggered:
        print("Error: Output is too long or promotional!")
    except Exception as e:
        print(f"Error: {str(e)}")

    print("\nTest 4: Order status check")
    try:
        input_text = "Where is my order #1234?"
        result = await Runner.run(shopping_agent, input_text)
        print(f"Response: {result.final_output.response}")
    except InputGuardrailTripwireTriggered:
        print("Error: Input is spammy or irrelevant!")
    except OutputGuardrailTripwireTriggered:
        print("Error: Output is too long or promotional!")
    except Exception as e:
        print(f"Error: {str(e)}")

    print("\nTest 5: Simple greeting")
    try:
        input_text = "hello"
        result = await Runner.run(shopping_agent, input_text)
        print(f"Response: {result.final_output.response}")
    except InputGuardrailTripwireTriggered:
        print("Error: Input is spammy or irrelevant!")
    except OutputGuardrailTripwireTriggered:
        print("Error: Output is too long or promotional!")
    except Exception as e:
        print(f"Error: {str(e)}")

    print("\nTest 6: Search for shoes")
    try:
        input_text = "Show me shoes"
        search_match = re.search(r"(?:show|search|find)\s+(?:me\s+)?([\w\s]+)", input_text.lower())
        if search_match:
            tool_result = await fetch_fake_store_products(query=search_match.group(1).strip())
            result = ShoppingResponse(response=f"Found: {tool_result}")
            print(f"Response: {result.response}")
        else:
            result = await Runner.run(shopping_agent, input_text)
            print(f"Response: {result.final_output.response}")
    except InputGuardrailTripwireTriggered:
        print("Error: Input is spammy or irrelevant!")
    except OutputGuardrailTripwireTriggered:
        print("Error: Output is too long or promotional!")
    except Exception as e:
        print(f"Error: {str(e)}")

    print("\nTest 7: Search for low price laptops")
    try:
        input_text = "Show me low price laptops"
        low_price = "low price" in input_text.lower()
        search_match = re.search(r"(?:show|search|find)\s+(?:me\s+)?([\w\s]+)", input_text.lower())
        if search_match:
            query = search_match.group(1).strip()
            if low_price:
                query = query.replace("low price", "").strip()
            tool_result = await fetch_fake_store_products(query=query, low_price=low_price)
            result = ShoppingResponse(response=f"Found: {tool_result}")
            print(f"Response: {result.response}")
        else:
            result = await Runner.run(shopping_agent, input_text)
            print(f"Response: {result.final_output.response}")
    except InputGuardrailTripwireTriggered:
        print("Error: Input is spammy or irrelevant!")
    except OutputGuardrailTripwireTriggered:
        print("Error: Output is too long or promotional!")
    except Exception as e:
        print(f"Error: {str(e)}")

    print("\nTest 8: Search for electronics")
    try:
        input_text = "Show me electronics"
        search_match = re.search(r"(?:show|search|find)\s+(?:me\s+)?([\w\s]+)", input_text.lower())
        if search_match:
            tool_result = await fetch_fake_store_products(query=search_match.group(1).strip())
            result = ShoppingResponse(response=f"Found: {tool_result}")
            print(f"Response: {result.response}")
        else:
            result = await Runner.run(shopping_agent, input_text)
            print(f"Response: {result.final_output.response}")
    except InputGuardrailTripwireTriggered:
        print("Error: Input is spammy or irrelevant!")
    except OutputGuardrailTripwireTriggered:
        print("Error: Output is too long or promotional!")
    except Exception as e:
        print(f"Error: {str(e)}")

    print("\nTest 9: Search for low price jackets")
    try:
        input_text = "Show me low price jackets"
        low_price = "low price" in input_text.lower()
        search_match = re.search(r"(?:show|search|find)\s+(?:me\s+)?([\w\s]+)", input_text.lower())
        if search_match:
            query = search_match.group(1).strip()
            if low_price:
                query = query.replace("low price", "").strip()
            tool_result = await fetch_fake_store_products(query=query, low_price=low_price)
            result = ShoppingResponse(response=f"Found: {tool_result}")
            print(f"Response: {result.response}")
        else:
            result = await Runner.run(shopping_agent, input_text)
            print(f"Response: {result.final_output.response}")
    except InputGuardrailTripwireTriggered:
        print("Error: Input is spammy or irrelevant!")
    except OutputGuardrailTripwireTriggered:
        print("Error: Output is too long or promotional!")
    except Exception as e:
        print(f"Error: {str(e)}")

    print("\nTest 10: Search for typo electronics (electronices)")
    try:
        input_text = "Show me electronices"
        search_match = re.search(r"(?:show|search|find)\s+(?:me\s+)?([\w\s]+)", input_text.lower())
        if search_match:
            tool_result = await fetch_fake_store_products(query=search_match.group(1).strip())
            result = ShoppingResponse(response=f"Found: {tool_result}")
            print(f"Response: {result.response}")
        else:
            result = await Runner.run(shopping_agent, input_text)
            print(f"Response: {result.final_output.response}")
    except InputGuardrailTripwireTriggered:
        print("Error: Input is spammy or irrelevant!")
    except OutputGuardrailTripwireTriggered:
        print("Error: Output is too long or promotional!")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)