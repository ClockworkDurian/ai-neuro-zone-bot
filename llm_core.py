import os
from typing import List, Dict
import openai

# ==========================================================
# CONFIG
# ==========================================================

openai.api_key = os.getenv("OPENAI_API_KEY")

XAI_API_KEY = os.getenv("GROK_API_KEY")
XAI_BASE_URL = "https://api.x.ai/v1"

# ==========================================================
# TEXT GENERATION
# ==========================================================

async def generate_text(
    provider: str,
    model: str,
    prompt: str,
    history: List[Dict],
):
    # Gemini — заглушка
    if provider == "gemini":
        return "Модель временно не поддерживается и будет добавлена позже."

    messages = []

    # история
    for msg in history:
        if "role" in msg and "content" in msg:
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

    # текущий запрос
    messages.append({
        "role": "user",
        "content": prompt,
    })

    if provider == "openai":
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=messages,
        )
        return response.choices[0].message["content"]

    if provider == "grok":
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=messages,
            api_key=XAI_API_KEY,
            api_base=XAI_BASE_URL,
        )
        return response.choices[0].message["content"]

    raise RuntimeError(f"Unknown provider: {provider}")

# ==========================================================
# IMAGE GENERATION
# ==========================================================

async def generate_image(
    provider: str,
    model: str,
    prompt: str,
):
    if provider != "openai":
        return None

    result = await openai.Image.acreate(
        model=model,
        prompt=prompt,
        size="1024x1024",
    )

    return result["data"][0]["url"]