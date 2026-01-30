import os
from typing import List, Dict
from openai import AsyncOpenAI

# ==========================================================
# CLIENTS
# ==========================================================

_openai_client = None
_grok_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
    return _openai_client


def get_grok_client():
    global _grok_client
    if _grok_client is None:
        _grok_client = AsyncOpenAI(
            api_key=os.getenv("GROK_API_KEY"),
            base_url="https://api.x.ai/v1",
        )
    return _grok_client


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

    client = get_openai_client() if provider == "openai" else get_grok_client()

    # ✅ ПРОСТОЙ Responses API — БЕЗ chat/messages
    response = await client.responses.create(
        model=model,
        input=prompt,
    )

    # Явно извлекаем текст
    try:
        return response.output[0].content[0].text
    except Exception:
        raise RuntimeError(f"Empty response from model {model}")


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

    client = get_openai_client()

    result = await client.images.generate(
        model=model,
        prompt=prompt,
        size="1024x1024",
    )

    return result.data[0].url
