import os
from typing import List, Dict, Optional

from openai import OpenAI
from groq import Groq


# ----------------------------------------------------------
# CLIENTS
# ----------------------------------------------------------

_openai_client: Optional[OpenAI] = None
_groq_client: Optional[Groq] = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


# ----------------------------------------------------------
# TEXT GENERATION
# ----------------------------------------------------------

async def generate_text(
    *,
    provider: str,
    model: str,
    prompt: str,
    history: List[Dict[str, str]] | None = None,
) -> str:
    """
    provider: "openai" | "grok"
    model: модель из существующего списка
    prompt: текст пользователя
    history: история сообщений (role/content)
    """

    messages: List[Dict[str, str]] = []

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": prompt})

    if provider == "openai":
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return resp.choices[0].message.content

    if provider == "grok":
        client = get_groq_client()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return resp.choices[0].message.content

    raise ValueError(f"Unknown provider: {provider}")


# ----------------------------------------------------------
# IMAGE GENERATION
# ----------------------------------------------------------

async def generate_image(
    *,
    provider: str,
    model: str,
    prompt: str,
) -> str:
    """
    Возвращает URL изображения
    """

    if provider == "openai":
        client = get_openai_client()
        result = client.images.generate(
            model=model,
            prompt=prompt,
            size="1024x1024",
        )
        return result.data[0].url

    if provider == "grok":
        # На данный момент Grok НЕ поддерживает генерацию изображений через API
        raise RuntimeError("Grok image generation is not supported via API")

    raise ValueError(f"Unknown provider: {provider}")