"""
llm_core.py
Универсальный слой для OpenAI (ChatGPT), xAI Grok и Google Gemini.
Поддерживает:
 - GPT-5 / GPT-5-mini / GPT-5-nano (через max_completion_tokens)
 - GPT-4.1 и другие обычные модели
 - xAI Grok через AsyncOpenAI (base_url="https://api.x.ai/v1")
 - Gemini 2.5 Flash / Flash-Lite (новый формат messages)
 - Эмулированный streaming (все провайдеры работают одинаково)
 - Генерацию изображений OpenAI + Grok
"""

import asyncio
import logging
from typing import List, Dict, AsyncIterator

import openai
import google.generativeai as genai

logger = logging.getLogger("llm_core")
logger.setLevel(logging.INFO)

# -------------------------------------------------------------------
# Вспомогательные функции
# -------------------------------------------------------------------

def _history_to_openai_messages(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Переводим историю в формат OpenAI/Grok messages."""
    messages = []
    for m in history:
        messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    return messages

def trim_history_by_tokens(history: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
    """Грубая оценка токенов: 1 слово = 1 токен."""
    if not history:
        return history
    tokens = 0
    out = []
    for m in reversed(history):
        tokens += len(m.get("content", "").split())
        if tokens > max_tokens:
            break
        out.append(m)
    return list(reversed(out))


# -------------------------------------------------------------------
# Основные функции генерации
# -------------------------------------------------------------------

async def generate_text(provider: str,
                        model: str,
                        history: List[Dict[str, str]],
                        user_input: str,
                        openai_key: str = None,
                        grok_key: str = None,
                        gemini_key: str = None,
                        max_history_tokens: int = 2000) -> str:
    """
    Генерирует финальный текст одним запросом (fallback).
    """

    # --------------------
    # История
    # --------------------
    history = trim_history_by_tokens(
        history + [{"role": "user", "content": user_input}],
        max_history_tokens
    )

    # --------------------
    # OPENAI (ChatGPT)
    # --------------------
    if provider == "openai":
        if not openai_key:
            raise ValueError("OPENAI_API_KEY отсутствует")

        client = openai.AsyncOpenAI(api_key=openai_key)
        messages = _history_to_openai_messages(history)

        # GPT-5 family → max_completion_tokens
        if model.startswith("gpt-5"):
            args = {"max_completion_tokens": 1000}
        else:
            args = {"max_tokens": 1000}

        try:
            r = await client.chat.completions.create(
                model=model,
                messages=messages,
                **args
            )
            return r.choices[0].message.content
        except Exception as e:
            logger.exception("OpenAI error: %s", e)
            raise

    # --------------------
    # GROK (xAI)
    # --------------------
    if provider == "grok":
        if not grok_key:
            raise ValueError("GROK_API_KEY отсутствует")

        # xAI API совместим с OpenAI client
        client = openai.AsyncOpenAI(
            api_key=grok_key,
            base_url="https://api.x.ai/v1"
        )

        messages = _history_to_openai_messages(history)

        try:
            r = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1000
            )
            return r.choices[0].message.content
        except Exception as e:
            logger.exception("Grok error: %s", e)
            raise

    # --------------------
    # GEMINI
    # --------------------
    if provider == "gemini":
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY отсутствует")

        genai.configure(api_key=gemini_key)

        # Новый формат Gemini: parts/text
        prompt_parts = [{"text": m["content"]} for m in history]
        model_obj = genai.GenerativeModel(model)

        try:
            chat = model_obj.start_chat(history=[
                {"role": "user", "parts": prompt_parts}
            ])
            resp = await chat.send_message_async(user_input)
            return resp.text
        except Exception as e:
            logger.exception("Gemini error: %s", e)
            raise

    raise ValueError("Unknown provider")


# -------------------------------------------------------------------
# STREAMING (эмуляция через чанки)
# -------------------------------------------------------------------

async def generate_text_stream(provider: str,
                               model: str,
                               history: List[Dict[str, str]],
                               user_input: str,
                               openai_key: str = None,
                               grok_key: str = None,
                               gemini_key: str = None,
                               max_history_tokens: int = 2000) -> AsyncIterator[str]:
    """
    Стриминг — даём пользователю имитацию стриминга,
    разбивая итоговый текст на кусочки.
    """

    final = await generate_text(
        provider=provider,
        model=model,
        history=history,
        user_input=user_input,
        openai_key=openai_key,
        grok_key=grok_key,
        gemini_key=gemini_key,
        max_history_tokens=max_history_tokens
    )

    chunk_size = 80
    i = 0
    while i < len(final):
        yield final[i:i + chunk_size]
        i += chunk_size
        await asyncio.sleep(0)


# -------------------------------------------------------------------
# IMAGE GENERATION
# -------------------------------------------------------------------

async def generate_image(provider: str,
                         prompt: str,
                         openai_key: str = None,
                         grok_key: str = None,
                         gemini_key: str = None) -> str:
    """
    Генерирует изображение → возвращает URL.
    """

    # OPENAI
    if provider == "openai":
        if not openai_key:
            raise ValueError("OPENAI_API_KEY отсутствует")
        client = openai.AsyncOpenAI(api_key=openai_key)
        try:
            r = await client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024"
            )
            return r.data[0].url
        except Exception as e:
            logger.exception("OpenAI image error: %s", e)
            raise

    # GROK (xAI)
    if provider == "grok":
        if not grok_key:
            raise ValueError("GROK_API_KEY отсутствует")

        # В API xAI изображение генерируется тоже через client.images
        client = openai.AsyncOpenAI(api_key=grok_key, base_url="https://api.x.ai/v1")
        try:
            r = await client.images.generate(
                model="grok-vision-alpha",
                prompt=prompt,
                size="1024x1024"
            )
            return r.data[0].url
        except Exception as e:
            logger.exception("Grok image error: %s", e)
            raise

    # GEMINI
    if provider == "gemini":
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY отсутствует")
        genai.configure(api_key=gemini_key)
        try:
            r = genai.generate_image(model="gemini-image-1", prompt=prompt)
            return r.output[0].images[0].uri
        except Exception as e:
            logger.exception("Gemini image error: %s", e)
            raise

    raise ValueError("Unknown provider")
