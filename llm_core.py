# llm_core.py
import base64
import os
import httpx
import openai

# ---------------------------------------------------------
# HISTORY
# ---------------------------------------------------------

def trim_history_by_tokens(history, max_tokens: int):
    tokens = 0
    result = []
    for msg in reversed(history):
        tokens += len(msg.get("content", "").split())
        if tokens > max_tokens:
            break
        result.append(msg)
    return list(reversed(result))

# ---------------------------------------------------------
# TEXT (NO STREAMING — STABLE)
# ---------------------------------------------------------

async def generate_text(
    provider: str,
    model: str,
    history: list,
    user_input: str,
    openai_key: str = None,
    grok_key: str = None,
    gemini_key: str = None,
    max_history_tokens: int = 8000,
):
    # history УЖЕ содержит user-сообщение — не добавляем повторно
    messages = trim_history_by_tokens(history, max_history_tokens)

    async with httpx.AsyncClient(timeout=60.0) as http_client:
        if provider == "openai":
            client = openai.AsyncOpenAI(
                api_key=openai_key,
                http_client=http_client,
            )

        elif provider == "grok":
            client = openai.AsyncOpenAI(
                api_key=grok_key,
                base_url="https://api.x.ai/v1",
                http_client=http_client,
            )

        else:
            raise RuntimeError("Unsupported provider")

        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
        )

        # ❗ SDK 1.x — доступ через .content
        return resp.choices[0].message.content

# ---------------------------------------------------------
# IMAGE
# ---------------------------------------------------------

async def generate_image(
    provider: str,
    model: str,
    prompt: str,
    openai_key: str = None,
    grok_key: str = None,
    gemini_key: str = None,
):
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        if provider == "openai":
            client = openai.AsyncOpenAI(
                api_key=openai_key,
                http_client=http_client,
            )

            r = await client.images.generate(
                model=model,
                prompt=prompt,
                size="1024x1024",
            )

            # OpenAI возвращает HTTPS URL — Telegram ОК
            return r.data[0].url

        elif provider == "grok":
            client = openai.AsyncOpenAI(
                api_key=grok_key,
                base_url="https://api.x.ai/v1",
                http_client=http_client,
            )

            r = await client.images.generate(
                model=model,
                prompt=prompt,
                width=1024,
                height=1024,
            )

            # xAI возвращает base64 → Telegram ждёт bytes
            image_bytes = base64.b64decode(r.data[0].b64_json)
            return image_bytes

        else:
            raise RuntimeError("Unsupported image provider")
