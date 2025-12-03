import asyncio
import base64
import logging
from typing import List, Dict, AsyncIterator

import openai
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted


logger = logging.getLogger("llm_core")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------
# HISTORY TRIMMER
# ---------------------------------------------------------
def trim_history_by_tokens(history: List[Dict], max_tokens: int) -> List[Dict]:
    """Грубая оценка — 1 слово = 1 токен"""
    tokens = 0
    out = []
    for msg in reversed(history):
        tokens += len(msg.get("content", "").split())
        if tokens > max_tokens:
            break
        out.append(msg)
    return list(reversed(out))


# ---------------------------------------------------------
# TEXT GENERATION (OpenAI, Grok, Gemini)
# ---------------------------------------------------------
async def generate_text(provider: str,
                        model: str,
                        history: List[Dict],
                        user_input: str,
                        openai_key=None,
                        grok_key=None,
                        gemini_key=None,
                        max_history_tokens=3000):

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_input})

    # ---------------------------------------
    # OPENAI
    # ---------------------------------------
    if provider == "openai":
        client = openai.AsyncOpenAI(api_key=openai_key)

        # GPT-5 family → uses max_completion_tokens
        params = {"max_completion_tokens": 1024} if model.startswith("gpt-5") else {"max_tokens": 1024}

        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            **params
        )
        return resp.choices[0].message.content

    # ---------------------------------------
    # GROK (xAI)
    # ---------------------------------------
    if provider == "grok":
        client = openai.AsyncOpenAI(api_key=grok_key, base_url="https://api.x.ai/v1")

        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1024
        )
        return resp.choices[0].message["content"]

    # ---------------------------------------
    # GEMINI (Text)
    # ---------------------------------------
    if provider == "gemini":
        try:
            genai.configure(api_key=gemini_key)

            # History → prompt
            prompt = ""
            for m in history:
                prompt += f"{m['role']}: {m['content']}\n"
            prompt += f"user: {user_input}"

            model_obj = genai.GenerativeModel(model)
            result = model_obj.generate_content(prompt)

            return result.text

        except ResourceExhausted:
            raise RuntimeError("GEMINI_QUOTA_EXCEEDED")

        except Exception as e:
            raise RuntimeError(f"Gemini error: {e}")

    raise RuntimeError("Unknown provider")


# ---------------------------------------------------------
# TEXT STREAMING — OpenAI & Grok only
# ---------------------------------------------------------
async def generate_text_stream(provider: str,
                               model: str,
                               history: List[Dict],
                               user_input: str,
                               openai_key=None,
                               grok_key=None,
                               gemini_key=None,
                               max_history_tokens=3000) -> AsyncIterator[str]:

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for h in history:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_input})

    # -----------------------------
    # OPENAI STREAM
    # -----------------------------
    if provider == "openai":
        client = openai.AsyncOpenAI(api_key=openai_key)
        params = {"max_completion_tokens": 1024} if model.startswith("gpt-5") else {"max_tokens": 1024}

        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **params
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

        return

    # -----------------------------
    # GROK STREAM
    # -----------------------------
    if provider == "grok":
        client = openai.AsyncOpenAI(api_key=grok_key, base_url="https://api.x.ai/v1")

        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=1024
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta.get("content")
                if delta:
                    yield delta

        return

    # -----------------------------
    # GEMINI (no stream — fallback)
    # -----------------------------
    text = await generate_text(provider, model, history, user_input,
                               openai_key, grok_key, gemini_key,
                               max_history_tokens)
    yield text


# ---------------------------------------------------------
# IMAGE GENERATION
# ---------------------------------------------------------
async def generate_image(provider: str,
                         model: str,
                         prompt: str,
                         openai_key=None,
                         grok_key=None,
                         gemini_key=None):

    # -----------------------------
    # OPENAI — DALL·E 3
    # -----------------------------
    if provider == "openai":
        client = openai.AsyncOpenAI(api_key=openai_key)
        resp = await client.images.generate(
            model=model,
            prompt=prompt,
            size="1024x1024"
        )
        return resp.data[0].url

    # -----------------------------
    # GROK — FIXED (uses width/height)
    # returns BASE64 → convert to data URL
    # -----------------------------
    if provider == "grok":
        client = openai.AsyncOpenAI(api_key=grok_key, base_url="https://api.x.ai/v1")

        resp = await client.images.generate(
            model=model,
            prompt=prompt,
            width=1024,
            height=1024
        )

        b64 = resp.data[0].b64_json
        return f"data:image/png;base64,{b64}"

    # -----------------------------
    # GEMINI IMAGE — working!
    # -----------------------------
    if provider == "gemini":
        genai.configure(api_key=gemini_key)

        model_obj = genai.GenerativeModel(model)
        result = model_obj.generate_image(prompt)

        # Gemini returns inline image data
        img = result.images[0]
        if img.mime_type.startswith("image/"):
            return f"data:{img.mime_type};base64,{base64.b64encode(img.data).decode()}"

        raise RuntimeError("Gemini image: unsupported response format")

    raise RuntimeError("Unknown provider")
