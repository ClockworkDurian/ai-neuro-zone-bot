import asyncio
import json
import time

import openai
import google.generativeai as genai


# ---------------------------------------------------------
# TEXT GENERATION
# ---------------------------------------------------------

async def generate_text(provider, model, history, user_input,
                        openai_key=None, grok_key=None, gemini_key=None,
                        max_history_tokens=5000):

    if provider == "openai":
        try:
            client = openai.AsyncOpenAI(api_key=openai_key)

            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            for h in history:
                messages.append({"role": h["role"], "content": h["content"]})
            messages.append({"role": "user", "content": user_input})

            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False
            )
            return resp.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"OpenAI error: {e}")

    # -------------------------------
    # GROK (xAI)
    # -------------------------------
    elif provider == "grok":
        try:
            import openai as grok_api
            client = grok_api.AsyncOpenAI(
                api_key=grok_key,
                base_url="https://api.x.ai/v1"
            )

            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            for h in history:
                messages.append({"role": h["role"], "content": h["content"]})
            messages.append({"role": "user", "content": user_input})

            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False
            )
            return resp.choices[0].message["content"]

        except Exception as e:
            raise RuntimeError(f"Grok error: {e}")

    # --------------------------------------------------
    # GEMINI — временно отключено (API изменилось)
    # --------------------------------------------------
    elif provider == "gemini":
        raise RuntimeError("Gemini временно недоступен: изменился API Google. Исправим позже.")

    else:
        raise RuntimeError(f"Unknown provider: {provider}")


# ---------------------------------------------------------
# STREAMING
# ---------------------------------------------------------

async def generate_text_stream(provider, model, history, user_input,
                               openai_key=None, grok_key=None, gemini_key=None,
                               max_history_tokens=5000):

    try:
        # Пока стриминг есть только у OpenAI и Grok
        if provider == "openai":
            client = openai.AsyncOpenAI(api_key=openai_key)

            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            for h in history:
                messages.append({"role": h["role"], "content": h["content"]})
            messages.append({"role": "user", "content": user_input})

            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        elif provider == "grok":
            import openai as grok_api
            client = grok_api.AsyncOpenAI(
                api_key=grok_key,
                base_url="https://api.x.ai/v1"
            )

            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            for h in history:
                messages.append({"role": h["role"], "content": h["content"]})
            messages.append({"role": "user", "content": user_input})

            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.get("content"):
                    yield chunk.choices[0].delta["content"]

        else:
            # Gemini пока отключен
            result = await generate_text(provider, model, history, user_input,
                                         openai_key, grok_key, gemini_key, max_history_tokens)
            yield result

    except Exception as e:
        raise RuntimeError(str(e))


# ---------------------------------------------------------
# IMAGE GENERATION
# ---------------------------------------------------------

async def generate_image(provider, model, prompt,
                         openai_key=None, grok_key=None, gemini_key=None):

    # -------------------------------
    # OPENAI (DALL-E 3)
    # -------------------------------
    if provider == "openai":
        try:
            client = openai.AsyncOpenAI(api_key=openai_key)

            resp = await client.images.generate(
                model=model,
                prompt=prompt,
                size="1024x1024"
            )

            url = resp.data[0].url
            return url

        except Exception as e:
            raise RuntimeError(f"OpenAI image error: {e}")

    # -------------------------------
    # GROK IMAGE — ИСПРАВЛЕНО
    # -------------------------------
    elif provider == "grok":
        try:
            import openai as grok_api
            client = grok_api.AsyncOpenAI(
                api_key=grok_key,
                base_url="https://api.x.ai/v1"
            )

            resp = await client.images.generate(
                model=model,          # grok-image-1
                prompt=prompt,
                width=1024,           # 🔥 ВАЖНО! вместо size
                height=1024
            )

            # Grok возвращает base64
            b64 = resp.data[0].b64_json
            return f"data:image/png;base64,{b64}"

        except Exception as e:
            raise RuntimeError(f"Grok image error: {e}")

    # -------------------------------
    # GEMINI IMAGE — временно отключен
    # -------------------------------
    elif provider == "gemini":
        raise RuntimeError("Gemini image временно недоступен: Google API поменялся")

    else:
        raise RuntimeError(f"Unknown provider: {provider}")
