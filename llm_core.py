import asyncio
import time
import openai
import google.generativeai as genai


# ---------------------------------------------------------
# HISTORY TRIM
# ---------------------------------------------------------

def trim_history_by_tokens(history, max_tokens):
    """Simple heuristic: 1 token ≈ 1 word."""
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


# ---------------------------------------------------------
# TEXT GENERATION
# ---------------------------------------------------------

async def generate_text(provider, model, history, user_input,
                        openai_key=None, grok_key=None, gemini_key=None,
                        max_history_tokens=5000):

    # merge history
    history = trim_history_by_tokens(history + [{"role": "user", "content": user_input}],
                                     max_history_tokens)

    # -----------------------------------------------------
    # OPENAI
    # -----------------------------------------------------
    if provider == "openai":
        client = openai.AsyncOpenAI(api_key=openai_key)

        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages += [{"role": h["role"], "content": h["content"]} for h in history]

        # GPT-5 family uses max_completion_tokens
        is_gpt5 = model.startswith("gpt-5")

        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=1000 if is_gpt5 else None,
                max_tokens=None if is_gpt5 else 1000,
            )
            return resp.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI error: {e}")

    # -----------------------------------------------------
    # GROK  (xAI)
    # -----------------------------------------------------
    elif provider == "grok":
        client = openai.AsyncOpenAI(
            api_key=grok_key,
            base_url="https://api.x.ai/v1"
        )

        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages += [{"role": h["role"], "content": h["content"]} for h in history]

        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1200
            )
            return resp.choices[0].message["content"]
        except Exception as e:
            raise RuntimeError(f"Grok error: {e}")

    # -----------------------------------------------------
    # GEMINI (new API removed generate_text → отключено)
    # -----------------------------------------------------
    elif provider == "gemini":
        raise RuntimeError("Gemini временно отключён: Google убрал старый API. Починим позже.")

    else:
        raise RuntimeError("Unknown provider")


# ---------------------------------------------------------
# STREAMING
# ---------------------------------------------------------

async def generate_text_stream(provider, model, history, user_input,
                               openai_key=None, grok_key=None, gemini_key=None,
                               max_history_tokens=5000):

    if provider not in ["openai", "grok"]:
        # Gemini — fallback (нет стриминга)
        result = await generate_text(provider, model, history, user_input,
                                     openai_key, grok_key, gemini_key, max_history_tokens)
        yield result
        return

    # -------------------- OPENAI STREAM --------------------
    if provider == "openai":
        client = openai.AsyncOpenAI(api_key=openai_key)

        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages += [{"role": h["role"], "content": h["content"]} for h in history]

        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                text = chunk.choices[0].delta.get("content")
                if text:
                    yield text
        return

    # -------------------- GROK STREAM --------------------
    if provider == "grok":
        client = openai.AsyncOpenAI(
            api_key=grok_key,
            base_url="https://api.x.ai/v1"
        )

        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages += [{"role": h["role"], "content": h["content"]} for h in history]

        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                text = chunk.choices[0].delta.get("content")
                if text:
                    yield text
        return


# ---------------------------------------------------------
# IMAGE GENERATION
# ---------------------------------------------------------

async def generate_image(provider, model, prompt,
                         openai_key=None, grok_key=None, gemini_key=None):

    # -----------------------
    # OPENAI (DALL·E 3)
    # -----------------------
    if provider == "openai":
        client = openai.AsyncOpenAI(api_key=openai_key)
        try:
            r = await client.images.generate(
                model=model,
                prompt=prompt,
                size="1024x1024"
            )
            return r.data[0].url
        except Exception as e:
            raise RuntimeError(f"OpenAI image error: {e}")

    # -----------------------
    # GROK — FIXED!
    # -----------------------
    elif provider == "grok":
        client = openai.AsyncOpenAI(
            api_key=grok_key,
            base_url="https://api.x.ai/v1"
        )
        try:
            r = await client.images.generate(
                model=model,
                prompt=prompt,
                width=1024,           # 🔥 ВАЖНО: xAI не принимает size
                height=1024
            )

            b64 = r.data[0].b64_json
            return f"data:image/png;base64,{b64}"

        except Exception as e:
            raise RuntimeError(f"Grok image error: {e}")

    # -----------------------
    # GEMINI — отключён пока
    # -----------------------
    elif provider == "gemini":
        raise RuntimeError("Gemini image временно отключён: Google сменил API")

    else:
        raise RuntimeError("Unknown provider")
