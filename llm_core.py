# llm_core.py
import base64
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
# TEXT (NON-STREAM, ОСНОВА)
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

        return resp.choices[0].message.content

# ---------------------------------------------------------
# TEXT STREAM (СОВМЕСТИМОСТЬ С bot.py)
# ---------------------------------------------------------

async def generate_text_stream(
    provider: str,
    model: str,
    history: list,
    user_input: str,
    openai_key: str = None,
    grok_key: str = None,
    gemini_key: str = None,
    max_history_tokens: int = 8000,
):
    """
    Совместимость с bot.py.
    Реального стриминга сейчас нет — отдаём весь текст одним чанком.
    """
    text = await generate_text(
        provider=provider,
        model=model,
        history=history,
        user_input=user_input,
        openai_key=openai_key,
        grok_key=grok_key,
        gemini_key=gemini_key,
        max_history_tokens=max_history_tokens,
    )
    yield text

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

            return base64.b64decode(r.data[0].b64_json)

        else:
            raise RuntimeError("Unsupported image provider")
