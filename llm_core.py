import base64
import httpx
import openai

# ---------------------------------------------------------
# CUSTOM HTTPX CLIENT (NO PROXIES!)
# ---------------------------------------------------------

def make_http_client():
    return httpx.AsyncClient(
        timeout=60.0,
        proxies=None,          # КЛЮЧЕВО
        trust_env=False        # ИГНОРИРУЕМ HTTP_PROXY из Railway
    )

# ---------------------------------------------------------
# HISTORY TRIM
# ---------------------------------------------------------

def trim_history_by_tokens(history, max_tokens):
    tokens = 0
    out = []
    for m in reversed(history):
        tokens += len(m.get("content", "").split())
        if tokens > max_tokens:
            break
        out.append(m)
    return list(reversed(out))

# ---------------------------------------------------------
# TEXT STREAM
# ---------------------------------------------------------

async def generate_text_stream(
    provider,
    model,
    history,
    user_input,
    openai_key=None,
    grok_key=None,
    gemini_key=None,
    max_history_tokens=5000
):
    history = trim_history_by_tokens(
        history + [{"role": "user", "content": user_input}],
        max_history_tokens
    )

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages += history

    http_client = make_http_client()

    if provider == "openai":
        client = openai.AsyncOpenAI(
            api_key=openai_key,
            http_client=http_client
        )

    elif provider == "grok":
        client = openai.AsyncOpenAI(
            api_key=grok_key,
            base_url="https://api.x.ai/v1",
            http_client=http_client
        )
    else:
        raise RuntimeError("Provider not supported")

    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta:
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                yield content

# ---------------------------------------------------------
# TEXT NON-STREAM (COMPAT)
# ---------------------------------------------------------

async def generate_text(
    provider,
    model,
    history,
    user_input,
    openai_key=None,
    grok_key=None,
    gemini_key=None,
    max_history_tokens=5000
):
    result = ""
    async for chunk in generate_text_stream(
        provider,
        model,
        history,
        user_input,
        openai_key,
        grok_key,
        gemini_key,
        max_history_tokens
    ):
        result += chunk
    return result

# ---------------------------------------------------------
# IMAGE
# ---------------------------------------------------------

async def generate_image(
    provider,
    model,
    prompt,
    openai_key=None,
    grok_key=None,
    gemini_key=None
):
    http_client = make_http_client()

    if provider == "openai":
        client = openai.AsyncOpenAI(
            api_key=openai_key,
            http_client=http_client
        )
        r = await client.images.generate(
            model=model,
            prompt=prompt,
            size="1024x1024"
        )
        return r.data[0].url

    elif provider == "grok":
        client = openai.AsyncOpenAI(
            api_key=grok_key,
            base_url="https://api.x.ai/v1",
            http_client=http_client
        )
        r = await client.images.generate(
            model=model,
            prompt=prompt,
            width=1024,
            height=1024
        )
        return base64.b64decode(r.data[0].b64_json)

    else:
        raise RuntimeError("Image provider not supported")
