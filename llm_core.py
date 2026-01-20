import openai
import base64

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
# TEXT
# ---------------------------------------------------------

async def generate_text_stream(provider, model, history, user_input,
                               openai_key=None, grok_key=None, gemini_key=None,
                               max_history_tokens=5000):

    history = trim_history_by_tokens(
        history + [{"role": "user", "content": user_input}],
        max_history_tokens
    )

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages += history

    if provider == "openai":
        client = openai.AsyncOpenAI(api_key=openai_key)

    elif provider == "grok":
        client = openai.AsyncOpenAI(
            api_key=grok_key,
            base_url="https://api.x.ai/v1"
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
# IMAGE
# ---------------------------------------------------------

async def generate_image(provider, model, prompt,
                         openai_key=None, grok_key=None, gemini_key=None):

    if provider == "openai":
        client = openai.AsyncOpenAI(api_key=openai_key)
        r = await client.images.generate(
            model=model,
            prompt=prompt,
            size="1024x1024"
        )
        return r.data[0].url

    elif provider == "grok":
        client = openai.AsyncOpenAI(
            api_key=grok_key,
            base_url="https://api.x.ai/v1"
        )

        r = await client.images.generate(
            model=model,
            prompt=prompt,
            width=1024,
            height=1024
        )

        img_bytes = base64.b64decode(r.data[0].b64_json)
        return img_bytes

    else:
        raise RuntimeError("Image provider not supported")
