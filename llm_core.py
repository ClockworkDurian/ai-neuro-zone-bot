import os
import base64
import httpx
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ==========================================================
# TEXT GENERATION (у тебя уже работает — НЕ ТРОГАЕМ)
# ==========================================================

async def generate_text(provider: str, model: str, messages: list[str]) -> str:
    if provider == "openai":
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": m} for m in messages],
        )
        return resp.choices[0].message.content

    if provider == "grok":
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": m} for m in messages],
                },
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    raise RuntimeError("Unknown provider")


# ==========================================================
# IMAGE GENERATION — ВОТ ТУТ БЫЛА ПРОБЛЕМА
# ==========================================================

async def generate_image(provider: str, model: str, prompt: str) -> str:
    # ------------------ OpenAI ------------------
    if provider == "openai":
        result = openai_client.images.generate(
            model=model,
            prompt=prompt,
            size="1024x1024",
        )
        return result.data[0].url

    # ------------------ GROK (xAI) ------------------
    if provider == "grok":
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.x.ai/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {GROK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "prompt": prompt,
                    "size": "1024x1024",
                },
            )

            r.raise_for_status()
            data = r.json()

            # xAI возвращает base64, а не url
            image_b64 = data["data"][0].get("b64_json")
            if not image_b64:
                raise RuntimeError("xAI image response has no b64_json")

            image_bytes = base64.b64decode(image_b64)
            filename = "grok_image.png"

            with open(filename, "wb") as f:
                f.write(image_bytes)

            # bot.py уже умеет отправлять локальный файл
            return filename

    raise RuntimeError("Unknown provider")