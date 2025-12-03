"""
llm_core.py
Унифицированный слой для OpenAI (ChatGPT), xAI Grok (через OpenAI-compatible client),
и Google Gemini (google.generativeai).
Реализовано:
 - generate_text(...)  -> final string (async)
 - generate_text_stream(...) -> async iterator of chunks (async)
 - generate_image(...) -> returns URL (async)
 - trim_history_by_tokens(...) helper
Поведение:
 - GPT-5 family -> max_completion_tokens
 - GPT-4.1 and others -> max_tokens
 - Grok uses AsyncOpenAI with base_url https://api.x.ai/v1
 - Gemini uses google.generativeai.generate_text where possible
"""

import asyncio
import logging
from typing import List, Dict, AsyncIterator

import openai
import google.generativeai as genai

logger = logging.getLogger("llm_core")
logger.setLevel(logging.INFO)

# ---------------------------
# Helpers
# ---------------------------
def _history_to_openai_messages(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    msgs = []
    for m in history:
        msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    return msgs

def trim_history_by_tokens(history: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
    """Грубая эвристика: 1 слово ~= 1 токен"""
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

# ---------------------------
# Core functions
# ---------------------------
async def generate_text(provider: str,
                        model: str,
                        history: List[Dict[str, str]],
                        user_input: str,
                        openai_key: str = None,
                        grok_key: str = None,
                        gemini_key: str = None,
                        max_history_tokens: int = 2000) -> str:
    """
    Выполняет итоговый (non-stream) запрос к провайдеру и возвращает текст.
    Может поднять RuntimeError("GEMINI_QUOTA_EXCEEDED") при 429 от Gemini.
    """
    history = trim_history_by_tokens(history + [{"role": "user", "content": user_input}], max_history_tokens)

    # ---------------- OpenAI (ChatGPT family) ----------------
    if provider == "openai":
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not provided")
        client = openai.AsyncOpenAI(api_key=openai_key)
        messages = _history_to_openai_messages(history)
        # decide token param
        if model and model.startswith("gpt-5"):
            params = {"max_completion_tokens": 1024}
        else:
            params = {"max_tokens": 1024}
        try:
            resp = await client.chat.completions.create(model=model, messages=messages, **params)
            # try to extract text
            try:
                return resp.choices[0].message.content or ""
            except Exception:
                # fallback to str
                return str(resp)
        except Exception as e:
            logger.exception("OpenAI error: %s", e)
            raise

    # ---------------- Grok (xAI via OpenAI-compatible client) ----------------
    if provider == "grok":
        if not grok_key:
            raise ValueError("GROK_API_KEY not provided")
        # use AsyncOpenAI with base_url
        client = openai.AsyncOpenAI(api_key=grok_key, base_url="https://api.x.ai/v1")
        messages = _history_to_openai_messages(history)
        # Grok expects max_tokens param typically
        try:
            resp = await client.chat.completions.create(model=model or "grok-2-mini", messages=messages, max_tokens=1024)
            try:
                return resp.choices[0].message.content or ""
            except Exception:
                return str(resp)
        except Exception as e:
            logger.exception("Grok error: %s", e)
            raise

    # ---------------- Gemini (google.generativeai) ----------------
    if provider == "gemini":
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY not provided")

        # build a simple prompt concatenating history (more predictable)
        parts = []
        for m in history:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                parts.append(f"User: {content}")
            else:
                parts.append(f"Assistant: {content}")
        parts.append(f"User: {user_input}")
        prompt = "\n".join(parts)

        try:
            genai.configure(api_key=gemini_key)
            # Try simple generate_text (more robust for short prompts)
            resp = genai.generate_text(model=model, prompt=prompt)
            # resp might have different structure; try common fields
            if resp is None:
                return ""
            # prefer resp.candidates[0].output or resp.candidates[0].content or resp.text
            # handle different SDK versions:
            if hasattr(resp, "candidates") and resp.candidates:
                cand = resp.candidates[0]
                # cand might be dict-like
                if isinstance(cand, dict):
                    out = cand.get("output") or cand.get("content") or cand.get("text")
                    if isinstance(out, str):
                        return out
                    return str(cand)
                else:
                    # object-like
                    out = getattr(cand, "output", None) or getattr(cand, "content", None) or getattr(cand, "text", None)
                    return out if isinstance(out, str) else str(cand)
            # try direct fields
            if hasattr(resp, "text"):
                return getattr(resp, "text") or ""
            # fallback
            return str(resp)
        except Exception as e:
            # detect quota/resource exhausted errors (best-effort)
            msg = str(e).lower()
            logger.exception("Gemini error: %s", e)
            if "quota" in msg or "resourceexhausted" in msg or "resource exhausted" in msg or "429" in msg:
                # raise sentinel for bot to mark provider unavailable
                raise RuntimeError("GEMINI_QUOTA_EXCEEDED")
            raise

    raise ValueError("Unknown provider: %s" % str(provider))

# ---------------------------
# Streaming wrapper (emulated)
# ---------------------------
async def generate_text_stream(provider: str,
                               model: str,
                               history: List[Dict[str, str]],
                               user_input: str,
                               openai_key: str = None,
                               grok_key: str = None,
                               gemini_key: str = None,
                               max_history_tokens: int = 2000) -> AsyncIterator[str]:
    """
    Эмулируемый стрим: получает итог от generate_text и разбивает на чанки.
    """
    final = await generate_text(provider=provider, model=model, history=history, user_input=user_input,
                                openai_key=openai_key, grok_key=grok_key, gemini_key=gemini_key,
                                max_history_tokens=max_history_tokens)
    # guard
    if not isinstance(final, str):
        final = str(final)
    chunk_size = 80
    i = 0
    while i < len(final):
        yield final[i:i + chunk_size]
        i += chunk_size
        await asyncio.sleep(0)

# ---------------------------
# Image generation
# ---------------------------
async def generate_image(provider: str,
                         prompt: str,
                         openai_key: str = None,
                         grok_key: str = None,
                         gemini_key: str = None) -> str:
    """
    Возвращает URL картинки.
    """
    # OPENAI images (DALL·E 3)
    if provider == "openai":
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not provided")
        client = openai.AsyncOpenAI(api_key=openai_key)
        try:
            resp = await client.images.generate(model="dall-e-3", prompt=prompt, size="1024x1024", n=1)
            # common structure: resp.data[0].url
            return resp.data[0].url
        except Exception as e:
            logger.exception("OpenAI image error: %s", e)
            raise

    # GROK images via xAI base_url
    if provider == "grok":
        if not grok_key:
            raise ValueError("GROK_API_KEY not provided")
        client = openai.AsyncOpenAI(api_key=grok_key, base_url="https://api.x.ai/v1")
        try:
            # try a reasonable model name; if your old working bot used different name, substitute here
            resp = await client.images.generate(model="grok-image-1", prompt=prompt, size="1024x1024", n=1)
            return resp.data[0].url
        except Exception as e:
            logger.exception("Grok image error: %s", e)
            raise

    # GEMINI image
    if provider == "gemini":
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY not provided")
        try:
            genai.configure(api_key=gemini_key)
            resp = genai.generate_image(model="gemini-image-1", prompt=prompt, size="1024x1024")
            # structure depends on sdk version:
            try:
                return resp.output[0].images[0].uri
            except Exception:
                # fallback
                return str(resp)
        except Exception as e:
            logger.exception("Gemini image error: %s", e)
            raise

    raise ValueError("Unknown provider for image generation")
