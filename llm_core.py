"""
llm_core.py
Унифицированный слой для работы с OpenAI (ChatGPT), xAI Grok и Google Gemini.
Функции:
- generate_text_stream(...) -> async generator (chunks)
- generate_text(...) -> final text (async)
- generate_image(...) -> returns image_url (async)
- trim_history_by_tokens(...) -> helper для обрезки истории (псевдо-токены)
Примечание: реализована максимально совместимая логика с теми SDK,
которые использовались в старом bot.py (xai_sdk, openai.AsyncOpenAI, google.generativeai).
"""

import asyncio
import logging
import time
from typing import AsyncIterator, List, Dict, Any

import openai
import google.generativeai as genai
from xai_sdk import Client as XAI_Client
from xai_sdk.chat import user as xai_user, assistant as xai_assistant

logger = logging.getLogger("llm_core")
logger.setLevel(logging.INFO)

# --- Утилиты ---
def _history_to_openai_messages(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Преобразовать историю в формат OpenAI messages."""
    msgs = []
    for m in history:
        r = m.get("role", "user")
        c = m.get("content", "")
        msgs.append({"role": r, "content": c})
    return msgs

def trim_history_by_tokens(history: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
    """
    Простая эвристика: считаем 1 слово ~= 1 токен (грубая), и обрезаем историю с головы.
    В реальном проекте можно заменить на более точный подсчёт токенов.
    """
    if not history:
        return history
    tokens = 0
    out = []
    # идём с конца, собираем пока не превысим max_tokens
    for m in reversed(history):
        tokens += len(m.get("content", "").split())
        if tokens > max_tokens:
            break
        out.append(m)
    return list(reversed(out))

# --- Основные функции ---

async def generate_text(provider: str,
                        model: str,
                        history: List[Dict[str, str]],
                        user_input: str,
                        openai_key: str = None,
                        grok_key: str = None,
                        gemini_key: str = None,
                        max_history_tokens: int = 2000) -> str:
    """
    Запрос к модели — возвращает итоговый текст.
    provider: "openai" | "grok" | "gemini"
    model: id модели (например "gpt-5", "gpt-4.1", "grok-4-fast-reasoning", "gemini-2.5-flash")
    history: list of {"role":"user"/"assistant", "content": "..."}
    """
    history = trim_history_by_tokens(history + [{"role": "user", "content": user_input}], max_history_tokens)
    try:
        if provider == "openai":
            if not openai_key:
                raise ValueError("OpenAI key not provided")
            client = openai.AsyncOpenAI(api_key=openai_key)
            messages = _history_to_openai_messages(history)
            # GPT-5 family expects max_completion_tokens, others - max_tokens
            kwargs = {}
            if model and model.startswith("gpt-5"):
                kwargs["max_completion_tokens"] = 1024
            else:
                kwargs["max_tokens"] = 1024
            resp = await client.chat.completions.create(model=model, messages=messages, **kwargs)
            # Поддержка разных структур ответа (защита)
            try:
                content = resp.choices[0].message.content
            except Exception:
                # fallback
                content = getattr(resp.choices[0].message, "content", str(resp))
            return content or ""

        elif provider == "grok":
            if not grok_key:
                raise ValueError("Grok (xAI) key not provided")
            xai = XAI_Client(api_key=grok_key)
            # Собираем сообщения в формате xai_sdk: user(...) / assistant(...)
            msgs = []
            for m in history:
                if m["role"] == "user":
                    msgs.append(xai_user(m["content"]))
                else:
                    msgs.append(xai_assistant(m["content"]))
            # Добавим последний юзерский
            msgs.append(xai_user(user_input))
            # Создаём чат и получаем ответ синхронно в отдельном тред (xai_sdk может быть sync)
            def _run_chat():
                chat = xai.chat.create(model=model, messages=msgs)
                # sample() возвращает ответ (sync)
                sample = chat.sample()
                return getattr(sample, "content", str(sample))
            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(None, _run_chat)
            return content or ""

        elif provider == "gemini":
            if not gemini_key:
                raise ValueError("Gemini key not provided")
            genai.configure(api_key=gemini_key)
            # Для Gemini используем простой подход: генai.generate_text (если доступно)
            # Но для совместимости с новым API используем интерфейс chat/parts
            # Попробуем сначала использовать generaciones через chat-like api:
            messages = [{"role": "user", "parts": [{"text": msg["content"]}]} for msg in history]
            # new api: genai.chat (if exists)
            try:
                model_obj = genai.GenerativeModel(model)
                chat = model_obj.start_chat(history=messages)
                response = await chat.send_message_async(user_input)
                return getattr(response, "text", str(response))
            except Exception as e:
                # fallback: try simple generate_text
                try:
                    resp = genai.generate_text(model=model, prompt=user_input)
                    return getattr(resp, "candidates", [])[0].get("output", "") if resp else ""
                except Exception as e2:
                    raise e2

        else:
            raise ValueError("Unknown provider")
    except Exception as e:
        logger.exception("generate_text failed: %s", e)
        raise

async def generate_text_stream(provider: str,
                               model: str,
                               history: List[Dict[str, str]],
                               user_input: str,
                               openai_key: str = None,
                               grok_key: str = None,
                               gemini_key: str = None,
                               max_history_tokens: int = 2000) -> AsyncIterator[str]:
    """
    Streaming-обёртка: по возможности стримим, иначе возвращаем итог одним чанком.
    Для совместимости некоторые SDK не поддерживают async streaming одинаково,
    поэтому мы используем generate_text() и возвращаем чанки.
    """
    try:
        # Попробуем получить итог
        final = await generate_text(provider, model, history, user_input,
                                    openai_key=openai_key, grok_key=grok_key,
                                    gemini_key=gemini_key, max_history_tokens=max_history_tokens)
        # Разбиваем на разумные чанки (симулируем стриминг)
        chunk_size = 80
        i = 0
        while i < len(final):
            chunk = final[i:i + chunk_size]
            yield chunk
            i += chunk_size
            await asyncio.sleep(0)  # уступаем планировщику
    except Exception as e:
        logger.exception("generate_text_stream failed: %s", e)
        # поднимем исключение наружу — вызывающий код обработает fallback
        raise

async def generate_image(provider: str,
                         prompt: str,
                         openai_key: str = None,
                         grok_key: str = None,
                         gemini_key: str = None) -> str:
    """
    Генерация изображения: возвращает URL (или выбрасывает исключение).
    Поддерживаем OpenAI (DALL-E 3) и xAI (Grok Image).
    """
    try:
        if provider == "openai":
            if not openai_key:
                raise ValueError("OpenAI key not provided")
            client = openai.AsyncOpenAI(api_key=openai_key)
            resp = await client.images.generate(model="dall-e-3", prompt=prompt, n=1, size="1024x1024")
            # структура: resp.data[0].url
            return resp.data[0].url

        elif provider == "grok":
            if not grok_key:
                raise ValueError("Grok key not provided")
            xai = XAI_Client(api_key=grok_key)
            def _run_image():
                # пример: xai_client.image.sample(...)
                res = xai.image.sample(model="grok-2-image-1212", prompt=prompt, image_format="url")
                return getattr(res, "url", None) or res
            loop = asyncio.get_running_loop()
            url = await loop.run_in_executor(None, _run_image)
            return url

        elif provider == "gemini":
            # Вариант: генерируем через Gemini image (если доступно)
            if not gemini_key:
                raise ValueError("Gemini key not provided")
            genai.configure(api_key=gemini_key)
            try:
                resp = genai.generate_image(model="gemini-image-1", prompt=prompt, size="1024x1024")
                # структура зависит от версии
                return resp.output[0].images[0].uri
            except Exception:
                raise RuntimeError("Gemini image generation not implemented or key/quotas exhausted")
        else:
            raise ValueError("Unknown image provider")
    except Exception as e:
        logger.exception("generate_image failed: %s", e)
        raise
