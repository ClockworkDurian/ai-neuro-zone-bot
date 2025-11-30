# llm_core.py
import asyncio
import logging
import math
import random
from typing import AsyncIterator, Dict, List, Tuple

# сторонние клиенты
# Убедись, что в requirements есть: openai, google-generative-ai, groq
try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None

# Grok client placeholder (SDKs меняются) — адаптируй при необходимости
try:
    from groq import Groq
except Exception:
    Groq = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

# --- LOGGER (без логирования пользовательского текста) ---
logger = logging.getLogger("llm_core")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)

# --- Настройки ---
DEFAULT_MAX_TOKENS_HISTORY = 3000  # лимит токенов на одного пользователя в истории
DEFAULT_MAX_MESSAGES = 10  # запасной лимит по числу сообщений, если нужно
RETRY_ATTEMPTS = 2
RETRY_BACKOFF = 1.0  # seconds base

# Простейшая оценка "токенов" — приближённая, для управления контекстом
def estimate_tokens(text: str) -> int:
    # грубая аппроксимация: 1 токен ~ 0.75 слова
    words = len(text.split())
    tokens = int(words / 0.75) if words > 0 else 0
    return tokens

def history_token_count(history: List[Dict]) -> int:
    total = 0
    for m in history:
        total += estimate_tokens(m.get("content", ""))
    return total

def trim_history_by_tokens(history: List[Dict], max_tokens: int) -> List[Dict]:
    """Обрезать историю, сохранив как можно больше последних сообщений, чтобы не превышать max_tokens."""
    if not history:
        return history
    # идём с конца
    rev = list(reversed(history))
    total = 0
    kept = []
    for m in rev:
        t = estimate_tokens(m.get("content", ""))
        if total + t > max_tokens and kept:
            break
        total += t
        kept.append(m)
    kept_rev = list(reversed(kept))
    return kept_rev

# --- Простая стратегия retry для сетевых ошибок ---
async def _with_retries(coro_func, *args, attempts=RETRY_ATTEMPTS, backoff=RETRY_BACKOFF, **kwargs):
    last_exc = None
    for i in range(attempts + 1):
        try:
            return await coro_func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            wait = backoff * (2 ** i) + random.random() * 0.2
            logger.warning("llm_core: попытка %d не удалась: %s, ждём %.2fs", i+1, repr(e), wait)
            await asyncio.sleep(wait)
    raise last_exc

# --- Асинхронные реализации (упрощённые) ---

async def generate_text(provider: str,
                        model: str,
                        history: List[Dict],
                        user_input: str,
                        openai_key: str = None,
                        grok_key: str = None,
                        gemini_key: str = None,
                        max_history_tokens: int = DEFAULT_MAX_TOKENS_HISTORY) -> str:
    """Возвращает итоговый ответ (строка). НЕТ streaming. Использует retry."""
    # Обрезаем историю по токенам
    trimmed_history = trim_history_by_tokens(history + [{"role": "user", "content": user_input}], max_history_tokens)
    # формируем messages для OpenAI-подобных
    messages = [{"role": m["role"], "content": m["content"]} for m in trimmed_history]

    if provider == "openai":
        if AsyncOpenAI is None:
            raise RuntimeError("OpenAI SDK не установлен (AsyncOpenAI отсутствует).")
        async def call_openai():
            client = AsyncOpenAI(api_key=openai_key)
            # пример использования API, возможно потребуются правки под версию SDK
            resp = await client.chat.completions.create(model=model, messages=messages, max_tokens=1024)
            # безопасно получить контент
            try:
                return resp.choices[0].message.content
            except Exception:
                return str(resp)
        return await _with_retries(call_openai)

    elif provider == "grok":
        if Groq is None:
            raise RuntimeError("Grok SDK не установлен.")
        async def call_grok():
            client = Groq(api_key=grok_key)
            # Синхронный/асинхронный клиент зависит от SDK — здесь упрощение
            resp = client.chat.completions.create(model=model, messages=messages, max_tokens=1024)
            try:
                return resp.choices[0].message["content"]
            except Exception:
                return str(resp)
        return await _with_retries(call_grok)

    elif provider == "gemini":
        if genai is None:
            raise RuntimeError("Google Generative SDK не установлен.")
        async def call_gemini():
            genai.configure(api_key=gemini_key)
            model_engine = genai.GenerativeModel(model)
            # Примерный flow - SDK может отличаться
            convo = model_engine.start_chat(history=[{"role": m["role"], "parts":[{"text": m["content"]}]} for m in trimmed_history if "content" in m])
            convo.send_message(user_input)
            # Вернуть последний текст
            return convo.last.text
        return await _with_retries(call_gemini)
    else:
        raise ValueError("Unknown provider")

async def generate_text_stream(provider: str,
                               model: str,
                               history: List[Dict],
                               user_input: str,
                               openai_key: str = None,
                               grok_key: str = None,
                               gemini_key: str = None,
                               max_history_tokens: int = DEFAULT_MAX_TOKENS_HISTORY) -> AsyncIterator[str]:
    """
    Асинхронный генератор, отдаёт куски текста, чтобы клиент (бот) мог отправлять/редактировать сообщение по мере готовности.
    Не все SDK поддерживают стриминг — если стриминга нет, отдаёт один большой кусок.
    """
    trimmed_history = trim_history_by_tokens(history + [{"role": "user", "content": user_input}], max_history_tokens)
    messages = [{"role": m["role"], "content": m["content"]} for m in trimmed_history]

    if provider == "openai" and AsyncOpenAI is not None:
        client = AsyncOpenAI(api_key=openai_key)
        # Попытка стриминга — синтаксис SDK может отличаться в зависимости от версии.
        # Если не поддерживается — падаем в блок except и вернём цельный ответ.
        try:
            # stream=True — псевдокод; подстрой под SDK версии
            stream = await client.chat.completions.create(model=model, messages=messages, max_tokens=1024, stream=True)
            async for chunk in stream:
                # chunk может содержать частичный текст в разных полях — упрощаем:
                try:
                    text = chunk.choices[0].delta.content
                except Exception:
                    text = getattr(chunk, "content", str(chunk))
                if text:
                    yield text
            return
        except Exception as e:
            logger.info("OpenAI streaming fallback: %s", e)
            # fallthrough to single-response

    # Grok / Gemini / fallback: вернуть единый ответ
    text = await generate_text(provider, model, history, user_input, openai_key, grok_key, gemini_key, max_history_tokens)
    # отдаём порциями
    chunk_size = 120
    for i in range(0, len(text), chunk_size):
        yield text[i:i+chunk_size]

async def generate_image(provider: str,
                         prompt: str,
                         openai_key: str = None,
                         grok_key: str = None,
                         max_retries: int = RETRY_ATTEMPTS) -> str:
    """Возвращает URL изображения (или data, если платформа возвращает)."""
    if provider == "openai":
        if AsyncOpenAI is None:
            raise RuntimeError("OpenAI SDK отсутствует.")
        async def call_images():
            client = AsyncOpenAI(api_key=openai_key)
            resp = await client.images.generate(model="gpt-image-1", prompt=prompt, size="1024x1024")
            try:
                return resp.data[0].url
            except Exception:
                return str(resp)
        return await _with_retries(call_images)

    elif provider == "grok":
        if Groq is None:
            raise RuntimeError("Grok SDK отсутствует.")
        async def call_grok_img():
            client = Groq(api_key=grok_key)
            r = client.images.generate(model="grok-image-1", prompt=prompt, size="1024x1024")
            try:
                return r.data[0].url
            except Exception:
                return str(r)
        return await _with_retries(call_grok_img)

    else:
        raise ValueError("Provider не поддерживает изображение или неизвестен.")

# --- Небольшие утилиты для тестирования/мониторинга ---
async def ping_provider(provider: str, openai_key=None, grok_key=None, gemini_key=None) -> Dict:
    """Простейшая проверка доступности провайдера."""
    try:
        if provider == "openai":
            if AsyncOpenAI is None:
                return {"ok": False, "reason": "openai-sdk-missing"}
            client = AsyncOpenAI(api_key=openai_key)
            # пробная простая операция
            await client.models.list()
            return {"ok": True}
        if provider == "grok":
            if Groq is None:
                return {"ok": False, "reason": "grok-sdk-missing"}
            return {"ok": True}
        if provider == "gemini":
            if genai is None:
                return {"ok": False, "reason": "gemini-sdk-missing"}
            return {"ok": True}
    except Exception as e:
        return {"ok": False, "reason": str(e)}

