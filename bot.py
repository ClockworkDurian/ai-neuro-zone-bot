# bot.py (обновлённый)
import asyncio
import logging
import os
import time
from collections import defaultdict, deque
from typing import Dict, List

from aiogram import Bot, Dispatcher, types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils import exceptions, executor

from llm_core import generate_text_stream, generate_text, generate_image, trim_history_by_tokens, estimate_tokens

# --- Настройка логирования (без логирования пользовательских запросов) ---
logger = logging.getLogger("neurozone_bot")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)

# --- Переменные окружения (Railway) ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not BOT_TOKEN:
    logger.error("BOT_TOKEN not set in envs. Exiting.")
    raise SystemExit("BOT_TOKEN is required")

# --- Параметры (можно поднимать/тюнить) ---
MAX_HISTORY_TOKENS_DEFAULT = 3000
RATE_LIMIT_PER_MINUTE = 30  # максимум запросов на пользователя в минуту
STREAM_EDIT_INTERVAL = 0.35  # сек между редактированием сообщения при стриминге

# --- Простая rate-limit реализация: sliding window timestamps per user ---
user_requests: Dict[int, deque] = defaultdict(lambda: deque())

def check_rate_limit(user_id: int, limit=RATE_LIMIT_PER_MINUTE) -> bool:
    now = time.time()
    window = 60
    dq = user_requests[user_id]
    while dq and dq[0] < now - window:
        dq.popleft()
    if len(dq) >= limit:
        return False
    dq.append(now)
    return True

# --- Состояния пользователей (в памяти) ---
user_state: Dict[int, Dict] = defaultdict(lambda: {
    "mode": "text",  # or "image"
    "provider": "openai",  # "openai" / "grok" / "gemini"
    "model": None,
    "history": [],  # list of {"role":"user"/"assistant", "content": "..."}
    "max_history_tokens": MAX_HISTORY_TOKENS_DEFAULT
})

# --- Клавиатуры (статичные версии) ---
def kb_main() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton("Текст", callback_data="mode:text"),
         InlineKeyboardButton("Картинки", callback_data="mode:image")],
        [InlineKeyboardButton("Провайдер: OpenAI", callback_data="provider:openai"),
         InlineKeyboardButton("Grok", callback_data="provider:grok"),
         InlineKeyboardButton("Gemini", callback_data="provider:gemini")],
        [InlineKeyboardButton("Сброс истории", callback_data="action:reset")]
    ])

def kb_models(provider: str) -> InlineKeyboardMarkup:
    buttons = []
    if provider == "openai":
        buttons = [
            InlineKeyboardButton("gpt-4o-mini", callback_data="model:gpt-4o-mini"),
            InlineKeyboardButton("gpt-4o", callback_data="model:gpt-4o")
        ]
    elif provider == "grok":
        buttons = [
            InlineKeyboardButton("grok-2-mini", callback_data="model:grok-2-mini"),
            InlineKeyboardButton("grok-2", callback_data="model:grok-2")
        ]
    elif provider == "gemini":
        buttons = [
            InlineKeyboardButton("gemini-1.5-flash", callback_data="model:gemini-1.5-flash")
        ]
    kb = InlineKeyboardMarkup()
    kb.row(*buttons)
    return kb

# --- Инициализация бота / dispatcher ---
bot = Bot(token=BOT_TOKEN, parse_mode="HTML")
dp = Dispatcher()

# --- HELPERS ---
def safe_log_event(user_id: int, event: str, extra: dict = None):
    """Логируем событие, НО НЕ хранит и не выводит пользовательский текст."""
    msg = {"user_id": user_id, "event": event}
    if extra:
        msg.update(extra)
    logger.info("EVENT %s", msg)

# Ограничение истории по токенам (используем функцию trim_history_by_tokens)
def trim_user_history(user_id: int):
    st = user_state[user_id]
    hist = st["history"]
    st["history"] = trim_history_by_tokens(hist, st.get("max_history_tokens", MAX_HISTORY_TOKENS_DEFAULT))

# --- Handlers ---

@dp.message(commands=["start", "help"])
async def cmd_start(message: types.Message):
    user_id = message.from_user.id
    safe_log_event(user_id, "start")
    await message.answer("Привет! Выбери режим и провайдера в меню.", reply_markup=kb_main())

@dp.callback_query()
async def cb_handler(q: types.CallbackQuery):
    user_id = q.from_user.id
    data = q.data or ""
    safe_log_event(user_id, "callback", {"data": data})
    if data.startswith("mode:"):
        mode = data.split(":",1)[1]
        user_state[user_id]["mode"] = mode
        await q.message.edit_text(f"Режим: {mode}. Выберите провайдера и модель.", reply_markup=kb_models(user_state[user_id].get("provider","openai")))
        await q.answer()
        return
    if data.startswith("provider:"):
        prov = data.split(":",1)[1]
        user_state[user_id]["provider"] = prov
        await q.message.edit_text(f"Провайдер: {prov}. Выберите модель.", reply_markup=kb_models(prov))
        await q.answer()
        return
    if data.startswith("model:"):
        model = data.split(":",1)[1]
        user_state[user_id]["model"] = model
        await q.message.edit_text(f"Модель установлена: {model}\nТеперь просто отправь сообщение с запросом.", reply_markup=kb_main())
        await q.answer()
        return
    if data == "action:reset":
        user_state[user_id]["history"] = []
        await q.answer("История сброшена.")
        await q.message.edit_text("История очищена.", reply_markup=kb_main())
        safe_log_event(user_id, "history_reset")
        return
    await q.answer()

@dp.message()
async def on_message(message: types.Message):
    user_id = message.from_user.id
    text = message.text or ""
    safe_log_event(user_id, "message_received", {"len": len(text)})
    # rate limit
    if not check_rate_limit(user_id):
        await message.answer("Слишком много запросов — подожди минуту.")
        safe_log_event(user_id, "rate_limited")
        return

    state = user_state[user_id]
    mode = state.get("mode", "text")
    provider = state.get("provider", "openai")
    model = state.get("model")
    if mode == "image":
        # генерация изображения
        # логируем событие (без prompt)
        safe_log_event(user_id, "image_request", {"provider": provider, "model": model})
        try:
            # короткая заглушка - вызов llm_core
            img_url = await generate_image(provider=provider, prompt=text,
                                           openai_key=OPENAI_API_KEY, grok_key=GROK_API_KEY)
            await message.answer_photo(img_url, caption="Готово (сгенерировано).")
            # записать в историю (не сохраняем сам prompt в лог)
            state["history"].append({"role": "user", "content": "[image-prompt]"})
            state["history"].append({"role": "assistant", "content": "[image-generated]"})
            trim_user_history(user_id)
            safe_log_event(user_id, "image_ok", {"provider": provider})
        except Exception as e:
            safe_log_event(user_id, "image_error", {"err": str(e)})
            await message.answer("Ошибка при генерации изображения. Повторите позже.")
        return

    # text mode
    # append user message to history (we store the content for actual requests, but logs don't print it)
    state["history"].append({"role": "user", "content": text})
    trim_user_history(user_id)

    # streaming: сначала отправляем "печатает..." и потом редактируем
    status_msg = await message.answer("Генерирую ответ...")
    try:
        # Try streaming if supported
        stream_gen = generate_text_stream(provider=provider,
                                          model=model or ("gpt-4o-mini" if provider=="openai" else "grok-2-mini"),
                                          history=state["history"],
                                          user_input=text,
                                          openai_key=OPENAI_API_KEY,
                                          grok_key=GROK_API_KEY,
                                          gemini_key=GEMINI_API_KEY,
                                          max_history_tokens=state.get("max_history_tokens", MAX_HISTORY_TOKENS_DEFAULT))
        # accumulate small chunks for edit
        full = ""
        last_edit_time = time.time()
        async for chunk in stream_gen:
            # chunk is str
            full += chunk
            # редактируем сообщение каждые STREAM_EDIT_INTERVAL
            if time.time() - last_edit_time >= STREAM_EDIT_INTERVAL:
                try:
                    await status_msg.edit_text(full)
                except exceptions.TelegramAPIError:
                    pass
                last_edit_time = time.time()
        # final edit
        try:
            await status_msg.edit_text(full)
        except exceptions.TelegramAPIError:
            pass

        # добавляем ассистента в историю (сохраняем текст, но не логируем)
        state["history"].append({"role": "assistant", "content": full})
        trim_user_history(user_id)
        safe_log_event(user_id, "text_ok", {"provider": provider, "model": model, "out_len": len(full)})
    except Exception as e:
        safe_log_event(user_id, "text_error", {"provider": provider, "err": str(e)})
        # fallback: пробуем non-stream генерацию (retry handled inside)
        try:
            resp = await generate_text(provider=provider,
                                       model=model or ("gpt-4o-mini" if provider=="openai" else "grok-2-mini"),
                                       history=state["history"],
                                       user_input=text,
                                       openai_key=OPENAI_API_KEY,
                                       grok_key=GROK_API_KEY,
                                       gemini_key=GEMINI_API_KEY,
                                       max_history_tokens=state.get("max_history_tokens", MAX_HISTORY_TOKENS_DEFAULT))
            await status_msg.edit_text(resp)
            state["history"].append({"role": "assistant", "content": resp})
            trim_user_history(user_id)
            safe_log_event(user_id, "text_ok_fallback", {"provider": provider})
        except Exception as e2:
            logger.exception("Final fallback error: %s", e2)
            await status_msg.edit_text("Не удалось получить ответ. Попробуйте позже.")

# --- Command to set max_history_tokens by user (for testing/tariffs) ---
@dp.message(commands=["set_history_tokens"])
async def cmd_set_history(message: types.Message):
    user_id = message.from_user.id
    args = (message.get_args() or "").strip()
    if not args.isdigit():
        await message.answer("Использование: /set_history_tokens 3000")
        return
    v = int(args)
    user_state[user_id]["max_history_tokens"] = v
    await message.answer(f"Лимит истории в токенах установлен: {v}")
    safe_log_event(user_id, "set_history", {"tokens": v})

# graceful shutdown handler not required for Railway but useful locally
async def on_startup(dp):
    logger.info("Bot started")

async def on_shutdown(dp):
    logger.info("Bot shutting down")

if __name__ == "__main__":
    # запускаем long-polling (Railway обычно запускает python bot.py)
    executor.start_polling(dp, on_startup=on_startup, on_shutdown=on_shutdown)
