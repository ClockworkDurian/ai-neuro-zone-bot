# bot.py — финальная версия (aiogram 3.13.1 + streaming + token limit + rate-limit + модели из старого бота)

import asyncio
import logging
import os
import time
from collections import defaultdict, deque

from aiogram import Bot, Dispatcher, types
from aiogram.exceptions import TelegramAPIError
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

# Импорт основной логики ИИ
from llm_core import (
    generate_text_stream,
    generate_text,
    generate_image,
    trim_history_by_tokens
)

# -------------------------------------------------------------------
# ЛОГИРОВАНИЕ
# -------------------------------------------------------------------
logger = logging.getLogger("neurozone_bot")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

def safe_log(user_id, event, extra=None):
    data = {"user_id": user_id, "event": event}
    if extra:
        data.update(extra)
    logger.info(data)

# -------------------------------------------------------------------
# ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ
# -------------------------------------------------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not BOT_TOKEN:
    raise SystemExit("❌ BOT_TOKEN отсутствует в окружении")

# -------------------------------------------------------------------
# ИНИЦИАЛИЗАЦИЯ aiogram 3.x
# -------------------------------------------------------------------
bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode="HTML")
)
dp = Dispatcher()

# -------------------------------------------------------------------
# RATE-LIMIT
# -------------------------------------------------------------------
RATE_LIMIT_PER_MINUTE = 30
user_requests = defaultdict(lambda: deque())

def check_rate_limit(user_id: int) -> bool:
    now = time.time()
    dq = user_requests[user_id]
    while dq and dq[0] < now - 60:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_PER_MINUTE:
        return False
    dq.append(now)
    return True

# -------------------------------------------------------------------
# ИСТОРИЯ ПОЛЬЗОВАТЕЛЯ
# -------------------------------------------------------------------
MAX_HISTORY_TOKENS_DEFAULT = 3000

user_state = defaultdict(lambda: {
    "mode": "text",
    "provider": "openai",
    "model": None,
    "history": [],
    "max_history_tokens": MAX_HISTORY_TOKENS_DEFAULT
})

def trim_user_history(uid: int):
    st = user_state[uid]
    st["history"] = trim_history_by_tokens(
        st["history"],
        st["max_history_tokens"]
    )

# -------------------------------------------------------------------
# МОДЕЛИ (из твоего старого bot.py)
# -------------------------------------------------------------------
openai_models = {
    "GPT-5": {"id": "gpt-5", "desc": "Лучшая модель для кода и агентных задач."},
    "GPT-5 mini": {"id": "gpt-5-mini", "desc": "Более быстрая, экономичная версия."},
    "GPT-5 nano": {"id": "gpt-5-nano", "desc": "Самая быстрая и экономичная версия."},
    "GPT-4.1": {"id": "gpt-4.1", "desc": "Самая умная модель без рассуждений."}
}

grok_models = {
    "Grok-code-fast-1": {"id": "grok-code-fast-1", "desc": "Быстрая и экономичная модель для кодирования."},
    "Grok-4-fast-reasoning": {"id": "grok-4-fast-reasoning", "desc": "Последнее достижение в экономичных моделях."},
    "Grok-4-fast-non-reasoning": {"id": "grok-4-fast-non-reasoning", "desc": "Последнее достижение в экономичных моделях."}
}

gemini_models = {
    "Gemini 2.5 Flash": {"id": "gemini-2.5-flash", "desc": "Лучшая по цене/производительности."},
    "Gemini 2.5 Flash-Lite": {"id": "gemini-2.5-flash-lite", "desc": "Самая быстрая flash-модель."}
}

# -------------------------------------------------------------------
# КЛАВИАТУРЫ
# -------------------------------------------------------------------

def kb_main() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Текст", callback_data="mode:text"),
            InlineKeyboardButton(text="Картинки", callback_data="mode:image")
        ],
        [
            InlineKeyboardButton(text="OpenAI", callback_data="provider:openai"),
            InlineKeyboardButton(text="Grok", callback_data="provider:grok"),
            InlineKeyboardButton(text="Gemini", callback_data="provider:gemini")
        ],
        [
            InlineKeyboardButton(text="Сброс истории", callback_data="reset:history")
        ]
    ])

async def show_models_for_provider(cb: types.CallbackQuery, provider_key: str):
    """Показ моделей с описаниями."""
    uid = cb.from_user.id

    if provider_key == "openai":
        models_dict = openai_models
        header = "🔵 <b>OpenAI — ChatGPT</b>"
    elif provider_key == "grok":
        models_dict = grok_models
        header = "🧠 <b>Grok — xAI</b>"
    else:
        models_dict = gemini_models
        header = "⚡ <b>Gemini — Google AI</b>"

    parts = [f"{header}\n\nВыберите модель:"]
    kb_rows = []

    for name, meta in models_dict.items():
        parts.append(f"\n<b>{name}</b> — <i>{meta['desc']}</i>")
        kb_rows.append([
            InlineKeyboardButton(text=name, callback_data=f"model:{meta['id']}")
        ])

    kb_rows.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="back:providers")])

    txt = "\n".join(parts)

    try:
        await cb.message.edit_text(txt, reply_markup=InlineKeyboardMarkup(inline_keyboard=kb_rows))
    except:
        await cb.message.answer(txt, reply_markup=InlineKeyboardMarkup(inline_keyboard=kb_rows))

    await cb.answer()

# -------------------------------------------------------------------
# CALLBACK HANDLERS
# -------------------------------------------------------------------

@dp.message(Command("start", "help"))
async def start_cmd(message: types.Message):
    uid = message.from_user.id
    safe_log(uid, "start")
    await message.answer("Привет! Выберите режим работы:", reply_markup=kb_main())

@dp.callback_query(lambda c: c.data and c.data.startswith("mode:"))
async def set_mode(cb: types.CallbackQuery):
    uid = cb.from_user.id
    mode = cb.data.split(":", 1)[1]
    user_state[uid]["mode"] = mode
    safe_log(uid, "mode_set", {"mode": mode})
    await cb.message.edit_text(
        f"Режим установлен: <b>{mode}</b>\nТеперь выберите провайдера:",
        reply_markup=kb_main()
    )
    await cb.answer()

@dp.callback_query(lambda c: c.data and c.data.startswith("provider:"))
async def set_provider(cb: types.CallbackQuery):
    uid = cb.from_user.id
    provider = cb.data.split(":", 1)[1]
    user_state[uid]["provider"] = provider
    safe_log(uid, "provider_set", {"provider": provider})
    await show_models_for_provider(cb, provider)

@dp.callback_query(lambda c: c.data == "back:providers")
async def back_to_providers(cb: types.CallbackQuery):
    await cb.message.edit_text("Выберите режим:", reply_markup=kb_main())
    await cb.answer()

@dp.callback_query(lambda c: c.data and c.data.startswith("model:"))
async def set_model(cb: types.CallbackQuery):
    uid = cb.from_user.id
    model_id = cb.data.split(":", 1)[1]
    user_state[uid]["model"] = model_id
    safe_log(uid, "model_set", {"model": model_id})
    await cb.message.edit_text(
        f"Вы выбрали модель:\n<b>{model_id}</b>\n\nТеперь отправьте ваш запрос.",
        reply_markup=kb_main()
    )
    await cb.answer()

@dp.callback_query(lambda c: c.data == "reset:history")
async def reset_history(cb: types.CallbackQuery):
    uid = cb.from_user.id
    user_state[uid]["history"] = []
    safe_log(uid, "history_reset")
    await cb.message.edit_text("🧹 История очищена.", reply_markup=kb_main())
    await cb.answer()

# -------------------------------------------------------------------
# MESSAGE HANDLER
# -------------------------------------------------------------------

@dp.message()
async def on_message(message: types.Message):
    uid = message.from_user.id
    text = message.text or ""
    safe_log(uid, "message_received", {"chars": len(text)})

    if not check_rate_limit(uid):
        await message.answer("Слишком много запросов. Подождите минуту.")
        return

    st = user_state[uid]
    mode = st["mode"]
    provider = st["provider"]
    model = st["model"]

    # ----- IMAGE MODE -----
    if mode == "image":
        try:
            img_url = await generate_image(
                provider=provider,
                prompt=text,
                openai_key=OPENAI_API_KEY,
                grok_key=GROK_API_KEY
            )
            await message.answer_photo(img_url, caption="Готово!")

            st["history"].append({"role": "user", "content": "[image_prompt]"})
            st["history"].append({"role": "assistant", "content": "[image_generated]"})
            trim_user_history(uid)

        except Exception as e:
            safe_log(uid, "image_error", {"err": str(e)})
            await message.answer("Ошибка при генерации изображения.")
        return

    # ----- TEXT MODE (STREAMING) -----
    st["history"].append({"role": "user", "content": text})
    trim_user_history(uid)

    status = await message.answer("Генерирую...")

    try:
        full = ""
        last_edit = time.time()

        stream = generate_text_stream(
            provider=provider,
            model=model,
            history=st["history"],
            user_input=text,
            openai_key=OPENAI_API_KEY,
            grok_key=GROK_API_KEY,
            gemini_key=GEMINI_API_KEY,
            max_history_tokens=st["max_history_tokens"]
        )

        async for chunk in stream:
            full += chunk
            if time.time() - last_edit >= 0.35:
                try:
                    await status.edit_text(full)
                except TelegramAPIError:
                    pass
                last_edit = time.time()

        try:
            await status.edit_text(full)
        except TelegramAPIError:
            pass

        st["history"].append({"role": "assistant", "content": full})
        trim_user_history(uid)

    except Exception as e:
        safe_log(uid, "stream_error", {"err": str(e)})

        try:
            fallback = await generate_text(
                provider=provider,
                model=model,
                history=st["history"],
                user_input=text,
                openai_key=OPENAI_API_KEY,
                grok_key=GROK_API_KEY,
                gemini_key=GEMINI_API_KEY,
                max_history_tokens=st["max_history_tokens"]
            )
            await status.edit_text(fallback)
        except:
            await status.edit_text("Ошибка при обращении к модели.")

# -------------------------------------------------------------------
# ЗАПУСК
# -------------------------------------------------------------------

async def main():
    logger.info("Bot started")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
