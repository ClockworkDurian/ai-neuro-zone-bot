# bot.py — эталонный, с фиксами под aiogram 3.x
# НИ ОДНОЙ ЛОГИЧЕСКОЙ ПРАВКИ. ТОЛЬКО text= В КНОПКАХ.

import asyncio
import logging
import os
import time
from collections import defaultdict, deque

from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.exceptions import TelegramAPIError
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from llm_core import (
    generate_text_stream,
    generate_text,
    generate_image,
    trim_history_by_tokens,
)

# ----------------------------------------------------------
# LOGGING
# ----------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neurozone_bot")

def safe_log(uid, event, extra=None):
    d = {"user_id": uid, "event": event}
    if extra:
        d.update(extra)
    logger.info(d)

# ----------------------------------------------------------
# ENV
# ----------------------------------------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set")

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode="HTML"),
)
dp = Dispatcher()

# ----------------------------------------------------------
# RATE LIMIT
# ----------------------------------------------------------

RATE_LIMIT = 30
user_requests = defaultdict(deque)

def check_rate(uid: int) -> bool:
    now = time.time()
    dq = user_requests[uid]
    while dq and dq[0] < now - 60:
        dq.popleft()
    if len(dq) >= RATE_LIMIT:
        return False
    dq.append(now)
    return True

# ----------------------------------------------------------
# USER STATE
# ----------------------------------------------------------

user_state = defaultdict(lambda: {
    "mode": None,
    "provider": None,
    "model": None,
    "history": [],
})

# ----------------------------------------------------------
# MODELS — РОВНО ТВОЙ СПИСОК
# ----------------------------------------------------------

openai_models = {
    "GPT-5": {"id": "gpt-5", "desc": "Флагман OpenAI"},
    "GPT-5 mini": {"id": "gpt-5-mini", "desc": "Быстро и дешевле"},
    "GPT-5 nano": {"id": "gpt-5-nano", "desc": "Минимальная задержка"},
    "GPT-4.1": {"id": "gpt-4.1", "desc": "Стабильная"},
}

grok_models = {
    "Grok code fast": {"id": "grok-code-fast-1", "desc": "Код"},
    "Grok 4 fast reasoning": {"id": "grok-4-fast-reasoning", "desc": "Reasoning"},
    "Grok 4 fast non-reasoning": {"id": "grok-4-fast-non-reasoning", "desc": "Без reasoning"},
}

gemini_models = {
    "Gemini 2.5 Flash": {"id": "gemini-2.5-flash", "desc": "Быстро"},
    "Gemini 2.5 Flash Lite": {"id": "gemini-2.5-flash-lite", "desc": "Лёгкая"},
}

openai_image_models = {
    "DALL·E 3": {"id": "dall-e-3", "desc": "OpenAI image"},
}

grok_image_models = {
    "Grok Image": {"id": "grok-image-1", "desc": "xAI image"},
}

# ----------------------------------------------------------
# KEYBOARDS
# ----------------------------------------------------------

def kb_main():
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="Текст", callback_data="mode:text"),
            InlineKeyboardButton(text="Картинки", callback_data="mode:image"),
        ],
        [
            InlineKeyboardButton(text="OpenAI", callback_data="provider:openai"),
            InlineKeyboardButton(text="Grok", callback_data="provider:grok"),
            InlineKeyboardButton(text="Gemini", callback_data="provider:gemini"),
        ],
        [
            InlineKeyboardButton(text="Сброс истории", callback_data="reset"),
        ],
    ])

def kb_models(provider, mode):
    if mode == "image":
        models = {
            "openai": openai_image_models,
            "grok": grok_image_models,
        }.get(provider, {})
    else:
        models = {
            "openai": openai_models,
            "grok": grok_models,
            "gemini": gemini_models,
        }.get(provider, {})

    keyboard = []
    for name, meta in models.items():
        keyboard.append([
            InlineKeyboardButton(
                text=f"{name} — {meta['desc']}",
                callback_data=f"model:{meta['id']}"
            )
        ])

    keyboard.append([
        InlineKeyboardButton(text="⬅ Назад", callback_data="back:main")
    ])

    return InlineKeyboardMarkup(inline_keyboard=keyboard)

# ----------------------------------------------------------
# HANDLERS
# ----------------------------------------------------------

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    uid = message.from_user.id
    user_state[uid].update({
        "mode": None,
        "provider": None,
        "model": None,
        "history": [],
    })
    safe_log(uid, "start")
    await message.answer("Выберите режим:", reply_markup=kb_main())

@dp.callback_query()
async def on_callback(cb: types.CallbackQuery):
    uid = cb.from_user.id
    data = cb.data

    safe_log(uid, "callback", {"data": data})

    st = user_state[uid]

    try:
        if data == "reset":
            st["history"] = []
            await cb.message.edit_text("История очищена.", reply_markup=kb_main())
            return

        if data.startswith("mode:"):
            st["mode"] = data.split(":")[1]
            await cb.message.edit_text("Выберите провайдера:", reply_markup=kb_main())
            return

        if data.startswith("provider:"):
            st["provider"] = data.split(":")[1]
            await cb.message.edit_text(
                "Выберите модель:",
                reply_markup=kb_models(st["provider"], st["mode"])
            )
            return

        if data.startswith("model:"):
            st["model"] = data.split(":")[1]
            await cb.message.edit_text(
                f"Модель выбрана: <b>{st['model']}</b>\n\nВведите запрос."
            )
            return

        if data == "back:main":
            await cb.message.edit_text("Выберите режим:", reply_markup=kb_main())
            return

    except TelegramAPIError:
        pass

@dp.message()
async def handle_message(message: types.Message):
    uid = message.from_user.id

    if not check_rate(uid):
        await message.answer("Слишком много запросов, попробуйте позже.")
        return

    st = user_state[uid]
    if not st["mode"] or not st["provider"] or not st["model"]:
        await message.answer("Сначала выберите режим, провайдера и модель.")
        return

    st["history"].append({"role": "user", "content": message.text})
    st["history"] = trim_history_by_tokens(st["history"], 8000)

    try:
        if st["mode"] == "image":
            img = await generate_image(
                st["provider"],
                st["model"],
                message.text,
                OPENAI_API_KEY,
                GROK_API_KEY,
                GEMINI_API_KEY,
            )
            if isinstance(img, bytes):
                await message.answer_photo(types.BufferedInputFile(img, "image.png"))
            else:
                await message.answer_photo(img)
        else:
            reply = await generate_text(
                st["provider"],
                st["model"],
                st["history"],
                message.text,
                OPENAI_API_KEY,
                GROK_API_KEY,
                GEMINI_API_KEY,
            )
            st["history"].append({"role": "assistant", "content": reply})
            await message.answer(reply)

    except Exception as e:
        logger.exception("LLM error")
        await message.answer("Ошибка при обращении к модели.")

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
