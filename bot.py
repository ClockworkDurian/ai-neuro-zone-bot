# bot.py
# ----------------------------------------------------------
# NeuroZone Telegram Bot
# ReplyKeyboard version (sticky bottom menu)
# ----------------------------------------------------------

import asyncio
import logging
import os
from collections import defaultdict, deque

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    ReplyKeyboardMarkup,
    KeyboardButton,
)

from llm_core import (
    generate_text,
    generate_image,
)

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neurozone")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

# ----------------------------------------------------------
# USER STATE (simple, predictable)
# ----------------------------------------------------------

user_state = defaultdict(lambda: {
    "mode": None,        # text | image
    "provider": None,    # openai | grok | gemini
    "model": None,
})

# ----------------------------------------------------------
# MODELS (НЕ МЕНЯЛ, БЕРЁМ ИЗ ТВОЕГО ЭТАЛОНА)
# ----------------------------------------------------------

TEXT_MODELS = {
    "openai": [
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o-mini",
    ],
    "grok": [
        "grok-beta",
        "grok-vision-beta",
    ],
    "gemini": [
        "gemini-2.5-flash-lite",
    ],
}

IMAGE_MODELS = {
    "openai": ["gpt-image-1"],
    "grok": ["grok-image"],
}

# ----------------------------------------------------------
# KEYBOARDS (ReplyKeyboard – STICKY)
# ----------------------------------------------------------

def kb_main():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📝 Текст"), KeyboardButton(text="🖼 Картинки")],
            [KeyboardButton(text="OpenAI"), KeyboardButton(text="Grok"), KeyboardButton(text="Gemini")],
            [KeyboardButton(text="🔄 Сброс истории")],
        ],
        resize_keyboard=True,
        persistent=True,  # FIX: меню залипает
    )

def kb_models(models):
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=m)] for m in models],
        resize_keyboard=True,
        persistent=True,
    )

# ----------------------------------------------------------
# COMMANDS
# ----------------------------------------------------------

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_state[message.from_user.id] = {
        "mode": None,
        "provider": None,
        "model": None,
    }
    await message.answer(
        "Привет! Выберите режим, провайдера и модель.",
        reply_markup=kb_main(),
    )

# ----------------------------------------------------------
# MENU HANDLERS
# ----------------------------------------------------------

@dp.message(lambda m: m.text in ["📝 Текст", "🖼 Картинки"])
async def choose_mode(message: types.Message):
    state = user_state[message.from_user.id]
    state["mode"] = "text" if "Текст" in message.text else "image"
    await message.answer("Выберите провайдера:", reply_markup=kb_main())

@dp.message(lambda m: m.text in ["OpenAI", "Grok", "Gemini"])
async def choose_provider(message: types.Message):
    uid = message.from_user.id
    state = user_state[uid]
    provider = message.text.lower()

    state["provider"] = provider

    # FIX: Gemini не падает
    if provider == "gemini":
        await message.answer(
            "Gemini временно недоступна.\nВыберите OpenAI или Grok.",
            reply_markup=kb_main(),
        )
        return

    if state["mode"] == "text":
        models = TEXT_MODELS.get(provider, [])
    else:
        models = IMAGE_MODELS.get(provider, [])

    if not models:
        await message.answer("Нет доступных моделей.", reply_markup=kb_main())
        return

    await message.answer(
        "Выберите модель:",
        reply_markup=kb_models(models),
    )

@dp.message(lambda m: any(m.text in v for v in TEXT_MODELS.values()) or
                      any(m.text in v for v in IMAGE_MODELS.values()))
async def choose_model(message: types.Message):
    uid = message.from_user.id
    state = user_state[uid]
    state["model"] = message.text

    await message.answer(
        f"Модель выбрана: {message.text}\nВведите запрос.",
        reply_markup=kb_main(),
    )

@dp.message(lambda m: m.text == "🔄 Сброс истории")
async def reset_state(message: types.Message):
    user_state[message.from_user.id] = {
        "mode": None,
        "provider": None,
        "model": None,
    }
    await message.answer("История сброшена.", reply_markup=kb_main())

# ----------------------------------------------------------
# MAIN MESSAGE HANDLER
# ----------------------------------------------------------

@dp.message()
async def handle_message(message: types.Message):
    uid = message.from_user.id
    state = user_state[uid]

    # FIX: НЕ сбрасываем состояние после ответа
    if not state["mode"] or not state["provider"] or not state["model"]:
        await message.answer(
            "Сначала выберите режим, провайдера и модель.",
            reply_markup=kb_main(),
        )
        return

    try:
        if state["mode"] == "text":
            result = await generate_text(
                provider=state["provider"],
                model=state["model"],
                prompt=message.text,
            )
            await message.answer(result, reply_markup=kb_main())

        elif state["mode"] == "image":
            img_url = await generate_image(
                provider=state["provider"],
                model=state["model"],
                prompt=message.text,
            )
            await message.answer_photo(img_url, reply_markup=kb_main())

    except Exception as e:
        logger.exception("LLM error")
        await message.answer(
            "Ошибка при обращении к модели.",
            reply_markup=kb_main(),
        )

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
