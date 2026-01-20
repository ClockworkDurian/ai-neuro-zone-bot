# bot.py — совместим с aiogram 3.13.x, без изменения логики и моделей

import asyncio
import logging
import os
from collections import defaultdict, deque

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from llm_core import (
    generate_text,
    generate_text_stream,
    generate_image,
    trim_history_by_tokens,
)

BOT_TOKEN = os.getenv("BOT_TOKEN")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neurozone_bot")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# ------------------------------------------------------------------
# USER STATE
# ------------------------------------------------------------------

user_mode = {}
user_provider = {}
user_model = {}
user_history = defaultdict(lambda: deque())

# ------------------------------------------------------------------
# KEYBOARDS
# ------------------------------------------------------------------

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
        ]
    ])

def kb_models(provider: str):
    models = {
        "openai": [
            ("GPT-4.1", "gpt-4.1"),
            ("GPT-4.1-mini", "gpt-4.1-mini"),
        ],
        "grok": [
            ("Grok-2-Vision", "grok-2-vision"),
            ("Grok-2", "grok-2"),
        ],
        "gemini": [
            ("Gemini 2.5 Flash", "gemini-2.5-flash"),
            ("Gemini 2.5 Flash Lite", "gemini-2.5-flash-lite"),
        ]
    }

    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=name, callback_data=f"model:{mid}")]
        for name, mid in models.get(provider, [])
    ])

# ------------------------------------------------------------------
# HANDLERS
# ------------------------------------------------------------------

@dp.message(Command("start"))
async def start(message: types.Message):
    user_history[message.from_user.id].clear()
    await message.answer("Привет! Выберите режим:", reply_markup=kb_main())

@dp.callback_query()
async def on_callback(call: types.CallbackQuery):
    uid = call.from_user.id
    data = call.data

    if data == "reset":
        user_history[uid].clear()
        await call.message.answer("История очищена.", reply_markup=kb_main())
        return

    if data.startswith("mode:"):
        user_mode[uid] = data.split(":")[1]
        await call.message.answer("Выберите провайдера:", reply_markup=kb_main())
        return

    if data.startswith("provider:"):
        prov = data.split(":")[1]
        user_provider[uid] = prov
        await call.message.answer("Выберите модель:", reply_markup=kb_models(prov))
        return

    if data.startswith("model:"):
        model = data.split(":")[1]
        user_model[uid] = model
        await call.message.answer(f"Вы выбрали модель:\n{model}\n\nТеперь отправьте запрос.")
        return

@dp.message()
async def handle_message(message: types.Message):
    uid = message.from_user.id

    if uid not in user_mode or uid not in user_provider or uid not in user_model:
        await message.answer("Сначала выберите режим, провайдера и модель.", reply_markup=kb_main())
        return

    text = message.text
    history = user_history[uid]
    history.append({"role": "user", "content": text})
    history = trim_history_by_tokens(history, max_tokens=8000)

    try:
        if user_mode[uid] == "text":
            reply = await generate_text(
                provider=user_provider[uid],
                model=user_model[uid],
                messages=list(history),
            )
            history.append({"role": "assistant", "content": reply})
            await message.answer(reply)

        else:
            img_url = await generate_image(
                provider=user_provider[uid],
                model=user_model[uid],
                prompt=text,
            )
            await message.answer_photo(img_url)

    except Exception as e:
        logger.exception("Model error")
        await message.answer("Ошибка при обращении к модели.")

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
