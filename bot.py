import asyncio
import logging
import os
import time
from collections import defaultdict, deque

from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, BufferedInputFile

from llm_core import (
    generate_text_stream,
    generate_image,
    trim_history_by_tokens
)

# ----------------------------------------------------------
# LOGGING
# ----------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neurozone")

# ----------------------------------------------------------
# ENV
# ----------------------------------------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN not set")

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode="HTML")
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
    "mode": None,       # text | image
    "provider": None,   # openai | grok | gemini
    "model": None,
    "history": []
})

# ----------------------------------------------------------
# MODELS (УТВЕРЖДЁННЫЙ ТОБОЙ СПИСОК)
# ----------------------------------------------------------

openai_models = {
    "GPT-5": {"id": "gpt-5", "desc": "Лучшая модель OpenAI"},
    "GPT-5 mini": {"id": "gpt-5-mini", "desc": "Быстрее и дешевле"},
    "GPT-5 nano": {"id": "gpt-5-nano", "desc": "Максимально быстрая"},
    "GPT-4.1": {"id": "gpt-4.1", "desc": "Стабильная и умная"},
}

grok_models = {
    "Grok code fast": {"id": "grok-code-fast-1", "desc": "Быстрый код"},
    "Grok 4 fast reasoning": {"id": "grok-4-fast-reasoning", "desc": "Reasoning модель"},
    "Grok 4 fast non-reasoning": {"id": "grok-4-fast-non-reasoning", "desc": "Без reasoning"},
}

gemini_models = {
    "Gemini 2.5 Flash": {"id": "gemini-2.5-flash", "desc": "Быстрая модель"},
    "Gemini 2.5 Flash Lite": {"id": "gemini-2.5-flash-lite", "desc": "Лёгкая версия"},
}

openai_image_models = {
    "DALL·E 3": {"id": "dall-e-3", "desc": "Генерация изображений OpenAI"},
}

grok_image_models = {
    "Grok Image": {"id": "grok-image-1", "desc": "Генерация изображений Grok"},
}

gemini_image_models = {
    "Gemini Image": {"id": "gemini-image-1", "desc": "Временно недоступно"},
}

# ----------------------------------------------------------
# KEYBOARDS
# ----------------------------------------------------------

def kb_main() -> InlineKeyboardMarkup:
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

def kb_back_to_providers() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⬅ Назад", callback_data="back:providers")],
        [InlineKeyboardButton(text="⬅ Главное меню", callback_data="back:main")],
    ])

async def repost_menu(chat_id: int):
    await bot.send_message(chat_id, "Меню:", reply_markup=kb_main())

# ----------------------------------------------------------
# START
# ----------------------------------------------------------

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_state[message.from_user.id].update({
        "mode": None,
        "provider": None,
        "model": None,
        "history": []
    })
    await message.answer("Привет! Выберите режим:", reply_markup=kb_main())

# ----------------------------------------------------------
# CALLBACKS
# ----------------------------------------------------------

@dp.callback_query(lambda c: c.data.startswith("mode:"))
async def cb_mode(cb: types.CallbackQuery):
    user_state[cb.from_user.id]["mode"] = cb.data.split(":")[1]
    await cb.message.edit_text("Выберите провайдера:", reply_markup=kb_main())
    await cb.answer()

@dp.callback_query(lambda c: c.data.startswith("provider:"))
async def cb_provider(cb: types.CallbackQuery):
    user_state[cb.from_user.id]["provider"] = cb.data.split(":")[1]
    await show_models(cb)
    await cb.answer()

async def show_models(cb: types.CallbackQuery):
    uid = cb.from_user.id
    st = user_state[uid]

    if st["mode"] == "image":
        models = {
            "openai": openai_image_models,
            "grok": grok_image_models,
            "gemini": gemini_image_models,
        }[st["provider"]]
    else:
        models = {
            "openai": openai_models,
            "grok": grok_models,
            "gemini": gemini_models,
        }[st["provider"]]

    text = "Выберите модель:\n\n"
    keyboard = []

    for name, meta in models.items():
        text += f"<b>{name}</b> — {meta['desc']}\n"
        keyboard.append([
            InlineKeyboardButton(
                text=name,
                callback_data=f"model:{meta['id']}"
            )
        ])

    keyboard.append([
        InlineKeyboardButton(text="⬅ Назад", callback_data="back:providers")
    ])

    await cb.message.edit_text(
        text,
        reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard)
    )

@dp.callback_query(lambda c: c.data.startswith("model:"))
async def cb_model(cb: types.CallbackQuery):
    user_state[cb.from_user.id]["model"] = cb.data.split(":")[1]
    await cb.message.edit_text(
        f"Модель выбрана: <b>{user_state[cb.from_user.id]['model']}</b>\n\nОтправьте запрос.",
        reply_markup=kb_back_to_providers()
    )
    await cb.answer()

@dp.callback_query(lambda c: c.data == "reset")
async def cb_reset(cb: types.CallbackQuery):
    user_state[cb.from_user.id]["history"] = []
    await cb.message.edit_text("История очищена.", reply_markup=kb_main())
    await cb.answer()

@dp.callback_query(lambda c: c.data == "back:main")
async def cb_back_main(cb: types.CallbackQuery):
    await cb.message.edit_text("Главное меню:", reply_markup=kb_main())
    await cb.answer()

@dp.callback_query(lambda c: c.data == "back:providers")
async def cb_back_providers(cb: types.CallbackQuery):
    await cb.message.edit_text("Выберите провайдера:", reply_markup=kb_main())
    await cb.answer()

# ----------------------------------------------------------
# MESSAGE HANDLER
# ----------------------------------------------------------

@dp.message()
async def handle_message(message: types.Message):
    uid = message.from_user.id

    if not check_rate(uid):
        await message.answer("Слишком много запросов, попробуйте позже.")
        return

    st = user_state[uid]
    if not all([st["mode"], st["provider"], st["model"]]):
        await message.answer("Сначала выберите режим, провайдера и модель.")
        return

    # ---------------- IMAGE ----------------
    if st["mode"] == "image":
        await message.answer("Генерирую изображение…")
        try:
            img = await generate_image(
                st["provider"],
                st["model"],
                message.text,
                OPENAI_API_KEY,
                GROK_API_KEY,
                GEMINI_API_KEY
            )

            if isinstance(img, bytes):
                await message.answer_photo(
                    BufferedInputFile(img, filename="image.png")
                )
            else:
                await message.answer_photo(img)

        except Exception as e:
            await message.answer(f"Ошибка: {e}")

        await repost_menu(message.chat.id)
        return

    # ---------------- TEXT ----------------
    st["history"].append({"role": "user", "content": message.text})
    st["history"] = trim_history_by_tokens(st["history"], 3000)

    status = await message.answer("Генерирую…")
    answer = ""

    try:
        async for chunk in generate_text_stream(
            st["provider"],
            st["model"],
            st["history"],
            message.text,
            OPENAI_API_KEY,
            GROK_API_KEY,
            GEMINI_API_KEY,
            3000
        ):
            answer += chunk
            await status.edit_text(answer)

        st["history"].append({"role": "assistant", "content": answer})

    except Exception as e:
        await status.edit_text(f"Ошибка: {e}")

    await repost_menu(message.chat.id)

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
