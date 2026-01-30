import asyncio
import logging
import os
import time
from collections import defaultdict, deque

from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

from llm_core import generate_text, generate_image

# ----------------------------------------------------------
# LOGGING
# ----------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neurozone_bot")

# ----------------------------------------------------------
# ENV
# ----------------------------------------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN")
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
# MODELS — БЕЗ ИЗМЕНЕНИЙ
# ----------------------------------------------------------

openai_models = {
    "GPT-5": {"id": "gpt-5"},
    "GPT-5 mini": {"id": "gpt-5-mini"},
    "GPT-5 nano": {"id": "gpt-5-nano"},
    "GPT-4.1": {"id": "gpt-4.1"},
}

grok_models = {
    "Grok code fast": {"id": "grok-code-fast-1"},
    "Grok 4 fast reasoning": {"id": "grok-4-fast-reasoning"},
    "Grok 4 fast non-reasoning": {"id": "grok-4-fast-non-reasoning"},
}

gemini_models = {
    "Gemini 2.5 Flash": {"id": "gemini-2.5-flash"},
    "Gemini 2.5 Flash Lite": {"id": "gemini-2.5-flash-lite"},
}

openai_image_models = {
    "DALL·E 3": {"id": "dall-e-3"},
}

grok_image_models = {
    "Grok Image": {"id": "grok-image-1"},
}

# ----------------------------------------------------------
# KEYBOARDS
# ----------------------------------------------------------

def main_menu():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Текст"), KeyboardButton(text="Картинки")],
            [KeyboardButton(text="OpenAI"), KeyboardButton(text="Grok"), KeyboardButton(text="Gemini")],
            [KeyboardButton(text="Сброс истории")],
        ],
        resize_keyboard=True,
        persistent=True,
    )

def models_menu(models: dict):
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=name)] for name in models.keys()],
        resize_keyboard=True,
        persistent=True,
    )

# ----------------------------------------------------------
# HANDLERS
# ----------------------------------------------------------

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    uid = message.from_user.id
    user_state[uid] = {
        "mode": None,
        "provider": None,
        "model": None,
        "history": [],
    }
    await message.answer("Выберите режим:", reply_markup=main_menu())

@dp.message()
async def handle_message(message: types.Message):
    uid = message.from_user.id
    text = message.text.strip()
    st = user_state[uid]

    # --- MENU ACTIONS ---

    if text == "Текст":
        st["mode"] = "text"
        await message.answer("Выберите провайдера:", reply_markup=main_menu())
        return

    if text == "Картинки":
        st["mode"] = "image"
        await message.answer("Выберите провайдера:", reply_markup=main_menu())
        return

    if text in ("OpenAI", "Grok", "Gemini"):
        st["provider"] = text.lower()

        if st["provider"] == "gemini":
            await message.answer(
                "Gemini временно недоступна.\nМодель будет добавлена позже.",
                reply_markup=main_menu()
            )
            return

        if st["mode"] == "text":
            models = openai_models if st["provider"] == "openai" else grok_models
        else:
            models = openai_image_models if st["provider"] == "openai" else grok_image_models

        await message.answer(
            "Выберите модель:",
            reply_markup=models_menu(models)
        )
        return

    # --- MODEL SELECTION ---

    all_models = {
        **openai_models,
        **grok_models,
        **openai_image_models,
        **grok_image_models,
        **gemini_models,
    }

    if text in all_models:
        st["model"] = all_models[text]["id"]
        await message.answer(
            f"Модель выбрана: <b>{st['model']}</b>\nВведите запрос.",
            reply_markup=main_menu()
        )
        return

    # --- RESET ---

    if text == "Сброс истории":
        st["history"] = []
        await message.answer("История очищена.", reply_markup=main_menu())
        return

    # --- QUERY ---

    if not st["mode"] or not st["provider"] or not st["model"]:
        await message.answer(
            "Сначала выберите режим, провайдера и модель.",
            reply_markup=main_menu()
        )
        return

    if not check_rate(uid):
        await message.answer("Слишком много запросов, попробуйте позже.", reply_markup=main_menu())
        return

    try:
        if st["mode"] == "text":
            reply = await generate_text(
                provider=st["provider"],
                model=st["model"],
                prompt=text,
                history=st["history"],
            )
            st["history"].append({"role": "user", "content": text})
            st["history"].append({"role": "assistant", "content": reply})
            await message.answer(reply, reply_markup=main_menu())
        else:
            img = await generate_image(
                provider=st["provider"],
                model=st["model"],
                prompt=text,
            )
            await message.answer_photo(img, reply_markup=main_menu())

    except Exception:
        logger.exception("LLM error")
        await message.answer("Ошибка при обращении к модели.", reply_markup=main_menu())

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
