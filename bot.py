# bot.py — полностью совместим с исправленным llm_core.py

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
    trim_history_by_tokens
)

# ------------------------------------------------
# LOGGING
# ------------------------------------------------

logger = logging.getLogger("neurozone_bot")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

def safe_log(uid, event, extra=None):
    d = {"user_id": uid, "event": event}
    if extra:
        d.update(extra)
    logger.info(d)


# ------------------------------------------------
# ENV VARS
# ------------------------------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not BOT_TOKEN:
    raise SystemExit("BOT_TOKEN is missing")

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp = Dispatcher()

# ------------------------------------------------
# RATE LIMITING
# ------------------------------------------------

RATE_LIMIT = 30  # per minute
user_requests = defaultdict(lambda: deque())

def check_rate(uid):
    now = time.time()
    dq = user_requests[uid]
    while dq and dq[0] < now - 60:
        dq.popleft()
    if len(dq) >= RATE_LIMIT:
        return False
    dq.append(now)
    return True


# ------------------------------------------------
# USER STATE
# ------------------------------------------------

user_state = defaultdict(lambda: {
    "mode": None,       # text or image
    "provider": None,   # openai / grok / gemini
    "model": None,
    "history": [],
})

provider_unavailable = {}  # provider -> timestamp available again


# ------------------------------------------------
# MODELS
# ------------------------------------------------

openai_models = {
    "GPT-5": {"id": "gpt-5", "desc": "Лучшая модель OpenAI."},
    "GPT-5 mini": {"id": "gpt-5-mini", "desc": "Быстрее и дешевле."},
    "GPT-5 nano": {"id": "gpt-5-nano", "desc": "Самая быстрая."},
    "GPT-4.1": {"id": "gpt-4.1", "desc": "Надёжная и умная."}
}

grok_models = {
    "Grok-code-fast-1": {"id": "grok-code-fast-1", "desc": "Быстрая модель для кода."},
    "Grok-4-fast-reasoning": {"id": "grok-4-fast-reasoning", "desc": "Экономичная reason-модель."},
    "Grok-4-fast-non-reasoning": {"id": "grok-4-fast-non-reasoning", "desc": "Экономичная no-reason модель."}
}

gemini_models = {
    "Gemini 2.5 Flash": {"id": "gemini-2.5-flash", "desc": "Flash модель."},
    "Gemini 2.5 Flash-Lite": {"id": "gemini-2.5-flash-lite", "desc": "Lite flash модель."}
}

# IMAGE MODELS
openai_image_models = {
    "DALL·E 3": {"id": "dall-e-3", "desc": "Генерация изображений DALL·E 3"}
}

grok_image_models = {
    "Grok Vision": {"id": "grok-image-1", "desc": "Генератор изображений Grok"}
}

gemini_image_models = {
    "Gemini Image": {"id": "gemini-image-1", "desc": "Генерация изображений в Gemini (отключена)"}
}


# ------------------------------------------------
# KEYBOARDS
# ------------------------------------------------

def kb_main():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton("Текст", callback_data="mode:text"),
         InlineKeyboardButton("Картинки", callback_data="mode:image")],
        [InlineKeyboardButton("OpenAI", callback_data="provider:openai"),
         InlineKeyboardButton("Grok", callback_data="provider:grok"),
         InlineKeyboardButton("Gemini", callback_data="provider:gemini")],
        [InlineKeyboardButton("Сброс истории", callback_data="reset")]
    ])

def model_menu():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton("⬅ Назад", callback_data="back:providers")],
        [InlineKeyboardButton("⬅ Главное меню", callback_data="back:main")]
    ])

async def repost_menu(chat_id):
    await bot.send_message(chat_id, "Меню:", reply_markup=kb_main())


# ------------------------------------------------
# MODEL LISTS
# ------------------------------------------------

async def show_models(cb, provider):
    uid = cb.from_user.id
    mode = user_state[uid]["mode"]

    if mode == "image":
        if provider == "openai":
            models = openai_image_models
            header = "🖼 OpenAI — изображения"
        elif provider == "grok":
            models = grok_image_models
            header = "🖼 Grok — изображения"
        else:
            models = gemini_image_models
            header = "🖼 Gemini — изображения"
    else:
        if provider == "openai":
            models = openai_models
            header = "🔵 OpenAI — текст"
        elif provider == "grok":
            models = grok_models
            header = "🧠 Grok — текст"
        else:
            models = gemini_models
            header = "⚡ Gemini — текст"

    text = f"{header}\n\nВыберите модель:"
    kb = []

    for name, meta in models.items():
        text += f"\n<b>{name}</b> — <i>{meta['desc']}</i>"
        kb.append([InlineKeyboardButton(name, callback_data=f"model:{meta['id']}")])

    kb.append([InlineKeyboardButton("⬅ Назад", callback_data="back:providers")])

    await cb.message.edit_text(text, reply_markup=InlineKeyboardMarkup(inline_keyboard=kb))
    await cb.answer()


# ------------------------------------------------
# CALLBACKS
# ------------------------------------------------

@dp.message(Command("start"))
async def start(message: types.Message):
    await message.answer("Привет! Выберите режим:", reply_markup=kb_main())

@dp.callback_query(lambda c: c.data.startswith("mode:"))
async def set_mode(cb):
    uid = cb.from_user.id
    mode = cb.data.split(":", 1)[1]
    user_state[uid]["mode"] = mode
    await cb.message.edit_text(f"Режим: <b>{mode}</b>. Теперь выберите провайдера:", reply_markup=kb_main())
    await cb.answer()

@dp.callback_query(lambda c: c.data.startswith("provider:"))
async def set_provider(cb):
    uid = cb.from_user.id
    provider = cb.data.split(":", 1)[1]
    user_state[uid]["provider"] = provider
    await show_models(cb, provider)

@dp.callback_query(lambda c: c.data.startswith("model:"))
async def set_model(cb):
    uid = cb.from_user.id
    model = cb.data.split(":", 1)[1]
    user_state[uid]["model"] = model
    await cb.message.edit_text(f"Модель выбрана: <b>{model}</b>\nТеперь отправьте запрос.", reply_markup=model_menu())
    await cb.answer()

@dp.callback_query(lambda c: c.data == "back:providers")
async def back_providers(cb):
    await cb.message.edit_text("Выберите провайдера:", reply_markup=kb_main())
    await cb.answer()

@dp.callback_query(lambda c: c.data == "back:main")
async def back_main(cb):
    await cb.message.edit_text("Главное меню:", reply_markup=kb_main())
    await cb.answer()

@dp.callback_query(lambda c: c.data == "reset")
async def reset_history(cb):
    uid = cb.from_user.id
    user_state[uid]["history"] = []
    await cb.message.edit_text("История очищена.", reply_markup=kb_main())
    await cb.answer()


# ------------------------------------------------
# MESSAGE HANDLER
# ------------------------------------------------

@dp.message()
async def handle_message(message: types.Message):

    uid = message.from_user.id

    if not check_rate(uid):
        await message.answer("Слишком много запросов.")
        return

    st = user_state[uid]
    mode = st["mode"]
    provider = st["provider"]
    model = st["model"]

    if not mode:
        await message.answer("Сначала выберите режим.")
        return
    if not provider:
        await message.answer("Сначала выберите провайдера.")
        return
    if not model:
        await message.answer("Сначала выберите модель.")
        return

    # ------------------------------------
    # IMAGE
    # ------------------------------------
    if mode == "image":
        await message.answer("Генерирую изображение...")

        try:
            url = await generate_image(provider, model, message.text,
                                       OPENAI_API_KEY, GROK_API_KEY, GEMINI_API_KEY)
            await message.answer_photo(url)
        except Exception as e:
            await message.answer(f"Ошибка: {e}")

        await repost_menu(message.chat.id)
        return

    # ------------------------------------
    # TEXT (stream)
    # ------------------------------------
    st["history"].append({"role": "user", "content": message.text})
    st["history"] = trim_history_by_tokens(st["history"], 3000)

    status = await message.answer("Генерирую...")

    try:

        full = ""
        async for chunk in generate_text_stream(
            provider, model, st["history"], message.text,
            OPENAI_API_KEY, GROK_API_KEY, GEMINI_API_KEY,
            3000
        ):
            full += chunk
            try:
                await status.edit_text(full)
            except:
                pass

        st["history"].append({"role": "assistant", "content": full})

    except Exception as e:
        err = str(e)
        if "GEMINI" in err and "временно" in err:
            await status.edit_text(err)
        else:
            await status.edit_text(f"Ошибка: {e}")

    await repost_menu(message.chat.id)


# ------------------------------------------------
# RUN
# ------------------------------------------------

async def main():
    print("Bot started")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
