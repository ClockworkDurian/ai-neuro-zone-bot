import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from llm_core import (
    generate_text,
    generate_text_stream,
    generate_image,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neurozone")

# ----------------------------------------------------------
# БОТ
# ----------------------------------------------------------

bot = Bot(token=os.getenv("BOT_TOKEN"))
dp = Dispatcher()

# ----------------------------------------------------------
# СОСТОЯНИЯ ПОЛЬЗОВАТЕЛЯ
# ----------------------------------------------------------

user_state = {}  # { user_id: { mode, provider, model, history } }


def get_state(user_id):
    if user_id not in user_state:
        user_state[user_id] = {
            "mode": None,
            "provider": None,
            "model": None,
            "history": [],
        }
    return user_state[user_id]


# ----------------------------------------------------------
# КНОПКИ
# ----------------------------------------------------------

def kb_main():
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Текст", callback_data="mode:text"),
                InlineKeyboardButton(text="Картинки", callback_data="mode:image")
            ],
            [
                InlineKeyboardButton(text="OpenAI", callback_data="provider:openai"),
                InlineKeyboardButton(text="Grok", callback_data="provider:grok"),
                InlineKeyboardButton(text="Gemini", callback_data="provider:gemini"),
            ],
            [InlineKeyboardButton(text="Сброс истории", callback_data="reset")],
        ]
    )


def kb_models(provider):
    if provider == "openai":
        buttons = [
            ("gpt-4o-mini", "model:gpt-4o-mini"),
            ("gpt-4.1", "model:gpt-4.1"),
            ("gpt-5", "model:gpt-5"),
            ("dall-e-3", "model:dall-e-3-image"),
        ]

    elif provider == "grok":
        buttons = [
            ("grok-2-mini", "model:grok-2-mini"),
            ("grok-2", "model:grok-2"),
            ("grok-image-1", "model:grok-image-1"),
        ]

    elif provider == "gemini":
        buttons = [
            ("gemini-2.0-flash", "model:gemini-2.0-flash"),
            ("gemini-2.0-pro", "model:gemini-2.0-pro"),
        ]

    else:
        buttons = []

    rows = []
    for text, data in buttons:
        rows.append([InlineKeyboardButton(text=text, callback_data=data)])

    return InlineKeyboardMarkup(inline_keyboard=rows)


# ----------------------------------------------------------
# START
# ----------------------------------------------------------

@dp.message(Command("start"))
async def start(message: types.Message):
    s = get_state(message.from_user.id)
    s["mode"] = None
    s["provider"] = None
    s["model"] = None
    s["history"] = []

    await message.answer("Привет! Выберите режим:", reply_markup=kb_main())


# ----------------------------------------------------------
# CALLBACK
# ----------------------------------------------------------

@dp.callback_query()
async def callback_handler(q: types.CallbackQuery):
    user_id = q.from_user.id
    s = get_state(user_id)

    # режим: текст / картинка
    if q.data.startswith("mode:"):
        s["mode"] = q.data.split(":")[1]
        s["provider"] = None
        s["model"] = None

        await q.message.answer(
            f"Режим выбран: {s['mode']}. Теперь выберите провайдера:",
            reply_markup=kb_main()
        )
        await q.answer()
        return

    # выбор провайдера
    if q.data.startswith("provider:"):
        s["provider"] = q.data.split(":")[1]
        s["model"] = None

        await q.message.answer(
            f"Провайдер выбран: {s['provider']}. Теперь выберите модель:",
            reply_markup=kb_models(s["provider"])
        )
        await q.answer()
        return

    # выбор модели
    if q.data.startswith("model:"):
        s["model"] = q.data.split(":")[1]

        await q.message.answer(
            f"Вы выбрали модель:\n<b>{s['model']}</b>\n\nТеперь отправьте ваш запрос.",
            parse_mode="HTML",
            reply_markup=kb_main()
        )
        await q.answer()
        return

    # сброс
    if q.data == "reset":
        s["history"] = []
        await q.message.answer("История очищена.", reply_markup=kb_main())
        await q.answer()
        return


# ----------------------------------------------------------
# TEXT HANDLER
# ----------------------------------------------------------

@dp.message()
async def text_handler(message: types.Message):
    user_id = message.from_user.id
    s = get_state(user_id)

    # Проверяем полноту настройки
    if not s["mode"]:
        await message.answer("Сначала выберите режим.", reply_markup=kb_main())
        return
    if not s["provider"]:
        await message.answer("Сначала выберите провайдера.", reply_markup=kb_main())
        return
    if not s["model"]:
        await message.answer("Сначала выберите модель.", reply_markup=kb_main())
        return

    # -------------------------------------------------
    # ТЕКСТОВЫЙ РЕЖИМ
    # -------------------------------------------------
    if s["mode"] == "text":
        s["history"].append({"role": "user", "content": message.text})

        try:
            async for chunk in generate_text_stream(
                provider=s["provider"],
                model=s["model"],
                history=s["history"],
                user_input=message.text,
                openai_key=os.getenv("OPENAI_API_KEY"),
                grok_key=os.getenv("GROK_API_KEY"),
                gemini_key=os.getenv("GEMINI_API_KEY"),
            ):
                await message.answer(chunk)

            s["history"].append({"role": "assistant", "content": chunk})

        except Exception as e:
            await message.answer(f"Ошибка: {e}")

        await message.answer("Готов!", reply_markup=kb_main())
        return

    # -------------------------------------------------
    # ИЗОБРАЖЕНИЯ
    # -------------------------------------------------
    if s["mode"] == "image":
        try:
            url = await generate_image(
                provider=s["provider"],
                model=s["model"],
                prompt=message.text,
                openai_key=os.getenv("OPENAI_API_KEY"),
                grok_key=os.getenv("GROK_API_KEY"),
                gemini_key=os.getenv("GEMINI_API_KEY"),
            )

            await message.answer_photo(url)
        except Exception as e:
            await message.answer(f"Ошибка изображения: {e}")

        await message.answer("Готов!", reply_markup=kb_main())


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
