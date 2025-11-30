# bot.py — финальная версия, совместимая с aiogram 3.13.1

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

# Импорт из нашего модуля логики
from llm_core import (
    generate_text_stream,
    generate_text,
    generate_image,
    trim_history_by_tokens
)

# -------------------------------------------------------------------
# ЛОГИРОВАНИЕ (без текста пользователей)
# -------------------------------------------------------------------
logger = logging.getLogger("neurozone_bot")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)


def safe_log(user_id, event, extra=None):
    """Логирование без текста пользователя."""
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
    raise SystemExit("❌ BOT_TOKEN отсутствует в переменных окружения")


# -------------------------------------------------------------------
# ИНИЦИАЛИЗАЦИЯ aiogram 3.x
# -------------------------------------------------------------------
bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode="HTML")
)
dp = Dispatcher()


# -------------------------------------------------------------------
# RATE LIMIT
# -------------------------------------------------------------------
RATE_LIMIT_PER_MINUTE = 30
user_requests = defaultdict(lambda: deque())


def check_rate_limit(user_id: int) -> bool:
    """Ограничение количества запросов в минуту."""
    now = time.time()
    dq = user_requests[user_id]

    # удаляем запросы старше 60 сек
    while dq and dq[0] < now - 60:
        dq.popleft()

    if len(dq) >= RATE_LIMIT_PER_MINUTE:
        return False

    dq.append(now)
    return True


# -------------------------------------------------------------------
# СОСТОЯНИЕ ПОЛЬЗОВАТЕЛЕЙ
# -------------------------------------------------------------------
MAX_HISTORY_TOKENS_DEFAULT = 3000

user_state = defaultdict(lambda: {
    "mode": "text",                # "text" или "image"
    "provider": "openai",          # openai / grok / gemini
    "model": None,                 # выбранная модель
    "history": [],                 # список словарей с контекстом
    "max_history_tokens": MAX_HISTORY_TOKENS_DEFAULT
})


def trim_user_history(uid: int):
    st = user_state[uid]
    st["history"] = trim_history_by_tokens(
        st["history"],
        st["max_history_tokens"]
    )


# -------------------------------------------------------------------
# КЛАВИАТУРЫ — полностью переписаны под aiogram 3.x
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
            InlineKeyboardButton(text="Gemini", callback_data="provider:gemini"),
        ],
        [
            InlineKeyboardButton(text="Сброс истории", callback_data="reset:history")
        ]
    ])


def kb_models(provider: str) -> InlineKeyboardMarkup:
    if provider == "openai":
        buttons = [
            InlineKeyboardButton(text="gpt-4o-mini", callback_data="model:gpt-4o-mini"),
            InlineKeyboardButton(text="gpt-4o", callback_data="model:gpt-4o"),
        ]
    elif provider == "grok":
        buttons = [
            InlineKeyboardButton(text="grok-2-mini", callback_data="model:grok-2-mini"),
            InlineKeyboardButton(text="grok-2", callback_data="model:grok-2"),
        ]
    else:  # gemini
        buttons = [
            InlineKeyboardButton(text="gemini-1.5-flash", callback_data="model:gemini-1.5-flash")
        ]

    return InlineKeyboardMarkup(inline_keyboard=[buttons])


# -------------------------------------------------------------------
# HANDLERS
# -------------------------------------------------------------------

@dp.message(Command("start", "help"))
async def cmd_start(message: types.Message):
    uid = message.from_user.id
    safe_log(uid, "start")
    await message.answer(
        "Привет! Выбери режим работы и провайдера:",
        reply_markup=kb_main()
    )


@dp.callback_query()
async def cb_handler(cb: types.CallbackQuery):
    uid = cb.from_user.id
    data = cb.data or ""
    safe_log(uid, "callback", {"data": data})

    # режим
    if data.startswith("mode:"):
        mode = data.split(":", 1)[1]
        user_state[uid]["mode"] = mode
        await cb.message.edit_text(
            f"Режим установлен: {mode}\nТеперь выбери провайдера:",
            reply_markup=kb_main()
        )
        await cb.answer()
        return

    # провайдер
    if data.startswith("provider:"):
        provider = data.split(":", 1)[1]
        user_state[uid]["provider"] = provider
        await cb.message.edit_text(
            f"Провайдер: {provider}\nВыберите модель:",
            reply_markup=kb_models(provider)
        )
        await cb.answer()
        return

    # модель
    if data.startswith("model:"):
        model = data.split(":", 1)[1]
        user_state[uid]["model"] = model
        await cb.message.edit_text(
            f"Модель установлена: {model}\nТеперь отправь запрос.",
            reply_markup=kb_main()
        )
        await cb.answer()
        return

    # сброс истории
    if data == "reset:history":
        user_state[uid]["history"] = []
        safe_log(uid, "history_reset")
        await cb.message.edit_text(
            "История очищена.",
            reply_markup=kb_main()
        )
        await cb.answer()
        return

    await cb.answer()


# -------------------------------------------------------------------
# MESSAGE HANDLER (основной)
# -------------------------------------------------------------------
@dp.message()
async def on_message(message: types.Message):
    uid = message.from_user.id
    text = message.text or ""

    safe_log(uid, "message_received", {"length": len(text)})

    # RATE LIMIT
    if not check_rate_limit(uid):
        await message.answer("Слишком много запросов, подожди минуту.")
        safe_log(uid, "rate_limited")
        return

    st = user_state[uid]
    mode = st["mode"]
    provider = st["provider"]
    model = st["model"]

    # ---------------------------------------------------------------
    # IMAGE MODE
    # ---------------------------------------------------------------
    if mode == "image":
        safe_log(uid, "image_request", {"provider": provider})

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
            await message.answer("Ошибка генерации изображения.")
        return

    # ---------------------------------------------------------------
    # TEXT MODE + STREAMING
    # ---------------------------------------------------------------
    st["history"].append({"role": "user", "content": text})
    trim_user_history(uid)

    status = await message.answer("Генерирую ответ...")

    try:
        # streaming
        full = ""
        last_edit = time.time()

        stream = generate_text_stream(
            provider=provider,
            model=model or ("gpt-4o-mini" if provider == "openai" else "grok-2-mini"),
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

        # финальное обновление
        try:
            await status.edit_text(full)
        except TelegramAPIError:
            pass

        st["history"].append({"role": "assistant", "content": full})
        trim_user_history(uid)
        safe_log(uid, "text_ok")

    except Exception as e:
        safe_log(uid, "text_stream_error", {"err": str(e)})

        try:
            fallback = await generate_text(
                provider=provider,
                model=model or ("gpt-4o-mini" if provider == "openai" else "grok-2-mini"),
                history=st["history"],
                user_input=text,
                openai_key=OPENAI_API_KEY,
                grok_key=GROK_API_KEY,
                gemini_key=GEMINI_API_KEY,
                max_history_tokens=st["max_history_tokens"]
            )
            await status.edit_text(fallback)
            st["history"].append({"role": "assistant", "content": fallback})
            trim_user_history(uid)
        except Exception:
            await status.edit_text("Ошибка. Попробуйте позже.")


# -------------------------------------------------------------------
# RUN POLLING
# -------------------------------------------------------------------
async def main():
    logger.info("Bot started")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
