# bot.py — aiogram 3.x совместимая версия
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

# наша логика ИИ
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
    raise SystemExit("❌ BOT_TOKEN не найден в переменных окружения")


# -------------------------------------------------------------------
# БОТ / ДИСПЕТЧЕР (aiogram 3.x)
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
    """Простой sliding window rate-limit."""
    now = time.time()
    window = 60
    dq = user_requests[user_id]

    # удалить старые запросы
    while dq and dq[0] < now - window:
        dq.popleft()

    if len(dq) >= RATE_LIMIT_PER_MINUTE:
        return False

    dq.append(now)
    return True


# -------------------------------------------------------------------
# СОСТОЯНИЕ ПОЛЬЗОВАТЕЛЯ
# -------------------------------------------------------------------
MAX_HISTORY_TOKENS_DEFAULT = 3000

user_state = defaultdict(lambda: {
    "mode": "text",         # text / image
    "provider": "openai",   # openai / grok / gemini
    "model": None,          # конкретная модель
    "history": [],          # [{"role": "...", "content": "..."}]
    "max_history_tokens": MAX_HISTORY_TOKENS_DEFAULT
})

def trim_user_history(user_id: int):
    st = user_state[user_id]
    st["history"] = trim_history_by_tokens(
        st["history"],
        st.get("max_history_tokens", MAX_HISTORY_TOKENS_DEFAULT)
    )


# -------------------------------------------------------------------
# INLINE-КНОПКИ
# -------------------------------------------------------------------
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
            InlineKeyboardButton(text="Сброс истории", callback_data="reset:history")
        ]
    ])

def kb_models(provider: str):
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
    else:  # gemini
        buttons = [
            InlineKeyboardButton("gemini-1.5-flash", callback_data="model:gemini-1.5-flash")
        ]

    kb = InlineKeyboardMarkup(inline_keyboard=[buttons])
    return kb


# -------------------------------------------------------------------
# HANDLERS
# -------------------------------------------------------------------

@dp.message(Command("start", "help"))
async def on_start(message: types.Message):
    uid = message.from_user.id
    safe_log(uid, "start")
    await message.answer("Привет! Выбери режим работы и провайдера:", reply_markup=kb_main())


@dp.callback_query()
async def on_callback(cb: types.CallbackQuery):
    uid = cb.from_user.id
    data = cb.data or ""
    safe_log(uid, "callback", {"data": data})

    # режим
    if data.startswith("mode:"):
        mode = data.split(":", 1)[1]
        user_state[uid]["mode"] = mode

        await cb.message.edit_text(
            f"Режим: {mode}\nТеперь выбери провайдера:",
            reply_markup=kb_main()
        )
        await cb.answer()
        return

    # провайдер
    if data.startswith("provider:"):
        prov = data.split(":", 1)[1]
        user_state[uid]["provider"] = prov

        await cb.message.edit_text(
            f"Провайдер: {prov}\nВыбери модель:",
            reply_markup=kb_models(prov)
        )
        await cb.answer()
        return

    # модель
    if data.startswith("model:"):
        model = data.split(":", 1)[1]
        user_state[uid]["model"] = model

        await cb.message.edit_text(
            f"Модель установлена: {model}\nТеперь отправь свой запрос.",
            reply_markup=kb_main()
        )
        await cb.answer()
        return

    # сброс истории
    if data == "reset:history":
        user_state[uid]["history"] = []
        await cb.message.edit_text("История очищена.", reply_markup=kb_main())
        safe_log(uid, "history_reset")
        await cb.answer()
        return

    await cb.answer()


@dp.message()
async def on_message(message: types.Message):
    uid = message.from_user.id
    text = message.text or ""

    safe_log(uid, "message_received", {"length": len(text)})

    # RATE LIMIT
    if not check_rate_limit(uid):
        await message.answer("Слишком много запросов. Подожди минуту.")
        safe_log(uid, "rate_limited")
        return

    st = user_state[uid]
    mode = st["mode"]
    provider = st["provider"]
    model = st["model"]

    # ----------------------------------------------------------
    # IMAGE MODE
    # ----------------------------------------------------------
    if mode == "image":
        safe_log(uid, "image_request", {"provider": provider, "model": model})

        try:
            img_url = await generate_image(
                provider=provider,
                prompt=text,
                openai_key=OPENAI_API_KEY,
                grok_key=GROK_API_KEY
            )

            await message.answer_photo(img_url, caption="Сгенерировано!")
            st["history"].append({"role": "user", "content": "[image_prompt]"})
            st["history"].append({"role": "assistant", "content": "[image_generated]"})
            trim_user_history(uid)
            safe_log(uid, "image_ok", {"provider": provider})

        except Exception as e:
            safe_log(uid, "image_error", {"error": str(e)})
            await message.answer("Ошибка генерации изображения. Попробуйте позже.")

        return

    # ----------------------------------------------------------
    # TEXT MODE + STREAMING
    # ----------------------------------------------------------
    # Добавить сообщение пользователя в историю
    st["history"].append({"role": "user", "content": text})
    trim_user_history(uid)

    status = await message.answer("Генерирую ответ...")

    try:
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

        full_text = ""
        last_edit = time.time()

        # STREAM LOOP
        async for chunk in stream:
            full_text += chunk
            if time.time() - last_edit >= 0.35:
                try:
                    await status.edit_text(full_text)
                except TelegramAPIError:
                    pass
                last_edit = time.time()

        # финальное редактирование
        try:
            await status.edit_text(full_text)
        except TelegramAPIError:
            pass

        st["history"].append({"role": "assistant", "content": full_text})
        trim_user_history(uid)
        safe_log(uid, "text_ok", {"provider": provider})

    except Exception as e:
        safe_log(uid, "text_stream_error", {"err": str(e)})

        # fallback — NON-STREAM RESPONSE
        try:
            resp = await generate_text(
                provider=provider,
                model=model or ("gpt-4o-mini" if provider == "openai" else "grok-2-mini"),
                history=st["history"],
                user_input=text,
                openai_key=OPENAI_API_KEY,
                grok_key=GROK_API_KEY,
                gemini_key=GEMINI_API_KEY,
                max_history_tokens=st["max_history_tokens"]
            )

            await status.edit_text(resp)
            st["history"].append({"role": "assistant", "content": resp})
            trim_user_history(uid)
            safe_log(uid, "text_fallback_ok", {"provider": provider})

        except Exception as e2:
            safe_log(uid, "text_fallback_error", {"err": str(e2)})
            await status.edit_text(
                "Ошибка. Провайдер перегружен или недоступен. Попробуйте позже."
            )


# -------------------------------------------------------------------
# START POLLING (aiogram 3.x)
# -------------------------------------------------------------------
async def main():
    logger.info("Bot started")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
