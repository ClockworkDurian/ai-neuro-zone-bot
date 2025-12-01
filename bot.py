# bot.py — полностью совместим с новым llm_core.py (OpenAI + Grok + Gemini)
# aiogram 3.13.1

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

# -------------------------------------------------------------------
# ЛОГИ
# -------------------------------------------------------------------

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

# -------------------------------------------------------------------
# ENV VARS
# -------------------------------------------------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not BOT_TOKEN:
    raise SystemExit("❌ BOT_TOKEN отсутствует")

# -------------------------------------------------------------------
# ИНИЦИАЛИЗАЦИЯ AIROGRAM
# -------------------------------------------------------------------

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp = Dispatcher()

# -------------------------------------------------------------------
# RATE LIMIT
# -------------------------------------------------------------------

RATE_LIMIT_PER_MINUTE = 30
user_requests = defaultdict(lambda: deque())

def check_rate_limit(uid: int) -> bool:
    now = time.time()
    dq = user_requests[uid]
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
    "mode": None,
    "provider": None,
    "model": None,
    "history": [],
    "max_history_tokens": MAX_HISTORY_TOKENS_DEFAULT
})

# -------------------------------------------------------------------
# МОДЕЛИ (КАК В СТАРОМ БОТЕ)
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

def kb_main():
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

def menu_after_answer():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⬅️ Главное меню", callback_data="back:main")]
    ])

def model_selected_menu():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⬅️ Назад к провайдерам", callback_data="back:providers")],
        [InlineKeyboardButton(text="⬅️ Главное меню", callback_data="back:main")]
    ])

async def repost_menu(chat_id: int):
    try:
        await bot.send_message(chat_id, "Меню:", reply_markup=kb_main())
    except Exception:
        pass

# -------------------------------------------------------------------
# ПОКАЗ МОДЕЛЕЙ
# -------------------------------------------------------------------

async def show_models_for_provider(cb: types.CallbackQuery, provider_key: str):
    if provider_key == "openai":
        models = openai_models
        header = "🔵 <b>OpenAI — ChatGPT модели</b>"
    elif provider_key == "grok":
        models = grok_models
        header = "🧠 <b>Grok — xAI модели</b>"
    else:
        models = gemini_models
        header = "⚡ <b>Gemini — Google AI модели</b>"

    txt = [f"{header}\n\nВыберите модель:"]
    kb = []

    for name, meta in models.items():
        txt.append(f"\n<b>{name}</b> — <i>{meta['desc']}</i>")
        kb.append([InlineKeyboardButton(text=name, callback_data=f"model:{meta['id']}")])

    kb.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="back:providers")])

    msg = "\n".join(txt)

    try:
        await cb.message.edit_text(msg, reply_markup=InlineKeyboardMarkup(inline_keyboard=kb))
    except:
        await cb.message.answer(msg, reply_markup=InlineKeyboardMarkup(inline_keyboard=kb))

    await cb.answer()

# -------------------------------------------------------------------
# CALLBACK HANDLERS
# -------------------------------------------------------------------

@dp.message(Command("start", "help"))
async def start_cmd(message: types.Message):
    await message.answer("Привет! Выберите режим:", reply_markup=kb_main())

@dp.callback_query(lambda c: c.data and c.data.startswith("mode:"))
async def choose_mode(cb: types.CallbackQuery):
    uid = cb.from_user.id
    mode = cb.data.split(":", 1)[1]
    user_state[uid]["mode"] = mode
    safe_log(uid, "mode_set", {"mode": mode})
    await cb.message.edit_text(f"Режим установлен: <b>{mode}</b>\nТеперь выберите провайдера:", reply_markup=kb_main())
    await cb.answer()

@dp.callback_query(lambda c: c.data and c.data.startswith("provider:"))
async def choose_provider(cb: types.CallbackQuery):
    uid = cb.from_user.id
    prov = cb.data.split(":", 1)[1]
    user_state[uid]["provider"] = prov
    safe_log(uid, "provider_set", {"provider": prov})
    await show_models_for_provider(cb, prov)

@dp.callback_query(lambda c: c.data == "back:providers")
async def back_to_providers(cb: types.CallbackQuery):
    await cb.message.edit_text("Выберите провайдера:", reply_markup=kb_main())
    await cb.answer()

@dp.callback_query(lambda c: c.data and c.data.startswith("model:"))
async def choose_model(cb: types.CallbackQuery):
    uid = cb.from_user.id
    model = cb.data.split(":", 1)[1]
    user_state[uid]["model"] = model
    safe_log(uid, "model_set", {"model": model})
    await cb.message.edit_text(
        f"Вы выбрали модель:\n<b>{model}</b>\n\nТеперь отправьте ваш запрос.",
        reply_markup=model_selected_menu()
    )
    await cb.answer()

@dp.callback_query(lambda c: c.data == "reset:history")
async def reset_history(cb: types.CallbackQuery):
    uid = cb.from_user.id
    user_state[uid]["history"] = []
    safe_log(uid, "history_reset")
    await cb.message.edit_text("История очищена.", reply_markup=kb_main())
    await cb.answer()

@dp.callback_query(lambda c: c.data == "back:main")
async def back_main(cb: types.CallbackQuery):
    await cb.message.edit_text("Главное меню:", reply_markup=kb_main())
    await cb.answer()

# -------------------------------------------------------------------
# MESSAGE HANDLER
# -------------------------------------------------------------------

@dp.message()
async def on_message(message: types.Message):
    uid = message.from_user.id
    text = message.text or ""
    safe_log(uid, "text_received", {"len": len(text)})

    if not check_rate_limit(uid):
        await message.answer("Слишком много запросов. Подождите минуту.")
        return

    st = user_state[uid]
    mode = st["mode"]
    provider = st["provider"]
    model = st["model"]

    if not mode:
        await message.answer("Выберите режим через /start")
        return
    if not provider:
        await message.answer("Выберите провайдера.")
        return
    if not model:
        await message.answer("Выберите модель.")
        return

    # ----------------------------------------
    # IMAGE MODE
    # ----------------------------------------
    if mode == "image":
        await message.answer("Генерирую изображение...")
        try:
            url = await generate_image(
                provider=provider,
                prompt=text,
                openai_key=OPENAI_API_KEY,
                grok_key=GROK_API_KEY,
                gemini_key=GEMINI_API_KEY
            )
            await message.answer_photo(url, caption="Готово!")
        except Exception as e:
            safe_log(uid, "image_error", {"err": str(e)})
            await message.answer("Ошибка при генерации изображения.")
        await repost_menu(message.chat.id)
        return

    # ----------------------------------------
    # TEXT MODE (STREAMING)
    # ----------------------------------------

    st["history"].append({"role": "user", "content": text})
    st["history"] = trim_history_by_tokens(st["history"], st["max_history_tokens"])

    status = await message.answer("Генерирую ответ...")

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
            # Обновляем раз в 0.35 секунды
            if time.time() - last_edit > 0.35:
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
        st["history"] = trim_history_by_tokens(st["history"], st["max_history_tokens"])

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
            st["history"].append({"role":"assistant", "content": fallback})
        except Exception as e2:
            safe_log(uid, "fallback_failed", {"err": str(e2)})
            await status.edit_text("Ошибка при обращении к модели.")

    await repost_menu(message.chat.id)

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

async def main():
    logger.info("Bot started")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
