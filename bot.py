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


logger = logging.getLogger("neurozone_bot")
logger.setLevel(logging.INFO)
h = logging.StreamHandler()
h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(h)

def safe_log(uid, event, extra=None):
    d = {"user_id": uid, "event": event}
    if extra:
        d.update(extra)
    logger.info(d)


# ---------------------------------------------------------
# ENV VARS
# ---------------------------------------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not BOT_TOKEN:
    raise SystemExit("BOT_TOKEN missing")

bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp = Dispatcher()


# ---------------------------------------------------------
# RATE LIMIT
# ---------------------------------------------------------
RATE_LIMIT_PER_MINUTE = 30
user_requests = defaultdict(lambda: deque())

def check_rate_limit(uid):
    now = time.time()
    dq = user_requests[uid]
    while dq and dq[0] < now - 60:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_PER_MINUTE:
        return False
    dq.append(now)
    return True


# ---------------------------------------------------------
# USER STATE
# ---------------------------------------------------------
MAX_HISTORY_TOKENS_DEFAULT = 3000

user_state = defaultdict(lambda: {
    "mode": None,
    "provider": None,
    "model": None,
    "history": [],
})

provider_unavailable = {}  # provider -> until timestamp


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------
openai_text_models = {
    "GPT-5": {"id": "gpt-5", "desc": "Сильная reasoning-модель."},
    "GPT-5 mini": {"id": "gpt-5-mini", "desc": "Быстрая экономичная версия."},
    "GPT-5 nano": {"id": "gpt-5-nano", "desc": "Максимальная скорость."},
    "GPT-4.1": {"id": "gpt-4.1", "desc": "Сильная модель без chain-of-thought."},
}

grok_text_models = {
    "Grok code": {"id": "grok-code-fast-1", "desc": "Быстрая модель для кода."},
    "Grok reasoning": {"id": "grok-4-fast-reasoning", "desc": "Мощная reasoning-модель."},
    "Grok non-reasoning": {"id": "grok-4-fast-non-reasoning", "desc": "Быстрая модель без reasoning."},
}

gemini_text_models = {
    "Gemini 2.5 Flash": {"id": "gemini-2.5-flash", "desc": "Быстрая и дешёвая."},
    "Gemini 2.5 Flash-Lite": {"id": "gemini-2.5-flash-lite", "desc": "Максимальная скорость."},
}

# Image
openai_image_models = {
    "DALL·E 3": {"id": "dall-e-3", "desc": "Генерация картинок OpenAI."}
}
grok_image_models = {
    "Grok Image": {"id": "grok-image-1", "desc": "Генерация картинок Grok."}
}
gemini_image_models = {
    "Gemini Image": {"id": "gemini-2.0-image", "desc": "Изображения Gemini."}
}


# ---------------------------------------------------------
# KEYBOARDS
# ---------------------------------------------------------
def kb_main():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton("Текст", callback_data="mode:text"),
         InlineKeyboardButton("Картинки", callback_data="mode:image")],
        [InlineKeyboardButton("OpenAI", callback_data="provider:openai"),
         InlineKeyboardButton("Grok", callback_data="provider:grok"),
         InlineKeyboardButton("Gemini", callback_data="provider:gemini")],
        [InlineKeyboardButton("Сброс истории", callback_data="reset:history")],
    ])


def model_selected_menu():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton("⬅ Назад", callback_data="back:providers")],
        [InlineKeyboardButton("⬅ Главное меню", callback_data="back:main")]
    ])


async def repost_menu(chat_id):
    try:
        await bot.send_message(chat_id, "Меню:", reply_markup=kb_main())
    except:
        pass


# ---------------------------------------------------------
# SHOW MODELS
# ---------------------------------------------------------
async def show_models_for_provider(cb, provider):
    uid = cb.from_user.id
    mode = user_state[uid]["mode"]

    if mode == "image":
        if provider == "openai":
            models = openai_image_models
            header = "🖼 <b>OpenAI — Images</b>"
        elif provider == "grok":
            models = grok_image_models
            header = "🖼 <b>Grok — Images</b>"
        else:
            models = gemini_image_models
            header = "🖼 <b>Gemini — Images</b>"
    else:
        if provider == "openai":
            models = openai_text_models
            header = "🔵 <b>OpenAI — Text models</b>"
        elif provider == "grok":
            models = grok_text_models
            header = "🧠 <b>Grok — Text models</b>"
        else:
            models = gemini_text_models
            header = "⚡ <b>Gemini — Text models</b>"

    text = header + "\n\nВыберите модель:\n"

    rows = []
    for name, meta in models.items():
        text += f"\n<b>{name}</b> — <i>{meta['desc']}</i>"
        rows.append([
            InlineKeyboardButton(name, callback_data=f"model:{meta['id']}")
        ])

    rows.append([InlineKeyboardButton("⬅ Назад", callback_data="back:providers")])

    try:
        await cb.message.edit_text(text, reply_markup=InlineKeyboardMarkup(inline_keyboard=rows))
    except:
        await cb.message.answer(text, reply_markup=InlineKeyboardMarkup(inline_keyboard=rows))

    await cb.answer()


# ---------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------
@dp.message(Command("start", "help"))
async def cmd_start(message):
    safe_log(message.from_user.id, "start")
    await message.answer("Привет! Выберите режим:", reply_markup=kb_main())


@dp.callback_query(lambda c: c.data.startswith("mode:"))
async def cb_mode(c):
    uid = c.from_user.id
    mode = c.data.split(":", 1)[1]
    user_state[uid]["mode"] = mode
    safe_log(uid, "set_mode", {"mode": mode})
    await c.message.edit_text("Режим выбран. Теперь выберите провайдера:", reply_markup=kb_main())
    await c.answer()


@dp.callback_query(lambda c: c.data.startswith("provider:"))
async def cb_provider(c):
    uid = c.from_user.id
    prov = c.data.split(":", 1)[1]
    user_state[uid]["provider"] = prov
    safe_log(uid, "set_provider", {"provider": prov})
    await show_models_for_provider(c, prov)


@dp.callback_query(lambda c: c.data == "back:providers")
async def cb_back_prov(c):
    await c.message.edit_text("Выберите провайдера:", reply_markup=kb_main())
    await c.answer()


@dp.callback_query(lambda c: c.data.startswith("model:"))
async def cb_model(c):
    uid = c.from_user.id
    model = c.data.split(":", 1)[1]
    user_state[uid]["model"] = model
    safe_log(uid, "set_model", {"model": model})
    await c.message.edit_text(f"Вы выбрали модель <b>{model}</b>\n\nТеперь отправьте ваш запрос.",
                              reply_markup=model_selected_menu())
    await c.answer()


@dp.callback_query(lambda c: c.data == "reset:history")
async def cb_reset(c):
    uid = c.from_user.id
    user_state[uid]["history"] = []
    safe_log(uid, "history_reset")
    await c.message.edit_text("История очищена.", reply_markup=kb_main())
    await c.answer()


@dp.callback_query(lambda c: c.data == "back:main")
async def cb_back_main(c):
    await c.message.edit_text("Главное меню:", reply_markup=kb_main())
    await c.answer()


# ---------------------------------------------------------
# MESSAGE HANDLER
# ---------------------------------------------------------
@dp.message()
async def on_message(message: types.Message):
    uid = message.from_user.id
    text = message.text or ""
    safe_log(uid, "msg", {"len": len(text)})

    if not check_rate_limit(uid):
        await message.answer("Слишком много запросов, подождите минуту.")
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

    # provider temporarily unavailable
    now = time.time()
    if provider in provider_unavailable and provider_unavailable[provider] > now:
        await message.answer(f"Провайдер {provider} временно недоступен.")
        return
    elif provider in provider_unavailable:
        del provider_unavailable[provider]

    # IMAGE MODE
    if mode == "image":
        await message.answer("Генерирую изображение…")

        try:
            url = await generate_image(provider, model, text,
                                       openai_key=OPENAI_API_KEY,
                                       grok_key=GROK_API_KEY,
                                       gemini_key=GEMINI_API_KEY)

            await message.answer_photo(url, caption="Готово!")

        except RuntimeError as e:
            err = str(e)
            safe_log(uid, "image_error", {"err": err})

            if "GEMINI_QUOTA_EXCEEDED" in err:
                provider_unavailable["gemini"] = time.time() + 600
                await message.answer("Gemini: превышена квота. Временно недоступен.")
            else:
                await message.answer(f"Ошибка: {err}")

        await repost_menu(message.chat.id)
        return

    # TEXT MODE
    st["history"].append({"role": "user", "content": text})
    st["history"] = trim_history_by_tokens(st["history"], MAX_HISTORY_TOKENS_DEFAULT)

    status = await message.answer("Генерирую ответ…")

    try:
        # STREAM (OpenAI + Grok)
        if provider in ("openai", "grok"):
            full = ""
            last_edit = time.time()

            stream = generate_text_stream(provider, model, st["history"], text,
                                          openai_key=OPENAI_API_KEY,
                                          grok_key=GROK_API_KEY,
                                          gemini_key=GEMINI_API_KEY)

            async for chunk in stream:
                full += chunk
                if time.time() - last_edit > 0.35:
                    try:
                        await status.edit_text(full)
                    except TelegramAPIError:
                        pass
                    last_edit = time.time()

            try:
                await status.edit_text(full)
            except:
                pass

            st["history"].append({"role": "assistant", "content": full})
            st["history"] = trim_history_by_tokens(st["history"], MAX_HISTORY_TOKENS_DEFAULT)

        # Gemini — non-stream
        else:
            full = await generate_text(provider, model, st["history"], text,
                                       openai_key=OPENAI_API_KEY,
                                       grok_key=GROK_API_KEY,
                                       gemini_key=GEMINI_API_KEY)

            await status.edit_text(full)

            st["history"].append({"role": "assistant", "content": full})
            st["history"] = trim_history_by_tokens(st["history"], MAX_HISTORY_TOKENS_DEFAULT)

    except RuntimeError as e:
        err = str(e)
        safe_log(uid, "text_error", {"err": err})

        if "GEMINI_QUOTA_EXCEEDED" in err:
            provider_unavailable["gemini"] = time.time() + 600
            await status.edit_text("Gemini: превышена квота (429).")
        else:
            await status.edit_text(f"Ошибка: {err}")

    await repost_menu(message.chat.id)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
async def main():
    logger.info("Bot started")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
