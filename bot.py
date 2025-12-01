# bot.py — интеграция с обновлённым llm_core.py (aiogram 3.x)

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

from llm_core import generate_text_stream, generate_text, generate_image, trim_history_by_tokens

logger = logging.getLogger("neurozone_bot")
logger.setLevel(logging.INFO)
h = logging.StreamHandler()
h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(h)

# env
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not BOT_TOKEN:
    raise SystemExit("BOT_TOKEN not set")

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp = Dispatcher()

# rate-limit
RATE_LIMIT_PER_MINUTE = 30
user_requests = defaultdict(lambda: deque())

def check_rate_limit(user_id: int) -> bool:
    now = time.time()
    dq = user_requests[user_id]
    while dq and dq[0] < now - 60:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_PER_MINUTE:
        return False
    dq.append(now)
    return True

# user state
MAX_HISTORY_TOKENS_DEFAULT = 3000
user_state = defaultdict(lambda: {
    "mode": None,
    "provider": None,
    "model": None,
    "history": [],
    "max_history_tokens": MAX_HISTORY_TOKENS_DEFAULT
})

def safe_log(uid, event, extra=None):
    d = {"user_id": uid, "event": event}
    if extra:
        d.update(extra)
    logger.info(d)

# models from your old bot
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

# keyboards
def kb_main():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Текст", callback_data="mode:text"),
         InlineKeyboardButton(text="Картинки", callback_data="mode:image")],
        [InlineKeyboardButton(text="OpenAI", callback_data="provider:openai"),
         InlineKeyboardButton(text="Grok", callback_data="provider:grok"),
         InlineKeyboardButton(text="Gemini", callback_data="provider:gemini")],
        [InlineKeyboardButton(text="Сброс истории", callback_data="reset:history")]
    ])

def model_selected_menu():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⬅️ Назад к провайдерам", callback_data="back:providers")],
        [InlineKeyboardButton(text="⬅️ Главное меню", callback_data="back:main")]
    ])

# helper to repost menu after answer so it stays at bottom
async def repost_menu(chat_id):
    try:
        await bot.send_message(chat_id, "Меню:", reply_markup=kb_main())
    except Exception:
        pass

# show models with descriptions
async def show_models_for_provider(cb: types.CallbackQuery, provider_key: str):
    uid = cb.from_user.id
    if provider_key == "openai":
        models_dict = openai_models; header = "🔹 <b>ChatGPT (OpenAI)</b>"
    elif provider_key == "grok":
        models_dict = grok_models; header = "🧠 <b>Grok (xAI)</b>"
    else:
        models_dict = gemini_models; header = "⚡ <b>Gemini (Google)</b>"

    parts = [f"{header}\n\nВыберите модель:"]
    kb_rows = []
    for name, meta in models_dict.items():
        parts.append(f"\n<b>{name}</b> — <i>{meta['desc']}</i>")
        kb_rows.append([InlineKeyboardButton(text=name, callback_data=f"model:{meta['id']}")])
    kb_rows.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="back:providers")])
    text = "\n".join(parts)
    try:
        await cb.message.edit_text(text, reply_markup=InlineKeyboardMarkup(inline_keyboard=kb_rows))
    except Exception:
        await cb.message.answer(text, reply_markup=InlineKeyboardMarkup(inline_keyboard=kb_rows))
    await cb.answer()

# handlers
@dp.message(Command("start", "help"))
async def cmd_start(message: types.Message):
    uid = message.from_user.id
    safe_log(uid, "start")
    await message.answer("Привет! Выбери режим:", reply_markup=kb_main())

@dp.callback_query(lambda c: c.data and c.data.startswith("mode:"))
async def cb_mode(cb: types.CallbackQuery):
    uid = cb.from_user.id
    mode = cb.data.split(":",1)[1]
    user_state[uid]["mode"] = mode
    safe_log(uid, "mode_set", {"mode": mode})
    await cb.message.edit_text(f"Режим: <b>{mode}</b>\nВыберите провайдера:", reply_markup=kb_main())
    await cb.answer()

@dp.callback_query(lambda c: c.data and c.data.startswith("provider:"))
async def cb_provider(cb: types.CallbackQuery):
    uid = cb.from_user.id
    prov = cb.data.split(":",1)[1]
    user_state[uid]["provider"] = prov
    safe_log(uid, "provider_set", {"provider": prov})
    await show_models_for_provider(cb, prov)

@dp.callback_query(lambda c: c.data == "back:providers")
async def cb_back_providers(cb: types.CallbackQuery):
    await cb.message.edit_text("Выберите провайдера:", reply_markup=kb_main())
    await cb.answer()

@dp.callback_query(lambda c: c.data and c.data.startswith("model:"))
async def cb_model(cb: types.CallbackQuery):
    uid = cb.from_user.id
    model_id = cb.data.split(":",1)[1]
    user_state[uid]["model"] = model_id
    safe_log(uid, "model_set", {"model": model_id})
    await cb.message.edit_text(f"Вы выбрали модель:\n<b>{model_id}</b>\n\nТеперь отправьте запрос.", reply_markup=model_selected_menu())
    await cb.answer()

@dp.callback_query(lambda c: c.data == "reset:history")
async def cb_reset(cb: types.CallbackQuery):
    uid = cb.from_user.id
    user_state[uid]["history"] = []
    safe_log(uid, "history_reset")
    await cb.message.edit_text("История очищена.", reply_markup=kb_main())
    await cb.answer()

@dp.callback_query(lambda c: c.data == "back:main")
async def cb_back_main(cb: types.CallbackQuery):
    await cb.message.edit_text("Главное меню:", reply_markup=kb_main())
    await cb.answer()

# message handler
@dp.message()
async def on_message(message: types.Message):
    uid = message.from_user.id
    text = message.text or ""
    safe_log(uid, "msg_received", {"len": len(text)})

    if not check_rate_limit(uid):
        safe_log(uid, "rate_limited")
        await message.answer("Слишком много запросов — подождите минуту.")
        return

    st = user_state[uid]
    mode = st.get("mode")
    provider = st.get("provider")
    model = st.get("model")

    if not mode:
        await message.answer("Выберите режим через /start")
        return

    # image mode
    if mode == "image":
        await message.answer("Генерирую изображение...")
        try:
            url = await generate_image(provider=provider, prompt=text,
                                       openai_key=OPENAI_API_KEY, grok_key=GROK_API_KEY,
                                       gemini_key=GEMINI_API_KEY)
            await message.answer_photo(url, caption="Готово!")
            # репост меню
            await repost_menu(message.chat.id)
        except Exception as e:
            safe_log(uid, "image_error", {"err": str(e)})
            await message.answer("Ошибка при обращении к изображению.")
            await repost_menu(message.chat.id)
        return

    # text mode
    # append to history
    st["history"].append({"role":"user", "content": text})
    st["history"] = trim_history_by_tokens(st["history"], st.get("max_history_tokens", MAX_HISTORY_TOKENS_DEFAULT))

    status = await message.answer("Генерирую ответ...")
    try:
        full = ""
        last_edit = time.time()
        stream = generate_text_stream(provider=provider, model=model,
                                      history=st["history"],
                                      user_input=text,
                                      openai_key=OPENAI_API_KEY,
                                      grok_key=GROK_API_KEY,
                                      gemini_key=GEMINI_API_KEY,
                                      max_history_tokens=st.get("max_history_tokens", MAX_HISTORY_TOKENS_DEFAULT))
        async for chunk in stream:
            full += chunk
            # edit every 0.4s
            if time.time() - last_edit >= 0.35:
                try:
                    await status.edit_text(full)
                except TelegramAPIError:
                    pass
                last_edit = time.time()

        try:
            await status.edit_text(full)
        except TelegramAPIError:
            pass

        st["history"].append({"role":"assistant", "content": full})
        st["history"] = trim_history_by_tokens(st["history"], st.get("max_history_tokens", MAX_HISTORY_TOKENS_DEFAULT))
        safe_log(uid, "text_ok")
        # репост меню, чтобы было внизу
        await repost_menu(message.chat.id)

    except Exception as e:
        safe_log(uid, "text_error", {"err": str(e)})
        # пытаемся fallback (однократный запрос)
        try:
            fallback = await generate_text(provider=provider, model=model,
                                           history=st["history"], user_input=text,
                                           openai_key=OPENAI_API_KEY, grok_key=GROK_API_KEY,
                                           gemini_key=GEMINI_API_KEY,
                                           max_history_tokens=st.get("max_history_tokens", MAX_HISTORY_TOKENS_DEFAULT))
            await status.edit_text(fallback)
            st["history"].append({"role":"assistant", "content": fallback})
            await repost_menu(message.chat.id)
        except Exception as e2:
            safe_log(uid, "fallback_failed", {"err": str(e2)})
            try:
                await status.edit_text("Ошибка при обращении к модели.")
            except Exception:
                pass
            await repost_menu(message.chat.id)

# run
async def main():
    logger.info("Bot started")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
