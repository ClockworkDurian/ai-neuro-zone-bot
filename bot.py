import os
import sys
import logging
import asyncio
import time

# --- ИЗМЕНЕНО: Импортируем официальные SDK ---
import openai
import google.generativeai as genai
from xai_sdk import Client as XAI_Client

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message
from dotenv import load_dotenv

# === Настройки кодировки и логов ===
sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Загрузка .env и конфигурация клиентов API ===
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN")
# --- ИЗМЕНЕНО: Конфигурация клиентов SDK ---
openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
xai_client = XAI_Client(api_key=os.getenv("GROK_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# === Состояние пользователей, история, константы ===
user_state = {}
MAX_HISTORY_LENGTH = 10

# === Модели с описаниями ===
# (Этот блок без изменений)
openai_models = { "GPT-5": {"id": "gpt-5", "desc": "Лучшая модель для кода и агентных задач."}, "GPT-5 mini": {"id": "gpt-5-mini", "desc": "Более быстрая, экономичная версия."}, "GPT-5 nano": {"id": "gpt-5-nano", "desc": "Самая быстрая и экономичная версия."}, "GPT-4.1": {"id": "gpt-4.1", "desc": "Самая умная модель без рассуждений."}}
grok_models = {"Grok-code-fast-1": {"id": "grok-code-fast-1", "desc": "Быстрая и экономичная модель для кодирования."}, "Grok-4-fast-reasoning": {"id": "grok-4-fast-reasoning", "desc": "Последнее достижение в экономичных моделях."}, "Grok-4-fast-non-reasoning": {"id": "grok-4-fast-non-reasoning", "desc": "Последнее достижение в экономичных моделях."}}
gemini_models = {"Gemini 2.5 Flash": {"id": "gemini-2.5-flash", "desc": "Лучшая по цене/производительности."}, "Gemini 2.5 Flash-Lite": {"id": "gemini-2.5-flash-lite", "desc": "Самая быстрая flash-модель."}}

# === Блок с меню (без изменений) ===
# (Этот блок без изменений, поэтому для краткости я его скрою)
def main_mode_menu():
    buttons = [[InlineKeyboardButton(text="✍️ Текстовый чат", callback_data="mode_textchat")], [InlineKeyboardButton(text="🎨 Сгенерировать изображение", callback_data="mode_imagegen")], [InlineKeyboardButton(text="🌐 Сайт", url="https://neurozone.pro/")], [InlineKeyboardButton(text="🔒 Политика конфиденциальности", url="https://neurozone.pro/privacy")]]
    return InlineKeyboardMarkup(inline_keyboard=buttons)
def text_provider_menu():
    buttons = [[InlineKeyboardButton(text="💭 ChatGPT (OpenAI)", callback_data="provider_openai")], [InlineKeyboardButton(text="🧠 Grok (xAI)", callback_data="provider_grok")], [InlineKeyboardButton(text="⚡ Gemini (Google)", callback_data="provider_gemini")], [InlineKeyboardButton(text="⬅️ Назад в главное меню", callback_data="back_to_main_menu")]]
    return InlineKeyboardMarkup(inline_keyboard=buttons)
def image_provider_menu():
    buttons = [[InlineKeyboardButton(text=" DALL-E 3 (OpenAI)", callback_data="image_provider_openai")], [InlineKeyboardButton(text=" Grok Image (xAI)", callback_data="image_provider_grok")], [InlineKeyboardButton(text="⬅️ Назад в главное меню", callback_data="back_to_main_menu")]]
    return InlineKeyboardMarkup(inline_keyboard=buttons)
def reset_user_state(user_id):
    user_state[user_id] = {"provider": None, "model": None, "history": [], "mode": None}

# === Команды и обработчики кнопок (без изменений) ===
# (Этот блок без изменений, поэтому для краткости я его скрою)
@dp.message(Command("start", "reset"))
async def start_reset_command(message: Message): await message.answer("👋 Привет! ...", parse_mode="Markdown", reply_markup=main_mode_menu())
@dp.callback_query(lambda c: c.data == "back_to_main_menu")
async def back_to_main_menu_handler(callback_query: types.CallbackQuery): reset_user_state(callback_query.from_user.id); await callback_query.message.edit_text("👋 Привет! ...", parse_mode="Markdown", reply_markup=main_mode_menu())
@dp.callback_query(lambda c: c.data.startswith("mode_"))
async def mode_selection_handler(callback_query: types.CallbackQuery):
    mode = callback_query.data.split("_")[1]; user_id = callback_query.from_user.id
    if user_id not in user_state: reset_user_state(user_id)
    user_state[user_id]["mode"] = mode
    if mode == "textchat": await callback_query.message.edit_text("...", reply_markup=text_provider_menu())
    elif mode == "imagegen": await callback_query.message.edit_text("...", reply_markup=image_provider_menu())
@dp.callback_query(lambda c: c.data.startswith("provider_"))
async def text_provider_selection(callback_query: types.CallbackQuery):
    provider = callback_query.data.split("_")[1]; user_id = callback_query.from_user.id; user_state[user_id]["provider"] = provider; models_dict, header = {}, ""
    if provider == "openai": models_dict, header = openai_models, "🔹 *ChatGPT (OpenAI)*"
    elif provider == "grok": models_dict, header = grok_models, "🧠 *Grok (xAI)*"
    elif provider == "gemini": models_dict, header = gemini_models, "⚡ *Gemini (Google)*"
    text_parts = [f"{header}\n\nВыберите модель:"]; buttons = []
    for name, data in models_dict.items(): text_parts.append(f"\n*{name}* - _{data['desc']}_"); buttons.append([InlineKeyboardButton(text=name, callback_data=f"model_{data['id']}")])
    buttons.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="mode_textchat")]); await callback_query.message.edit_text("\n".join(text_parts), parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons))
@dp.callback_query(lambda c: c.data.startswith("model_"))
async def model_selection(callback_query: types.CallbackQuery):
    model_id = callback_query.data.split("_", 1)[1]; user_id = callback_query.from_user.id
    if user_id not in user_state or not user_state[user_id].get("provider"): await callback_query.answer("Ошибка. Начните с /start"); return
    user_state[user_id]["model"] = model_id; await callback_query.message.edit_text(f"✅ Модель *{model_id}* выбрана.\n\nОтправьте свой вопрос.", parse_mode="Markdown")
@dp.callback_query(lambda c: c.data.startswith("image_provider_"))
async def image_provider_selection(callback_query: types.CallbackQuery):
    provider = callback_query.data.split("_")[1]; user_id = callback_query.from_user.id; user_state[user_id]["provider"] = provider; provider_name = ""
    if provider == "openai": provider_name = "DALL-E 3 (OpenAI)"
    elif provider == "grok": provider_name = "Grok Image (xAI)"
    await callback_query.message.edit_text(f"✅ Выбрана технология: *{provider_name}*.\n\nОтправьте промпт для генерации.", parse_mode="Markdown")


# --- ГЛАВНЫЙ ОБРАБОТЧИК СООБЩЕНИЙ ---

@dp.message()
async def main_message_handler(message: Message):
    user_id = message.from_user.id
    if user_id not in user_state or not user_state[user_id].get("mode"):
        await message.answer("Пожалуйста, выберите режим работы через /start")
        return
    mode = user_state[user_id]["mode"]
    if mode == "textchat": await handle_text_chat(message)
    elif mode == "imagegen": await handle_image_generation(message)
    else: await message.answer("Неизвестный режим. Начните с /start")

# --- ИЗМЕНЕНО: Логика для текстового чата полностью на SDK ---

async def handle_text_chat(message: Message):
    user_id = message.from_user.id
    if not user_state[user_id].get("model"):
        await message.answer("Сначала выберите модель, которой хотите задать вопрос.")
        return

    logging.info(f"Received text message from user_id: {user_id}")
    start_time = time.time()
    provider = user_state[user_id]["provider"]
    model_id = user_state[user_id]["model"]
    user_input = message.text.strip()
    history = user_state[user_id].get("history", [])
    if len(history) > MAX_HISTORY_LENGTH: history = history[-MAX_HISTORY_LENGTH:]

    await message.chat.do("typing")
    answer = ""
    
    try:
        if provider == "openai":
            history.append({"role": "user", "content": user_input})
            response = await openai_client.chat.completions.create(model=model_id, messages=history)
            answer = response.choices[0].message.content
            history.append({"role": "assistant", "content": answer})

        elif provider == "grok":
            history.append({"role": "user", "content": user_input})
            # SDK Grok синхронный, запускаем в отдельном потоке
            def _generate_grok_chat():
                chat_completion = xai_client.chat.create(model=model_id, messages=history)
                return chat_completion.choices[0].message.content
            answer = await asyncio.to_thread(_generate_grok_chat)
            history.append({"role": "assistant", "content": answer})
            
        elif provider == "gemini":
            # Gemini SDK сам управляет историей
            gemini_model = genai.GenerativeModel(model_id)
            chat_session = gemini_model.start_chat(history=history)
            response = await chat_session.send_message_async(user_input)
            answer = response.text
            # Обновляем историю из сессии
            history = [{"role": item.role, "parts": [part.text for part in item.parts]} for item in chat_session.history]

        duration = time.time() - start_time
        logging.info(f"SUCCESS text chat for user_id: {user_id}. Provider: {provider}, Model: {model_id}. Duration: {duration:.2f}s")
    
    except Exception as e:
        duration = time.time() - start_time
        logging.exception(f"ERROR during text chat for user_id: {user_id}. Provider: {provider}. Error: {e}")
        answer = f"❌ *Произошла ошибка при общении с нейросетью.*\n\n_{str(e)}_."

    user_state[user_id]["history"] = history
    await message.answer(answer, parse_mode="Markdown")

# --- ИЗМЕНЕНО: Логика для генерации изображений полностью на SDK ---

async def handle_image_generation(message: Message):
    user_id = message.from_user.id
    provider = user_state[user_id].get("provider")
    if not provider: await message.answer("Сначала выберите технологию для генерации."); return

    prompt = message.text.strip()
    logging.info(f"User {user_id} requested an image with provider {provider}.")
    await message.answer("🎨 Создаю изображение... Это может занять до минуты."); await message.chat.do("upload_photo")
    start_time = time.time()

    try:
        image_url = ""
        caption = ""

        if provider == "openai":
            response = await openai_client.images.generate(model="dall-e-3", prompt=prompt, n=1, size="1024x1024")
            image_url = response.data[0].url
            caption = f"Изображение от DALL-E 3:\n«{prompt}»"

        elif provider == "grok":
            def _generate_grok_image():
                response = xai_client.image.sample(model="grok-2-image-1212", prompt=prompt, image_format="url")
                return response.url
            image_url = await asyncio.to_thread(_generate_grok_image)
            caption = f"Изображение от Grok Image:\n«{prompt}»"
        
        duration = time.time() - start_time
        logging.info(f"SUCCESS image generation for user_id: {user_id}. Provider: {provider}. Duration: {duration:.2f}s")
        await message.answer_photo(photo=image_url, caption=caption)

    except Exception as e:
        duration = time.time() - start_time
        logging.exception(f"ERROR during image generation for user_id: {user_id}. Provider: {provider}")
        await message.answer(f"❌ *Ошибка при генерации изображения.*\n\n_{str(e)}_", parse_mode="Markdown")

# === Основная точка входа ===
async def main():
    if not TELEGRAM_TOKEN: logging.error("Токен бота не найден!"); return
    logging.info("🤖 Бот запущен")
    await bot.set_my_commands([
        types.BotCommand(command="start", description="Перезапустить бота и выбрать режим"),
        types.BotCommand(command="reset", description="Перезапустить бота и выбрать режим")
    ])
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
