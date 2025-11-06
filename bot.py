import os
import sys
import logging
import asyncio
import httpx
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, Message
from dotenv import load_dotenv

# === Настройки кодировки для Windows ===
sys.stdout.reconfigure(encoding='utf-8')

# === Логи ===
logging.basicConfig(level=logging.INFO)

# === Загрузка .env ===
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN") or os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
GROK_API_BASE = os.getenv("GROK_API_BASE", "https://api.x.ai/v1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PROXY_URL = os.getenv("PROXY_URL")

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# === Состояние пользователей и история ===
user_state = {}

# ИЗМЕНЕНО: Константа для ограничения размера истории (10 реплик = 5 пар вопрос-ответ)
MAX_HISTORY_LENGTH = 10

# === Главное меню и Модели ===
# (Этот блок кода не изменился, поэтому для краткости я его скрою)
def main_menu():
    buttons = [
        [InlineKeyboardButton(text="💭 ChatGPT (OpenAI)", callback_data="provider_openai")],
        [InlineKeyboardButton(text="🧠 Grok (xAI)", callback_data="provider_grok")],
        [InlineKeyboardButton(text="⚡ Gemini (Google)", callback_data="provider_gemini")],
        [InlineKeyboardButton(text="🌐 Сайт", url="https://neurozone.pro/")],
        [InlineKeyboardButton(text="🔒 Политика конфиденциальности", url="https://neurozone.pro/privacy")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)
openai_models = { "GPT-5": {"id": "gpt-5", "desc": "Лучшая модель..."}, "GPT-5 mini": {"id": "gpt-5-mini", "desc": "Более быстрая..."}, "GPT-5 nano": {"id": "gpt-5-nano", "desc": "Самая быстрая..."}, "GPT-4.1": {"id": "gpt-4.1", "desc": "Самая умная..."}}
grok_models = {"Grok-code-fast-1": {"id": "grok-code-fast-1", "desc": "Быстрая..."}, "Grok-4-fast-reasoning": {"id": "grok-4-fast-reasoning", "desc": "Последнее..."}, "Grok-4-fast-non-reasoning": {"id": "grok-4-fast-non-reasoning", "desc": "Последнее..."}}
gemini_models = {"Gemini 2.5 Flash": {"id": "gemini-2.5-flash", "desc": "Лучшая..."},"Gemini 2.5 Flash-Lite": {"id": "gemini-2.5-flash-lite", "desc": "Самая быстрая..."}}
# === Конец скрытого блока ===


# ИЗМЕНЕНО: Функция для сброса состояния и истории
def reset_user_state(user_id):
    user_state[user_id] = {"provider": None, "model": None, "history": []}


# === /start ===
@dp.message(Command("start"))
async def start_command(message: Message):
    reset_user_state(message.from_user.id) # Сбрасываем состояние при старте
    await message.answer(
        "👋 Привет! Это бот *NeuroZone*.\n\n"
        "Я запоминаю контекст нашего разговора. Чтобы начать новый диалог, используй команду /reset.\n\n"
        "Выбери нейросеть, с которой хочешь работать:",
        parse_mode="Markdown",
        reply_markup=main_menu()
    )

# ИЗМЕНЕНО: Новая команда /reset для сброса контекста
@dp.message(Command("reset"))
async def reset_command(message: Message):
    reset_user_state(message.from_user.id)
    await message.answer(
        "✅ Контекст разговора сброшен. Начинаем новый диалог!\n\nВыбери нейросеть:",
        parse_mode="Markdown",
        reply_markup=main_menu()
    )


# === Выбор провайдера ===
@dp.callback_query(lambda c: c.data.startswith("provider_"))
async def provider_selection(callback_query: types.CallbackQuery):
    provider = callback_query.data.split("_")[1]
    user_id = callback_query.from_user.id
    # ИЗМЕНЕНО: Убеждаемся, что история существует
    if user_id not in user_state:
        reset_user_state(user_id)
    user_state[user_id]["provider"] = provider
    
    # ... остальной код выбора провайдера не изменился ...
    buttons, text_parts, models_dict, header = [], [], {}, ""
    if provider == "openai": models_dict, header = openai_models, "🔹 *Выбран ChatGPT (OpenAI)*\n\n"
    elif provider == "grok": models_dict, header = grok_models, "🧠 *Выбран Grok (xAI)*\n\n"
    elif provider == "gemini": models_dict, header = gemini_models, "⚡ *Выбран Gemini (Google)*\n\n"
    text_parts.append(header + "Выберите модель из списка ниже:")
    for name, data in models_dict.items():
        text_parts.append(f"\n\n*{name}*\n_{data['desc']}_")
        buttons.append([InlineKeyboardButton(text=name, callback_data=f"model_{data['id']}")])
    buttons.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_main")])
    await callback_query.message.edit_text("\n".join(text_parts), parse_mode="Markdown", reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons))


# ... обработчики back_to_main и model_selection не изменились ...
@dp.callback_query(lambda c: c.data == "back_to_main")
async def back_to_main_menu(callback_query: types.CallbackQuery):
    reset_user_state(callback_query.from_user.id)
    await callback_query.message.edit_text("👋 Привет! ...", parse_mode="Markdown", reply_markup=main_menu())
@dp.callback_query(lambda c: c.data.startswith("model_"))
async def model_selection(callback_query: types.CallbackQuery):
    model_id = callback_query.data.split("_", 1)[1]
    user_id = callback_query.from_user.id
    if user_id not in user_state or not user_state[user_id].get("provider"): await callback_query.answer("Пожалуйста, сначала выбери провайдера.", show_alert=True); return
    user_state[user_id]["model"] = model_id
    provider = user_state[user_id]['provider'].capitalize()
    await callback_query.message.edit_text(f"✅ Провайдер: *{provider}*\n✅ Модель: *{model_id}*\n\nТеперь отправь свой вопрос.", parse_mode="Markdown")


# === Обработка текстовых сообщений (КЛЮЧЕВЫЕ ИЗМЕНЕНИЯ) ===
@dp.message()
async def handle_message(message: Message):
    user_id = message.from_user.id
    if user_id not in user_state or not user_state[user_id].get("model"):
        await message.answer("⚙️ Сначала выбери провайдера и модель через /start или /reset")
        return

    provider = user_state[user_id]["provider"]
    model = user_state[user_id]["model"]
    user_input = message.text.strip()
    
    # ИЗМЕНЕНО: Получаем историю пользователя
    history = user_state[user_id].get("history", [])

    # ИЗМЕНЕНО: Ограничиваем длину истории, чтобы избежать переполнения
    if len(history) > MAX_HISTORY_LENGTH:
        history = history[-MAX_HISTORY_LENGTH:]

    await message.chat.do("typing")
    
    answer = ""
    proxies = {"all://": PROXY_URL} if PROXY_URL else None
    
    try:
        async with httpx.AsyncClient(timeout=90.0, proxies=proxies) as client:
            if provider == "openai" or provider == "grok":
                # ИЗМЕНЕНО: Добавляем текущий вопрос в историю
                history.append({"role": "user", "content": user_input})
                
                api_url = "https://api.openai.com/v1/chat/completions" if provider == "openai" else f"{GROK_API_BASE}/chat/completions"
                api_key = OPENAI_API_KEY if provider == "openai" else GROK_API_KEY
                
                response = await client.post(
                    api_url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    # ИЗМЕНЕНО: Отправляем всю историю
                    json={"model": model, "messages": history}
                )
                response.raise_for_status()
                data = response.json()
                answer = data["choices"][0]["message"]["content"]
                # ИЗМЕНЕНО: Добавляем ответ бота в историю
                history.append({"role": "assistant", "content": answer})

            elif provider == "gemini":
                # Gemini имеет другой формат истории, конвертируем его
                gemini_history = []
                for msg in history:
                    role = "user" if msg["role"] == "user" else "model"
                    gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})
                
                # Добавляем текущий вопрос
                gemini_history.append({"role": "user", "parts": [{"text": user_input}]})

                gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
                response = await client.post(
                    gemini_url,
                    headers={"Content-Type": "application/json"},
                    json={"contents": gemini_history} # Отправляем сконвертированную историю
                )
                response.raise_for_status()
                data = response.json()
                answer = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Не удалось извлечь ответ.")
                
                # ИЗМЕНЕНО: Сохраняем историю в нашем универсальном формате
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": answer})

            else:
                answer = "Неизвестный провайдер."
    
    # ... блок except остался без изменений ...
    except httpx.HTTPStatusError as http_err: logging.error(f"Ошибка HTTP: {http_err.response.status_code} - {http_err.response.text}"); answer = f"❌ *Ошибка API ({http_err.response.status_code})*"
    except Exception as e: logging.exception("Непредвиденная ошибка"); answer = f"❌ Произошла непредвиденная ошибка: {str(e)}"

    # ИЗМЕНЕНО: Сохраняем обновленную историю
    user_state[user_id]["history"] = history
    
    await message.answer(answer, parse_mode="Markdown")


# === Основная точка входа ===
async def main():
    if not TELEGRAM_TOKEN: logging.error("Токен бота не найден!"); return
    logging.info("🤖 Бот запущен")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
