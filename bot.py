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

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# === Состояние пользователей ===
user_state = {}

# === Главное меню ===
def main_menu():
    buttons = [
        [InlineKeyboardButton(text="💭 ChatGPT (OpenAI)", callback_data="provider_openai")],
        [InlineKeyboardButton(text="🧠 Grok (xAI)", callback_data="provider_grok")],
        [InlineKeyboardButton(text="⚡ Gemini (Google)", callback_data="provider_gemini")],
        [InlineKeyboardButton(text="🌐 Сайт", url="https://neurozone.pro/")],
        [InlineKeyboardButton(text="🔒 Политика конфиденциальности", url="https://neurozone.pro/privacy")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


# === Модели с описаниями ===
# Ваш оригинальный, рабочий список моделей OpenAI, за исключением gpt-5-pro.
openai_models = {
    "GPT-5": {
        "id": "gpt-5",
        "desc": "Лучшая модель для кода и агентных задач в разных областях."
    },
    "GPT-5 mini": {
        "id": "gpt-5-mini",
        "desc": "Более быстрая, экономичная версия GPT-5 для четко определенных задач."
    },
    "GPT-5 nano": {
        "id": "gpt-5-nano",
        "desc": "Самая быстрая и экономичная версия GPT-5."
    },
    "GPT-4.1": {
        "id": "gpt-4.1",
        "desc": "Самая умная модель без рассуждений."
    }
}

# Ваш оригинальный список моделей Grok
grok_models = {
    "Grok-code-fast-1": {
        "id": "grok-code-fast-1",
        "desc": "Быстрая и экономичная модель для рассуждений, которая отлично справляется с агентным кодированием."
    },
    "Grok-4-fast-reasoning": {
        "id": "grok-4-fast-reasoning",
        "desc": "Последнее достижение в области экономичных моделей для рассуждений."
    },
    "Grok-4-fast-non-reasoning": {
        "id": "grok-4-fast-non-reasoning",
        "desc": "Последнее достижение в области экономичных моделей для рассуждений."
    }
}

# ИСПРАВЛЕНО: Оставлены только подтвержденные рабочие модели
gemini_models = {
    "Gemini 2.5 Flash": {
        "id": "gemini-2.5-flash",
        "desc": "Лучшая модель по соотношению цены и производительности, предлагающая разносторонние возможности для крупномасштабной обработки и задач с низкой задержкой."
    },
    "Gemini 2.5 Flash-Lite": {
        "id": "gemini-2.5-flash-lite",
        "desc": "Самая быстрая flash-модель, оптимизированная для экономии и высокой пропускной способности."
    }
}


# === /start ===
@dp.message(Command("start"))
async def start_command(message: Message):
    user_state[message.from_user.id] = {"provider": None, "model": None}
    await message.answer(
        "👋 Привет! Это бот *NeuroZone*.\n\nВыбери нейросеть, с которой хочешь работать:",
        parse_mode="Markdown",
        reply_markup=main_menu()
    )

# === Выбор провайдера ===
@dp.callback_query(lambda c: c.data.startswith("provider_"))
async def provider_selection(callback_query: types.CallbackQuery):
    provider = callback_query.data.split("_")[1]
    user_id = callback_query.from_user.id
    user_state[user_id] = {"provider": provider, "model": None}

    buttons = []
    text_parts = []
    models_dict = {}
    header = ""

    if provider == "openai":
        models_dict = openai_models
        header = "🔹 *Выбран ChatGPT (OpenAI)*\n\n"
    elif provider == "grok":
        models_dict = grok_models
        header = "🧠 *Выбран Grok (xAI)*\n\n"
    elif provider == "gemini":
        models_dict = gemini_models
        header = "⚡ *Выбран Gemini (Google)*\n\n"

    text_parts.append(header)
    text_parts.append("Выберите модель из списка ниже:")

    for name, data in models_dict.items():
        text_parts.append(f"\n\n*{name}*")
        text_parts.append(f"_{data['desc']}_")
        buttons.append([InlineKeyboardButton(text=name, callback_data=f"model_{data['id']}")])
    
    buttons.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_main")])
    
    await callback_query.message.edit_text(
        "\n".join(text_parts), 
        parse_mode="Markdown", 
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons)
    )

# === Кнопка "Назад" ===
@dp.callback_query(lambda c: c.data == "back_to_main")
async def back_to_main_menu(callback_query: types.CallbackQuery):
    user_state[callback_query.from_user.id] = {"provider": None, "model": None}
    await callback_query.message.edit_text(
        "👋 Привет! Это бот *NeuroZone*.\n\nВыбери нейросеть, с которой хочешь работать:",
        parse_mode="Markdown",
        reply_markup=main_menu()
    )


# === Выбор модели ===
@dp.callback_query(lambda c: c.data.startswith("model_"))
async def model_selection(callback_query: types.CallbackQuery):
    model_id = callback_query.data.split("_", 1)[1]
    user_id = callback_query.from_user.id
    if user_id not in user_state or not user_state[user_id].get("provider"):
        await callback_query.answer("Пожалуйста, сначала выбери провайдера.", show_alert=True)
        return

    user_state[user_id]["model"] = model_id
    provider = user_state[user_id]['provider'].capitalize()

    await callback_query.message.edit_text(
        f"✅ Провайдер: *{provider}*\n"
        f"✅ Модель: *{model_id}*\n\n"
        f"Теперь отправь свой вопрос. Для смены нейросети введи /start",
        parse_mode="Markdown"
    )

# === Обработка текстовых сообщений ===
@dp.message()
async def handle_message(message: Message):
    user_id = message.from_user.id
    if user_id not in user_state or not user_state[user_id].get("model"):
        await message.answer("⚙️ Сначала выбери провайдера и модель через /start")
        return

    provider = user_state[user_id]["provider"]
    model = user_state[user_id]["model"]
    user_input = message.text.strip()

    await message.chat.do("typing")
    
    answer = ""

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            if provider == "openai":
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json={"model": model, "messages": [{"role": "user", "content": user_input}]}
                )
                response.raise_for_status()
                data = response.json()
                answer = data["choices"][0]["message"]["content"]

            elif provider == "grok":
                response = await client.post(
                    f"{GROK_API_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {GROK_API_KEY}"},
                    json={"model": model, "messages": [{"role": "user", "content": user_input}]}
                )
                response.raise_for_status()
                data = response.json()
                answer = data["choices"][0]["message"]["content"]

            elif provider == "gemini":
                # ИСПРАВЛЕНО: Возвращаемся к v1beta, так как рабочие модели Flash используют его.
                gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
                response = await client.post(
                    gemini_url,
                    headers={"Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": user_input}]}]}
                )
                response.raise_for_status()
                data = response.json()
                answer = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Не удалось извлечь ответ из API Gemini.")

            else:
                answer = "Неизвестный провайдер."

    except httpx.HTTPStatusError as http_err:
        logging.error(f"Ошибка HTTP: {http_err.response.status_code} - {http_err.response.text}")
        error_text = f"❌ *Ошибка API ({http_err.response.status_code})*"
        details = ""
        try:
            details = http_err.response.json().get("error", {}).get("message", "")
        except Exception:
            pass 
        
        if details:
            error_text += f"\n_{details}_"
        
        if http_err.response.status_code in [403, 401]:
             error_text += "\n\n*Причина:* Проверьте ваш API-ключ или права доступа к данной модели."
        elif http_err.response.status_code == 404:
            error_text += f"\n\n*Причина:* Модель `{model}` не найдена. Возможно, она не существует или у вас нет к ней доступа через используемый API."
        elif http_err.response.status_code == 503:
            error_text += "\n\n*Причина:* Сервис временно недоступен или перегружен. Попробуйте позже."

        answer = error_text
            
    except Exception as e:
        logging.exception("Непредвиденная ошибка")
        answer = f"❌ Произошла непредвиденная ошибка: {str(e)}"

    await message.answer(answer, parse_mode="Markdown")


# === Основная точка входа ===
async def main():
    if not TELEGRAM_TOKEN:
        logging.error("Токен бота не найден!")
        return
        
    logging.info("🤖 Бот запущен")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
