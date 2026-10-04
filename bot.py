"""Telegram-бот с контекстом диалога (aiogram 3 + OpenAI/GenAPI)."""
import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import Message

import config
from api_client import create_client
from context_manager import context_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("bot")

dp = Dispatcher()
llm = None  # создаётся в main() после проверки конфига

TG_LIMIT = 4096
HELP_TEXT = (
    "Просто пишите сообщения — я помню контекст диалога.\n\n"
    "Команды:\n"
    "/reset или «очистить контекст» — забыть историю\n"
    "/settings — текущие параметры\n"
    "/model <имя> — сменить модель (например /model gpt-4.1-mini)\n"
    "/temp <0..2> — temperature\n"
    "/max <число> — max_tokens\n"
    "/help — эта справка"
)


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(f"Привет, {message.from_user.first_name}! Я бот с памятью диалога.\n\n{HELP_TEXT}")


@dp.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(HELP_TEXT)


@dp.message(Command("reset"))
@dp.message(F.text.lower().in_({"очистить контекст", "очистить", "сброс"}))
async def cmd_reset(message: Message):
    context_manager.reset(message.from_user.id)
    logger.info("user=%s context reset", message.from_user.id)
    await message.answer("🧹 Контекст очищен. Начинаем с чистого листа.")


@dp.message(Command("settings"))
async def cmd_settings(message: Message):
    uid = message.from_user.id
    s = context_manager.settings(uid)
    await message.answer(
        f"Провайдер: {config.API_PROVIDER}\n"
        f"Модель: {s.model}\n"
        f"temperature: {s.temperature}\n"
        f"max_tokens: {s.max_tokens}\n"
        f"Сообщений в контексте: {context_manager.history_len(uid)}/{config.MAX_CONTEXT_MESSAGES}"
    )


@dp.message(Command("model"))
async def cmd_model(message: Message, command: CommandObject):
    if not command.args:
        await message.answer("Укажите модель: /model gpt-4o-mini")
        return
    context_manager.settings(message.from_user.id).model = command.args.strip()
    logger.info("user=%s model=%s", message.from_user.id, command.args.strip())
    await message.answer(f"✅ Модель: {command.args.strip()}")


@dp.message(Command("temp"))
async def cmd_temp(message: Message, command: CommandObject):
    try:
        value = float((command.args or "").replace(",", "."))
        if not 0.0 <= value <= 2.0:
            raise ValueError
    except ValueError:
        await message.answer("temperature — число от 0 до 2, например /temp 0.2")
        return
    context_manager.settings(message.from_user.id).temperature = value
    logger.info("user=%s temperature=%s", message.from_user.id, value)
    await message.answer(f"✅ temperature = {value}")


@dp.message(Command("max"))
async def cmd_max(message: Message, command: CommandObject):
    try:
        value = int(command.args or "")
        if not 1 <= value <= 16000:
            raise ValueError
    except ValueError:
        await message.answer("max_tokens — целое число от 1 до 16000, например /max 150")
        return
    context_manager.settings(message.from_user.id).max_tokens = value
    logger.info("user=%s max_tokens=%s", message.from_user.id, value)
    await message.answer(f"✅ max_tokens = {value}")


@dp.message(F.text)
async def handle_text(message: Message):
    uid = message.from_user.id
    text = message.text
    s = context_manager.settings(uid)
    messages = context_manager.build_messages(uid, text)
    logger.info("user=%s message len=%d context=%d", uid, len(text), context_manager.history_len(uid))

    await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    try:
        result = await llm.chat(messages, model=s.model, temperature=s.temperature, max_tokens=s.max_tokens)
    except Exception as e:
        logger.exception("user=%s LLM request failed: %s", uid, e)
        await message.answer(f"⚠️ Ошибка при обращении к API: {type(e).__name__}. Подробности в логе.")
        return

    answer = result.text or "(пустой ответ — возможно, не хватило max_tokens)"
    context_manager.add_exchange(uid, text, answer)

    cost = f" ≈ ${result.cost_usd:.6f}" if result.cost_usd is not None else ""
    footer = (f"\n\n— {result.model} · t={s.temperature} · max={s.max_tokens} · "
              f"токены {result.prompt_tokens}+{result.completion_tokens}={result.total_tokens}{cost}")
    full = answer + footer
    for i in range(0, len(full), TG_LIMIT):
        await message.answer(full[i:i + TG_LIMIT])


@dp.message()
async def handle_other(message: Message):
    await message.answer("Я понимаю только текстовые сообщения.")


async def main():
    global llm
    config.validate()
    llm = create_client()
    bot = Bot(token=config.BOT_TOKEN)
    logger.info("Бот запущен: provider=%s model=%s", config.API_PROVIDER, config.DEFAULT_MODEL)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Бот остановлен")
