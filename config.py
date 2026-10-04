"""Настройки бота: токены из .env и параметры модели по умолчанию."""
import os

from dotenv import load_dotenv

load_dotenv()

# --- Telegram ---
BOT_TOKEN = os.getenv("BOT_TOKEN", "")

# --- Провайдер API: "openai" (OpenAI напрямую или ProxyAPI) или "genapi" ---
API_PROVIDER = os.getenv("API_PROVIDER", "openai").lower()

# OpenAI / ProxyAPI (официальная библиотека openai)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Пусто -> api.openai.com; для ProxyAPI: https://api.proxyapi.ru/openai/v1
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None

# GenAPI (HTTP-запросы)
GENAPI_KEY = os.getenv("GENAPI_KEY", "")
GENAPI_URL = os.getenv("GENAPI_URL", "https://api.gen-api.ru/api/v1/networks")

# --- Параметры модели по умолчанию (каждый пользователь может поменять командами) ---
DEFAULT_MODEL = os.getenv("MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "Ты — дружелюбный помощник в Telegram. Отвечай по-русски, по делу.",
)

# Сколько последних сообщений (user + assistant) хранить в контексте
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "20"))

# Ориентировочные цены, USD за 1 млн токенов: (input, output).
# Сверяйте с https://platform.openai.com/docs/pricing — цены меняются.
PRICES_PER_1M = {
    "gpt-3.5-turbo": (0.50, 1.50),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
}

LOG_FILE = os.getenv("LOG_FILE", "bot.log")


def validate() -> None:
    """Проверяет, что нужные ключи заданы. Бросает RuntimeError, если нет."""
    missing = []
    if not BOT_TOKEN:
        missing.append("BOT_TOKEN")
    if API_PROVIDER == "openai" and not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if API_PROVIDER == "genapi" and not GENAPI_KEY:
        missing.append("GENAPI_KEY")
    if API_PROVIDER not in ("openai", "genapi"):
        raise RuntimeError(f"Неизвестный API_PROVIDER={API_PROVIDER!r} (нужно openai или genapi)")
    if missing:
        raise RuntimeError(f"Не заданы переменные в .env: {', '.join(missing)}")
