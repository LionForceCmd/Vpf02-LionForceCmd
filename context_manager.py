"""Хранение контекста диалога и настроек пользователя в оперативной памяти (dict)."""
from collections import deque
from dataclasses import dataclass, field

import config


@dataclass
class UserSettings:
    model: str = config.DEFAULT_MODEL
    temperature: float = config.DEFAULT_TEMPERATURE
    max_tokens: int = config.DEFAULT_MAX_TOKENS


@dataclass
class UserSession:
    history: deque = field(default_factory=lambda: deque(maxlen=config.MAX_CONTEXT_MESSAGES))
    settings: UserSettings = field(default_factory=UserSettings)


class ContextManager:
    def __init__(self, system_prompt: str = config.SYSTEM_PROMPT):
        self.system_prompt = system_prompt
        self._sessions: dict[int, UserSession] = {}

    def _session(self, user_id: int) -> UserSession:
        if user_id not in self._sessions:
            self._sessions[user_id] = UserSession()
        return self._sessions[user_id]

    def build_messages(self, user_id: int, user_text: str) -> list[dict]:
        """System prompt + история + новое сообщение пользователя."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._session(user_id).history)
        messages.append({"role": "user", "content": user_text})
        return messages

    def add_exchange(self, user_id: int, user_text: str, answer: str) -> None:
        """Сохраняем пару вопрос/ответ только после успешного ответа модели."""
        history = self._session(user_id).history
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": answer})

    def reset(self, user_id: int) -> None:
        self._session(user_id).history.clear()

    def history_len(self, user_id: int) -> int:
        return len(self._session(user_id).history)

    def settings(self, user_id: int) -> UserSettings:
        return self._session(user_id).settings


context_manager = ContextManager()
