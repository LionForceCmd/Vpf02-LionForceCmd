"""Клиент LLM: OpenAI (официальная библиотека, в т.ч. через ProxyAPI) или GenAPI (HTTP)."""
import json
import logging
from dataclasses import dataclass

import httpx
from openai import AsyncOpenAI

import config

logger = logging.getLogger("api_client")


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    request_id: str | None
    cost_usd: float | None


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Ищем цену по самому длинному совпадающему префиксу имени модели."""
    key = max((k for k in config.PRICES_PER_1M if model.startswith(k)), key=len, default=None)
    if key is None:
        return None
    price_in, price_out = config.PRICES_PER_1M[key]
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


def _is_reasoning_model(model: str) -> bool:
    # gpt-5* и o-серия не принимают temperature и max_tokens (нужен max_completion_tokens)
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def _build_params(model: str, messages: list[dict], temperature: float, max_tokens: int) -> dict:
    params = {"model": model, "messages": messages}
    if _is_reasoning_model(model):
        params["max_completion_tokens"] = max_tokens
        if temperature != 1.0:
            logger.warning("Модель %s не поддерживает temperature — параметр не отправляется", model)
    else:
        params["temperature"] = temperature
        params["max_tokens"] = max_tokens
    return params


def _log_params(provider: str, params: dict) -> None:
    """Логируем отправляемые параметры; из messages — только роли и длины, без полного текста."""
    shown = {k: v for k, v in params.items() if k != "messages"}
    shown["messages"] = [f"{m['role']}:{len(m['content'])}ch" for m in params["messages"]]
    logger.info("-> %s request: %s", provider, json.dumps(shown, ensure_ascii=False))


class OpenAIClient:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            timeout=config.REQUEST_TIMEOUT,
        )

    async def chat(self, messages: list[dict], model: str, temperature: float, max_tokens: int) -> LLMResult:
        params = _build_params(model, messages, temperature, max_tokens)
        _log_params("openai", params)
        response = await self.client.chat.completions.create(**params)

        usage = response.usage
        prompt_t = usage.prompt_tokens if usage else 0
        completion_t = usage.completion_tokens if usage else 0
        choice = response.choices[0]
        result = LLMResult(
            text=(choice.message.content or "").strip(),
            model=response.model,
            prompt_tokens=prompt_t,
            completion_tokens=completion_t,
            total_tokens=prompt_t + completion_t,
            request_id=response.id,
            cost_usd=estimate_cost(response.model, prompt_t, completion_t),
        )
        logger.info(
            "<- openai response id=%s model=%s finish=%s tokens=%d/%d/%d cost≈$%s",
            result.request_id, result.model, choice.finish_reason,
            prompt_t, completion_t, result.total_tokens,
            f"{result.cost_usd:.6f}" if result.cost_usd is not None else "?",
        )
        return result


class GenAPIClient:
    """GenAPI: POST {GENAPI_URL}/{network_id} с is_sync=true. network_id = MODEL (например gpt-4-1-mini)."""

    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {config.GENAPI_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def chat(self, messages: list[dict], model: str, temperature: float, max_tokens: int) -> LLMResult:
        params = {"messages": messages, "temperature": temperature,
                  "max_tokens": max_tokens, "is_sync": True}
        _log_params("genapi", {"model": model, **params})
        url = f"{config.GENAPI_URL.rstrip('/')}/{model}"
        async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT) as client:
            resp = await client.post(url, headers=self.headers, json=params)
        if resp.status_code != 200:
            raise RuntimeError(f"GenAPI HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        logger.debug("<- genapi raw: %s", json.dumps(data, ensure_ascii=False)[:2000])

        text = self._extract_text(data)
        usage = self._find(data, "usage") or {}
        prompt_t = int(usage.get("prompt_tokens", 0))
        completion_t = int(usage.get("completion_tokens", 0))
        cost = data.get("cost")  # GenAPI возвращает стоимость в рублях, если есть
        result = LLMResult(
            text=text,
            model=model,
            prompt_tokens=prompt_t,
            completion_tokens=completion_t,
            total_tokens=int(usage.get("total_tokens", prompt_t + completion_t)),
            request_id=str(data.get("request_id") or data.get("id") or ""),
            cost_usd=None,
        )
        logger.info("<- genapi response request_id=%s tokens=%d/%d/%d cost=%s",
                    result.request_id, prompt_t, completion_t, result.total_tokens, cost)
        return result

    @classmethod
    def _find(cls, obj, key):
        """Рекурсивный поиск ключа — формат ответа GenAPI отличается у разных моделей."""
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            for v in obj.values():
                found = cls._find(v, key)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for v in obj:
                found = cls._find(v, key)
                if found is not None:
                    return found
        return None

    @classmethod
    def _extract_text(cls, data: dict) -> str:
        message = cls._find(data, "message")
        if isinstance(message, dict) and message.get("content"):
            return message["content"].strip()
        content = cls._find(data, "content")
        if isinstance(content, str):
            return content.strip()
        raise RuntimeError(f"Не удалось разобрать ответ GenAPI: {json.dumps(data, ensure_ascii=False)[:500]}")


def create_client():
    return GenAPIClient() if config.API_PROVIDER == "genapi" else OpenAIClient()
