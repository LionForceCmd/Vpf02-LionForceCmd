"""Прогоняет один и тот же запрос с разными model/temperature/max_tokens
и печатает Markdown-таблицу для README. Ответы сохраняет в experiments.md.

Запуск: python run_experiments.py
"""
import asyncio
import logging
import sys

import config
from api_client import create_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(config.LOG_FILE, encoding="utf-8")])

PROMPT = "Придумай слоган и короткое описание для кофейни у моря."

# (модель, temperature, max_tokens) — для GenAPI замените имена на ID сетей, например gpt-4-1-mini
RUNS = [
    ("gpt-4o-mini", 0.2, 60),
    ("gpt-4o-mini", 0.2, 300),
    ("gpt-4o-mini", 1.0, 300),
    ("gpt-4o-mini", 1.5, 300),
    ("gpt-4.1-mini", 0.2, 300),
    ("gpt-4.1-mini", 1.0, 300),
]


async def main():
    if not (config.GENAPI_KEY if config.API_PROVIDER == "genapi" else config.OPENAI_API_KEY):
        sys.exit("Не задан ключ API в .env")
    llm = create_client()
    rows, details = [], []
    for n, (model, temp, max_t) in enumerate(RUNS, 1):
        messages = [{"role": "system", "content": config.SYSTEM_PROMPT},
                    {"role": "user", "content": PROMPT}]
        try:
            r = await llm.chat(messages, model=model, temperature=temp, max_tokens=max_t)
        except Exception as e:
            logging.exception("Прогон %d упал", n)
            rows.append(f"| {model} | {temp} | {max_t} | {n} | ошибка: {type(e).__name__} | — | — | — |")
            continue
        cost = f"${r.cost_usd:.6f}" if r.cost_usd is not None else "см. кабинет"
        rows.append(f"| {model} | {temp} | {max_t} | {n} | _опишите эффект_ | "
                    f"{r.prompt_tokens}+{r.completion_tokens}={r.total_tokens} | {cost} | `{r.request_id}` |")
        details.append(f"### Прогон {n}: {model}, t={temp}, max_tokens={max_t}\n\n{r.text}\n")

    table = ("| Модель | temperature | max_tokens | № прогона | Эффект (сжатость/креатив) "
             "| Токены (in+out=total) | Стоимость | ID запроса |\n"
             "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows))
    with open("experiments.md", "w", encoding="utf-8") as f:
        f.write(f"# Результаты прогонов\n\nЗапрос: «{PROMPT}»\n\n{table}\n\n## Ответы\n\n" + "\n".join(details))
    print("\n" + table + "\n\nОтветы сохранены в experiments.md")


if __name__ == "__main__":
    asyncio.run(main())
