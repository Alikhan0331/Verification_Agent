"""Тонкий слой над LLM: «дай промпт и схему — получи объект pydantic».

Провайдеры: gemini (по умолчанию, есть бесплатный тариф) и anthropic.
Добавить свой — реализовать `_call(system, user, schema) -> str | dict`.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class LLMClient:
    provider = "base"

    def __init__(self, model: str, min_interval: float = 0.0, max_retries: int = 5):
        self.model = model
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._last_call = 0.0

    # --- реализуется в наследниках ---
    def _call(self, system: str, user: str, schema: type[BaseModel]) -> str | dict:
        raise NotImplementedError

    # --- общее ---
    def generate(self, system: str, user: str, schema: type[T], label: str = "") -> T:
        """Вызов модели с ретраями (429/5xx) и повтором при невалидном JSON."""
        hint = ""
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                self._last_call = time.monotonic()
                self.calls += 1
                raw = self._call(system, user + hint, schema)
                data = raw if isinstance(raw, dict) else json.loads(_strip_fence(raw))
                return schema.model_validate(data)
            except (ValidationError, json.JSONDecodeError) as e:
                last_err = e
                _log(f"  [{label}] ответ не прошёл валидацию (попытка {attempt}): {str(e)[:200]}")
                hint = ("\n\nПРЕДЫДУЩИЙ ОТВЕТ НЕ ПРОШЁЛ ВАЛИДАЦИЮ JSON-схемы. Верни строго по схеме, "
                        f"все поля обязательны. Ошибка: {str(e)[:400]}")
            except Exception as e:  # сетевые/квотные ошибки провайдера
                last_err = e
                s = str(e)
                retryable = any(t in s for t in ("429", "RESOURCE_EXHAUSTED", "500", "502", "503", "504",
                                                  "UNAVAILABLE", "overloaded", "timeout", "Timeout"))
                if not retryable or attempt == self.max_retries:
                    raise LLMError(f"Ошибка провайдера {self.provider}: {s[:500]}") from e
                delay = min(5 * 2 ** (attempt - 1), 90)
                _log(f"  [{label}] {s[:120]} — повтор через {delay} с")
                time.sleep(delay)
        raise LLMError(f"Не удалось получить валидный ответ ({label}): {last_err}")


def _strip_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s


class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(self, model: str, api_key: str, **kw):
        super().__init__(model, **kw)
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:  # pragma: no cover
            raise LLMError("Нужен пакет google-genai (pip install -r requirements.txt)") from e
        self._types = types
        self._client = genai.Client(api_key=api_key)

    def _call(self, system, user, schema):
        t = self._types
        resp = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=t.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.1,
                max_output_tokens=32000,
            ),
        )
        usage = getattr(resp, "usage_metadata", None)
        if usage:
            self.input_tokens += getattr(usage, "prompt_token_count", 0) or 0
            self.output_tokens += getattr(usage, "candidates_token_count", 0) or 0
        if not resp.text:
            raise json.JSONDecodeError("пустой ответ модели", "", 0)
        return resp.text


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str, api_key: str, **kw):
        super().__init__(model, **kw)
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise LLMError("Нужен пакет anthropic (pip install -r requirements.txt)") from e
        self._client = anthropic.Anthropic(api_key=api_key)

    def _call(self, system, user, schema):
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=16000,
            temperature=0.1,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[{
                "name": "submit_result",
                "description": "Вернуть результат проверки строго по схеме.",
                "input_schema": schema.model_json_schema(),
            }],
            tool_choice={"type": "tool", "name": "submit_result"},
        )
        self.input_tokens += msg.usage.input_tokens
        self.output_tokens += msg.usage.output_tokens
        for block in msg.content:
            if block.type == "tool_use":
                return block.input
        raise json.JSONDecodeError("модель не вернула tool_use", "", 0)


def make_client(provider: str | None = None, model: str | None = None, min_interval: float | None = None) -> LLMClient:
    provider = (provider or os.getenv("LLM_PROVIDER") or "gemini").lower()
    if min_interval is None:
        min_interval = float(os.getenv("LLM_MIN_INTERVAL", "0") or 0)
    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise LLMError("Не задан GEMINI_API_KEY (см. README и .env.example)")
        model = model or os.getenv("GEMINI_MODEL")
        if not model:
            raise LLMError("Не задана модель: GEMINI_MODEL в .env или --model")
        return GeminiClient(model, key, min_interval=min_interval)
    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("Не задан ANTHROPIC_API_KEY (см. README и .env.example)")
        model = model or os.getenv("ANTHROPIC_MODEL")
        if not model:
            raise LLMError("Не задана модель: ANTHROPIC_MODEL в .env или --model")
        return AnthropicClient(model, key, min_interval=min_interval)
    raise LLMError(f"Неизвестный провайдер {provider!r} (gemini | anthropic)")
