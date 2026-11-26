"""Тонкий слой над LLM: «дай промпт и схему — получи объект pydantic».

Провайдеры: gemini (по умолчанию, есть бесплатный тариф) и anthropic.
Добавить свой — реализовать `_call(system, user, schema) -> str | dict`.
"""

from __future__ import annotations

import json
import os
import sys
import threading
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

    def __init__(self, model: str, min_interval: float = 0.0, max_retries: int = 5,
                 fallbacks: list[str] | None = None):
        self.model = model
        self.models = [model] + [m for m in (fallbacks or []) if m and m != model]
        self.models_used = [model]      # какие модели реально отвечали в этом прогоне
        self._dead: set[str] = set()    # модели с исчерпанной суточной квотой или закрытые
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._last_call = 0.0
        self._lock = threading.Lock()  # агент вызывает модель из нескольких потоков

    # --- реализуется в наследниках ---
    def _call(self, system: str, user: str, schema: type[BaseModel], effort: str | None) -> str | dict:
        raise NotImplementedError

    def _throttle(self) -> None:
        """Выдерживает минимальный интервал между стартами вызовов (общий для всех потоков)."""
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            self.calls += 1

    def _add_usage(self, inp: int, out: int) -> None:
        with self._lock:
            self.input_tokens += inp or 0
            self.output_tokens += out or 0

    # --- общее ---
    def generate(self, system: str, user: str, schema: type[T], label: str = "",
                 effort: str | None = None) -> T:
        """Вызов модели с ретраями (429/5xx) и повтором при невалидном JSON.

        effort: "low" | "medium" | "high" — глубина размышлений модели, если провайдер это умеет.
        """
        hint = ""
        last_err: Exception | None = None
        attempt = 0
        overload_streak = 0
        while attempt < self.max_retries:
            attempt += 1
            used = self.model
            try:
                self._throttle()
                raw = self._call(system, user + hint, schema, effort)
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
                if _model_unusable(s):
                    # суточная квота исчерпана или модель закрыта — повторять бессмысленно, берём запасную
                    if self._switch_model(used, s):
                        attempt -= 1
                        continue
                    raise LLMError(
                        f"Суточная квота исчерпана (или модель закрыта, либо перегружена) у всех моделей: {', '.join(self.models)}. "
                        "Подождите сброса квоты (у Google — в полночь по тихоокеанскому времени), "
                        "добавьте запасные модели в GEMINI_FALLBACK_MODELS или используйте другой ключ.") from e
                retryable = _is_transient(e, s)
                if not retryable or attempt == self.max_retries:
                    raise LLMError(f"Ошибка провайдера {self.provider}: {s[:500]}") from e
                overloaded = "503" in s or "UNAVAILABLE" in s or "overloaded" in s
                overload_streak = overload_streak + 1 if overloaded else 0
                if overload_streak >= OVERLOAD_SWITCH_AFTER and self._switch_model(used, "перегрузка"):
                    overload_streak = 0      # модель перегружена надолго — не ждём, берём запасную
                    continue
                delay = min(5 * 2 ** (attempt - 1), 90)
                _log(f"  [{label}] {s[:120]} — повтор через {delay} с")
                time.sleep(delay)
        raise LLMError(f"Не удалось получить валидный ответ ({label}): {last_err}")

    def _switch_model(self, failed: str, reason: str) -> bool:
        """Переключает клиент на следующую живую модель. False — запасных не осталось."""
        with self._lock:
            if self.model != failed:      # другой поток уже переключил — просто повторяем вызов
                return True
            self._dead.add(failed)
            alive = [m for m in self.models if m not in self._dead]
            if not alive:
                return False
            self.model = alive[0]
            self.models_used.append(self.model)
            _log(f"  модель {failed} недоступна ({_short_reason(reason)}) — переключаюсь на {self.model}")
            return True


OVERLOAD_SWITCH_AFTER = 2      # столько подряд ответов 503 на одной модели — и переходим на запасную

_TRANSIENT_TOKENS = ("429", "RESOURCE_EXHAUSTED", "500", "502", "503", "504", "UNAVAILABLE", "overloaded",
                     "timeout", "Timeout", "timed out", "10053", "10054", "10060", "Connection", "connection",
                     "RemoteProtocol", "ReadError", "ConnectError", "SSLError", "EOF occurred")


def _is_transient(e: Exception, s: str) -> bool:
    """Ошибка, которая лечится повтором: лимиты, перегрузка, сбой сервера, обрыв соединения."""
    return isinstance(e, (ConnectionError, TimeoutError)) or any(t in s for t in _TRANSIENT_TOKENS)


def _model_unusable(s: str) -> bool:
    """Ошибка не лечится повтором: исчерпана суточная квота либо модель закрыта для аккаунта."""
    low = s.lower()
    daily = "perday" in low.replace(" ", "").replace("_", "") or "per day" in low or "daily" in low
    gone = "no longer available" in low or ("404" in s and "NOT_FOUND" in s)
    return daily or gone


def reason_is_overload(s: str) -> bool:
    return s == "перегрузка"


def _short_reason(s: str) -> str:
    low = s.lower()
    if "no longer available" in low or "404" in s:
        return "закрыта для аккаунта"
    if reason_is_overload(s):
        return "перегружена"
    return "суточная квота исчерпана"


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
        self._thinking_ok = True  # станет False, если модель не принимает thinking_level

    def _config(self, system, schema, effort):
        t = self._types
        kw = dict(system_instruction=system, response_mime_type="application/json",
                  response_schema=schema, temperature=0.1, max_output_tokens=32000)
        level = (effort or os.getenv("GEMINI_THINKING") or "").strip().upper()
        if level and self._thinking_ok and hasattr(t, "ThinkingLevel") and level in t.ThinkingLevel.__members__:
            kw["thinking_config"] = t.ThinkingConfig(thinking_level=level)
        return t.GenerateContentConfig(**kw)

    def _call(self, system, user, schema, effort=None):
        try:
            resp = self._client.models.generate_content(
                model=self.model, contents=user, config=self._config(system, schema, effort))
        except Exception as e:
            if self._thinking_ok and "thinking" in str(e).lower() and "400" in str(e):
                self._thinking_ok = False  # модель не поддерживает управление размышлениями — работаем без него
                _log("  модель не принимает thinking_level — продолжаю без него")
                resp = self._client.models.generate_content(
                    model=self.model, contents=user, config=self._config(system, schema, effort))
            else:
                raise
        usage = getattr(resp, "usage_metadata", None)
        if usage:
            self._add_usage(getattr(usage, "prompt_token_count", 0), getattr(usage, "candidates_token_count", 0))
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

    def _call(self, system, user, schema, effort=None):
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
        self._add_usage(msg.usage.input_tokens, msg.usage.output_tokens)
        for block in msg.content:
            if block.type == "tool_use":
                return block.input
        raise json.JSONDecodeError("модель не вернула tool_use", "", 0)


def _env_list(name: str) -> list[str]:
    return [m.strip() for m in (os.getenv(name) or "").split(",") if m.strip()]


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
        return GeminiClient(model, key, min_interval=min_interval,
                            fallbacks=_env_list("GEMINI_FALLBACK_MODELS"))
    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("Не задан ANTHROPIC_API_KEY (см. README и .env.example)")
        model = model or os.getenv("ANTHROPIC_MODEL")
        if not model:
            raise LLMError("Не задана модель: ANTHROPIC_MODEL в .env или --model")
        return AnthropicClient(model, key, min_interval=min_interval,
                               fallbacks=_env_list("ANTHROPIC_FALLBACK_MODELS"))
    raise LLMError(f"Неизвестный провайдер {provider!r} (gemini | anthropic)")
