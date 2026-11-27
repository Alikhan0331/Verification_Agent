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


class RefusalError(LLMError):
    """Модель отказалась отвечать (stop_reason=refusal у Claude): повтор и другая модель цепочки могут помочь."""


def _err_text(e: Exception) -> str:
    """Текст ошибки провайдера с причиной из тела ответа (у Claude str(e) бывает просто «Error code: 400»)."""
    s = str(e)
    body = getattr(e, "body", None)
    err = body.get("error") if isinstance(body, dict) else None
    msg = (err.get("message") if isinstance(err, dict) else None) or (body.get("message") if isinstance(body, dict) else None)
    if msg and msg not in s:
        s += f" — {msg}"
    low = s.lower()
    if "credit balance" in low:
        s += " [Недостаточно средств на счёте Anthropic: пополните баланс в Billing]"
    return s


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class LLMClient:
    provider = "base"

    def __init__(self, model: str, min_interval: float = 0.0, max_retries: int = 5,
                 fallbacks: list[str] | None = None, n_keys: int = 1, max_overload: int = 16):
        self.model = model
        self.models = [model] + [m for m in (fallbacks or []) if m and m != model]
        self.models_used = [model]      # какие модели реально отвечали в этом прогоне
        self.calls_by_model: dict[str, int] = {}   # сколько успешных ответов дала каждая модель
        self._dead: set[str] = set()    # модели с исчерпанной суточной квотой у всех ключей или закрытые
        self.max_overload = max_overload   # сколько ответов 503 терпим за один вызов, прежде чем сдаться
        self.n_keys = max(n_keys, 1)    # ключей API (у каждого проекта своя квота); вызовы чередуются по кругу
        self._key_dead: set[tuple[str, int]] = set()   # (модель, номер ключа) с исчерпанной суточной квотой
        self._rr = 0
        self._tls = threading.local()   # номер ключа для текущего вызова (читает _call провайдера)
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
        self._tls.model = None          # модель только для этого запроса (после отказа первой модели цепочки)
        try:
            return self._generate(system, user, schema, label, effort, [])
        finally:
            self._tls.model = None

    def _generate(self, system, user, schema, label, effort, refused):
        hint = ""
        last_err: Exception | None = None
        attempt = 0
        overload_streak = overload_total = 0
        while attempt < self.max_retries:
            attempt += 1
            used = self._tls.model or self.model
            kidx = self._next_key(used)
            self._tls.key = kidx
            try:
                self._throttle()
                raw = self._call(system, user + hint, schema, effort)
                data = raw if isinstance(raw, dict) else json.loads(_strip_fence(raw))
                parsed = schema.model_validate(data)
                with self._lock:
                    self.calls_by_model[used] = self.calls_by_model.get(used, 0) + 1
                return parsed
            except (ValidationError, json.JSONDecodeError) as e:
                last_err = e
                _log(f"  [{label}] ответ не прошёл валидацию (попытка {attempt}): {str(e)[:200]}")
                hint = ("\n\nПРЕДЫДУЩИЙ ОТВЕТ НЕ ПРОШЁЛ ВАЛИДАЦИЮ JSON-схемы. Верни строго по схеме, "
                        f"все поля обязательны. Ошибка: {str(e)[:400]}")
            except RefusalError as e:
                last_err = e
                refused.append(used)
                if len(refused) == 1:
                    _log(f"  [{label}] {used} отказалась отвечать — повторяю запрос")
                    continue
                other = [m for m in self.models if m not in refused and m not in self._dead]
                if not other:
                    raise LLMError("Все модели отказались отвечать на запрос «" + label + "» (" + ", ".join(dict.fromkeys(refused)) +
                                   "). Это сработала защита провайдера; попробуйте другую модель (--model).") from e
                self._tls.model = other[0]
                _log(f"  [{label}] {used} отказалась отвечать — пробую {other[0]}")
            except Exception as e:  # сетевые/квотные ошибки провайдера
                last_err = e
                s = _err_text(e)
                if _model_unusable(s):
                    if not _model_closed(s) and self._kill_key(used, kidx):
                        attempt -= 1         # квота кончилась у одного ключа, другие ещё живы — повторяем другим
                        continue
                    # суточная квота исчерпана у всех ключей или модель закрыта — берём запасную модель
                    if self._switch_model(used, s):
                        attempt -= 1
                        continue
                    raise LLMError(
                        f"Суточная квота исчерпана (или модель закрыта, либо перегружена) у всех моделей: {', '.join(self.models)}. "
                        "Подождите сброса квоты (у Google — в полночь по тихоокеанскому времени), "
                        f"добавьте запасные модели в {self.provider.upper()}_FALLBACK_MODELS или используйте другой ключ/модель.") from e
                retryable = _is_transient(e, s)
                if retryable and _is_overload(s):
                    # перегрузка (503) временная: модели не списываем, у неё свой бюджет попыток.
                    # Пробуем другую модель цепочки, при общей перегрузке ждём и возвращаемся по кругу.
                    overload_total += 1
                    if overload_total > self.max_overload:
                        raise LLMError(f"Модели перегружены (503), {overload_total - 1} попыток подряд без ответа: "
                                       f"{s[:300]}") from e
                    attempt -= 1
                    overload_streak += 1
                    if overload_streak >= OVERLOAD_SWITCH_AFTER and self._rotate_model(used):
                        overload_streak = 0
                        time.sleep(2)
                        continue
                    delay = min(5 * 2 ** min(overload_total // 2, 4), 60)
                    _log(f"  [{label}] перегрузка (503) — повтор через {delay} с")
                    time.sleep(delay)
                    continue
                overload_streak = 0
                if not retryable or attempt == self.max_retries:
                    raise LLMError(f"Ошибка провайдера {self.provider}: {s[:500]}") from e
                delay = min(5 * 2 ** (attempt - 1), 90)
                _log(f"  [{label}] {s[:120]} — повтор через {delay} с")
                time.sleep(delay)
        raise LLMError(f"Не удалось получить валидный ответ ({label}): {last_err}")

    def _rotate_model(self, failed: str) -> bool:
        """Перегрузка: переходим к следующей модели цепочки, не списывая текущую. False — другой модели нет."""
        with self._lock:
            if self.model != failed:
                return True
            cands = [m for m in self.models if m not in self._dead]
            if len(cands) < 2:
                return False
            nxt = cands[(cands.index(failed) + 1) % len(cands)] if failed in cands else cands[0]
            self.model = nxt
            if nxt not in self.models_used:
                self.models_used.append(nxt)
            _log(f"  модель {failed} перегружена (503) — пробую {nxt}")
            return True

    def _next_key(self, model: str) -> int:
        """Следующий живой ключ по кругу — вызовы чередуются между ключами."""
        with self._lock:
            alive = [i for i in range(self.n_keys) if (model, i) not in self._key_dead]
            if not alive:
                return 0
            self._rr += 1
            return alive[self._rr % len(alive)]

    def _kill_key(self, model: str, kidx: int) -> bool:
        """Ключ исчерпал суточную квоту на модель. True — у модели остались другие живые ключи."""
        with self._lock:
            if self.model != model:           # другой поток уже перешёл на запасную модель
                return True
            if (model, kidx) not in self._key_dead:
                self._key_dead.add((model, kidx))
                left = sum(1 for i in range(self.n_keys) if (model, i) not in self._key_dead)
                if self.n_keys > 1:
                    _log(f"  ключ №{kidx + 1}: суточная квота на {model} исчерпана, живых ключей: {left}")
            return any((model, i) not in self._key_dead for i in range(self.n_keys))

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


OVERLOAD_SWITCH_AFTER = 2      # столько подряд ответов 503 на одной модели — и пробуем следующую модель цепочки


def _is_overload(s: str) -> bool:
    return "503" in s or "UNAVAILABLE" in s or "overloaded" in s

_TRANSIENT_TOKENS = ("429", "RESOURCE_EXHAUSTED", "500", "502", "503", "504", "UNAVAILABLE", "overloaded",
                     "timeout", "Timeout", "timed out", "10053", "10054", "10060", "Connection", "connection",
                     "RemoteProtocol", "ReadError", "ConnectError", "SSLError", "EOF occurred", "disconnected",
                     "getaddrinfo", "11001", "Name or service", "Temporary failure", "reset by peer",
                     "Broken pipe")


_NET_MODULES = ("httpx", "httpcore", "socket", "ssl", "urllib3", "requests", "aiohttp")


def _is_transient(e: Exception, s: str) -> bool:
    """Ошибка, которая лечится повтором: лимиты, перегрузка, сбой сервера, обрыв соединения, сбой DNS."""
    if isinstance(e, OSError):                       # ConnectionError, TimeoutError, socket.gaierror (DNS)…
        return True
    if type(e).__module__.split(".")[0] in _NET_MODULES:   # httpx.RemoteProtocolError, ReadTimeout и т. п.
        return True
    return any(t in s for t in _TRANSIENT_TOKENS)


def _model_closed(s: str) -> bool:
    low = s.lower()
    return "no longer available" in low or ("404" in s and "not_found" in low)


def _model_unusable(s: str) -> bool:
    """Ошибка не лечится повтором: исчерпана суточная квота либо модель закрыта для аккаунта."""
    low = s.lower()
    daily = "perday" in low.replace(" ", "").replace("_", "") or "per day" in low or "daily" in low
    return daily or _model_closed(s)


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

    def __init__(self, model: str, api_key: str | list[str], **kw):
        keys = [api_key] if isinstance(api_key, str) else [k for k in api_key if k]
        super().__init__(model, n_keys=len(keys), **kw)
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:  # pragma: no cover
            raise LLMError("Нужен пакет google-genai (pip install -r requirements.txt)") from e
        self._types = types
        timeout_ms = int(float(os.getenv("GEMINI_TIMEOUT_S", "240") or 240) * 1000)   # зависший запрос не должен висеть вечно
        self._clients = [genai.Client(api_key=k, http_options=types.HttpOptions(timeout=timeout_ms)) for k in keys]
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
        client = self._clients[getattr(self._tls, "key", 0) % len(self._clients)]
        try:
            resp = client.models.generate_content(
                model=self.model, contents=user, config=self._config(system, schema, effort))
        except Exception as e:
            if self._thinking_ok and "thinking" in str(e).lower() and "400" in str(e):
                self._thinking_ok = False  # модель не поддерживает управление размышлениями — работаем без него
                _log("  модель не принимает thinking_level — продолжаю без него")
                resp = client.models.generate_content(
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
    """Claude через официальный SDK.

    Основной режим: структурированный вывод (output_config.format, JSON-схема). У новых моделей (Opus 5.5, Sonnet 5.5,
    Fable) принудительный tool_choice и параметр temperature отклоняются (400), поэтому их здесь нет.
    Запасной режим: если API отверг саму схему (например, как слишком сложную), клиент переключается на вызов инструмента
    submit_result с tool_choice=auto и дальше работает так.
    """

    provider = "anthropic"

    def __init__(self, model: str, api_key: str, **kw):
        super().__init__(model, **kw)
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise LLMError("Нужен пакет anthropic (pip install -r requirements.txt)") from e
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "32000") or 32000)
        self._effort_ok = True        # станет False, если модель не принимает output_config.effort
        self._mode = "json"           # "json" — структурированный вывод; "tool" — запасной режим через инструмент

    def _stream(self, system, user, cfg, **extra):
        # стриминг: большой max_tokens без него упирается в таймаут HTTP-запроса
        if cfg:
            extra["output_config"] = cfg
        with self._client.messages.stream(
                model=getattr(self._tls, "model", None) or self.model, max_tokens=self._max_tokens, system=system,
                messages=[{"role": "user", "content": user}], **extra) as stream:
            return stream.get_final_message()

    def _run(self, system, user, cfg, effort, **extra):
        """Один запрос; если модель не принимает effort, повторяет без него."""
        cfg = dict(cfg)
        level = (effort or os.getenv("ANTHROPIC_EFFORT") or "").strip().lower()
        if level in ("low", "medium", "high", "xhigh", "max") and self._effort_ok:
            cfg["effort"] = level
        try:
            return self._stream(system, user, cfg, **extra)
        except self._anthropic.BadRequestError as e:
            if "effort" in cfg and "effort" in str(e).lower():
                self._effort_ok = False                 # старая модель без управления усилием — работаем без него
                _log("  модель не принимает effort — продолжаю без него")
                cfg.pop("effort")
                return self._stream(system, user, cfg, **extra)
            raise

    def _call(self, system, user, schema, effort=None):
        if self._mode == "json":
            try:
                msg = self._run(system, user,
                                {"format": {"type": "json_schema", "schema": self._anthropic.transform_schema(schema)}}, effort)
            except self._anthropic.BadRequestError as e:
                low = str(e).lower()
                if not any(w in low for w in ("schema", "output_config", "structured", "too complex")):
                    raise
                self._mode = "tool"
                _log("  API не принял схему структурированного вывода — перехожу на вызов инструмента")
                return self._call(system, user, schema, effort)
        else:
            tool = {"name": "submit_result", "description": "Вернуть результат проверки строго по схеме.",
                    "input_schema": schema.model_json_schema()}
            msg = self._run(system + "\n\nОтвет верни только вызовом инструмента submit_result.", user, {}, effort,
                            tools=[tool], tool_choice={"type": "auto"})
        self._add_usage(msg.usage.input_tokens, msg.usage.output_tokens)
        if msg.stop_reason == "refusal":
            raise RefusalError("Модель отказалась отвечать (stop_reason=refusal)")
        if msg.stop_reason == "max_tokens":
            raise json.JSONDecodeError("ответ обрезан по max_tokens (увеличьте ANTHROPIC_MAX_TOKENS)", "", 0)
        for block in msg.content:
            if block.type == "tool_use":
                return block.input
        text = "".join(b.text for b in msg.content if b.type == "text")
        if not text.strip():
            raise json.JSONDecodeError("пустой ответ модели", "", 0)
        return text


DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"                 # основная модель Gemini, если не задана GEMINI_MODEL или --model
DEFAULT_GEMINI_FALLBACKS = ("gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-4-hydrogen",
                            "gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"             # основная модель Claude, если не задана ANTHROPIC_MODEL или --model
DEFAULT_ANTHROPIC_FALLBACKS = ("claude-sonnet-5-5",)    # запасная при недоступности основной (ANTHROPIC_FALLBACK_MODELS)


def _env_list(name: str) -> list[str]:
    return [m.strip() for m in (os.getenv(name) or "").split(",") if m.strip()]


def make_client(provider: str | None = None, model: str | None = None, min_interval: float | None = None) -> LLMClient:
    provider = (provider or os.getenv("LLM_PROVIDER") or "gemini").lower()
    if min_interval is None:
        # паузу между вызовами задаём для бесплатного Gemini; у Claude отдельная настройка (по умолчанию без паузы)
        var = "ANTHROPIC_MIN_INTERVAL" if provider == "anthropic" else "LLM_MIN_INTERVAL"
        min_interval = float(os.getenv(var, "0") or 0)
    if provider == "gemini":
        key = _env_list("GEMINI_API_KEYS") or [k for k in [os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")] if k]
        if not key:
            raise LLMError("Не задан GEMINI_API_KEY (см. README и .env.example)")
        key = list(dict.fromkeys(key))       # одинаковые ключи — это одна квота, дубликаты не нужны
        model = model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        fallbacks = _env_list("GEMINI_FALLBACK_MODELS") if os.getenv("GEMINI_FALLBACK_MODELS") is not None             else list(DEFAULT_GEMINI_FALLBACKS)
        return GeminiClient(model, key, min_interval=min_interval, fallbacks=fallbacks)
    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("Не задан ANTHROPIC_API_KEY (см. README и .env.example)")
        model = model or os.getenv("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC_MODEL
        fallbacks = _env_list("ANTHROPIC_FALLBACK_MODELS") if os.getenv("ANTHROPIC_FALLBACK_MODELS") is not None \
            else list(DEFAULT_ANTHROPIC_FALLBACKS)
        return AnthropicClient(model, key, min_interval=min_interval, fallbacks=fallbacks)
    raise LLMError(f"Неизвестный провайдер {provider!r} (gemini | anthropic)")
