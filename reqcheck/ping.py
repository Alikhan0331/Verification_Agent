"""Проверка связи с провайдером: python -m reqcheck.ping [--provider anthropic] [--model ID]

Отправляет три крошечных запроса (несколько центов не стоят): обычный, потоковый и потоковый в том виде, в котором их
шлёт агент. Если какой-то из них не проходит, печатает, что именно ответил сервер. Ключ не печатается.
"""

from __future__ import annotations

import argparse
import os
import re
import sys


def _clean(s: str) -> str:
    return re.sub(r"(sk-ant-\S+|AIza\S+|AQ\.\S+)", "<ключ>", s)


def _show(e: Exception) -> str:
    resp = getattr(e, "response", None)
    raw = ""
    try:
        raw = " ".join(str(resp.text).split())[:400]
    except Exception:
        pass
    return _clean(f"{type(e).__name__}: {e} | тело: {getattr(e, 'body', None)} | ответ сервера: {raw or '—'}")


def ping_anthropic(model: str) -> int:
    import anthropic

    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        print("Нет ANTHROPIC_API_KEY (ни в окружении, ни в .env).")
        return 1
    print(f"anthropic SDK {anthropic.__version__}, модель {model}, ключ длиной {len(key)} символов")
    proxies = [v for v in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY") if os.getenv(v)]
    print("прокси в окружении:", ", ".join(proxies) or "нет")
    c = anthropic.Anthropic(api_key=key, max_retries=0)
    msgs = [{"role": "user", "content": "ответь одним словом: да"}]
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": False}
    bad = 0

    def attempt(name, fn):
        nonlocal bad
        try:
            print(f"[{name}] OK:", fn())
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"[{name}] ОШИБКА:", _show(e))

    attempt("1. обычный запрос", lambda: "".join(b.text for b in c.messages.create(model=model, max_tokens=300, messages=msgs).content
                                                  if b.type == "text").strip())

    def stream(**extra):
        with c.messages.stream(model=model, max_tokens=2000, messages=msgs, **extra) as st:
            m = st.get_final_message()
        return f"stop={m.stop_reason}, токенов {m.usage.input_tokens}+{m.usage.output_tokens}"

    attempt("2. потоковый запрос", stream)
    attempt("3. потоковый, формат и effort как у агента",
            lambda: stream(output_config={"format": {"type": "json_schema", "schema": schema}, "effort": "high"}))
    print("Итог:", "всё работает" if not bad else f"не прошло запросов: {bad} из 3")
    return 1 if bad else 0


def ping_gemini(model: str) -> int:
    from google import genai

    keys = [k.strip() for k in (os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or "").split(",") if k.strip()]
    if not keys:
        print("Нет GEMINI_API_KEY (ни в окружении, ни в .env).")
        return 1
    try:
        r = genai.Client(api_key=keys[0]).models.generate_content(model=model, contents="ответь одним словом: да")
        print(f"[gemini {model}] OK:", (r.text or "").strip())
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"[gemini {model}] ОШИБКА:", _clean(str(e))[:500])
        return 1


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    ap = argparse.ArgumentParser(prog="reqcheck.ping", description="Проверка связи с моделью")
    ap.add_argument("--provider", default=None, help="gemini | anthropic (иначе LLM_PROVIDER из .env)")
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)
    provider = (args.provider or os.getenv("LLM_PROVIDER") or "gemini").lower()
    if provider == "anthropic":
        return ping_anthropic(args.model or os.getenv("ANTHROPIC_MODEL") or "claude-opus-5-5")
    return ping_gemini(args.model or os.getenv("GEMINI_MODEL") or "gemini-3.8-flash")


if __name__ == "__main__":
    raise SystemExit(main())
