"""Первый запуск: выбор провайдера модели и ввод ключа.

Если ключа нет ни в окружении, ни в `.env`, агент в интерактивном терминале спрашивает, какую модель использовать
(Google Gemini или Anthropic Claude), просит вставить ключ (ввод скрыт) и сохраняет его в `.env`, который в репозиторий
не попадает. Без терминала (автоматический запуск) ничего не спрашивается: ключ берётся из окружения или `.env`.
"""

from __future__ import annotations

import getpass
import os
import re
import sys
from pathlib import Path
from typing import Callable, Mapping, MutableMapping

REPO_ROOT = Path(__file__).resolve().parent.parent
KEY_VARS = {
    "gemini": ("GEMINI_API_KEYS", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
}
WRITE_VAR = {"gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
KEY_HELP = {
    "gemini": "Google Gemini, бесплатный ключ: https://aistudio.google.com/apikey (Create API key)",
    "anthropic": "Anthropic Claude, ключ: https://console.anthropic.com/settings/keys (платный API)",
}
_PLACEHOLDER = re.compile(r"^your_|_here$", re.I)


def has_key(provider: str, environ: Mapping[str, str]) -> bool:
    for name in KEY_VARS.get(provider, ()):
        v = (environ.get(name) or "").strip()
        if v and not _PLACEHOLDER.search(v):
            return True
    return False


def pick_provider(explicit: str | None, environ: Mapping[str, str]) -> str:
    """Провайдер: параметр --provider, затем LLM_PROVIDER; если для него ключа нет, а для другого есть — другой."""
    if explicit:
        return explicit.lower()
    chosen = (environ.get("LLM_PROVIDER") or "gemini").lower()
    if not has_key(chosen, environ):
        for other in KEY_VARS:
            if other != chosen and has_key(other, environ):
                return other
    return chosen


def set_env_values(path: Path, updates: Mapping[str, str]) -> None:
    """Меняет (или добавляет) значения в .env, остальные строки не трогает. Нет файла: берём за основу .env.example."""
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
    else:
        example = path.with_name(".env.example")
        lines = example.read_text(encoding="utf-8").splitlines() if example.exists() else []
    for name, value in updates.items():
        pat = re.compile(rf"^\s*#?\s*{re.escape(name)}\s*=")
        for i, ln in enumerate(lines):
            if pat.match(ln):
                lines[i] = f"{name}={value}"
                break
        else:
            lines.append(f"{name}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")


def ensure_credentials(explicit: str | None = None, env_file: Path | None = None,
                       environ: MutableMapping[str, str] | None = None,
                       input_fn: Callable[[str], str] = input,
                       secret_fn: Callable[[str], str] = getpass.getpass,
                       out=None, interactive: bool | None = None) -> str:
    """Возвращает провайдера; при отсутствии ключа в терминале спрашивает его и сохраняет в .env."""
    environ = os.environ if environ is None else environ
    out = out or sys.stderr
    env_file = env_file or (REPO_ROOT / ".env")
    provider = pick_provider(explicit, environ)
    if has_key(provider, environ):
        return provider
    if interactive is None:
        interactive = sys.stdin is not None and sys.stdin.isatty()
    if not interactive or provider not in KEY_VARS:
        return provider                        # дальше make_client сообщит, чего не хватает

    print("\nНе найден ключ API для модели.", file=out)
    if not explicit:
        print("Какую модель использовать?", file=out)
        print(f"  1 - {KEY_HELP['gemini']}", file=out)
        print(f"  2 - {KEY_HELP['anthropic']}", file=out)
        try:
            ans = input_fn("Введите 1 или 2 (Enter — 1): ").strip()
        except EOFError:
            return provider
        provider = "anthropic" if ans.startswith("2") else "gemini"
    else:
        print(KEY_HELP.get(provider, provider), file=out)
    try:
        key = secret_fn("Вставьте ключ и нажмите Enter (ввод скрыт): ").strip()
    except (EOFError, KeyboardInterrupt):
        return provider
    if not key:
        return provider
    updates = {WRITE_VAR[provider]: key, "LLM_PROVIDER": provider}
    set_env_values(env_file, updates)
    for k, v in updates.items():
        environ[k] = v
    print("Ключ сохранён в .env (в репозиторий он не попадает).", file=out)
    return provider
