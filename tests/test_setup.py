"""Первый запуск: выбор провайдера (Gemini или Claude) и сохранение ключа в .env."""

from __future__ import annotations

import io

import pytest

from reqcheck import setup

EXAMPLE = """# Скопируйте этот файл в .env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash

# --- Anthropic (по желанию) ---
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
# ANTHROPIC_MODEL=claude-opus-5-5
LLM_MIN_INTERVAL=3
"""


def _never(prompt=""):
    raise AssertionError("не должно спрашивать: " + prompt)


@pytest.fixture
def env_file(tmp_path):
    (tmp_path / ".env.example").write_text(EXAMPLE, encoding="utf-8")
    return tmp_path / ".env"


def test_placeholder_is_not_a_key():
    assert not setup.has_key("gemini", {"GEMINI_API_KEY": "your_gemini_api_key_here"})
    assert not setup.has_key("anthropic", {"ANTHROPIC_API_KEY": "  "})
    assert setup.has_key("gemini", {"GEMINI_API_KEYS": "k1,k2"}) and setup.has_key("anthropic", {"ANTHROPIC_API_KEY": "sk-ant-x"})


@pytest.mark.parametrize("explicit,env,expected", [
    (None, {}, "gemini"),
    (None, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "k"}, "gemini"),
    (None, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "your_x", "ANTHROPIC_API_KEY": "sk"}, "anthropic"),   # только ключ Claude
    (None, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "sk", "GEMINI_API_KEY": "k"}, "anthropic"),
    (None, {"ANTHROPIC_API_KEY": "sk"}, "anthropic"),
    ("anthropic", {"GEMINI_API_KEY": "k"}, "anthropic"),                                  # явный параметр сильнее
    ("Gemini", {"ANTHROPIC_API_KEY": "sk"}, "gemini"),
])
def test_pick_provider(explicit, env, expected):
    assert setup.pick_provider(explicit, env) == expected


def test_existing_key_means_no_questions(env_file):
    env = {"GEMINI_API_KEY": "k"}
    assert setup.ensure_credentials(None, env_file, env, _never, _never, io.StringIO(), interactive=True) == "gemini"
    assert not env_file.exists()


def test_non_interactive_never_prompts_or_writes(env_file):
    assert setup.ensure_credentials(None, env_file, {}, _never, _never, io.StringIO(), interactive=False) == "gemini"
    assert not env_file.exists()


def test_user_chooses_claude_and_key_is_saved(env_file):
    env: dict[str, str] = {}
    out = io.StringIO()
    got = setup.ensure_credentials(None, env_file, env, lambda p: "2", lambda p: "  sk-ant-secret$1  ", out, interactive=True)
    assert got == "anthropic" and env["ANTHROPIC_API_KEY"] == "sk-ant-secret$1" and env["LLM_PROVIDER"] == "anthropic"
    text = env_file.read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY=sk-ant-secret$1" in text and "# ANTHROPIC_API_KEY" not in text      # строка раскомментирована
    assert "LLM_PROVIDER=anthropic" in text and "GEMINI_MODEL=gemini-3.8-flash" in text and "LLM_MIN_INTERVAL=3" in text
    assert "sk-ant-secret" not in out.getvalue()                                                   # ключ не печатается
    assert not text.startswith("﻿") and "\r" not in text


def test_default_choice_is_gemini_and_replaces_placeholder(env_file):
    env: dict[str, str] = {}
    got = setup.ensure_credentials(None, env_file, env, lambda p: "", lambda p: "AIza-test", io.StringIO(), interactive=True)
    assert got == "gemini"
    lines = env_file.read_text(encoding="utf-8").splitlines()
    assert "GEMINI_API_KEY=AIza-test" in lines and "your_gemini_api_key_here" not in "\n".join(lines)
    assert lines.count("LLM_PROVIDER=gemini") == 1


def test_explicit_provider_asks_only_for_the_key(env_file):
    got = setup.ensure_credentials("anthropic", env_file, {}, _never, lambda p: "sk-ant-test-123456", io.StringIO(), interactive=True)
    assert got == "anthropic" and "ANTHROPIC_API_KEY=sk-ant-test-123456" in env_file.read_text(encoding="utf-8")


def test_empty_key_changes_nothing(env_file):
    got = setup.ensure_credentials(None, env_file, {}, lambda p: "1", lambda p: "   ", io.StringIO(), interactive=True)
    assert got == "gemini" and not env_file.exists()


def test_set_env_values_updates_existing_file_in_place(tmp_path):
    p = tmp_path / ".env"
    p.write_text("A=1\n# B=old\nC=3\n", encoding="utf-8")
    setup.set_env_values(p, {"B": "2", "D": "4"})
    assert p.read_text(encoding="utf-8") == "A=1\nB=2\nC=3\nD=4\n"


def test_other_env_settings_apply_right_after_first_launch(env_file):
    """Регрессия: после ввода ключа в .env попадали модель и паузы, но в текущем запуске они не действовали
    («Не задана модель: GEMINI_MODEL»)."""
    from reqcheck import llm

    env: dict[str, str] = {}
    setup.ensure_credentials(None, env_file, env, lambda p: "", lambda p: "AIza-test", io.StringIO(), interactive=True)
    assert env["GEMINI_MODEL"] == "gemini-3.8-flash" and env["LLM_MIN_INTERVAL"] == "3"
    assert "# ANTHROPIC_MODEL" not in env and "ANTHROPIC_MODEL" not in env


def test_gemini_model_has_defaults_without_env(monkeypatch):
    from reqcheck import llm

    for v in ("GEMINI_MODEL", "GEMINI_FALLBACK_MODELS", "GEMINI_API_KEYS", "GOOGLE_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    c = llm.make_client("gemini")
    assert c.models[0] == llm.DEFAULT_GEMINI_MODEL and c.models[1:] == list(llm.DEFAULT_GEMINI_FALLBACKS)
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "")                 # пусто — запасные отключены
    assert llm.make_client("gemini").models == [llm.DEFAULT_GEMINI_MODEL]


def test_key_problem_catches_ctrl_v_and_garbage():
    """Регрессия: Ctrl+V в скрытом поле сохранял в .env один символ «▬», и Gemini отвечал «400 Bad Request»."""
    assert setup.key_problem("\x16") and setup.key_problem("▬") and setup.key_problem("ключ-кириллицей-1234567")
    assert setup.key_problem("AIza SyA1234567890") and setup.key_problem("AIzaSyA\n1234567890") and setup.key_problem("abc")
    assert setup.key_problem("AIzaSyA1234567890-abc_DEF.xyz") is None and setup.key_problem("sk-ant-api03-abcDEF123_-xyz") is None


def test_broken_key_in_env_is_not_a_key_and_user_is_asked_again(env_file):
    env = {"GEMINI_API_KEY": "▬"}
    assert not setup.has_key("gemini", env)
    answers = iter(["\x16", "AIza-good-key-123"])                   # первая попытка — Ctrl+V, вторая — нормальный ключ
    out = io.StringIO()
    got = setup.ensure_credentials(None, env_file, env, lambda p: "", lambda p: next(answers), out, interactive=True)
    assert got == "gemini" and env["GEMINI_API_KEY"] == "AIza-good-key-123"
    assert "Ключ не принят" in out.getvalue() and "правой кнопкой" in out.getvalue()
    assert "GEMINI_API_KEY=AIza-good-key-123" in env_file.read_text(encoding="utf-8").splitlines()


def test_make_client_reports_damaged_key_clearly(monkeypatch):
    from reqcheck import llm

    for v in ("GEMINI_API_KEYS", "GOOGLE_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "▬")
    with pytest.raises(llm.LLMError, match="повреждён"):
        llm.make_client("gemini")
