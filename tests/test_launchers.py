"""Скрипты запуска: кодировка, от которой зависит, запустятся ли они у пользователя."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOM = b"\xef\xbb\xbf"


def test_run_ps1_has_exactly_one_bom_and_crlf():
    """Windows PowerShell 5.1 читает кириллицу только при BOM; второй BOM ломает первую строку-комментарий."""
    raw = (ROOT / "run.ps1").read_bytes()
    assert raw.startswith(BOM) and not raw[3:].startswith(BOM)
    assert BOM not in raw[3:]


def test_run_sh_has_no_bom_and_lf_endings():
    raw = (ROOT / "run.sh").read_bytes()
    assert not raw.startswith(BOM) and b"\r\n" not in raw and raw.startswith(b"#!")


def test_launchers_add_ask_only_in_a_live_terminal():
    """Запуск одной командой идёт с --ask, но не в пайпе и не при явном --ask / --no-ask."""
    ps1 = (ROOT / "run.ps1").read_text(encoding="utf-8-sig")
    sh = (ROOT / "run.sh").read_text(encoding="utf-8")
    assert "IsInputRedirected" in ps1 and "IsOutputRedirected" in ps1 and '"--no-ask"' in ps1 and '+= "--ask"' in ps1
    assert "[ -t 0 ] && [ -t 1 ]" in sh and "--no-ask" in sh and '--ask' in sh
