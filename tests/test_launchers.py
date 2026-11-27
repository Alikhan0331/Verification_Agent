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
