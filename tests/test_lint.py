"""Детерминированные проверки шаблона: находят заглушки и пустые поля, не шумят на чистых документах."""

from __future__ import annotations

from pathlib import Path

import pytest

from reqcheck.lint import lint
from reqcheck.loader import Document, load_document

ROOT = Path(__file__).resolve().parent.parent


def _doc(text: str) -> Document:
    return Document(name="x.md", path="", sha256="", lines=text.split("\n"))


def test_task_document_placeholder_and_empty_fields():
    found = lint(load_document(ROOT / "docs" / "requirements.md"))
    kinds = [f.kind for f in found]
    assert kinds.count("placeholder_section") == 1 and kinds.count("empty_field") == 2
    ph = next(f for f in found if f.kind == "placeholder_section")
    assert ph.line_from == 57 and "заполняются тестировщиками" in ph.quote
    text = " ".join(f.problem for f in found if f.kind == "empty_field")
    assert "Prod" in text and "Swagger" in text


@pytest.mark.parametrize("name", ["card_statement_clean.md", "otp_login.md", "loan_prepayment.md", "p2p_limits.md"])
def test_no_noise_on_corpus(name):
    assert lint(load_document(ROOT / "tests" / "corpus" / name)) == []


def test_empty_section_but_not_container():
    found = lint(_doc("# T\n## Контейнер\n### Вложенный\nТекст раздела с содержанием.\n## Пустой\n## Последний\nесть"))
    assert [f.kind for f in found] == ["empty_section"]
    assert found[0].quote == "## Пустой" and found[0].line_from == 5


def test_tbd_section_and_legit_dash_in_other_tables():
    text = ("# T\n## Тесты\nTBD\n## Ответ\n| Параметр | Тип | Условие |\n| :-- | :-- | :-- |\n| `a` | string | — |\n"
            "## Общие\n| Поле | Значение |\n| :-- | :-- |\n| **Автор** | Иванов |\n| **URL** | — |")
    found = lint(_doc(text))
    assert sorted(f.kind for f in found) == ["empty_field", "placeholder_section"]   # тире в таблице ответа — норма
