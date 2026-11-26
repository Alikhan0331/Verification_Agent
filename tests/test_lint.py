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
    found = [f for f in lint(load_document(ROOT / "docs" / "requirements.md")) if f.kind != "no_security"]
    kinds = [f.kind for f in found]
    assert kinds.count("placeholder_section") == 1 and kinds.count("empty_field") == 2
    ph = next(f for f in found if f.kind == "placeholder_section")
    assert ph.line_from == 57 and "заполняются тестировщиками" in ph.quote
    text = " ".join(f.problem for f in found if f.kind == "empty_field")
    assert "Prod" in text and "Swagger" in text


@pytest.mark.parametrize("name", ["card_statement_clean.md", "otp_login.md"])
def test_no_noise_on_corpus(name):
    assert lint(load_document(ROOT / "tests" / "corpus" / name)) == []


def test_security_gap_found_when_no_requirement_mentions_security():
    found = lint(load_document(ROOT / "docs" / "requirements.md"))
    gap = [f for f in found if f.kind == "no_security"]
    assert len(gap) == 1 and "Нефункциональные" in gap[0].quote           # «безопасност» в разделе Ревью не считается
    assert not [f for f in lint(load_document(ROOT / "tests" / "corpus" / "card_statement_clean.md"))
                if f.kind == "no_security"]


def test_empty_section_but_not_container():
    found = lint(_doc("# T\n## Контейнер\n### Вложенный\nТекст раздела с содержанием.\n## Пустой\n## Последний\nесть"))
    assert [f.kind for f in found] == ["empty_section"]
    assert found[0].quote == "## Пустой" and found[0].line_from == 5


def test_tbd_section_and_legit_dash_in_other_tables():
    text = ("# T\n## Тесты\nTBD\n## Ответ\n| Параметр | Тип | Условие |\n| :-- | :-- | :-- |\n| `a` | string | — |\n"
            "## Общие\n| Поле | Значение |\n| :-- | :-- |\n| **Автор** | Иванов |\n| **URL** | — |")
    found = lint(_doc(text))
    assert sorted(f.kind for f in found) == ["empty_field", "placeholder_section"]   # тире в таблице ответа — норма


def test_deferred_decision_and_dangling_section_reference():
    found = lint(load_document(ROOT / "tests" / "corpus" / "loan_prepayment.md"))
    by = {f.kind: f for f in found}
    assert "определяется позже" in by["deferred"].quote and by["deferred"].line_from == 24
    assert by["dangling_ref"].criterion == "S-TRA" and "7.3" in by["dangling_ref"].problem


def test_dangling_requirement_id_only_for_known_prefixes():
    text = ("# T\n## Цели\n* Цель один.\n## Функциональные требования\n"
            "* **FR-1.** Система показывает сообщение из FR-9 и хранит данные в UTF-8 и AES-256.\n"
            "* **FR-2.** Система делает второе действие.\n* **FR-3.** Система делает третье действие.\n"
            "* **FR-4.** Система делает четвёртое действие.\n* **FR-5.** Система делает пятое действие.")
    refs = [f for f in lint(_doc(text)) if f.kind == "dangling_ref"]
    assert [f.quote for f in refs] == ["FR-9"]                # UTF-8 и AES-256 — не ссылки на требования


def test_sum_error_and_missing_appendix_on_cashback_corpus():
    found = lint(load_document(ROOT / "tests" / "corpus" / "cashback_accrual.md"))
    ar = [f for f in found if f.kind == "arithmetic"]
    assert len(ar) == 1 and "2 400 ₸ + 1 200 ₸ = 3 400 ₸" in ar[0].quote and "3 600" in ar[0].problem
    ap = [f for f in found if f.kind == "dangling_ref" and "приложени" in f.quote]
    assert len(ap) == 1 and "Б" in ap[0].problem


def test_correct_sums_and_products_are_not_flagged():
    text = ("# T\n## Цели\n* Цель один.\n## Расчёт\nПример: 2 400 ₸ + 1 200 ₸ = 3 600 ₸. Ещё: 1,5 + 2,5 = 4. "
            "Формула: 10 + 5 = 15 × 2.\n* a один\n* b два\n* c три\n* d четыре\n* e пять")
    assert [f for f in lint(_doc(text)) if f.kind == "arithmetic"] == []
