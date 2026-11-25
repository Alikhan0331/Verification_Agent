"""Детерминированная нарезка: границы требований, идентификаторы, разделы, пропуски."""

from __future__ import annotations

from pathlib import Path

import pytest

from reqcheck import segmenter
from reqcheck.grounding import Grounder
from reqcheck.loader import Document, load_document

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "tests" / "corpus"


def _units(path):
    doc = load_document(path)
    return doc, segmenter.segment(doc)[1]


def test_task_document_ids_and_lines():
    doc, units = _units(ROOT / "docs" / "requirements.md")
    by_id = {u.doc_id: u for u in units if u.doc_id}
    assert {"Ф-1", "Ф-2", "Ф-3", "Ф-4"} <= set(by_id)
    assert by_id["Ф-3"].line_from == 39 and by_id["Ф-4"].line_from == 40
    assert by_id["Ф-1"].kind == "functional"
    assert any(u.kind == "nonfunctional" for u in units)
    assert any(u.kind == "acceptance" for u in units)


def test_quotes_are_found_in_document():
    """Цитата каждой единицы должна находиться агентом-привязчиком в тех же строках."""
    for name in ["otp_login.md", "loan_prepayment.md", "p2p_limits.md", "card_statement_clean.md"]:
        doc, units = _units(CORPUS / name)
        g = Grounder(doc)
        assert units, name
        for u in units:
            loc = g.locate(u.quote, prefer=[(u.line_from, u.line_to)])
            assert loc is not None, (name, u.quote[:60])
            assert loc.line_from >= u.line_from - 1 and loc.line_to <= u.line_to + 1, (name, u.quote[:60])


def test_template_sections_skipped():
    _, units = _units(CORPUS / "card_statement_clean.md")
    sections = " ".join(u.section.lower() for u in units)
    assert "общие сведения" not in sections and "термин" not in sections
    assert [u.doc_id for u in units if u.kind == "functional"] == [f"FR-{i}" for i in range(1, 8)]
    assert sum(1 for u in units if u.kind == "acceptance") == 9        # по строке таблицы на критерий
    assert {u.doc_id for u in units if u.kind == "acceptance"} == {f"AC-{i}" for i in range(1, 10)}


def test_continuation_lines_glued_to_bullet():
    doc = Document(name="x.md", lines=["# T", "## Функциональные требования", "* **Ф-1.** Система должна",
                                       "  делать нечто важное за 5 секунд.", "* **Ф-2.** Система должна молчать.",
                                       "## Цели", "* Цель один.", "* Цель два.", "* Цель три."], path="", sha256="")
    _, units = segmenter.segment(doc)
    assert units[0].doc_id == "Ф-1" and (units[0].line_from, units[0].line_to) == (3, 4)
    assert "нечто важное" in units[0].quote


def test_plain_text_is_not_structured():
    doc = Document(name="x.txt", lines=["Просто абзац текста без разметки."] * 30, path="", sha256="")
    assert not segmenter.looks_structured(doc)


@pytest.mark.parametrize("name", ["otp_login.md", "loan_prepayment.md", "p2p_limits.md", "card_statement_clean.md"])
def test_corpus_is_structured(name):
    assert segmenter.looks_structured(load_document(CORPUS / name))
