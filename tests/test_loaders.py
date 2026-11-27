"""Чтение PDF и DOCX: организаторы выдают документ, в том числе в PDF."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from reqcheck.grounding import Grounder
from reqcheck.loader import LoadError, load_document

DOC = Path(__file__).resolve().parent.parent / "docs" / "requirements.md"
FONT = "C:/Windows/Fonts/arial.ttf"
Q_F3 = "На карту приходится максимум один такой push за календарный месяц."
Q_F4 = "Если она ниже нового порога, следующее достижение вызывает push независимо от истории предыдущей настройки."


def _make_pdf(path: Path, lines: list[str]) -> None:
    pytest.importorskip("reportlab")
    if not os.path.exists(FONT):
        pytest.skip("нет шрифта с кириллицей для генерации PDF")
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    pdfmetrics.registerFont(TTFont("Arial", FONT))
    c = canvas.Canvas(str(path))
    c.setFont("Arial", 9)
    y = 800
    for ln in lines:
        for part in textwrap.wrap(ln, 105) or [""]:
            c.drawString(40, y, part)
            y -= 12
            if y < 40:
                c.showPage()
                c.setFont("Arial", 9)
                y = 800
    c.save()


def test_pdf_text_extracted_grounded_and_paged(tmp_path):
    pdf = tmp_path / "req.pdf"
    _make_pdf(pdf, DOC.read_text(encoding="utf-8").splitlines())
    doc = load_document(pdf)
    assert len(doc.lines) > 50 and doc.line_pages[0] == 1 and max(doc.line_pages) >= 2
    g = Grounder(doc)
    for q in (Q_F3, Q_F4):                       # цитата может переноситься между строками PDF
        loc = g.locate(q)
        assert loc and not loc.approx, q
        assert doc.page_of(loc.line_from) >= 1


def test_pdf_without_text_layer_gives_clear_error(tmp_path):
    pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas

    p = tmp_path / "empty.pdf"
    c = canvas.Canvas(str(p))
    c.rect(10, 10, 100, 100)                     # только графика, текста нет
    c.save()
    with pytest.raises(LoadError, match="текстового слоя"):
        load_document(p)


def test_docx_paragraphs_headings_and_tables(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Заголовок раздела", level=1)
    d.add_paragraph("Система должна уведомлять клиента.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Поле", "Значение"
    t.cell(1, 0).text, t.cell(1, 1).text = "Порог", "1 000"
    p = tmp_path / "req.docx"
    d.save(str(p))
    doc = load_document(p)
    assert doc.lines[0] == "# Заголовок раздела"
    assert "Система должна уведомлять клиента." in doc.lines
    assert "| Порог | 1 000 |" in doc.lines


def test_sha256_does_not_depend_on_line_endings(tmp_path):
    lf, crlf = tmp_path / "a.md", tmp_path / "b.md"
    lf.write_bytes("# Док\n* Система должна работать.\n".encode("utf-8"))
    crlf.write_bytes("# Док\r\n* Система должна работать.\r\n".encode("utf-8"))
    assert load_document(lf).sha256 == load_document(crlf).sha256
