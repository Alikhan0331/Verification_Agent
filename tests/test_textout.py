"""Отчёт в .txt: обычный текст без разметки Markdown, все идентификаторы и цитаты на месте."""

from __future__ import annotations

import re
import sys

import test_pipeline as tp
from reqcheck import textout


def _report_md():
    from reqcheck import pipeline, report
    from reqcheck.loader import load_document

    doc = load_document(tp.DOC)
    return report.render(pipeline.run(doc, tp.FakeLLM(), batch_size=3, workers=2, set_passes=1))


def test_markup_is_gone_but_content_stays():
    md = _report_md()
    txt = textout.to_text(md)
    assert not re.search(r"\|\s*:?-{3,}", txt)                          # строки-разделители таблиц
    assert not re.search(r"^#{1,6} ", txt, re.M) and "```" not in txt
    assert max(len(line) for line in txt.split("\n")) <= textout.WIDTH
    for pat in (r"D-\d{3}", r"Q-\d{2}", r"R-\d{2}"):
        assert set(re.findall(pat, md)) == set(re.findall(pat, txt))
    assert txt.endswith("\n") and "\r" not in txt


def test_small_markdown_sample():
    md = ("# Заголовок\n\n| Поле | Значение |\n|---|---|\n| **Модель** | `m` \\| x |\n\n"
          "- пункт **жирный**\n  - вложенный\n\n> «Цитата **как есть**» — строка 5\n")
    txt = textout.to_text(md)
    assert txt.startswith("Заголовок\n=========\n")
    assert "Модель: m | x" in txt and "- пункт жирный" in txt and "  - вложенный" in txt
    assert "«Цитата **как есть**» — строка 5" in txt                     # цитата документа дословно


def test_long_line_is_wrapped_with_hanging_indent():
    txt = textout.to_text("- " + "слово " * 60)
    lines = txt.rstrip("\n").split("\n")
    assert len(lines) > 1 and all(len(x) <= textout.WIDTH for x in lines)
    assert lines[0].startswith("- ") and lines[1].startswith("  ")


def test_cli_writes_txt_twin_and_respects_flags(tmp_path, monkeypatch):
    from reqcheck import cli

    monkeypatch.setattr(cli, "make_client", lambda *a, **k: tp.FakeLLM())
    base = [str(tp.DOC), "--batch-size", "3", "--workers", "2", "--set-passes", "1", "--no-ask"]
    out = tmp_path / "r.md"
    assert cli.main([base[0], str(out)] + base[1:]) == 0
    assert out.read_text(encoding="utf-8").startswith("# ") and (tmp_path / "r.txt").exists()
    assert not (tmp_path / "r.txt").read_text(encoding="utf-8").startswith("#")

    only = tmp_path / "only"
    assert cli.main([base[0], str(only / "a.md")] + base[1:] + ["--no-txt"]) == 0
    assert (only / "a.md").exists() and not (only / "a.txt").exists()

    assert cli.main([base[0], str(only / "b.txt")] + base[1:]) == 0       # .txt вместо .md: только простой текст
    assert (only / "b.txt").exists() and not (only / "b.md").exists()
    assert "Паспорт прогона" in (only / "b.txt").read_text(encoding="utf-8")


def test_module_cli(tmp_path):
    src = tmp_path / "x.md"
    src.write_text("# A\n\ntext\n", encoding="utf-8")
    dst = tmp_path / "x.txt"
    assert textout.main([str(src), str(dst)]) == 0
    assert dst.read_text(encoding="utf-8") == "A\n=\n\ntext\n"
    assert textout.main([]) == 2
