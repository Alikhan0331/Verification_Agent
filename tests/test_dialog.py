"""Диалог в терминале (--ask): ввод ответов, круги уточнений, запись в отчёт. Модель заменена заглушкой."""

from __future__ import annotations

import io

import pytest

import test_pipeline as tp
from test_reply import ReplyLLM
from reqcheck import dialog, pipeline, reply, report
from reqcheck.conformance import check_report
from reqcheck.loader import load_document


def _fresh():
    return pipeline.run(load_document(tp.DOC), tp.FakeLLM(), batch_size=2, workers=2, segmentation="llm", set_passes=1)


def _script(*lines):
    it = iter(lines)

    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    return fake_input


@pytest.mark.parametrize("raw,n,expected", [
    ("А", 3, 0), ("б", 3, 1), ("В)", 3, 2), ("a", 2, 0), ("B", 2, 1), ("2", 3, 1),
    ("В", 2, None),            # третьего варианта нет — это не выбор варианта
    ("Позже определим", 3, None), ("А", 0, None), ("", 3, None),
])
def test_option_index(raw, n, expected):
    assert dialog._option_index(raw, n) == expected


def test_collect_answers_letter_free_text_skip_and_stop():
    res = _fresh()
    assert len(res.questions) >= 2 and res.questions[0].options
    out = io.StringIO()
    got = dialog.collect_answers(res, [0, 1], _script("А", "q"), out)           # q на втором вопросе заканчивает опрос
    assert list(got) == ["Q-01"] and got["Q-01"].startswith("А — ") and res.questions[0].options[0] in got["Q-01"]
    shown = out.getvalue()
    assert "Вопрос 1 из 2" in shown and res.questions[0].question in shown and "строка" in shown
    assert dialog.collect_answers(res, [0, 1], _script("", "своими словами"), io.StringIO()) == {"Q-02": "своими словами"}
    assert dialog.collect_answers(res, [0, 1], _script("q", "не дойдёт"), io.StringIO()) == {}


def test_collect_answers_survives_end_of_input():
    res = _fresh()
    assert dialog.collect_answers(res, [0, 1], _script("Ответ на первый"), io.StringIO()) == {"Q-01": "Ответ на первый"}


def test_answers_markdown_is_parsable():
    text = dialog.answers_markdown({"Q-01": "А — вариант\nещё строка", "Q-04": "Фиксируем 50 000"})
    assert reply.parse_answers(text) == {"Q-01": "А — вариант ещё строка", "Q-04": "Фиксируем 50 000"}


def test_dialog_two_rounds_closes_and_reports():
    res = _fresh()
    n0 = len(res.questions)
    active0 = len(res.active)
    llm = ReplyLLM(["closes", "insufficient"])
    out = io.StringIO()
    assert n0 == 2
    # круг 1: Q-01 — выбираем вариант (закроет), Q-02 — отписка (не закроет, агент задаст уточнение);
    # круг 2: отвечаем на уточняющий вопрос по Q-02 (по плану заглушки он закроется)
    res, given = dialog.run_dialog(res, llm, _script("А", "Позже определим", "Лимит фиксируем: 50 000 ₸"), out)
    assert list(given) == ["Q-01", "Q-02", f"Q-{n0 + 1:02d}"]
    assert len(res.questions) == n0 + 1                                       # одно уточнение появилось и закрыто
    assert [a["verdict"] for a in res.answers] == ["closes", "insufficient", "closes"]
    assert res.meta["reply"]["answered"] == 3
    assert len(res.active) < active0
    shown = out.getvalue()
    assert "Агент задаёт уточняющих вопросов: 1" in shown and "недостаточно для закрытия" in shown

    md = report.render(res)
    assert "Позже определим" in md and "Лимит фиксируем" in md
    assert [p for p in check_report(res, md) if "версия агента" not in p] == []


def test_dialog_without_answers_changes_nothing():
    res = _fresh()
    before = report.to_json(res)
    res2, given = dialog.run_dialog(res, ReplyLLM([]), _script("", "q"), io.StringIO())
    assert given == {} and report.to_json(res2) == before
