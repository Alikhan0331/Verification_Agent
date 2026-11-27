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


@pytest.mark.parametrize("answer,expected", [("", True), ("y", True), ("да", True), ("n", False), ("Н", False), ("нет", False)])
def test_offer_asks_before_starting_the_dialog(answer, expected):
    res = _fresh()
    out = io.StringIO()
    assert dialog.offer(res, lambda p: answer, out) is expected
    assert f"вопросов: {len(res.questions)}" in out.getvalue() and "отчёт сохранён" in out.getvalue()


def test_offer_without_input_means_no():
    assert dialog.offer(_fresh(), _script(), io.StringIO()) is False        # конец ввода: не зависаем и не опрашиваем


class _Tty(io.StringIO):
    def isatty(self):
        return True


def test_cli_offers_dialog_only_in_a_live_terminal(tmp_path, monkeypatch):
    import sys

    from reqcheck import cli

    monkeypatch.setattr(cli, "make_client", lambda *a, **k: tp.FakeLLM())
    offered = []
    monkeypatch.setattr(cli.dialog, "offer", lambda res: offered.append(len(res.questions)) or False)
    argv = [str(tp.DOC), str(tmp_path / "r.md"), "--batch-size", "3", "--workers", "2", "--set-passes", "1"]

    assert cli.main(argv) == 0 and offered == []                      # pytest: терминала нет, ничего не спрашиваем
    monkeypatch.setattr(sys, "stdin", _Tty())
    monkeypatch.setattr(sys, "stdout", _Tty())
    assert cli.main(argv) == 0 and len(offered) == 1 and offered[0] >= 1   # живой терминал: предлагаем диалог
    assert cli.main(argv + ["--no-ask"]) == 0 and len(offered) == 1        # --no-ask отключает предложение
