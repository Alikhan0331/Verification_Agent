"""Целостность корпуса и логика оценки (без вызовов модели)."""

from __future__ import annotations

from pathlib import Path

import pytest

from reqcheck import eval as ev
from reqcheck.grounding import Grounder
from reqcheck.loader import load_document

CORPUS = Path(__file__).resolve().parent / "corpus"
EXPECTED = sorted(CORPUS.glob("*.expected.json"))


@pytest.mark.parametrize("exp", EXPECTED, ids=lambda p: p.name)
def test_every_anchor_is_found_verbatim_in_its_document(exp):
    doc_path, clean, seeds = ev.load_expected(exp)
    g = Grounder(load_document(doc_path))
    for s in seeds:
        spans = ev.anchor_lines(g, s)               # бросит ValueError, если якорь не найден дословно
        assert len(spans) == len(s.anchors) >= s.min_anchors >= 1
    ids = [s.id for s in seeds]
    assert len(ids) == len(set(ids)), "повторяющиеся id семян"
    if clean:
        assert not seeds, "в контрольном документе не должно быть семян"


def test_matching_rules():
    seed = ev.Seed("S", "t", "critical", ["S-CNS"], ["a", "b"], 2)
    spans = [(10, 10), (20, 20)]
    D = lambda crit, *lines, **kw: dict(id="D-1", criterion=crit, severity="critical", rejected=False,
                                        quotes=[dict(line_from=n, line_to=n) for n in lines], **kw)
    assert ev.matches(D("S-CNS", 10, 20), seed, spans)            # обе стороны противоречия
    assert not ev.matches(D("S-CNS", 10), seed, spans)            # только одна сторона
    assert not ev.matches(D("UNA", 10, 20), seed, spans)          # не тот критерий


def test_evaluate_counts_found_missed_and_unmatched():
    seeds = [ev.Seed("S1", "первое", "critical", ["VER"], ["x"]), ev.Seed("S2", "второе", "major", ["COM"], ["y"])]
    spans = {"S1": [(5, 5)], "S2": [(9, 9)]}
    defects = [
        dict(id="D-1", criterion="VER", severity="critical", rejected=False, quotes=[dict(line_from=5, line_to=5)]),
        dict(id="D-2", criterion="UNA", severity="major", rejected=False, quotes=[dict(line_from=9, line_to=9)]),
        dict(id="D-3", criterion="COM", severity="major", rejected=True, quotes=[dict(line_from=9, line_to=9)]),
        dict(id="D-4", criterion="COM", severity="major", rejected=False, quotes=[dict(line_from=30, line_to=30)]),
    ]
    r = ev.evaluate("doc.md", seeds, spans, defects, clean=False)
    assert (r["found"], r["seeds"], r["recall"]) == (1, 2, 0.5)
    assert r["missed"][0]["id"] == "S2" and r["missed"][0]["near"] == ["D-2"]   # место найдено, критерий другой
    assert r["unmatched_important"] == ["D-2", "D-4"]                          # D-3 отклонён и не считается
    md = ev.render([r])
    assert "Итоговая полнота: 1/2" in md and "S2" in md


def test_baseline_report_scored_offline():
    """Первый боевой отчёт (если он лежит рядом) оценивается без вызовов модели."""
    rj = Path(__file__).resolve().parent.parent / "report.json"
    if not rj.exists():
        pytest.skip("нет report.json")
    res = ev.run_corpus(CORPUS, rj.parent, run=False, only="task_requirements", batch_size=10, workers=1, verify=True)
    # report.json называется иначе, чем ожидает run_corpus (NAME.json), поэтому оценим напрямую
    import json
    doc_path, clean, seeds = ev.load_expected(CORPUS / "task_requirements.expected.json")
    g = Grounder(load_document(doc_path))
    spans = {s.id: ev.anchor_lines(g, s) for s in seeds}
    r = ev.evaluate(doc_path.name, seeds, spans, json.loads(rj.read_text(encoding="utf-8"))["defects"], False)
    assert 0 <= r["found"] <= r["seeds"] == 24
    assert res == []   # нет task_requirements.json — без результата ничего не оценивается
