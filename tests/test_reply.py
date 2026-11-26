"""Режим ответов автора: разбор файла ответов, оценка (заглушка модели), пересчёт вердикта, отчёт."""

from __future__ import annotations

import re

import pytest

import test_pipeline as tp
from reqcheck import pipeline, reply, report
from reqcheck.conformance import check_report
from reqcheck.loader import load_document
from reqcheck.schemas import AnswerAssessment, AnswersResult, ConclusionText, FixItem


class ReplyLLM(tp.FakeLLM):
    """Оценки по порядку вопросов задаются планом: closes | insufficient | contradicts."""

    def __init__(self, plan):
        super().__init__()
        self.plan = plan

    def _AnswersResult(self, user):
        ids = re.findall(r"^(Q-\d+)\. Вопрос:", user, flags=re.M)
        assert ids and "ОТВЕТ АВТОРА" in user
        out = []
        for qid, verdict in zip(ids, self.plan):
            kw = dict(question_id=qid, verdict=verdict, reason="Оценка заглушки.", closed_defect_ids=[],
                      proposed_text="", follow_up="", conflict_quote="", conflict_problem="")
            if verdict == "closes":
                kw["proposed_text"] = "Повторный push после смены порога не отправляется."
            elif verdict == "insufficient":
                kw.update(closed_defect_ids=["D-001"], proposed_text="выдумка",       # модель ошиблась — код обнуляет
                          follow_up="Какое именно значение лимита нужно зафиксировать?")
            else:
                kw.update(follow_up="Как согласовать ответ с правилом одного push в месяц?",
                          conflict_quote=tp.Q_F3, conflict_problem="Ф-3 запрещает второй push.")
            out.append(AnswerAssessment(**kw))
        return AnswersResult(assessments=out)

    def _ConclusionText(self, user):
        first = re.search(r"^- (D-\d+) \[", user, flags=re.M)
        fixes = [FixItem(defect_ids=[first.group(1)], action="Внести формулировки из ответов.")] if first else []
        return ConclusionText(rationale="После ответов осталось меньше дефектов.", top_fixes=fixes)


def _fill(tpl: str, answers: dict[str, str]) -> str:
    parts = re.split(r"(?m)^(## Q-\d+)$", tpl)
    out = [parts[0]]
    for head, body in zip(parts[1::2], parts[2::2]):
        out.append(head + body.replace("Ответ:\n", f"Ответ: {answers.get(head[3:], '')}\n", 1))
    return "".join(out)


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    res = pipeline.run(load_document(tp.DOC), tp.FakeLLM(), batch_size=2, workers=2, segmentation="llm")
    p = tmp_path_factory.mktemp("rp") / "report.json"
    p.write_text(report.to_json(res), encoding="utf-8")
    return res, p


def test_template_and_parsing_roundtrip(base):
    res, _ = base
    assert len(res.questions) >= 2
    tpl = reply.make_template(res)
    assert "## Q-01" in tpl and "Ответ:" in tpl and "Вариант А:" in tpl
    assert reply.parse_answers(tpl) == {}                                   # пустой шаблон — ни одного ответа
    filled = tpl.replace("Ответ:\n", "Ответ: Вариант А, сохранить приоритет Ф-3.\nПоясню: иначе спор.\n", 1)
    got = reply.parse_answers(filled)
    assert list(got) == ["Q-01"] and "приоритет Ф-3" in got["Q-01"] and "Поясню" in got["Q-01"]
    assert "Вопрос" not in got["Q-01"] and "Закрывает" not in got["Q-01"]
    assert reply.parse_answers("Q-02: тест\nпродолжение\n- Q-03) да")["Q-02"] == "тест\nпродолжение"
    assert reply.parse_answers("Q-02: —") == {}


def test_load_result_roundtrip_and_sha_guard(base, tmp_path):
    res, p = base
    again = reply.load_result(p, tp.DOC)
    assert [d.id for d in again.defects] == [d.id for d in res.defects]
    assert [q.question for q in again.questions] == [q.question for q in res.questions]
    assert report.render(again).count("**Q-0") == report.render(res).count("**Q-0")
    other = tmp_path / "other.md"
    other.write_text("# Другой документ\n* Система должна работать.\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256"):
        reply.load_result(p, other)


def test_closing_answer_and_conflicting_answer(base):
    res0, p = base
    res = reply.load_result(p, tp.DOC)
    answers_md = _fill(reply.make_template(res), {"Q-01": "А", "Q-02": "Разрешить второй push."})
    res = reply.apply_answers(res, ReplyLLM(["closes", "contradicts"]), answers_md)

    by_q = {a["question_id"]: a for a in res.answers}
    assert by_q["Q-01"]["verdict"] == "closes" and by_q["Q-01"]["proposed_text"]
    assert by_q["Q-02"]["verdict"] == "contradicts" and by_q["Q-02"]["closed_defect_ids"] == []
    for x in res0.questions[0].defect_ids:
        d = next(d for d in res.defects if d.id == x)
        assert d.resolved_by == "Q-01" and d not in res.active
    conflict = [d for d in res.defects if d.source == "answer"]
    assert len(conflict) == 1 and conflict[0].quotes[0].line_from > 0 and "Q-02" in conflict[0].problem
    assert conflict[0] in res.active and conflict[0].id not in {d.id for d in res0.defects}
    assert len(res.questions) == len(res0.questions) + 1                    # уточняющий вопрос к противоречию
    assert res.questions[-1].defect_ids[-1] == conflict[0].id
    assert res.verdict == pipeline.verdict_of(res.defects)
    assert res.meta["reply"]["answered"] == 2

    md = report.render(res)
    assert "**Ответ автора:**" in md and "Предлагаемая формулировка для документа" in md
    assert "Приложение В. Закрыто ответами автора" in md and "закрыт ответом на Q-01" in md
    assert "Ответы автора |" in md and "Закрыто ответами автора |" in md
    assert [p for p in check_report(res, md) if "версия агента" not in p] == []
    assert '"answers"' in report.to_json(res)


def test_insufficient_answer_closes_nothing_even_if_model_claims(base):
    _, p = base
    res = reply.load_result(p, tp.DOC)
    answers_md = _fill(reply.make_template(res), {"Q-01": "не знаю, решим позже"})
    n_active = len(res.active)
    res = reply.apply_answers(res, ReplyLLM(["insufficient"]), answers_md)
    a = res.answers[0]
    assert a["verdict"] == "insufficient" and a["closed_defect_ids"] == [] and a["proposed_text"] == ""
    assert len(res.active) == n_active and not any(d.resolved_by for d in res.defects)
    assert res.questions[-1].question.startswith("Какое именно значение")


def test_apply_requires_at_least_one_answer(base):
    _, p = base
    res = reply.load_result(p, tp.DOC)
    with pytest.raises(RuntimeError, match="нет ни одного"):
        reply.apply_answers(res, ReplyLLM([]), reply.make_template(res))
