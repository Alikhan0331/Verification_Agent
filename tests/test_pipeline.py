"""Офлайн-тесты: модель заменена заглушкой, проверяется всё остальное
(загрузка, привязка цитат к строкам, сборка отчёта, обработка «галлюцинаций»)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from reqcheck import cli, pipeline, report
from reqcheck.criteria import REQ_CODES
from reqcheck.grounding import Grounder
from reqcheck.loader import LoadError, load_document
from reqcheck.schemas import (ConclusionText, CriterionVerdict, Decision, FixItem, QuoteRef,
                              RequirementBatchResult, RequirementCheck, SegmentationResult, Segment,
                              SetDefect, SetResult, ChecklistItem, VerificationResult)

DOC = Path(__file__).resolve().parent.parent / "docs" / "requirements.md"

Q_F1 = "Держатель активной карты KZT задаёт целый порог от 1 000 до 1 000 000 ₸ включительно."
Q_F3 = "На карту приходится максимум один такой push за календарный месяц."
Q_F4 = "Если она ниже нового порога, следующее достижение вызывает push независимо от истории предыдущей настройки."
Q_NFR = "Сервис передаёт push на отправку в течение минуты после получения события."


class FakeLLM:
    provider, model = "fake", "fake-model"
    calls = input_tokens = output_tokens = 0

    def generate(self, system, user, schema, label=""):
        self.calls += 1
        return getattr(self, "_" + schema.__name__)(user)

    def _SegmentationResult(self, user):
        S = lambda did, q, kind: Segment(doc_id=did, section="X", kind=kind, quote=q, summary=q[:60],
                                         line_from=0, line_to=0)
        return SegmentationResult(document_title="Уведомление о расходах по карте", requirements=[
            S("Ф-4", Q_F4, "functional"),          # намеренно не по порядку — агент отсортирует по строкам
            S("Ф-1", Q_F1, "functional"),
            S("Ф-3", Q_F3, "functional"),
            S("", Q_NFR, "nonfunctional"),
            S("Ф-99", "Этой цитаты нет в документе вообще, система должна летать", "functional"),  # выдумка
        ])

    def _RequirementBatchResult(self, user):
        block = user.split("ПРОВЕРЬ ТОЛЬКО ЭТИ ТРЕБОВАНИЯ:")[1].split("КРИТЕРИИ")[0]
        ids = re.findall(r"^- (R-\d+)", block, flags=re.M)
        out = []
        for rid in ids:
            crit = []
            for code in REQ_CODES:
                v = dict(criterion=code, status="ok", severity="minor", quote="", problem="", recommendation="")
                if rid == "R-04" and code == "VER":      # НФТ
                    v.update(status="violated", severity="critical", quote="в течение минуты",
                             problem="Нет условий и перцентиля измерения.", recommendation="Задать p95 ≤ 60 с.")
                if rid == "R-03" and code == "COR":      # Ф-4, цитата выдумана
                    v.update(status="violated", severity="major", quote="цитата, которой нет",
                             problem="Что-то неверно.", recommendation="Исправить.")
                crit.append(CriterionVerdict(**v))
            out.append(RequirementCheck(req_id=rid, criteria=crit))
        return RequirementBatchResult(checks=out)

    def _SetResult(self, user):
        return SetResult(defects=[
            SetDefect(criterion="S-CNS", kind="contradiction", severity="critical", related_req_ids=["R-02", "R-03", "R-99"],
                      quotes=[QuoteRef(text=Q_F3), QuoteRef(text="независимо от истории предыдущей настройки")],
                      problem="Один push в месяц и повторный push после смены порога несовместимы.",
                      recommendation="Определить приоритет правил."),
            SetDefect(criterion="S-COM", kind="omission", severity="major", related_req_ids=[],
                      quotes=[QuoteRef(text="Выдуманная цитата без привязки")],
                      problem="Пропуск без привязки к тексту.", recommendation="—"),
        ], checklist=[ChecklistItem(item="Требование может быть протестировано", status="partial",
                                     justification="Раздел тестирования пуст.")])

    def _VerificationResult(self, user):
        ids = re.findall(r"^- (D-\d+) \[\w+, COR\]", user, flags=re.M)
        return VerificationResult(decisions=[Decision(defect_id=i, keep=False, reason="не подтверждается") for i in ids])

    def _ConclusionText(self, user):
        return ConclusionText(rationale="Есть критичные дефекты.", top_fixes=[
            FixItem(defect_ids=["D-001", "D-777"], action="Определить измеримый критерий.")])


@pytest.fixture(scope="module")
def result():
    return pipeline.run(load_document(DOC), FakeLLM(), batch_size=2)


def test_segmentation_sorted_grounded_and_numbered(result):
    assert [r.id for r in result.reqs] == ["R-01", "R-02", "R-03", "R-04"]
    assert [r.doc_id for r in result.reqs] == ["Ф-1", "Ф-3", "Ф-4", ""]       # порядок документа
    assert result.reqs[2].line_from == 40                                       # Ф-4 — строка 40
    assert any("Этой цитаты нет" in w for w in result.warnings)                 # выдумка отброшена


def test_requirement_defects_grounded(result):
    ver = [d for d in result.defects if d.criterion == "VER"]
    assert len(ver) == 1 and ver[0].quotes[0].line_from == 46 and ver[0].severity == "critical"
    cor = [d for d in result.defects if d.criterion == "COR"]
    assert len(cor) == 1 and cor[0].rejected and cor[0].quotes[0].approx      # привязан к требованию, отклонён верификатором
    assert result.matrix["R-03"]["COR"] == "ok" and result.matrix["R-04"]["VER"] == "violated"


def test_set_defects_and_dropped_hallucination(result):
    sets = [d for d in result.defects if d.scope == "set"]
    assert len(sets) == 1 and sets[0].kind == "contradiction"
    assert sets[0].req_ids == ["R-02", "R-03"]                                  # R-99 отфильтрован
    assert {q.line_from for q in sets[0].quotes} == {39, 40}
    assert any("без проверяемой привязки" in w for w in result.warnings)


def test_verdict_and_conclusion(result):
    assert result.verdict == "no"
    assert [f.defect_ids for f in result.conclusion.top_fixes] == [["D-001"]] or \
        all("D-777" not in f.defect_ids for f in result.conclusion.top_fixes)


def test_report_structure(result):
    md = report.render(result)
    for h in ("## 1. Паспорт прогона", "## 3. Как агент нарезал требования", "## 4. Проверка каждого требования",
              "## 5. Проверка всего набора требований", "## 6. Сводка", "## 7. Итог",
              "Приложение А", "Приложение Б"):
        assert h in md, h
    assert "| R-03 | Ф-4 |" in md and "fake-model" in md and "Уведомление о расходах по карте" in md
    assert "Выдуманная цитата" not in md.split("Приложение Б")[0]
    assert "НЕТ" in md
    assert "строка 39" in md and "строка 40" in md
    assert report.to_json(result).startswith("{")


def test_grounding_tolerates_markdown_and_ellipsis_and_fuzzy():
    g = Grounder(load_document(DOC))
    assert g.locate("**Ф-3.** Когда расходы переходят от суммы ниже порога")
    loc = g.locate("Месяц определяется по дате совершения покупки… Переводы и снятие наличных исключаются.")
    assert loc and loc.line_from == loc.line_to == 38 and not loc.approx
    fuzzy = g.locate("Если она ниже нового порога, следующее достижение вызывает push, независимо от истории прежней настройки.")
    assert fuzzy and fuzzy.approx and fuzzy.line_from == 40
    assert g.locate("Совершенно постороннее предложение про космос и котиков") is None
    assert g.locate("") is None


def test_loader_errors(tmp_path):
    with pytest.raises(LoadError):
        load_document(tmp_path / "net.md")
    (tmp_path / "x.xyz").write_text("a")
    with pytest.raises(LoadError):
        load_document(tmp_path / "x.xyz")
    p = tmp_path / "cp.txt"
    p.write_bytes("Требование: система должна работать".encode("cp1251"))
    assert "Требование" in load_document(p).lines[0]


def test_cli_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "make_client", lambda *a, **k: FakeLLM())
    out, js = tmp_path / "out" / "report.md", tmp_path / "r.json"
    assert cli.main([str(DOC), str(out), "--json", str(js), "--batch-size", "3"]) == 0
    assert out.read_text(encoding="utf-8").startswith("# Отчёт о проверке требований") and js.exists()
    assert cli.main([str(tmp_path / "missing.md"), str(out)]) == 1
