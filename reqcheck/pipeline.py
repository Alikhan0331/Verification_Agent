"""Конвейер проверки: нарезка -> проверка требований -> проверка набора -> верификация -> итог.

Один агент, одна модель, несколько вызовов с разными задачами. Всё, что можно посчитать
кодом (нумерация, строки цитат, счётчики, вердикт), считается кодом, а не моделью.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__, prompts
from .criteria import REQ_CODES, SEVERITY_ORDER
from .grounding import Grounder
from .llm import LLMClient
from .loader import Document
from .schemas import (ConclusionText, RequirementBatchResult, SegmentationResult, SetResult,
                      VerificationResult)

VERDICT_TEXT = {
    "no": "НЕТ — начинать разработку и тестирование по документу в текущем виде нельзя",
    "conditional": "УСЛОВНО — начинать можно только после устранения значительных дефектов",
    "yes": "ДА — документ пригоден для начала разработки и тестирования",
}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass
class Req:
    id: str
    doc_id: str
    section: str
    kind: str
    quote: str
    summary: str
    line_from: int
    line_to: int


@dataclass
class Quote:
    text: str
    line_from: int
    line_to: int
    approx: bool = False


@dataclass
class Defect:
    id: str
    scope: str                 # "req" | "set"
    criterion: str
    severity: str
    kind: str                  # contradiction | omission | ... | "requirement"
    req_ids: list[str]
    quotes: list[Quote]
    problem: str
    recommendation: str
    rejected: bool = False
    reject_reason: str = ""


@dataclass
class RunResult:
    doc: Document
    title: str
    reqs: list[Req]
    matrix: dict[str, dict[str, str]]
    defects: list[Defect]
    checklist: list
    conclusion: ConclusionText | None
    verdict: str
    warnings: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def active(self) -> list[Defect]:
        return [d for d in self.defects if not d.rejected]


def agent_version() -> str:
    root = Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], cwd=root,
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return f"{out.stdout.strip()} (reqcheck {__version__})"
    except (OSError, subprocess.SubprocessError):
        pass
    return f"{__version__} (git-информация недоступна)"


def _req_view(reqs: list[Req]) -> list[dict]:
    return [dict(id=r.id, doc_id=r.doc_id, section=r.section, summary=r.summary, quote=r.quote) for r in reqs]


# --- Этап 1 ------------------------------------------------------------------
def _segment(doc: Document, llm: LLMClient, grounder: Grounder, warnings: list[str]) -> tuple[str, list[Req]]:
    system, user = prompts.segmentation_prompt(doc.name, doc.numbered())
    seg = llm.generate(system, user, SegmentationResult, "нарезка")
    found: list[tuple[int, int, str, object]] = []
    seen: set[str] = set()
    for s in seg.requirements:
        loc = grounder.locate(s.quote)
        if loc is None:
            warnings.append(f"Нарезка: цитата не найдена в документе, элемент отброшен: «{s.quote[:80]}»")
            continue
        key = f"{loc.line_from}:{loc.line_to}:{s.doc_id}"
        if key in seen:
            continue
        seen.add(key)
        found.append((loc.line_from, loc.line_to, key, (s, loc)))
    found.sort(key=lambda t: (t[0], t[1]))
    if not found:
        raise RuntimeError("Агент не выделил ни одного требования — проверьте документ и модель")
    width = 3 if len(found) > 99 else 2
    reqs = []
    for i, (_, _, _, (s, loc)) in enumerate(found, start=1):
        reqs.append(Req(f"R-{i:0{width}d}", s.doc_id.strip(), s.section.strip(), s.kind, loc.text,
                        s.summary.strip(), loc.line_from, loc.line_to))
    return seg.document_title.strip() or doc.name, reqs


# --- Этап 2 ------------------------------------------------------------------
def _check_requirements(doc, llm, grounder, reqs, batch_size, warnings):
    by_id = {r.id: r for r in reqs}
    results: dict[str, dict] = {}   # req_id -> {code: CriterionVerdict}

    def run_batch(batch: list[Req], label: str):
        system, user = prompts.requirement_check_prompt(doc.name, doc.numbered(), _req_view(reqs), _req_view(batch))
        res = llm.generate(system, user, RequirementBatchResult, label)
        for chk in res.checks:
            rid = chk.req_id.strip()
            if rid in by_id and rid not in results:
                results[rid] = {c.criterion.strip().upper(): c for c in chk.criteria}

    batches = [reqs[i:i + batch_size] for i in range(0, len(reqs), batch_size)]
    for n, b in enumerate(batches, start=1):
        _log(f"[2/5] Проверка требований: пачка {n}/{len(batches)} ({b[0].id}…{b[-1].id})")
        run_batch(b, f"требования {b[0].id}–{b[-1].id}")
    missing = [r for r in reqs if r.id not in results]
    if missing:
        _log(f"      повтор для пропущенных: {', '.join(r.id for r in missing)}")
        run_batch(missing, "повтор пропущенных")

    matrix: dict[str, dict[str, str]] = {}
    defects: list[Defect] = []
    for r in reqs:
        verdicts = results.get(r.id)
        if verdicts is None:
            warnings.append(f"{r.id}: модель не вернула проверку — требование НЕ проверено")
            matrix[r.id] = {c: "?" for c in REQ_CODES}
            continue
        row = {}
        for code in REQ_CODES:
            v = verdicts.get(code)
            if v is None:
                warnings.append(f"{r.id}: нет оценки по критерию {code}")
                row[code] = "?"
                continue
            if v.status != "violated":
                row[code] = v.status
                continue
            if not v.problem.strip():
                warnings.append(f"{r.id}/{code}: нарушение без описания проблемы — отброшено")
                row[code] = "ok"
                continue
            loc = grounder.locate(v.quote)
            if loc is None:
                warnings.append(f"{r.id}/{code}: цитата дефекта не найдена в документе, "
                                f"дефект привязан к тексту требования")
                q = Quote(r.quote, r.line_from, r.line_to, True)
            else:
                q = Quote(loc.text, loc.line_from, loc.line_to, loc.approx)
            row[code] = "violated"
            defects.append(Defect("", "req", code, v.severity, "requirement", [r.id], [q],
                                  v.problem.strip(), v.recommendation.strip()))
        matrix[r.id] = row
    return matrix, defects


# --- Этап 3 ------------------------------------------------------------------
def _check_set(doc, llm, grounder, reqs, warnings):
    by_id = {r.id: r for r in reqs}
    system, user = prompts.set_check_prompt(doc.name, doc.numbered(), _req_view(reqs))
    res = llm.generate(system, user, SetResult, "набор")
    defects: list[Defect] = []
    for d in res.defects:
        rids = [x.strip() for x in d.related_req_ids if x.strip() in by_id]
        quotes: list[Quote] = []
        for q in d.quotes:
            loc = grounder.locate(q.text)
            if loc is None:
                warnings.append(f"Набор/{d.criterion}: цитата не найдена в документе и отброшена: «{q.text[:80]}»")
                continue
            quotes.append(Quote(loc.text, loc.line_from, loc.line_to, loc.approx))
        if not quotes and rids:
            quotes = [Quote(by_id[r].quote, by_id[r].line_from, by_id[r].line_to, True) for r in rids[:2]]
        if not quotes or not d.problem.strip():
            warnings.append(f"Набор/{d.criterion}: дефект без проверяемой привязки к тексту отброшен: "
                            f"{d.problem[:100]}")
            continue
        defects.append(Defect("", "set", d.criterion.strip().upper(), d.severity, d.kind, rids, quotes,
                              d.problem.strip(), d.recommendation.strip()))
    return defects, res.checklist


# --- Этап 4 ------------------------------------------------------------------
def _verify(doc, llm, defects: list[Defect], matrix: dict | None = None):
    targets = [d for d in defects if d.severity in ("critical", "major")]
    if not targets:
        return
    by_id = {d.id: d for d in defects}
    chunk = 25
    for i in range(0, len(targets), chunk):
        part = targets[i:i + chunk]
        _log(f"[4/5] Верификация дефектов {part[0].id}…{part[-1].id}")
        view = [dict(id=d.id, severity=d.severity, criterion=d.criterion,
                     scope=("требование " + ", ".join(d.req_ids)) if d.scope == "req" else "набор в целом",
                     quotes=[q.text for q in d.quotes], problem=d.problem) for d in part]
        system, user = prompts.verification_prompt(doc.name, doc.numbered(), view)
        res = llm.generate(system, user, VerificationResult, "верификация")
        for dec in res.decisions:
            d = by_id.get(dec.defect_id.strip())
            if d is not None and not dec.keep and d in part:
                d.rejected, d.reject_reason = True, dec.reason.strip() or "отклонён верификатором"
    # отклонённый дефект по требованию не должен светиться ✗ в матрице критериев
    if matrix is not None:
        for d in defects:
            if d.rejected and d.scope == "req":
                matrix[d.req_ids[0]][d.criterion] = "ok"


# --- Итог --------------------------------------------------------------------
def verdict_of(defects: list[Defect]) -> str:
    active = [d for d in defects if not d.rejected]
    if any(d.severity == "critical" for d in active):
        return "no"
    if any(d.severity == "major" for d in active):
        return "conditional"
    return "yes"


def _conclude(doc, llm, result: RunResult, warnings):
    active = result.active
    important = [d for d in active if d.severity in ("critical", "major")]
    reqs = {r.id: r for r in result.reqs}

    def where(d: Defect) -> str:
        if d.req_ids:
            return ", ".join(f"{r}({reqs[r].doc_id})" if reqs[r].doc_id else r for r in d.req_ids)
        return "набор в целом"

    stats = (f"требований {len(result.reqs)}, дефектов {len(active)} "
             f"(критичных {sum(d.severity == 'critical' for d in active)}, "
             f"значительных {sum(d.severity == 'major' for d in active)}, "
             f"незначительных {sum(d.severity == 'minor' for d in active)})")
    view = [dict(id=d.id, severity=d.severity, criterion=d.criterion, where=where(d), problem=d.problem)
            for d in sorted(important, key=lambda d: SEVERITY_ORDER[d.severity])]
    system, user = prompts.conclusion_prompt(VERDICT_TEXT[result.verdict], stats, view)
    concl = llm.generate(system, user, ConclusionText, "итог")
    valid = {d.id for d in active}
    for fx in concl.top_fixes:
        bad = [x for x in fx.defect_ids if x not in valid]
        if bad:
            warnings.append(f"Итог: убраны несуществующие ссылки на дефекты: {', '.join(bad)}")
            fx.defect_ids = [x for x in fx.defect_ids if x in valid]
    return concl


def run(doc: Document, llm: LLMClient, batch_size: int = 6, verify: bool = True) -> RunResult:
    t0 = time.monotonic()
    warnings: list[str] = []
    grounder = Grounder(doc)

    _log("[1/5] Нарезка на требования…")
    title, reqs = _segment(doc, llm, grounder, warnings)
    _log(f"      выделено требований: {len(reqs)}")

    matrix, req_defects = _check_requirements(doc, llm, grounder, reqs, batch_size, warnings)

    _log("[3/5] Проверка набора в целом…")
    set_defects, checklist = _check_set(doc, llm, grounder, reqs, warnings)

    defects = req_defects + set_defects
    defects.sort(key=lambda d: (0 if d.scope == "req" else 1, d.req_ids[0] if d.req_ids else "~",
                                SEVERITY_ORDER[d.severity]))
    for i, d in enumerate(defects, start=1):
        d.id = f"D-{i:03d}"

    if verify:
        _verify(doc, llm, defects, matrix)
    else:
        _log("[4/5] Верификация отключена (--no-verify)")

    result = RunResult(doc=doc, title=title, reqs=reqs, matrix=matrix, defects=defects, checklist=checklist,
                       conclusion=None, verdict="yes", warnings=warnings)
    result.verdict = verdict_of(defects)

    _log("[5/5] Итог…")
    result.conclusion = _conclude(doc, llm, result, warnings)

    result.meta = dict(
        provider=llm.provider, model=llm.model, calls=llm.calls,
        input_tokens=llm.input_tokens, output_tokens=llm.output_tokens,
        seconds=round(time.monotonic() - t0, 1), batch_size=batch_size, verify=verify,
        agent_version=agent_version(),
    )
    return result
