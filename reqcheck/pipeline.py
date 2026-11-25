"""Конвейер проверки.

  1. нарезка на требования
  2. проверка каждого требования по критериям (пачками, параллельно)
  3. проверка набора целиком + отдельный проход по противоречиям
  4. объединение дублей
  5. верификация критичных и значительных дефектов
  6. итог

Один агент, одна модель, несколько вызовов с разными задачами. Всё, что можно посчитать кодом
(нумерация, строки цитат, счётчики, вердикт), считается кодом, а не моделью.
"""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__, prompts
from .criteria import REQ_CODES, SEVERITY_ORDER
from .grounding import Grounder
from .llm import LLMClient
from .loader import Document
from .schemas import (ConclusionText, ConflictScanResult, DuplicateResult, RequirementBatchResult,
                      SegmentationResult, SetDefect, SetResult, VerificationResult)

VERDICT_TEXT = {
    "no": "НЕТ — начинать разработку и тестирование по документу в текущем виде нельзя",
    "conditional": "УСЛОВНО — начинать можно только после устранения значительных дефектов",
    "yes": "ДА — документ пригоден для начала разработки и тестирования",
}
TOTAL_STEPS = 6


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
    merged_into: str = ""      # если дефект признан дублем — id дефекта, в который он влит


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
    seg = llm.generate(system, user, SegmentationResult, "нарезка", effort="low")
    found: list[tuple[int, int, str, object]] = []
    seen: set[str] = set()
    for s in seg.requirements:
        loc = grounder.locate(s.quote, prefer=[(s.line_from, s.line_to)] if s.line_from > 0 else None)
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
def _check_requirements(doc, llm, grounder, reqs, batch_size, workers, warnings):
    by_id = {r.id: r for r in reqs}
    results: dict[str, dict] = {}   # req_id -> {code: CriterionVerdict}

    def call(batch: list[Req], label: str):
        system, user = prompts.requirement_check_prompt(doc.name, doc.numbered(), _req_view(reqs), _req_view(batch))
        return llm.generate(system, user, RequirementBatchResult, label, effort="low")

    def absorb(res: RequirementBatchResult):
        for chk in res.checks:
            rid = chk.req_id.strip()
            if rid in by_id and rid not in results:
                results[rid] = {c.criterion.strip().upper(): c for c in chk.criteria}

    batches = [reqs[i:i + batch_size] for i in range(0, len(reqs), batch_size)]
    _log(f"[2/{TOTAL_STEPS}] Проверка требований: {len(batches)} пачек, потоков: {workers}")
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        futs = {pool.submit(call, b, f"требования {b[0].id}–{b[-1].id}"): b for b in batches}
        for done, fut in enumerate(as_completed(futs), start=1):
            b = futs[fut]
            absorb(fut.result())
            _log(f"      готово {done}/{len(batches)} ({b[0].id}…{b[-1].id})")
    missing = [r for r in reqs if r.id not in results]
    if missing:
        _log(f"      повтор для пропущенных: {', '.join(r.id for r in missing)}")
        absorb(call(missing, "повтор пропущенных"))

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
            # ищем цитату рядом с самим требованием, а не по всему документу
            loc = grounder.locate(v.quote, prefer=[(r.line_from, r.line_to)])
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
def _to_defects(raw: list[SetDefect], reqs: list[Req], grounder: Grounder, warnings: list[str],
                src: str) -> list[Defect]:
    by_id = {r.id: r for r in reqs}
    out: list[Defect] = []
    for d in raw:
        rids = []
        for x in d.related_req_ids:
            x = x.strip()
            if x in by_id and x not in rids:
                rids.append(x)
        prefer = [(by_id[r].line_from, by_id[r].line_to) for r in rids] or None
        quotes: list[Quote] = []
        for q in d.quotes:
            loc = grounder.locate(q.text, prefer=prefer)
            if loc is None:
                warnings.append(f"{src}/{d.criterion}: цитата не найдена в документе и отброшена: «{q.text[:80]}»")
                continue
            quotes.append(Quote(loc.text, loc.line_from, loc.line_to, loc.approx))
        if not quotes and rids:
            quotes = [Quote(by_id[r].quote, by_id[r].line_from, by_id[r].line_to, True) for r in rids[:2]]
        if not quotes or not d.problem.strip():
            warnings.append(f"{src}/{d.criterion}: дефект без проверяемой привязки к тексту отброшен: "
                            f"{d.problem[:100]}")
            continue
        out.append(Defect("", "set", d.criterion.strip().upper(), d.severity, d.kind, rids, quotes,
                          d.problem.strip(), d.recommendation.strip()))
    return out


def _check_set(doc, llm, grounder, reqs, warnings):
    """Два независимых прохода по набору — общий и специальный (противоречия) — идут параллельно."""
    sys1, user1 = prompts.set_check_prompt(doc.name, doc.numbered(), _req_view(reqs))
    sys2, user2 = prompts.conflict_scan_prompt(doc.name, doc.numbered(), _req_view(reqs))
    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(llm.generate, sys1, user1, SetResult, "набор", "high")
        f2 = pool.submit(llm.generate, sys2, user2, ConflictScanResult, "противоречия", "high")
        res, scan = f1.result(), f2.result()
    defects = _to_defects(res.defects, reqs, grounder, warnings, "Набор")
    defects += _to_defects(scan.defects, reqs, grounder, warnings, "Противоречия")
    return defects, res.checklist


# --- Этап 4: дубли -------------------------------------------------------------
def _dedup(doc, llm, defects: list[Defect], warnings: list[str]) -> None:
    if len(defects) < 2:
        return
    by_id = {d.id: d for d in defects}
    view = [dict(id=d.id, severity=d.severity, criterion=d.criterion,
                 scope=("требование " + ", ".join(d.req_ids)) if d.scope == "req" else "набор в целом",
                 quotes=[q.text for q in d.quotes], problem=d.problem) for d in defects]
    system, user = prompts.dedup_prompt(view)
    res = llm.generate(system, user, DuplicateResult, "дубли", effort="medium")
    used: set[str] = set()
    merged = 0
    for g in res.groups:
        ids = [x.strip() for x in [g.keep_id, *g.drop_ids] if x.strip() in by_id]
        ids = [x for i, x in enumerate(ids) if x not in ids[:i] and x not in used]
        if len(ids) < 2:
            continue
        group = [by_id[x] for x in ids]
        # остаётся дефект уровня набора, если он есть в группе, иначе тот, что назвала модель
        keep = next((d for d in group if d.scope == "set"), group[0])
        for d in group:
            if d is keep:
                continue
            for rid in d.req_ids:
                if rid not in keep.req_ids:
                    keep.req_ids.append(rid)
            for q in d.quotes:
                if all((q.line_from, q.line_to) != (k.line_from, k.line_to) for k in keep.quotes):
                    keep.quotes.append(q)
            if SEVERITY_ORDER[d.severity] < SEVERITY_ORDER[keep.severity]:
                keep.severity = d.severity
            d.rejected, d.merged_into = True, keep.id
            d.reject_reason = f"дубль {keep.id}"
            merged += 1
        used.update(ids)
    _log(f"      объединено дублей: {merged}")


# --- Этап 5: верификация -------------------------------------------------------
def _verify(doc, llm, defects: list[Defect], matrix: dict | None = None):
    targets = [d for d in defects if not d.rejected and d.severity in ("critical", "major")]
    if not targets:
        return
    by_id = {d.id: d for d in defects}
    chunk = 25
    for i in range(0, len(targets), chunk):
        part = targets[i:i + chunk]
        _log(f"[5/{TOTAL_STEPS}] Верификация дефектов {part[0].id}…{part[-1].id}")
        view = [dict(id=d.id, severity=d.severity, criterion=d.criterion,
                     scope=("требование " + ", ".join(d.req_ids)) if d.scope == "req" else "набор в целом",
                     quotes=[q.text for q in d.quotes], problem=d.problem) for d in part]
        system, user = prompts.verification_prompt(doc.name, doc.numbered(), view)
        res = llm.generate(system, user, VerificationResult, "верификация", effort="medium")
        for dec in res.decisions:
            d = by_id.get(dec.defect_id.strip())
            if d is not None and not dec.keep and d in part:
                d.rejected, d.reject_reason = True, dec.reason.strip() or "отклонён верификатором"
    # отклонённый верификатором дефект по требованию не должен светиться ✗ в матрице критериев
    if matrix is not None:
        for d in defects:
            if d.rejected and not d.merged_into and d.scope == "req":
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
    concl = llm.generate(system, user, ConclusionText, "итог", effort="medium")
    valid = {d.id for d in active}
    alias = {d.id: d.merged_into for d in result.defects if d.merged_into}
    for fx in concl.top_fixes:
        fixed: list[str] = []
        for x in fx.defect_ids:
            x = alias.get(x, x)               # дубль -> дефект, в который он влит
            if x in valid:
                if x not in fixed:
                    fixed.append(x)
            else:
                warnings.append(f"Итог: убрана ссылка на отсутствующий или отклонённый дефект {x}")
        fx.defect_ids = fixed
    return concl


def run(doc: Document, llm: LLMClient, batch_size: int = 10, verify: bool = True, workers: int = 3) -> RunResult:
    t0 = time.monotonic()
    warnings: list[str] = []
    grounder = Grounder(doc)

    _log(f"[1/{TOTAL_STEPS}] Нарезка на требования…")
    title, reqs = _segment(doc, llm, grounder, warnings)
    _log(f"      выделено требований: {len(reqs)}")

    matrix, req_defects = _check_requirements(doc, llm, grounder, reqs, batch_size, workers, warnings)

    _log(f"[3/{TOTAL_STEPS}] Проверка набора в целом и поиск противоречий…")
    set_defects, checklist = _check_set(doc, llm, grounder, reqs, warnings)

    defects = req_defects + set_defects
    defects.sort(key=lambda d: (0 if d.scope == "req" else 1, d.req_ids[0] if d.req_ids else "~",
                                SEVERITY_ORDER[d.severity]))
    for i, d in enumerate(defects, start=1):
        d.id = f"D-{i:03d}"

    _log(f"[4/{TOTAL_STEPS}] Объединение дублей среди {len(defects)} дефектов…")
    _dedup(doc, llm, defects, warnings)

    if verify:
        _verify(doc, llm, defects, matrix)
    else:
        _log(f"[5/{TOTAL_STEPS}] Верификация отключена (--no-verify)")

    result = RunResult(doc=doc, title=title, reqs=reqs, matrix=matrix, defects=defects, checklist=checklist,
                       conclusion=None, verdict="yes", warnings=warnings)
    result.verdict = verdict_of(defects)

    _log(f"[6/{TOTAL_STEPS}] Итог…")
    result.conclusion = _conclude(doc, llm, result, warnings)

    result.meta = dict(
        provider=llm.provider, model=" → ".join(getattr(llm, "models_used", None) or [llm.model]), calls=llm.calls,
        input_tokens=llm.input_tokens, output_tokens=llm.output_tokens,
        seconds=round(time.monotonic() - t0, 1), batch_size=batch_size, workers=workers, verify=verify,
        agent_version=agent_version(),
    )
    return result
