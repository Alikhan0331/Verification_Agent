"""Оценка качества агента на корпусе документов с заложенными дефектами.

Корпус: пары `NAME.md` + `NAME.expected.json`. В эталоне перечислены заложенные дефекты («семена»):

    {"document": "NAME.md",             # путь относительно файла эталона (по умолчанию NAME.md)
     "clean": false,                      # true — контрольный документ без заложенных дефектов
     "seeds": [{"id": "S01", "title": "...", "severity": "critical",
                "criteria": ["S-CNS", "COR"],       # какие критерии считаются «правильным» попаданием
                "anchors": ["дословная цитата 1", "цитата 2"],
                "min_anchors": 2}]}               # сколько якорей замечание должно покрыть своими цитатами

Дефект агента «находит» семя, если его критерий входит в список семени и цитаты дефекта
(по номерам строк) покрывают не меньше min_anchors якорей. Для противоречий поэтому нужны обе стороны.

Запуск:
    python -m reqcheck.eval tests/corpus --run --out eval_out        # прогнать агента и оценить
    python -m reqcheck.eval tests/corpus --results eval_out          # оценить уже готовые NAME.json
    python -m reqcheck.eval tests/corpus --results . --only task_requirements
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .grounding import Grounder
from .loader import load_document


@dataclass
class Seed:
    id: str
    title: str
    severity: str
    criteria: list[str]
    anchors: list[str]
    min_anchors: int = 1


def load_expected(path: Path) -> tuple[Path, bool, list[Seed]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    doc_path = (path.parent / data.get("document", path.name.replace(".expected.json", ".md"))).resolve()
    seeds = [Seed(s["id"], s["title"], s.get("severity", "major"), s["criteria"], s["anchors"],
                  s.get("min_anchors", 1)) for s in data.get("seeds", [])]
    return doc_path, bool(data.get("clean", False)), seeds


def anchor_lines(grounder: Grounder, seed: Seed) -> list[tuple[int, int]]:
    spans = []
    for a in seed.anchors:
        loc = grounder.locate(a)
        if loc is None or loc.approx:
            raise ValueError(f"{seed.id}: якорь не найден в документе дословно: «{a[:70]}»")
        spans.append((loc.line_from, loc.line_to))
    return spans


def _covers(defect: dict, span: tuple[int, int]) -> bool:
    return any(q["line_from"] <= span[1] and span[0] <= q["line_to"] for q in defect["quotes"])


def matches(defect: dict, seed: Seed, spans: list[tuple[int, int]]) -> bool:
    if defect["criterion"] not in seed.criteria:
        return False
    return sum(_covers(defect, s) for s in spans) >= min(seed.min_anchors, len(spans))


def evaluate(doc_name: str, seeds: list[Seed], spans_by_seed: dict[str, list[tuple[int, int]]],
             defects: list[dict], clean: bool) -> dict:
    active = [d for d in defects if not d.get("rejected")]
    found, missed = [], []
    used: set[str] = set()
    for s in seeds:
        hit = [d for d in active if matches(d, s, spans_by_seed[s.id])]
        if hit:
            found.append((s, [d["id"] for d in hit]))
            used.update(d["id"] for d in hit)
        else:
            missed.append(s)
    important = [d for d in active if d["severity"] in ("critical", "major")]
    unmatched = [d for d in important if d["id"] not in used]
    # «общее» попадание по семени любым критерием, но не тем — подсказка для разбора промптов
    near = {s.id: [d["id"] for d in active if d["criterion"] not in s.criteria
                   and sum(_covers(d, sp) for sp in spans_by_seed[s.id]) >= min(s.min_anchors, len(spans_by_seed[s.id]))]
            for s in missed}
    n_crit = sum(s.severity == "critical" for s in seeds)
    return dict(
        document=doc_name, clean=clean, seeds=len(seeds), found=len(found),
        recall=(len(found) / len(seeds)) if seeds else None,
        critical_seeds=n_crit,
        critical_found=sum(s.severity == "critical" for s, _ in found),
        found_ids=[(s.id, ids) for s, ids in found],
        missed=[dict(id=s.id, title=s.title, severity=s.severity, near=near.get(s.id, [])) for s in missed],
        active_defects=len(active), critical_defects=sum(d["severity"] == "critical" for d in active),
        important_defects=len(important), unmatched_important=[d["id"] for d in unmatched],
        unmatched_ratio=(len(unmatched) / len(important)) if important else 0.0,
    )


def render(results: list[dict]) -> str:
    L = ["# Оценка агента на корпусе", "",
         "| Документ | Заложено | Найдено | Полнота | Критичных найдено | Критичных дефектов у агента | "
         "Вне эталона (крит.+знач.) |",
         "| :-- | :-: | :-: | :-: | :-: | :-: | :-: |"]
    for r in results:
        if r["clean"]:
            L.append(f"| {r['document']} (контроль) | — | — | — | — | {r['critical_defects']} | "
                     f"{r['important_defects']} из {r['active_defects']} |")
        else:
            L.append(f"| {r['document']} | {r['seeds']} | {r['found']} | {r['recall']:.0%} | "
                     f"{r['critical_found']}/{r['critical_seeds']} | {r['critical_defects']} | "
                     f"{len(r['unmatched_important'])} из {r['important_defects']} |")
    tot_s = sum(r["seeds"] for r in results if not r["clean"])
    tot_f = sum(r["found"] for r in results if not r["clean"])
    if tot_s:
        L += ["", f"**Итоговая полнота: {tot_f}/{tot_s} = {tot_f / tot_s:.0%}**", ""]
    for r in results:
        if r["missed"]:
            L += [f"### Пропущено: {r['document']}", ""]
            for m in r["missed"]:
                hint = f" (замечания по этому месту есть, но с другим критерием: {', '.join(m['near'])})" if m["near"] else ""
                L.append(f"- **{m['id']}** [{m['severity']}] {m['title']}{hint}")
            L.append("")
        if r["unmatched_important"] and not r["clean"]:
            L += [f"Критичные и значительные дефекты вне эталона ({r['document']}) — проверить вручную, "
                  f"это либо ложные срабатывания, либо реальные находки сверх эталона: "
                  f"{', '.join(r['unmatched_important'])}", ""]
        if r["clean"] and r["important_defects"]:
            L += [f"Контрольный документ: у агента {r['important_defects']} критичных/значительных дефектов "
                  f"({', '.join(r['unmatched_important'])}) — проверить на ложные срабатывания.", ""]
    return "\n".join(L).rstrip() + "\n"


def run_corpus(corpus: Path, results_dir: Path | None, run: bool, only: str | None,
               batch_size: int, workers: int, verify: bool) -> list[dict]:
    out: list[dict] = []
    for exp in sorted(corpus.glob("*.expected.json")):
        name = exp.name.replace(".expected.json", "")
        if only and name != only:
            continue
        doc_path, clean, seeds = load_expected(exp)
        doc = load_document(doc_path)
        grounder = Grounder(doc)
        spans = {s.id: anchor_lines(grounder, s) for s in seeds}
        rj = (results_dir or Path(".")) / f"{name}.json"
        if run:
            from . import pipeline, report
            from .llm import LLMError, make_client
            llm = make_client()
            print(f"== {name}: прогон агента ({llm.provider}/{llm.model})", file=sys.stderr)
            try:
                res = pipeline.run(doc, llm, batch_size=batch_size, verify=verify, workers=workers)
            except (LLMError, RuntimeError) as e:     # не оцениваем по устаревшему результату прошлого прогона
                print(f"== {name}: прогон не удался, документ пропущен: {str(e)[:200]}", file=sys.stderr)
                continue
            rj.parent.mkdir(parents=True, exist_ok=True)
            (rj.parent / f"{name}.md").write_text(report.render(res), encoding="utf-8")
            rj.write_text(report.to_json(res), encoding="utf-8")
        if not rj.exists():
            print(f"нет результата {rj} — пропускаю {name}", file=sys.stderr)
            continue
        defects = json.loads(rj.read_text(encoding="utf-8"))["defects"]
        out.append(evaluate(doc.name, seeds, spans, defects, clean))
    return out


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    ap = argparse.ArgumentParser(prog="reqcheck.eval", description="Оценка агента на корпусе с заложенными дефектами")
    ap.add_argument("corpus", help="папка с NAME.md и NAME.expected.json")
    ap.add_argument("--run", action="store_true", help="прогнать агента на документах (тратит вызовы модели)")
    ap.add_argument("--results", "--out", dest="results", help="папка с NAME.json (куда писать при --run)")
    ap.add_argument("--only", help="оценить только один документ по имени (без расширения)")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--md", metavar="PATH", help="сохранить сводку оценки в файл")
    a = ap.parse_args(argv)
    results = run_corpus(Path(a.corpus), Path(a.results) if a.results else None, a.run, a.only,
                         a.batch_size, a.workers, not a.no_verify)
    if not results:
        print("Нечего оценивать.", file=sys.stderr)
        return 1
    md = render(results)
    print(md)
    if a.md:
        Path(a.md).write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
