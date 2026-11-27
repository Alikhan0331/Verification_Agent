"""Режим ответов автора: вопросы агента → ответы автора → оценка ответов → обновлённый отчёт.

Два шага, оба поверх готового прогона (`report.json`), без повторной проверки всего документа:

    python -m reqcheck.reply template report.json answers.md
        — шаблон файла ответов: по блоку на каждый вопрос агента;
    python -m reqcheck.reply apply requirements.md report.json answers.md report_after.md
        — агент оценивает каждый ответ: закрывает ли он дефекты, достаточно ли он конкретен,
          не противоречит ли документу; закрытые дефекты уходят из сводки, вердикт и итог пересчитываются,
          к недостаточным ответам добавляются более узкие вопросы, а к закрывающим — готовая формулировка
          для вставки в документ.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import __version__, pipeline, prompts, report, textout
from .conformance import check_report
from .grounding import Grounder
from .llm import LLMError, make_client
from .loader import LoadError, load_document
from .pipeline import Defect, Quote, Req, RunResult
from .schemas import AnswersResult, ChecklistItem, ConclusionText, Question

_HEAD = re.compile(r"^\s*#{1,6}\s*\**\s*(Q-\d+)\b")
_COMPACT = re.compile(r"^\s*(?:[-*]\s*)?\**\s*(Q-\d+)\s*\**\s*[:.)]\s*(.*)$")
_ANSWER = re.compile(r"^\s*\**\s*Ответ\s*\**\s*:\s*\**\s*(.*)$", re.IGNORECASE)
_SKIP = re.compile(r"^\s*(?:\**\s*(?:Вопрос|Варианты|Закрывает|Адресат|Блокирует|Цитата)\b|[-*>]?\s*Вариант\s+[АБВ]\b|>)",
                   re.IGNORECASE)
_EMPTY = re.compile(r"^[\s—–\-_.…]*$")


# --- Загрузка результата прогона --------------------------------------------------
def load_result(json_path: str | Path, doc_path: str | Path) -> RunResult:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    doc = load_document(doc_path)
    if data.get("sha256") and data["sha256"] != doc.sha256:
        raise RuntimeError("Документ изменился с момента прогона (не совпадает SHA-256): "
                           "сначала повторите проверку документа, затем применяйте ответы")
    defects = []
    for d in data["defects"]:
        d = dict(d)
        d["quotes"] = [Quote(**q) for q in d["quotes"]]
        defects.append(Defect(**d))
    questions = [Question(**{k: v for k, v in q.items() if k != "id"}) for q in data.get("questions", [])]
    concl = ConclusionText(**data["conclusion"]) if data.get("conclusion") else None
    return RunResult(doc=doc, title=data["title"], reqs=[Req(**r) for r in data["requirements"]],
                     matrix=data["matrix"], defects=defects,
                     checklist=[ChecklistItem(**c) for c in data.get("checklist", [])], conclusion=concl,
                     verdict=data["verdict"], warnings=list(data.get("warnings", [])), meta=dict(data["meta"]),
                     questions=questions, answers=list(data.get("answers", [])))


# --- Шаблон ответов ---------------------------------------------------------------
def make_template(res: RunResult) -> str:
    by_id = {d.id: d for d in res.defects}
    out = ["# Ответы автора на уточняющие вопросы",
           "",
           f"Документ: {res.title}. Впишите ответ после «Ответ:». Можно ответить буквой варианта (А, Б, В) или своими словами; "
           "пустой ответ означает «пока без ответа». Блоки `## Q-NN` не переименовывайте.",
           ""]
    for i, q in enumerate(res.questions, start=1):
        first = by_id.get(q.defect_ids[0])
        out += [f"## Q-{i:02d}", "", f"Вопрос: {q.question}", ""]
        for k, o in enumerate(q.options):
            out.append(f"Вариант {'АБВ'[k]}: {o}")
        if q.options:
            out.append("")
        out.append("Закрывает: " + ", ".join(q.defect_ids) + ("; блокирует разработку и тестирование" if q.blocking else ""))
        if first and first.quotes:
            out.append("Цитата: " + " ".join(first.quotes[0].text.split()))
        out += ["", "Ответ:", ""]
    return "\n".join(out).rstrip() + "\n"


def parse_answers(text: str) -> dict[str, str]:
    """Q-NN → текст ответа. Понимает шаблон (`## Q-01` … `Ответ: …`) и краткую форму `Q-01: …`."""
    answers: dict[str, list[str]] = {}
    cur: str | None = None
    mode = ""                      # "block" — после «Ответ:» в шаблоне; "compact" — краткая форма
    for ln in text.splitlines():
        m = _HEAD.match(ln)
        if m:
            cur, mode = m.group(1), "head"
            answers.setdefault(cur, [])
            continue
        m = _COMPACT.match(ln)
        if m and not (mode == "block"):
            cur, mode = m.group(1), "compact"
            answers[cur] = [m.group(2)]
            continue
        if cur is None:
            continue
        a = _ANSWER.match(ln)
        if mode == "head" and a:
            mode = "block"
            answers[cur].append(a.group(1))
            continue
        if mode == "block" or mode == "compact":
            if _SKIP.match(ln):
                continue
            answers[cur].append(ln)
    out = {}
    for qid, lines in answers.items():
        txt = "\n".join(lines).strip()
        if txt and not _EMPTY.match(txt):
            out[qid] = txt
    return out


# --- Применение ответов -----------------------------------------------------------
def apply_answers(res: RunResult, llm, answers_text: str) -> RunResult:
    """Оценивает ответы автора и обновляет результат: закрывает дефекты, добавляет вопросы, пересчитывает итог."""
    given = parse_answers(answers_text)
    n_q = len(res.questions)
    qids = {f"Q-{i:02d}": i - 1 for i in range(1, n_q + 1)}
    for qid in given:
        if qid not in qids:
            res.warnings.append(f"Ответы: в файле есть {qid}, а такого вопроса в отчёте нет — пропущен")
    given = {k: v for k, v in given.items() if k in qids}
    if not given:
        raise RuntimeError("В файле ответов нет ни одного заполненного ответа на вопросы этого отчёта")

    by_id = {d.id: d for d in res.defects}
    items = []
    for qid, ans in given.items():
        q = res.questions[qids[qid]]
        items.append(dict(id=qid, question=q.question, options=q.options, answer=ans,
                          defects=[dict(id=x, severity=by_id[x].severity, criterion=by_id[x].criterion,
                                        problem=by_id[x].problem, quotes=[t.text for t in by_id[x].quotes])
                                   for x in q.defect_ids if x in by_id]))
    system, user = prompts.answers_prompt(res.doc.numbered(), items)
    parsed: AnswersResult = llm.generate(system, user, AnswersResult, "разбор ответов", effort="medium")

    grounder = Grounder(res.doc)
    next_id = max((int(d.id.split("-")[1]) for d in res.defects if d.id.startswith("D-")), default=0) + 1
    new_questions: list[Question] = []
    answered: list[dict] = []
    seen: set[str] = set()
    for a in parsed.assessments:
        qid = a.question_id.strip()
        if qid not in given or qid in seen:
            continue
        seen.add(qid)
        q = res.questions[qids[qid]]
        verdict = a.verdict
        closed = [x for x in dict.fromkeys(a.closed_defect_ids) if x in q.defect_ids]
        if verdict == "closes":
            closed = list(q.defect_ids)
        elif verdict == "partial" and not closed:
            verdict = "insufficient"
        elif verdict in ("insufficient", "contradicts"):
            closed = []
        proposed = a.proposed_text.strip() if closed else ""
        follow = " ".join(a.follow_up.split())
        open_ids = [x for x in q.defect_ids if x not in closed]

        if verdict == "contradicts" and a.conflict_quote.strip():
            loc = grounder.locate(a.conflict_quote)
            if loc is not None:
                nd = Defect(f"D-{next_id:03d}", "set", "S-CNS", "major", "contradiction", [],
                            [Quote(loc.text, loc.line_from, loc.line_to, loc.approx)],
                            f"Ответ автора на {qid} («{' '.join(given[qid].split())[:200]}») противоречит документу: "
                            f"{' '.join(a.conflict_problem.split())}",
                            "Согласовать ответ с указанным местом документа: изменить ответ или исправить документ.",
                            source="answer")
                next_id += 1
                res.defects.append(nd)
                open_ids.append(nd.id)
            else:
                res.warnings.append(f"Ответы: цитата конфликта для {qid} не найдена в документе и отброшена")

        for x in closed:
            d = by_id[x]
            if not d.resolved_by:
                d.resolved_by = qid
                if d.scope == "req" and d.req_ids:
                    still = any(o is not d and not o.rejected and not o.resolved_by and o.scope == "req"
                                and o.req_ids[:1] == d.req_ids[:1] and o.criterion == d.criterion
                                for o in res.defects)
                    if not still and d.criterion in res.matrix.get(d.req_ids[0], {}):
                        res.matrix[d.req_ids[0]][d.criterion] = "ok"

        if verdict != "closes" and open_ids and "?" in follow and len(follow) >= 15:
            new_questions.append(Question(defect_ids=open_ids, owner=q.owner, question=follow, options=[],
                                          blocking=q.blocking))
        answered.append(dict(question_id=qid, answer=given[qid], verdict=verdict, closed_defect_ids=closed,
                             reason=" ".join(a.reason.split()), proposed_text=proposed))

    for qid in given:
        if qid not in seen:
            res.warnings.append(f"Ответы: модель не оценила ответ на {qid} — вопрос остаётся открытым")

    res.answers = answered
    res.questions = res.questions + new_questions
    res.defects.sort(key=lambda d: int(d.id.split("-")[1]))
    res.verdict = pipeline.verdict_of(res.defects)
    res.conclusion = pipeline._conclude(res.doc, llm, res, res.warnings)
    res.meta["reply"] = dict(answered=len(answered), closed=sum(1 for d in res.defects if d.resolved_by and not d.rejected),
                             model=" → ".join(getattr(llm, "models_used", None) or [llm.model]), calls=llm.calls,
                             agent_version=pipeline.agent_version())
    return res


# --- CLI --------------------------------------------------------------------------
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
    ap = argparse.ArgumentParser(prog="reqcheck.reply", description="Режим ответов автора на вопросы агента")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template", help="создать файл для ответов по готовому report.json")
    t.add_argument("document", help="проверенный документ (для сверки SHA-256)")
    t.add_argument("report_json")
    t.add_argument("answers_md")
    a = sub.add_parser("apply", help="оценить ответы и выпустить обновлённый отчёт")
    a.add_argument("document")
    a.add_argument("report_json")
    a.add_argument("answers_md")
    a.add_argument("output", help="путь к обновлённому отчёту (.md)")
    a.add_argument("--json", metavar="PATH", help="дополнительно сохранить результат в JSON")
    a.add_argument("--provider")
    a.add_argument("--model")
    a.add_argument("--min-interval", type=float)
    args = ap.parse_args(argv)

    try:
        res = load_result(args.report_json, args.document)
        if args.cmd == "template":
            Path(args.answers_md).write_text(make_template(res), encoding="utf-8")
            print(f"Шаблон ответов: {args.answers_md} (вопросов: {len(res.questions)})", file=sys.stderr)
            return 0
        llm = make_client(args.provider, args.model, args.min_interval)
        text = Path(args.answers_md).read_text(encoding="utf-8")
        res = apply_answers(res, llm, text)
    except (LoadError, LLMError, RuntimeError, OSError, KeyError, json.JSONDecodeError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    md = report.render(res)
    saved = textout.save_report(out, md)
    for p in check_report(res, md):
        print(f"Предупреждение (шаблон отчёта): {p}", file=sys.stderr)
    if args.json:
        Path(args.json).write_text(report.to_json(res), encoding="utf-8")
    rp = res.meta["reply"]
    print(f"Готово: {', '.join(str(p) for p in saved)} (ответов учтено: {rp['answered']}, закрыто дефектов: {rp['closed']}, "
          f"осталось: {len(res.active)}, вердикт: {pipeline.VERDICT_TEXT[res.verdict].split(' — ')[0]})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
