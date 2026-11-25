"""Сборка отчёта (Markdown, по-русски) из структурированного результата.

Шаблон отчёта фиксирован в коде; модель пишет только содержимое замечаний и итоговый текст.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from collections import Counter

from .criteria import ALL_CRITERIA, REQ_CODES, REQ_CRITERIA, SEVERITY_ORDER, SEVERITY_RU, SET_CRITERIA
from .pipeline import VERDICT_TEXT, Defect, Quote, RunResult

_MARK = {"ok": "✓", "violated": "✗", "na": "–", "?": "?"}
_CHECKLIST_RU = {"yes": "Да", "partial": "Частично", "no": "Нет", "not_required": "Не требуется"}
_KIND_RU = {"contradiction": "противоречие", "omission": "пропуск", "terminology": "терминология",
            "traceability": "прослеживаемость", "validation": "валидируемость", "other": "прочее",
            "requirement": "формулировка"}


def _c(s: str) -> str:
    """Безопасно для ячейки таблицы."""
    return " ".join(str(s).split()).replace("|", "\\|")


def _crit(code: str) -> str:
    c = ALL_CRITERIA.get(code)
    return f"{code} «{c.name}»" if c else code


def _lines(q: Quote) -> str:
    return f"строка {q.line_from}" if q.line_from == q.line_to else f"строки {q.line_from}–{q.line_to}"


def _where(res: RunResult, q: Quote) -> str:
    page = res.doc.page_of(q.line_from)
    return f"{_lines(q)}" + (f", стр. {page}" if page else "")


def _quote_md(res: RunResult, q: Quote) -> str:
    approx = " *(приблизительное совпадение)*" if q.approx else ""
    return f"> «{' '.join(q.text.split())}» — {_where(res, q)}{approx}"


def _req_label(res: RunResult, rid: str) -> str:
    r = next((x for x in res.reqs if x.id == rid), None)
    return f"{rid} ({r.doc_id})" if r and r.doc_id else rid


def _defect_md(res: RunResult, d: Defect) -> list[str]:
    head = f"- **{d.id}** · {_crit(d.criterion)} · {SEVERITY_RU[d.severity]}"
    if d.scope == "set":
        head += f" · {_KIND_RU.get(d.kind, d.kind)}"
    if d.scope == "set" or len(d.req_ids) > 1:
        if d.req_ids:
            head += " · затронуты: " + ", ".join(_req_label(res, r) for r in d.req_ids)
    merged = [x.id for x in res.defects if x.merged_into == d.id]
    if merged:
        head += " · объединяет также: " + ", ".join(merged)
    # незначительные замечания по одному требованию — коротко: цитата уже приведена над списком
    if d.scope == "req" and d.severity == "minor" and len(d.quotes) == 1:
        q = d.quotes[0]
        line = f"{head} ({_where(res, q)}): {d.problem}"
        if d.recommendation:
            line += f" *Как исправить:* {d.recommendation}"
        return [line]
    out = [head]
    for q in d.quotes:
        out.append("  " + _quote_md(res, q))
    out.append(f"  - **Проблема:** {d.problem}")
    if d.recommendation:
        out.append(f"  - **Как исправить:** {d.recommendation}")
    return out


def render(res: RunResult) -> str:
    m = res.meta
    active = res.active
    L: list[str] = []
    add = L.append

    add("# Отчёт о проверке требований")
    add("")
    add(f"**{res.title}**")
    add("")
    add("> Отчёт сформирован автоматически агентом `reqcheck` по входному документу. Вручную не редактировался.")
    add("")

    # 1. Паспорт
    add("## 1. Паспорт прогона")
    add("")
    add("| Поле | Значение |")
    add("| :--- | :--- |")
    add(f"| Проверяемый документ | {_c(res.title)} (файл `{_c(res.doc.name)}`) |")
    add(f"| SHA-256 документа | `{res.doc.sha256[:16]}…` |")
    add(f"| Строк в документе | {len(res.doc.lines)} |")
    add(f"| Дата прогона | {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} |")
    add(f"| Модель | {m['provider']} / `{m['model']}` |")
    add(f"| Версия агента | {_c(m['agent_version'])} |")
    add(f"| Вызовов модели / токенов (вход+выход) | {m['calls']} / {m['input_tokens']}+{m['output_tokens']} |")
    add(f"| Время работы | {m['seconds']} с |")
    add(f"| Параметры | пачка проверки = {m['batch_size']}, потоков = {m.get('workers', 1)}, "
        f"верификация дефектов = {'вкл' if m['verify'] else 'выкл'} |")
    add("")

    # 2. Критерии
    add("## 2. Критерии проверки")
    add("")
    add("Свойства качественного требования и набора требований: ISO/IEC/IEEE 29148:2018 (п. 5.2.5, 5.2.6) "
        "и IEEE Std 830-1998 (п. 4.3).")
    add("")
    add("| Код | Критерий | Источник | Область |")
    add("| :--- | :--- | :--- | :--- |")
    for c in REQ_CRITERIA:
        add(f"| {c.code} | {c.name} | {c.source} | отдельное требование |")
    for c in SET_CRITERIA:
        add(f"| {c.code} | {c.name} | {c.source} | набор в целом |")
    add("")
    add("Важность дефекта: **критичное** — по формулировке нельзя однозначно реализовать или проверить "
        "(непроверяемость, противоречие, неполнота); **значительное** — вероятна ошибка или спор, смысл в целом "
        "восстанавливается; **незначительное** — оформление, прослеживаемость.")
    add("")

    # 3. Нарезка
    add("## 3. Как агент нарезал требования")
    add("")
    add("Документ разобран на отдельные пункты. Если в документе у пункта был номер, он указан; если номера не было, "
        "агент присвоил свой идентификатор (колонка «ID в документе» — «—»), а дословная цитата приведена в таблице.")
    add("")
    add("| ID в отчёте | ID в документе | Раздел | Тип | Строки | Краткая формулировка |")
    add("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in res.reqs:
        lines = str(r.line_from) if r.line_from == r.line_to else f"{r.line_from}–{r.line_to}"
        text = _c(r.summary)
        if not r.doc_id:
            text += f"<br>*Цитата:* «{_c(r.quote)}»"
        add(f"| {r.id} | {_c(r.doc_id) or '—'} | {_c(r.section)} | {r.kind} | {lines} | {text} |")
    add("")

    # 4. Требования
    add("## 4. Проверка каждого требования")
    add("")
    add("### 4.1. Матрица критериев")
    add("")
    add("✓ — соответствует, ✗ — нарушение, – — не применимо, ? — модель не оценила.")
    add("")
    add("| Требование | " + " | ".join(REQ_CODES) + " |")
    add("| :--- | " + " | ".join(":---:" for _ in REQ_CODES) + " |")
    for r in res.reqs:
        label = r.id + (f" / {_c(r.doc_id)}" if r.doc_id else "")
        add(f"| {label} | " + " | ".join(_MARK[res.matrix[r.id][c]] for c in REQ_CODES) + " |")
    add("")
    add("### 4.2. Замечания по требованиям")
    add("")
    req_defects: dict[str, list[Defect]] = {}
    for d in active:
        if d.scope == "req":
            req_defects.setdefault(d.req_ids[0], []).append(d)
    merged_refs: dict[str, list[Defect]] = {}
    for d in res.defects:
        if d.merged_into and d.scope == "req":
            merged_refs.setdefault(d.req_ids[0], []).append(d)
    for r in res.reqs:
        ds = sorted(req_defects.get(r.id, []), key=lambda d: SEVERITY_ORDER[d.severity])
        title = f"#### {r.id}" + (f" ({r.doc_id})" if r.doc_id else "") + f" — {r.summary}"
        add(title)
        add("")
        add(f"> «{' '.join(r.quote.split())}» — {_where(res, Quote(r.quote, r.line_from, r.line_to))}")
        add("")
        refs = merged_refs.get(r.id, [])
        if not ds and refs:
            for d in refs:
                add(f"- **{d.id}** · {_crit(d.criterion)}: {d.problem} → учтено в **{d.merged_into}** (см. ниже)")
            add("")
            continue
        if not ds:
            unknown = [c for c in REQ_CODES if res.matrix[r.id][c] == "?"]
            add("Нарушений не выявлено." if not unknown else
                f"Нарушений не выявлено по оценённым критериям (не оценены: {', '.join(unknown)}).")
            add("")
            continue
        for d in ds:
            L.extend(_defect_md(res, d))
        for d in refs:
            add(f"- **{d.id}** · {_crit(d.criterion)}: {d.problem} → учтено в **{d.merged_into}** (см. ниже)")
        add("")

    # 5. Набор
    add("## 5. Проверка всего набора требований")
    add("")
    set_defects = sorted((d for d in active if d.scope == "set"),
                         key=lambda d: (SEVERITY_ORDER[d.severity], d.id))
    add("### 5.1. Дефекты набора")
    add("")
    if set_defects:
        for d in set_defects:
            L.extend(_defect_md(res, d))
    else:
        add("Дефектов на уровне набора не выявлено.")
    add("")
    if res.checklist:
        add("### 5.2. Чек-лист ревью из документа")
        add("")
        add("| Пункт чек-листа | Оценка | Обоснование |")
        add("| :--- | :--- | :--- |")
        for it in res.checklist:
            add(f"| {_c(it.item)} | {_CHECKLIST_RU[it.status]} | {_c(it.justification)} |")
        add("")

    # 6. Сводка
    add("## 6. Сводка")
    add("")
    sev = Counter(d.severity for d in active)
    with_viol = sum(1 for row in res.matrix.values() if "violated" in row.values())
    add("| Показатель | Значение |")
    add("| :--- | :--- |")
    add(f"| Требований выделено | {len(res.reqs)} |")
    add(f"| Требований с нарушениями | {with_viol} |")
    add(f"| Дефектов всего | {len(active)} (по требованиям: {sum(d.scope == 'req' for d in active)}, "
        f"по набору: {sum(d.scope == 'set' for d in active)}) |")
    add(f"| — критичных | {sev['critical']} |")
    add(f"| — значительных | {sev['major']} |")
    add(f"| — незначительных | {sev['minor']} |")
    n_merged = sum(1 for d in res.defects if d.merged_into)
    if n_merged:
        add(f"| Объединено дублей | {n_merged} |")
    add("")
    by_crit = Counter(d.criterion for d in active)
    if by_crit:
        add("| Критерий | Дефектов |")
        add("| :--- | :--- |")
        for code, n in sorted(by_crit.items(), key=lambda kv: -kv[1]):
            add(f"| {_crit(code)} | {n} |")
        add("")

    crit = [d for d in active if d.severity == "critical"]
    add("**Критичные дефекты**")
    add("")
    if not crit:
        add("Критичных дефектов нет.")
    groups = [
        ("Непроверяемые формулировки", lambda d: d.criterion == "VER" or d.criterion == "S-VAL"),
        ("Противоречия", lambda d: d.kind == "contradiction" or d.criterion == "S-CNS"),
        ("Неполные формулировки и пропуски", lambda d: d.criterion in ("COM", "S-COM")),
    ]
    used: set[str] = set()
    for name, pred in groups + [("Прочие критичные", lambda d: True)]:
        part = [d for d in crit if d.id not in used and pred(d)]
        if not part:
            continue
        used.update(d.id for d in part)
        add(f"*{name}* ({len(part)}):")
        add("")
        for d in part:
            where = ", ".join(_req_label(res, r) for r in d.req_ids) if d.req_ids else "набор в целом"
            add(f"- **{d.id}** ({where}, {_where(res, d.quotes[0])}): {d.problem}")
        add("")

    # 7. Итог
    add("## 7. Итог")
    add("")
    add(f"**Можно ли начинать разработку и тестирование: {VERDICT_TEXT[res.verdict]}.**")
    add("")
    if res.conclusion:
        add(res.conclusion.rationale)
        add("")
        add("**Что исправить в первую очередь:**")
        add("")
        for i, fx in enumerate(res.conclusion.top_fixes, start=1):
            ids = f" *({', '.join(fx.defect_ids)})*" if fx.defect_ids else ""
            add(f"{i}. {fx.action}{ids}")
        add("")

    # Приложения
    rejected = [d for d in res.defects if d.rejected and not d.merged_into]
    if rejected:
        add("## Приложение А. Отклонено этапом верификации")
        add("")
        add("Эти замечания первичный проход нашёл, но проверяющий проход их не подтвердил; в сводку и итог они не входят.")
        add("")
        for d in rejected:
            add(f"- **{d.id}** · {_crit(d.criterion)} · {SEVERITY_RU[d.severity]}: {d.problem} "
                f"— *отклонено: {d.reject_reason}*")
        add("")
    if res.warnings:
        add("## Приложение Б. Предупреждения агента")
        add("")
        add("Технические замечания о качестве самого прогона (цитаты, которых нет в документе; пропуски в ответах модели).")
        add("")
        for w in res.warnings:
            add(f"- {w}")
        add("")
    return "\n".join(L).rstrip() + "\n"


def to_json(res: RunResult) -> str:
    data = dict(
        title=res.title, document=res.doc.name, sha256=res.doc.sha256, meta=res.meta, verdict=res.verdict,
        requirements=[dataclasses.asdict(r) for r in res.reqs], matrix=res.matrix,
        defects=[dataclasses.asdict(d) for d in res.defects],
        checklist=[c.model_dump() for c in res.checklist],
        conclusion=res.conclusion.model_dump() if res.conclusion else None,
        warnings=res.warnings,
    )
    return json.dumps(data, ensure_ascii=False, indent=2)
