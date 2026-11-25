"""Самопроверка отчёта на соответствие шаблону из ТЗ (раздел 6).

Не оценивает качество замечаний — только то, что отчёт содержит все обязательные части
и что каждое нарушение привязано к конкретному месту документа.
"""

from __future__ import annotations

import re

from .pipeline import RunResult

REQUIRED_HEADINGS = [
    "## 1. Паспорт прогона",
    "## 3. Как агент нарезал требования",
    "## 4. Проверка каждого требования",
    "## 5. Проверка всего набора требований",
    "## 6. Сводка",
    "## 7. Итог",
]
PASSPORT_FIELDS = ["Проверяемый документ", "Дата прогона", "Модель", "Версия агента"]
SUMMARY_FIELDS = ["Требований выделено", "Дефектов всего", "критичных"]


def _section(md: str, start: str) -> str:
    i = md.find(start)
    if i < 0:
        return ""
    m = re.search(r"^## ", md[i + len(start):], flags=re.M)
    return md[i: i + len(start) + (m.start() if m else len(md))]


def check_report(res: RunResult, md: str) -> list[str]:
    """Возвращает список нарушений шаблона; пустой список — отчёт соответствует ТЗ."""
    problems: list[str] = []

    for h in REQUIRED_HEADINGS:
        if h not in md:
            problems.append(f"нет раздела «{h[3:]}»")

    passport = _section(md, "## 1. Паспорт прогона")
    for f in PASSPORT_FIELDS:
        if f not in passport:
            problems.append(f"в паспорте прогона нет поля «{f}»")
    if res.meta.get("agent_version", "").endswith("недоступна)"):
        problems.append("версия агента не привязана к коммиту (репозиторий не git или git недоступен)")

    slicing = _section(md, "## 3. Как агент нарезал требования")
    checks = _section(md, "## 4. Проверка каждого требования")
    for r in res.reqs:
        if f"| {r.id} |" not in slicing:
            problems.append(f"{r.id}: нет в таблице нарезки")
        if f"#### {r.id}" not in checks:
            problems.append(f"{r.id}: нет в разделе проверки требований")
        if not r.doc_id and r.quote.strip() and r.quote.strip()[:30] not in slicing.replace("\\|", "|"):
            problems.append(f"{r.id}: у требования без номера в таблице нарезки нет цитаты")

    summary = _section(md, "## 6. Сводка")
    for f in SUMMARY_FIELDS:
        if f not in summary:
            problems.append(f"в сводке нет «{f}»")

    final = _section(md, "## 7. Итог")
    if "Можно ли начинать разработку и тестирование" not in final:
        problems.append("в итоге нет ответа, можно ли начинать разработку и тестирование")
    if "первую очередь" not in final:
        problems.append("в итоге нет списка «что исправить в первую очередь»")

    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", md)
    cyr = sum(1 for c in letters if "А" <= c <= "я" or c in "Ёё")
    if letters and cyr / len(letters) < 0.6:
        problems.append("отчёт написан не на русском языке (доля кириллицы ниже 60%)")

    n = len(res.doc.lines)
    for d in res.active:
        if f"**{d.id}**" not in md:
            problems.append(f"{d.id}: дефект не попал в отчёт")
        if not d.quotes or not d.problem.strip():
            problems.append(f"{d.id}: нет цитаты или описания проблемы")
        for q in d.quotes:
            if not (1 <= q.line_from <= q.line_to <= n):
                problems.append(f"{d.id}: номера строк цитаты вне документа")
    return problems
