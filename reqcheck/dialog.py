"""Диалог с автором документа прямо во время запуска: python -m reqcheck док.md отчёт.md --ask

После проверки агент по очереди задаёт уточняющие вопросы в терминале. Ответ можно дать буквой варианта,
цифрой или своими словами; Enter пропускает вопрос, `q` заканчивает опрос. Затем ответы оцениваются тем же кодом,
что и в режиме файла ответов (reply.apply_answers): дефекты закрываются, вердикт и итог пересчитываются.
Если ответ неполный, агент задаёт узкий уточняющий вопрос, и опрос продолжается (до `max_rounds` кругов).

Весь диалог попадает в отчёт: раздел 7 показывает вопрос, ответ автора и оценку ответа.
"""

from __future__ import annotations

import sys
from typing import Callable

from . import reply
from .pipeline import RunResult

_OWNER_RU = {"owner": "владелец продукта", "analyst": "аналитик", "architect": "архитектор / разработка",
             "security": "безопасность", "qa": "тестирование"}
_LETTERS = "АБВ"
_LATIN = {"A": 0, "B": 1, "C": 2}
_STOP = {"q", "й", "/q", "стоп", "хватит"}


def _option_index(raw: str, n_options: int) -> int | None:
    """Ответ вида «А», «б», «2», «a» → номер варианта; иначе None (это свободный текст)."""
    t = raw.strip().rstrip(").:").upper()
    if len(t) != 1 or not n_options:
        return None
    if t in _LETTERS and _LETTERS.index(t) < n_options:
        return _LETTERS.index(t)
    if t in _LATIN and _LATIN[t] < n_options:
        return _LATIN[t]
    if t.isdigit() and 1 <= int(t) <= n_options:
        return int(t) - 1
    return None


def collect_answers(res: RunResult, indexes: list[int], input_fn: Callable[[str], str] = input,
                    out=None) -> dict[str, str]:
    """Задаёт вопросы с номерами `indexes` (с нуля) и возвращает {Q-NN: ответ}."""
    out = out or sys.stdout
    by_id = {d.id: d for d in res.defects}
    answers: dict[str, str] = {}
    total = len(indexes)
    for pos, i in enumerate(indexes, start=1):
        q = res.questions[i]
        qid = f"Q-{i + 1:02d}"
        tag = "блокирует разработку и тестирование" if q.blocking else "уточнение"
        print(f"\n─── Вопрос {pos} из {total} · {qid} · {tag} · адресат: {_OWNER_RU.get(q.owner, q.owner)} ───", file=out)
        first = by_id.get(q.defect_ids[0])
        if first and first.quotes:
            qt = first.quotes[0]
            quote = " ".join(qt.text.split())
            quote = quote if len(quote) <= 220 else quote[:217] + "…"
            print(f"В документе (строка {qt.line_from}): «{quote}»", file=out)
        print(f"{q.question}", file=out)
        for k, o in enumerate(q.options):
            print(f"  {_LETTERS[k]}) {o}", file=out)
        hint = "буква варианта или свой ответ; " if q.options else ""
        try:
            raw = input_fn(f"Ваш ответ ({hint}Enter — пропустить, q — закончить): ")
        except EOFError:
            print("\n(ввод закончился, опрос остановлен)", file=out)
            break
        raw = raw.strip()
        if raw.lower() in _STOP:
            break
        if not raw:
            continue
        k = _option_index(raw, len(q.options))
        answers[qid] = f"{_LETTERS[k]} — {q.options[k]}" if k is not None else raw
    return answers


def answers_markdown(items: dict[str, str]) -> str:
    """Краткая форма `Q-01: …` — её понимает reply.parse_answers; файл сохраняется рядом с отчётом."""
    lines = ["# Ответы автора, данные в диалоге", ""]
    for qid, text in items.items():
        lines.append(f"{qid}: {' '.join(text.split())}")
    return "\n".join(lines) + "\n"


def run_dialog(res: RunResult, llm, input_fn: Callable[[str], str] = input, out=None,
               max_rounds: int = 2) -> tuple[RunResult, dict[str, str]]:
    """Опрос в терминале и применение ответов. Возвращает (результат, все данные ответы)."""
    out = out or sys.stdout
    all_given: dict[str, str] = {}
    if not res.questions:
        print("\nУточняющих вопросов нет: проверка не выявила критичных и значительных замечаний.", file=out)
        return res, all_given
    pending = list(range(len(res.questions)))
    print(f"\nАгент подготовил вопросов: {len(pending)}. Сначала идут те, что блокируют разработку.", file=out)
    for rnd in range(1, max_rounds + 1):
        if not pending:
            break
        given = collect_answers(res, pending, input_fn, out)
        if not given:
            if rnd == 1:
                print("\nОтветов нет — отчёт остаётся без изменений.", file=out)
            break
        n_before = len(res.questions)
        prev_answers = list(res.answers)
        prev_meta = dict(res.meta.get("reply", {}))
        print(f"\nОцениваю ответов: {len(given)}…", file=out)
        res = reply.apply_answers(res, llm, answers_markdown(given))
        res.answers = prev_answers + list(res.answers)          # круги не затирают друг друга
        if prev_meta:
            res.meta["reply"]["answered"] += prev_meta.get("answered", 0)
        all_given.update(given)
        for a in res.answers[len(prev_answers):]:
            mark = {"closes": "закрывает дефект", "partial": "закрывает частично",
                    "insufficient": "недостаточно для закрытия", "contradicts": "противоречит документу"}[a["verdict"]]
            print(f"  {a['question_id']}: {mark}. {a['reason']}", file=out)
        pending = list(range(n_before, len(res.questions)))      # уточняющие вопросы, заданные агентом по итогам ответов
        if pending and rnd < max_rounds:
            print(f"\nАгент задаёт уточняющих вопросов: {len(pending)}.", file=out)
    return res, all_given
