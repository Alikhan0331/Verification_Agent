"""Детерминированные проверки шаблона: пустые разделы, заглушки, незаполненные поля.

Это то, что надёжнее и дешевле найти кодом, чем просить модель: раздел без содержимого, раздел с
текстом-заглушкой («заполняется тестировщиками», TBD), поле таблицы «Поле | Значение» со значением «—»,
запись вида «Prod: —». Каждая находка привязана к дословной цитате и строке документа.
Возвращается список `Finding`; пайплайн превращает его в дефекты набора (S-COM) до слияния дублей.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .loader import Document
from .segmenter import HEADING, ID_CORE, TABLE_ROW, TABLE_SEP, _cells, _strip_md, looks_structured, segment

PLACEHOLDER_TEXT = re.compile(
    r"заполня(?:ю|е)тся|заполнит[ье]|не\s+заполнен|\btbd\b|\btba\b|будет\s+(?:добавлен|уточнен|определен)|"
    r"в\s+разработке|to\s+be\s+(?:defined|determined)", re.IGNORECASE)
EMPTY_VALUE = re.compile(r"^[\s*_`]*(?:[—–-]|tbd|tba|\?+|n/?a|н/?д)?[\s*_`]*$", re.IGNORECASE)
INLINE_EMPTY = re.compile(r"(?<![\w/])([A-Za-zА-Яа-яЁё][\w ]{1,18}):\s*(?:—|–)\s*(?=[|.;,]|$)")
SECURITY_WORDS = re.compile(
    r"безопасн|шифр|аудит|журнал|персональн|конфиденциальн|маскир|аутентифик|токен|хэш|хеш|"
    r"прав[аы]? доступа|encrypt|audit|security", re.IGNORECASE)
KV_LEFT = {"поле", "параметр", "field", "parameter", "свойство"}
KV_RIGHT = {"значение", "value"}


@dataclass
class Finding:
    kind: str            # empty_section | placeholder_section | empty_field
    severity: str
    quote: str           # дословно из документа
    line_from: int
    line_to: int
    problem: str
    recommendation: str
    criterion: str = "S-COM"
    dkind: str = "omission"


def lint(doc: Document) -> list[Finding]:
    lines = doc.lines
    n = len(lines)
    out: list[Finding] = []

    heads = [(i, len(m.group(1)), _strip_md(m.group(2)))
             for i, ln in enumerate(lines) if (m := HEADING.match(ln))]
    for k, (i, lvl, title) in enumerate(heads):
        if lvl < 2:
            continue
        nxt = heads[k + 1] if k + 1 < len(heads) else None
        end = nxt[0] if nxt else n
        body = [(j, lines[j].strip()) for j in range(i + 1, end) if lines[j].strip()]
        if not body:
            if nxt is None or nxt[1] <= lvl:                     # под заголовком ничего нет, и это не контейнер
                out.append(Finding(
                    "empty_section", "major", lines[i].strip(), i + 1, i + 1,
                    f"Раздел «{title}» пуст: содержимого нет, требования и условия не определены.",
                    f"Заполнить раздел «{title}» либо явно указать, что он не применяется, с обоснованием."))
            continue
        if len(body) <= 2 and all(PLACEHOLDER_TEXT.search(_strip_md(t)) for _, t in body):
            j0, text = body[0]
            out.append(Finding(
                "placeholder_section", "major", text, j0 + 1, body[-1][0] + 1,
                f"Раздел «{title}» не заполнен: вместо содержимого стоит текст-заглушка.",
                f"Заменить заглушку содержательными пунктами раздела «{title}» или явно исключить раздел."))

    i = 0
    while i < n:                                                  # таблицы «Поле | Значение»
        if not TABLE_ROW.match(lines[i]):
            i += 1
            continue
        j = i
        while j < n and TABLE_ROW.match(lines[j]):
            j += 1
        header = [_strip_md(c).lower() for c in _cells(lines[i])]
        if len(header) == 2 and header[0] in KV_LEFT and header[1] in KV_RIGHT:
            for k in range(i + 1, j):
                if TABLE_SEP.match(lines[k]):
                    continue
                cells = _cells(lines[k])
                if len(cells) != 2:
                    continue
                name, value = _strip_md(cells[0]), cells[1]
                if EMPTY_VALUE.match(value):
                    out.append(Finding(
                        "empty_field", "minor", lines[k].strip(), k + 1, k + 1,
                        f"Поле «{name}» не заполнено (значение «—» или пусто).",
                        f"Указать значение поля «{name}» или явно написать, почему оно не требуется."))
                else:
                    for m in INLINE_EMPTY.finditer(value):
                        out.append(Finding(
                            "empty_field", "minor", lines[k].strip(), k + 1, k + 1,
                            f"Поле «{name}»: для «{m.group(1).strip()}» значение не указано («—»).",
                            f"Указать значение для «{m.group(1).strip()}» или обосновать отсутствие."))
        i = j

    out.extend(_security_gap(doc, heads))
    out.extend(_deferred_and_dangling(doc, heads))
    out.extend(_appendix_refs(doc, heads))
    out.extend(_sum_errors(doc))
    return out


DEFERRED = re.compile(
    r"[^,;.:]*\b(?:определя\w*\s+позже|определит\w*\s+позже|будет\s+(?:определен\w*|уточнен\w*|согласован\w*)|"
    r"уточня\w*\s+позже|уточнит\w*\s+позже|определ\w+\s+в\s+дальнейшем|tbd|tbc)\b[^,;.]*", re.IGNORECASE)
SECTION_REF = re.compile(r"(?:раздел\w*|разд\.|пункт\w*|п\.)\s*(\d+(?:\.\d+)+)", re.IGNORECASE)
HEAD_NUM = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]?\s")
ID_TOKEN = re.compile(rf"(?<![\w-])({ID_CORE})(?![\w-])")


APPENDIX_REF = re.compile(r"приложени\w*\s+([А-ЯA-Z])\b")
NUM = r"\d{1,3}(?:[ \u00a0]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?"
UNIT = r"(?:\s*(?:₸|тг\.?|тенге|руб\w*\.?|\$|€|%))?"
SUM_EXPR = re.compile(rf"((?:{NUM}){UNIT}(?:\s*\+\s*(?:{NUM}){UNIT})+)\s*=\s*({NUM}){UNIT}(?![\d,.]*\s*[×x*/])")


def _num(s: str) -> float:
    return float(re.sub(r"[ \u00a0]", "", s).replace(",", "."))


def _appendix_refs(doc: Document, heads) -> list[Finding]:
    """«см. приложение Б», а такого приложения (заголовка или строки «Приложение Б») в документе нет."""
    if not looks_structured(doc):
        return []
    have = {m.group(1) for ln in doc.lines
            if (m := re.match(r"^\s*#*\s*\**\s*приложени\w*\s+([А-ЯA-Z])\b", ln, re.IGNORECASE))}
    out: list[Finding] = []
    seen: set[str] = set()
    _, units = segment(doc)
    for u in units:
        for m in APPENDIX_REF.finditer(u.quote):
            letter = m.group(1)
            if letter in have or letter in seen:
                continue
            seen.add(letter)
            phrase = m.group(0)
            out.append(Finding(
                "dangling_ref", "major", phrase, _line_of(doc, phrase, u.line_from, u.line_to), u.line_to,
                f"Ссылка на приложение {letter}, которого в документе нет: содержимое, на которое опирается требование, "
                "недоступно для разработки и проверки.",
                f"Добавить приложение {letter} в документ либо перенести его содержимое в требования.",
                criterion="S-TRA", dkind="traceability"))
    return out


def _sum_errors(doc: Document) -> list[Finding]:
    """Простая арифметика в тексте: «a + b = c» с неверным итогом (примеры расчётов, критерии приёмки)."""
    out: list[Finding] = []
    for k, ln in enumerate(doc.lines):
        for m in SUM_EXPR.finditer(ln):
            parts = [_num(x) for x in re.findall(NUM, m.group(1))]
            total = _num(m.group(2))
            if len(parts) >= 2 and abs(sum(parts) - total) > 0.5:
                right = f"{sum(parts):,.2f}".rstrip("0").rstrip(".").replace(",", " ")
                out.append(Finding(
                    "arithmetic", "major", m.group(0).strip(), k + 1, k + 1,
                    f"Арифметическая ошибка в примере: сумма слагаемых равна {right}, а в тексте указано "
                    f"{m.group(2).strip()}. Пример противоречит правилам расчёта и не может служить эталоном для тестов.",
                    "Пересчитать пример по правилам документа и исправить итог либо уточнить правило.",
                    criterion="S-CNS", dkind="contradiction"))
    return out


def _line_of(doc: Document, phrase: str, lo: int, hi: int) -> int:
    for k in range(lo - 1, min(hi, len(doc.lines))):
        if phrase in doc.lines[k]:
            return k + 1
    return lo


def _deferred_and_dangling(doc: Document, heads) -> list[Finding]:
    """Отложенные решения («определяется позже»), ссылки на несуществующие разделы и требования."""
    if not looks_structured(doc):
        return []
    _, units = segment(doc)
    out: list[Finding] = []
    for u in units:
        m = DEFERRED.search(u.quote)
        if m:
            phrase = m.group(0).strip(" *_`|")
            out.append(Finding(
                "deferred", "major", phrase, _line_of(doc, phrase, u.line_from, u.line_to), u.line_to,
                f"Решение отложено: «{phrase}». Требование нельзя реализовать и проверить, пока значение не определено.",
                "Зафиксировать значение (или правило его вычисления) сейчас либо вынести вопрос в открытые вопросы "
                "с владельцем и сроком; до решения не считать требование готовым к разработке."))
    numbers = {m.group(1) for _, _, t in heads if (m := HEAD_NUM.match(t))}
    defined = {u.doc_id.replace("–", "-") for u in units if u.doc_id}
    prefixes = {d.split("-")[0] for d in defined}
    seen: set[str] = set()
    for u in units:
        for m in SECTION_REF.finditer(u.quote):
            ref = m.group(1)
            if ref in numbers or ref in seen:
                continue
            seen.add(ref)
            phrase = u.quote[max(0, m.start() - 30):m.end()].strip(" *_`|")
            phrase = phrase if phrase in doc.lines[u.line_from - 1] else m.group(0)
            out.append(Finding(
                "dangling_ref", "major", phrase, _line_of(doc, phrase, u.line_from, u.line_to), u.line_to,
                f"Ссылка на раздел {ref}, которого в документе нет: порядок или правило, на которое ссылается требование, "
                "нельзя найти и проверить.",
                f"Добавить раздел {ref} либо исправить ссылку и привести правило в документ.",
                criterion="S-TRA", dkind="traceability"))
        for m in ID_TOKEN.finditer(u.quote):
            tok = m.group(1).replace("–", "-")
            if tok.split("-")[0] in prefixes and tok not in defined and tok not in seen:
                seen.add(tok)
                out.append(Finding(
                    "dangling_ref", "major", tok, _line_of(doc, m.group(1), u.line_from, u.line_to), u.line_to,
                    f"Ссылка на требование {tok}, которого в документе нет.",
                    f"Исправить ссылку на {tok} или добавить недостающее требование.",
                    criterion="S-TRA", dkind="traceability"))
    return out


def _security_gap(doc: Document, heads) -> list[Finding]:
    """Ни одно требование не касается безопасности/аудита/защиты данных (раздел «Ревью» шаблона не в счёт)."""
    if not looks_structured(doc):
        return []
    _, units = segment(doc)
    if not units or any(SECURITY_WORDS.search(u.quote) for u in units):
        return []
    anchor = next(((i, t) for i, lvl, t in heads if "нефункц" in t.lower()), None) or         next(((i, t) for i, lvl, t in heads if lvl >= 2 and "функц" in t.lower()), None)
    if anchor is None:
        return []
    i, title = anchor
    return [Finding(
        "no_security", "major", doc.lines[i].strip(), i + 1, i + 1,
        "В требованиях нет ничего о безопасности и защите данных: не заданы аутентификация и права доступа, "
        "защита и маскирование чувствительных данных (в том числе в уведомлениях и URL), журналирование и аудит действий.",
        f"Добавить в раздел «{title}» требования информационной безопасности и аудита либо явно обосновать их отсутствие.")]
