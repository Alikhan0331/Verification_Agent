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
from .segmenter import HEADING, TABLE_ROW, TABLE_SEP, _cells, _strip_md, looks_structured, segment

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
