"""Детерминированная нарезка структурированного документа (Markdown, DOCX) на требования.

Зачем: нарезка через модель недетерминирована (число требований плавает от прогона к прогону) и
стоит вызова модели, а при бесплатных лимитах вызовы дороги. У документа со структурой
(заголовки, списки, таблицы) границы требований видны без модели:

  * пункт списка (в том числе нумерованный) — одно требование; продолжение строки приклеивается;
  * строка таблицы — одно требование (критерии приёмки, параметры, статусы);
    таблица «Поле | Значение» целиком — одно требование (общие параметры);
  * абзац — одно требование, блок кода — одно требование;
  * разделы-шаблоны (общие сведения, ревью, согласующие, термины) и пустые заглушки пропускаются —
    их разбирает проверка набора;
  * идентификатор берётся из текста (`**Ф-1.**`, `FR-3`, первая ячейка строки таблицы), раздел —
    из заголовков, тип — из названия раздела.

Для плоского текста (PDF) этих признаков нет — тогда агент использует нарезку моделью.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .loader import Document

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
BULLET = re.compile(r"^\s*(?:[*\-+]|\d+[.)])\s+(.*)$")
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEP = re.compile(r"^\s*\|?[\s:\-|]+\|?\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
QUOTE_LINE = re.compile(r"^\s*>")
ID_CORE = r"[A-Za-zА-Яа-яЁё]{1,5}[-–]\d+(?:\.\d+)*"
ID_BOLD = re.compile(rf"^\s*(?:[*\-+]\s+)?\*\*\s*({ID_CORE})\.?\s*\*\*")
ID_PLAIN = re.compile(rf"^\s*(?:[*\-+]\s+)?({ID_CORE})[.:)]\s")
ID_CELL = re.compile(rf"^\W*({ID_CORE})\W*$")

SKIP_SECTIONS = ("общие сведения", "ревью", "согласующ", "термин", "оглавлен", "содержание",
                 "история изменений", "лист согласования")
PLACEHOLDER = re.compile(r"заполня|заполнит|\btbd\b|\btba\b|^[\W_]*$|^не\s+(используются|используется|требуется)\W*$",
                         re.IGNORECASE)
KV_HEADERS = ({"поле", "параметр", "field", "parameter", "свойство"}, {"значение", "value"})

KIND_RULES = [  # порядок важен: «нефункц» раньше «функц»
    ("приемк", "acceptance"), ("приёмк", "acceptance"), ("нефункц", "nonfunctional"), ("функц", "functional"),
    ("цел", "goal"), ("интеграц", "integration"), ("данн", "data"), ("bpmn", "process"), ("sequence", "process"),
    ("процесс", "process"), ("ui", "ui"), ("интерфейс", "ui"), ("дизайн", "ui"), ("rest", "api"), ("api", "api"),
    ("запрос", "api"), ("ответ", "api"), ("статус", "api"), ("парамет", "api"), ("логик", "api"),
    ("правил", "constraint"), ("ограничен", "constraint"),
]


@dataclass
class Unit:
    doc_id: str
    section: str
    kind: str
    quote: str
    summary: str
    line_from: int
    line_to: int


def _strip_md(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[*_`]+", "", s)).strip()


def _kind(section: str) -> str:
    low = section.lower()
    for key, kind in KIND_RULES:
        if key in low:
            return kind
    return "other"


def _cells(row: str) -> list[str]:
    inner = row.strip().strip("|")
    return [c.strip() for c in re.split(r"(?<!\\)\|", inner)]


def _summary(text: str, limit: int = 140) -> str:
    t = _strip_md(text)
    t = re.sub(rf"^{ID_CORE}\.?\s*", "", t)
    first = re.split(r"(?<=[.!?])\s", t, maxsplit=1)[0]
    s = first if 8 <= len(first) <= limit else t
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def _is_structural(line: str) -> bool:
    return bool(HEADING.match(line) or BULLET.match(line) or TABLE_ROW.match(line) or FENCE.match(line)
                or QUOTE_LINE.match(line))


def looks_structured(doc: Document) -> bool:
    """Есть ли у документа достаточно разметки, чтобы нарезать его правилами."""
    heads = sum(1 for ln in doc.lines if HEADING.match(ln))
    items = sum(1 for ln in doc.lines if BULLET.match(ln) or (TABLE_ROW.match(ln) and not TABLE_SEP.match(ln)))
    return heads >= 3 and items >= 5


def segment(doc: Document) -> tuple[str, list[Unit]]:
    lines = doc.lines
    n = len(lines)
    title = doc.name
    stack: list[tuple[int, str]] = []   # (уровень, текст заголовка)
    skip_level: int | None = None       # пропускаем раздел, пока не встретим заголовок того же или выше уровня
    seen_h2 = False
    units: list[Unit] = []

    def section() -> str:
        return " / ".join(t for lvl, t in stack if lvl >= 2)

    def emit(lf: int, lt: int, text: str, doc_id: str = "", kind: str | None = None, summary: str | None = None):
        if skip_level is not None or not seen_h2:
            return
        body = _strip_md(text)
        if len(body) < 12 or PLACEHOLDER.search(body):   # «или», пустые строки таблиц, заглушки
            return
        units.append(Unit(doc_id, section(), kind or _kind(section()), text.strip(), summary or _summary(text), lf, lt))

    i = 0
    while i < n:
        raw = lines[i]
        ln = i + 1
        if not raw.strip():
            i += 1
            continue

        m = HEADING.match(raw)
        if m:
            lvl, text = len(m.group(1)), _strip_md(m.group(2))
            if lvl == 1 and title == doc.name:
                title = text
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, text))
            if skip_level is not None and lvl <= skip_level:
                skip_level = None
            if skip_level is None and any(k in text.lower() for k in SKIP_SECTIONS):
                skip_level = lvl
            if lvl >= 2:
                seen_h2 = True
            i += 1
            continue

        if FENCE.match(raw):                                   # блок кода — одно требование
            j = i + 1
            while j < n and not FENCE.match(lines[j]):
                j += 1
            block = " ".join(s.strip() for s in lines[i + 1:j] if s.strip())
            if block:
                emit(ln, min(j + 1, n), block, kind=_kind(section()) if _kind(section()) != "other" else "process",
                     summary=_summary(block))
            i = j + 1
            continue

        if QUOTE_LINE.match(raw):
            i += 1
            continue

        if TABLE_ROW.match(raw):
            j = i
            while j < n and TABLE_ROW.match(lines[j]):
                j += 1
            rows = [(k + 1, lines[k]) for k in range(i, j)]
            header = _cells(rows[0][1])
            body_rows = [(k, r) for k, r in rows[1:] if not TABLE_SEP.match(r)]
            is_kv = (len(header) == 2 and header[0].lower().strip("* ") in KV_HEADERS[0]
                     and header[1].lower().strip("* ") in KV_HEADERS[1])
            if is_kv and body_rows:                            # «Поле | Значение» — таблица целиком
                text = "\n".join(r for _, r in rows if not TABLE_SEP.match(r))
                names = ", ".join(_strip_md(_cells(r)[0]) for _, r in body_rows[:4])
                emit(rows[0][0], rows[-1][0], text, summary=f"Таблица «{_strip_md(stack[-1][1]) if stack else ''}»: {names}…")
            else:
                for k, r in body_rows:
                    cells = [c for c in _cells(r)]
                    if not any(_strip_md(c) for c in cells):
                        continue
                    did = ""
                    cm = ID_CELL.match(_strip_md(cells[0])) if cells else None
                    if cm:
                        did = cm.group(1)
                    emit(k, k, r, doc_id=did, summary=_summary(" → ".join(_strip_md(c) for c in cells if _strip_md(c))))
            i = j
            continue

        b = BULLET.match(raw)
        if b:                                                  # пункт списка + продолжения строки
            j = i + 1
            while j < n and lines[j].strip() and not _is_structural(lines[j]):
                j += 1
            text = " ".join(s.strip() for s in lines[i:j])
            mid = ID_BOLD.match(raw) or ID_PLAIN.match(raw)
            emit(ln, j, text, doc_id=mid.group(1).replace("–", "-") if mid else "")
            i = j
            continue

        j = i + 1                                              # абзац
        while j < n and lines[j].strip() and not _is_structural(lines[j]):
            j += 1
        text = " ".join(s.strip() for s in lines[i:j])
        mid = ID_BOLD.match(raw) or ID_PLAIN.match(raw)
        emit(ln, j, text, doc_id=mid.group(1).replace("–", "-") if mid else "")
        i = j

    return title, units
