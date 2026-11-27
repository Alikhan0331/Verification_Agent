"""Отчёт в виде простого текста (.txt) для редакторов, которые не показывают Markdown.

Текст строит код из того же отчёта, что и report.md, без участия модели: заголовки подчёркиваются, таблицы
превращаются в строки «поле: значение» или выровненные колонки, строки переносятся по ширине 100 символов.
Цитаты из документа (строки с «>») остаются дословными: их разметка не снимается.

    python -m reqcheck.textout report.md report.txt
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

WIDTH = 100
_SEP_CELL = re.compile(r"^:?-{2,}:?$")


def _split_row(line: str) -> list[str]:
    s = line.strip()
    s = s[1:] if s.startswith("|") else s
    s = s[:-1] if s.endswith("|") and not s.endswith("\\|") else s
    cells = re.split(r"(?<!\\)\|", s)
    return [c.strip().replace("\\|", "|") for c in cells]


def _inline(s: str) -> str:
    """Снимает разметку Markdown, но не трогает цитату документа между первой ««» и последней «»»:
    она остаётся дословной, вместе с разметкой источника."""
    a, b = s.find("«"), s.rfind("»")
    if 0 <= a < b:
        return _strip(s[:a]) + s[a:b + 1] + _strip(s[b + 1:])
    return _strip(s)


def _strip(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"__(.+?)__", r"\1", s)
    s = re.sub(r"(?<![\w*])\*(?![\s*])([^*\n]+?)(?<![\s*])\*(?![\w*])", r"\1", s)
    s = re.sub(r"`([^`\n]*)`", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", s)
    s = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|])", r"\1", s)
    return s


def _wrap(text: str, indent: str = "", hang: str | None = None, width: int = WIDTH) -> list[str]:
    hang = indent if hang is None else hang
    out = textwrap.wrap(text, width=width, initial_indent=indent, subsequent_indent=hang,
                        break_long_words=False, break_on_hyphens=False)
    return out or [indent.rstrip()]


def _cells(rows: list[list[str]]) -> list[str]:
    """Одна запись на строку таблицы; <br> внутри ячейки — перенос строки с отступом."""
    out: list[str] = []
    for r in rows:
        parts = " | ".join(c for c in r if c).split("<br>")
        for k, part in enumerate(parts):
            out += _wrap(part.strip(), "" if k == 0 else "    ", "    ")
    return out


def _table(rows: list[list[str]]) -> list[str]:
    rows = [r for r in rows if not all(_SEP_CELL.match(c) for c in r if c) or not any(r)]
    rows = [[_inline(c) if "<br>" not in c else "<br>".join(_inline(x) for x in c.split("<br>")) for c in r]
            for r in rows if any(c for c in r)]
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    ncol = max(len(r) for r in rows)
    out: list[str] = []
    if ncol == 2:                                            # «поле: значение»
        for r in body:
            val = r[1] if len(r) > 1 else ""
            out += _wrap(f"{r[0]}: {val}".replace("<br>", " "), "", "    ")
        return out
    short = max((len(c) for r in rows for c in r[1:]), default=0) <= 22 and ncol >= 5
    if short:                                                # матрица: выровненные колонки
        widths = [max(len(r[i]) if i < len(r) else 0 for r in rows) for i in range(ncol)]
        for r in rows:
            r = r + [""] * (ncol - len(r))
            out.append(" | ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())
        return out
    return _cells(body or rows)                              # длинные ячейки: одна запись на строку таблицы


def to_text(md: str) -> str:
    lines = md.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("|"):                                # таблица целиком
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(_split_row(lines[i]))
                i += 1
            out += _table(block)
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            level, title = len(m.group(1)), _inline(m.group(2)).strip()
            if level == 1:
                out += ["", title, "=" * min(len(title), WIDTH), ""]
            elif level == 2:
                out += ["", title, "-" * min(len(title), WIDTH), ""]
            elif level == 3:
                out += ["", title, "~" * min(len(title), WIDTH)]
            else:
                out += ["", *_wrap("■ " + title, "", "  ")]
            i += 1
            continue
        q = re.match(r"^(\s*)>\s?(.*)$", ln)
        if q:                                                # цитата документа: дословно, только снимаем «\|»
            body = q.group(2).replace("\\|", "|")
            body = body if body.startswith("«") else _inline(body)      # цитата документа дословно, прочее без разметки
            out += _wrap(body, q.group(1) + "  ", q.group(1) + "  ")
            i += 1
            continue
        b = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", ln)
        if b and not s.startswith("**"):
            lead = b.group(1)
            mark = "-" if b.group(2) in "-*+" else b.group(2)
            out += _wrap(f"{mark} {_inline(b.group(3))}", lead, lead + " " * (len(mark) + 1))
            i += 1
            continue
        if s in ("---", "***", "___"):
            out.append("-" * 40)
        elif s == "":
            out.append("")
        else:
            out += _wrap(_inline(s), "", "")
        i += 1
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip("\n")
    return text + "\n"


def save_report(out: Path, md: str, twin: bool = True) -> list[Path]:
    """Пишет отчёт. Путь .txt — простой текст; иначе Markdown и (если twin) рядом такой же отчёт в .txt:
    его читает любой редактор, а Markdown в части программ показывается с лишними знаками."""
    if out.suffix.lower() == ".txt":
        out.write_text(to_text(md), encoding="utf-8")
        return [out]
    out.write_text(md, encoding="utf-8")
    if not twin:
        return [out]
    txt = out.with_suffix(".txt")
    txt.write_text(to_text(md), encoding="utf-8")
    return [out, txt]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("Использование: python -m reqcheck.textout report.md report.txt", file=sys.stderr)
        return 2
    src, dst = Path(argv[0]), Path(argv[1])
    dst.write_text(to_text(src.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"Готово: {dst}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
