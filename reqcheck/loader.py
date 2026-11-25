"""Чтение документа требований: .md/.txt, .pdf, .docx -> пронумерованные строки.

Каждой строке даём номер L001, L002… — на них модель ссылается, а агент потом
проверяет цитаты по исходному тексту.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


class LoadError(RuntimeError):
    pass


@dataclass
class Document:
    name: str
    path: str
    sha256: str
    lines: list[str]
    line_pages: list[int | None] = field(default_factory=list)  # страница PDF для каждой строки

    def numbered(self) -> str:
        return "\n".join(f"L{i:03d}| {ln}" for i, ln in enumerate(self.lines, start=1))

    def page_of(self, line_no: int) -> int | None:
        if 1 <= line_no <= len(self.line_pages):
            return self.line_pages[line_no - 1]
        return None

    def excerpt(self, line_from: int, line_to: int) -> str:
        return " ".join(s.strip() for s in self.lines[max(line_from, 1) - 1:line_to] if s.strip())


def _read_text_file(path: Path) -> list[str]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc).splitlines()
        except UnicodeDecodeError:
            continue
    raise LoadError(f"Не удалось определить кодировку файла {path}")


def _read_pdf(path: Path) -> tuple[list[str], list[int | None]]:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise LoadError("Для PDF нужен пакет pypdf (pip install -r requirements.txt)") from e
    reader = PdfReader(str(path))
    lines: list[str] = []
    pages: list[int | None] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for ln in text.splitlines():
            if ln.strip():
                lines.append(ln.rstrip())
                pages.append(page_no)
    if not lines:
        raise LoadError("В PDF нет текстового слоя (скан?). Нужен OCR — в черновике не поддерживается.")
    return lines, pages


def _read_docx(path: Path) -> list[str]:
    try:
        import docx  # python-docx
    except ImportError as e:  # pragma: no cover
        raise LoadError("Для .docx нужен пакет python-docx (pip install -r requirements.txt)") from e
    d = docx.Document(str(path))
    lines: list[str] = []
    # Параграфы и таблицы в порядке следования в документе
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in d.element.body.iterchildren():
        if child.tag.endswith("}p"):
            p = Paragraph(child, d)
            if p.text.strip():
                prefix = "# " if p.style is not None and p.style.name.lower().startswith(("heading", "заголовок")) else ""
                lines.append(prefix + p.text.strip())
        elif child.tag.endswith("}tbl"):
            t = Table(child, d)
            for row in t.rows:
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                lines.append("| " + " | ".join(cells) + " |")
    return lines


def load_document(path: str | Path) -> Document:
    p = Path(path)
    if not p.is_file():
        raise LoadError(f"Файл не найден: {p}")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    ext = p.suffix.lower()
    pages: list[int | None]
    if ext in (".md", ".markdown", ".txt", ".text", ""):
        lines = _read_text_file(p)
        pages = [None] * len(lines)
    elif ext == ".pdf":
        lines, pages = _read_pdf(p)
    elif ext == ".docx":
        lines = _read_docx(p)
        pages = [None] * len(lines)
    else:
        raise LoadError(f"Формат {ext!r} не поддерживается (md, txt, pdf, docx)")
    if not any(s.strip() for s in lines):
        raise LoadError("Документ пуст")
    return Document(name=p.name, path=str(p), sha256=sha, lines=lines, line_pages=pages)
