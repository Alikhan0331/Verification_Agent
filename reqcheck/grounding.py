"""Привязка цитат к тексту документа.

Модель может «дорисовать» цитату. Поэтому каждую цитату агент ищет в исходном
тексте, сам вычисляет номера строк и отбрасывает то, чего в документе нет.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from difflib import SequenceMatcher

from .loader import Document

_NORM_RE = re.compile(r"[\W_]+", re.UNICODE)
_ELLIPSIS_RE = re.compile(r"\s*(?:…|\.{3})\s*")


def normalize(s: str) -> str:
    """Оставляем только буквы и цифры в нижнем регистре: устойчиво к markdown, пробелам, кавычкам."""
    return _NORM_RE.sub("", s.lower().replace("ё", "е"))


@dataclass
class Located:
    line_from: int
    line_to: int
    approx: bool  # True — совпадение нечёткое, цитата заменена текстом из источника
    text: str     # цитата, как она будет показана в отчёте


class Grounder:
    FUZZY_THRESHOLD = 0.85

    def __init__(self, doc: Document):
        self.doc = doc
        parts: list[str] = []
        starts: list[int] = []  # позиция начала каждой строки в общем нормализованном потоке
        pos = 0
        for ln in doc.lines:
            n = normalize(ln)
            starts.append(pos)
            parts.append(n)
            pos += len(n)
        self.stream = "".join(parts)
        self._starts = starts

    def _line_at(self, pos: int) -> int:
        return max(bisect_right(self._starts, pos), 1)  # 1-based

    def _exact(self, nq: str) -> tuple[int, int] | None:
        i = self.stream.find(nq)
        if i < 0:
            return None
        return self._line_at(i), self._line_at(i + len(nq) - 1)

    def _fuzzy(self, nq: str) -> tuple[int, int] | None:
        if len(nq) < 12 or len(self.stream) == 0:
            return None
        w = int(len(nq) * 1.25)
        step = max(len(nq) // 6, 1)
        best, best_i = 0.0, -1
        for i in range(0, max(len(self.stream) - w, 0) + 1, step):
            r = SequenceMatcher(None, nq, self.stream[i:i + w], autojunk=False).ratio()
            if r > best:
                best, best_i = r, i
        if best >= self.FUZZY_THRESHOLD and best_i >= 0:
            return self._line_at(best_i), self._line_at(min(best_i + w, len(self.stream)) - 1)
        return None

    def locate(self, quote: str) -> Located | None:
        """Находит цитату в документе. None — цитаты в документе нет."""
        q = (quote or "").strip()
        if not q:
            return None
        # допускаем «…» внутри цитаты: каждая часть должна быть найдена
        parts = [p for p in _ELLIPSIS_RE.split(q) if normalize(p)]
        if not parts:
            return None
        spans = []
        for p in parts:
            span = self._exact(normalize(p))
            if span is None:
                break
            spans.append(span)
        else:
            return Located(min(s[0] for s in spans), max(s[1] for s in spans), False, q)

        fuzzy = self._fuzzy(normalize(q))
        if fuzzy:
            lf, lt = fuzzy
            return Located(lf, lt, True, self.doc.excerpt(lf, lt))
        return None
