"""Привязка цитат к тексту документа.

Модель может «дорисовать» цитату. Поэтому каждую цитату агент ищет в исходном
тексте, сам вычисляет номера строк и отбрасывает то, чего в документе нет.

Если цитата встречается в документе несколько раз (короткие фразы вроде «0 или 1»),
берётся вхождение, ближайшее к строкам требования, к которому относится замечание.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from difflib import SequenceMatcher

from .loader import Document

_NORM_RE = re.compile(r"[\W_]+", re.UNICODE)
_ELLIPSIS_RE = re.compile(r"\s*(?:…|\.{3})\s*")

Span = tuple[int, int]


def normalize(s: str) -> str:
    """Оставляем только буквы и цифры в нижнем регистре: устойчиво к markdown, пробелам, кавычкам."""
    return _NORM_RE.sub("", s.lower().replace("ё", "е"))


def _distance(span: Span, prefer: list[Span] | None) -> int:
    if not prefer:
        return 0
    best = 10 ** 9
    for lo, hi in prefer:
        if span[1] < lo:
            d = lo - span[1]
        elif span[0] > hi:
            d = span[0] - hi
        else:
            d = 0
        best = min(best, d)
    return best


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

    def _exact_all(self, nq: str) -> list[Span]:
        spans: list[Span] = []
        i = self.stream.find(nq)
        while i >= 0:
            spans.append((self._line_at(i), self._line_at(i + len(nq) - 1)))
            i = self.stream.find(nq, i + 1)
        return spans

    def _fuzzy(self, nq: str, prefer: list[Span] | None) -> Span | None:
        if len(nq) < 12 or len(self.stream) == 0:
            return None
        w = int(len(nq) * 1.25)
        step = max(len(nq) // 6, 1)
        scored: list[tuple[float, int]] = []
        for i in range(0, max(len(self.stream) - w, 0) + 1, step):
            r = SequenceMatcher(None, nq, self.stream[i:i + w], autojunk=False).ratio()
            if r >= self.FUZZY_THRESHOLD:
                scored.append((r, i))
        if not scored:
            return None
        top = max(r for r, _ in scored)
        # среди почти лучших совпадений предпочитаем ближайшее к требованию
        cands = [i for r, i in scored if r >= top - 0.03]
        spans = [(self._line_at(i), self._line_at(min(i + w, len(self.stream)) - 1)) for i in cands]
        return min(spans, key=lambda s: _distance(s, prefer))

    def locate(self, quote: str, prefer: list[Span] | None = None) -> Located | None:
        """Находит цитату в документе. None — цитаты в документе нет.

        prefer — диапазоны строк (from, to), рядом с которыми нужно искать при нескольких вхождениях.
        """
        q = (quote or "").strip()
        if not q:
            return None
        # допускаем «…» внутри цитаты: каждая часть должна быть найдена
        parts = [p for p in _ELLIPSIS_RE.split(q) if normalize(p)]
        if not parts:
            return None
        spans: list[Span] = []
        for p in parts:
            occ = self._exact_all(normalize(p))
            if not occ:
                break
            spans.append(min(occ, key=lambda s: _distance(s, prefer)))
        else:
            return Located(min(s[0] for s in spans), max(s[1] for s in spans), False, q)

        fuzzy = self._fuzzy(normalize(q), prefer)
        if fuzzy:
            lf, lt = fuzzy
            return Located(lf, lt, True, self.doc.excerpt(lf, lt))
        return None
