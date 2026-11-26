"""Поведение клиента модели: ретраи, запасные модели при суточной квоте, повтор при невалидном JSON."""

from __future__ import annotations

import pytest

from reqcheck import llm as llm_mod
from reqcheck.llm import LLMClient, LLMError
from reqcheck.schemas import ConclusionText

OK = {"rationale": "x", "top_fixes": []}
DAILY = Exception("429 RESOURCE_EXHAUSTED. Quota exceeded: generate_content_free_tier_requests, limit: 20, "
                  "description: Generate content requests per day limit exceeded")
MINUTE = Exception("429 RESOURCE_EXHAUSTED. Quota exceeded: requests per minute limit exceeded")
CLOSED = Exception("404 NOT_FOUND. This model models/old is no longer available to new users")


class Scripted(LLMClient):
    provider = "scripted"

    def __init__(self, script, **kw):
        super().__init__(**kw)
        self.script = {m: list(v) if isinstance(v, list) else [v] for m, v in script.items()}
        self.seen: list[str] = []

    def _call(self, system, user, schema, effort):
        self.seen.append(self.model)
        q = self.script[self.model]
        r = q.pop(0) if len(q) > 1 else q[0]
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)


def test_daily_quota_switches_to_fallback():
    c = Scripted({"a": DAILY, "b": OK}, model="a", fallbacks=["b"])
    assert c.generate("s", "u", ConclusionText).rationale == "x"
    assert c.seen == ["a", "b"] and c.models_used == ["a", "b"] and c.model == "b"


def test_closed_model_switches_too_and_stays_switched():
    c = Scripted({"a": CLOSED, "b": OK}, model="a", fallbacks=["b"])
    c.generate("s", "u", ConclusionText)
    c.generate("s", "u", ConclusionText)
    assert c.seen == ["a", "b", "b"]


def test_all_models_dead_gives_clear_error():
    c = Scripted({"a": DAILY, "b": DAILY}, model="a", fallbacks=["b"])
    with pytest.raises(LLMError, match="Суточная квота исчерпана"):
        c.generate("s", "u", ConclusionText)


def test_per_minute_limit_is_retried_on_same_model():
    c = Scripted({"a": [MINUTE, MINUTE, OK]}, model="a", fallbacks=["b"])
    c.script["b"] = [OK]
    assert c.generate("s", "u", ConclusionText).rationale == "x"
    assert c.seen == ["a", "a", "a"] and c.models_used == ["a"]


def test_invalid_json_is_retried_with_hint():
    c = Scripted({"a": ["не json", '{"rationale": "y", "top_fixes": []}']}, model="a")
    assert c.generate("s", "u", ConclusionText).rationale == "y"
    assert c.seen == ["a", "a"]


def test_gives_up_after_max_retries_on_persistent_server_error():
    c = Scripted({"a": Exception("503 UNAVAILABLE high demand")}, model="a", max_retries=3)
    with pytest.raises(LLMError):
        c.generate("s", "u", ConclusionText)
    assert len(c.seen) == 3


def test_connection_reset_is_retried():
    reset = Exception("[WinError 10053] Программа на вашем хост-компьютере разорвала установленное подключение")
    c = Scripted({"a": [reset, OK]}, model="a")
    assert c.generate("s", "u", ConclusionText).rationale == "x"
    assert c.seen == ["a", "a"]


def test_persistent_overload_switches_to_fallback_quickly():
    busy = Exception("503 UNAVAILABLE high demand")
    c = Scripted({"a": busy, "b": OK}, model="a", fallbacks=["b"])
    assert c.generate("s", "u", ConclusionText).rationale == "x"
    assert c.seen == ["a", "a", "b"] and c.models_used == ["a", "b"]
