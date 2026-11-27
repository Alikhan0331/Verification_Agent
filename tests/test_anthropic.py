"""Клиент Anthropic без сети: форма запроса (то, что принимают актуальные модели Claude) и разбор ответа."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import anthropic
import pytest

from reqcheck import llm as llm_mod
from reqcheck.llm import AnthropicClient, LLMError, make_client
from reqcheck.schemas import ConclusionText, QuestionsResult

GOOD = json.dumps({"rationale": "ок", "top_fixes": []})


class FakeStream:
    def __init__(self, owner, msg):
        self.owner, self.msg = owner, msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


def _msg(text=GOOD, stop="end_turn"):
    return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
                           stop_reason=stop, usage=SimpleNamespace(input_tokens=100, output_tokens=20))


def _tool_msg(payload, stop="tool_use"):
    return SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="submit_result", input=payload)],
                           stop_reason=stop, usage=SimpleNamespace(input_tokens=100, output_tokens=20))


class BadRequest(anthropic.BadRequestError):
    def __init__(self, text):
        Exception.__init__(self, f"Error code: 400 - {text}")


def BadEffort():
    return BadRequest("effort is not supported for this model")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    c = AnthropicClient("claude-sonnet-5-5", "test-key", max_retries=3)
    c.sent = []
    c.script = [_msg()]

    def fake_stream(**kw):
        c.sent.append(copy.deepcopy(kw))        # код меняет словарь настроек после ошибки: фиксируем снимок
        r = c.script.pop(0) if len(c.script) > 1 else c.script[0]
        if isinstance(r, Exception):
            raise r
        return FakeStream(c, r)

    monkeypatch.setattr(c._client.messages, "stream", fake_stream)
    return c


def test_request_has_only_what_current_claude_models_accept(client):
    res = client.generate("система", "запрос", ConclusionText, "итог", effort="medium")
    assert isinstance(res, ConclusionText) and res.rationale == "ок"
    kw = client.sent[0]
    # у Sonnet 5.5 / Opus 5.5 / Fable эти параметры дают 400
    assert "temperature" not in kw and "tools" not in kw and "tool_choice" not in kw
    assert kw["model"] == "claude-sonnet-5-5" and kw["max_tokens"] == 32000 and kw["system"] == "система"
    cfg = kw["output_config"]
    assert cfg["effort"] == "medium" and cfg["format"]["type"] == "json_schema"
    schema = cfg["format"]["schema"]
    assert schema["type"] == "object" and schema["additionalProperties"] is False
    assert (client.input_tokens, client.output_tokens, client.calls_by_model) == (100, 20, {"claude-sonnet-5-5": 1})


def test_all_response_schemas_are_accepted_by_structured_outputs():
    """Каждая схема ответа после преобразования SDK: у всех объектов additionalProperties=false."""
    import inspect

    from pydantic import BaseModel

    import reqcheck.schemas as S

    def bad(x):
        if isinstance(x, dict):
            if x.get("type") == "object" and x.get("additionalProperties") is not False:
                return True
            return any(bad(v) for v in x.values())
        if isinstance(x, list):
            return any(bad(v) for v in x)
        return False

    models = [c for c in vars(S).values() if inspect.isclass(c) and issubclass(c, BaseModel) and c.__module__ == S.__name__]
    assert len(models) >= 15
    for m in models:
        assert not bad(anthropic.transform_schema(m)), m.__name__


def test_model_without_effort_support_is_retried_without_it(client):
    client.script = [BadEffort(), _msg()]
    res = client.generate("s", "u", ConclusionText, "итог", effort="high")
    assert res.rationale == "ок"
    assert "effort" in client.sent[0]["output_config"] and "effort" not in client.sent[1]["output_config"]
    client.generate("s", "u", ConclusionText, "итог", effort="high")        # запоминает: больше не просит effort
    assert "effort" not in client.sent[-1]["output_config"]


def test_refusal_and_truncation_are_reported(client):
    client.script = [_msg(text="", stop="refusal")]
    with pytest.raises(LLMError, match="отказались"):
        client.generate("s", "u", ConclusionText, "итог")
    client.script = [_msg(text='{"rationale": "обрыв', stop="max_tokens")]
    with pytest.raises(LLMError):
        client.generate("s", "u", ConclusionText, "итог")


def test_invalid_json_is_retried_with_hint(client):
    client.script = [_msg(text="не json"), _msg()]
    assert client.generate("s", "u", ConclusionText, "итог").rationale == "ок"
    assert "НЕ ПРОШЁЛ ВАЛИДАЦИЮ" in client.sent[1]["messages"][0]["content"]


def test_make_client_reads_anthropic_settings(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-5-5")
    monkeypatch.setenv("ANTHROPIC_FALLBACK_MODELS", "claude-sonnet-5-5")
    c = make_client("anthropic")
    assert isinstance(c, AnthropicClient) and c.models == ["claude-opus-5-5", "claude-sonnet-5-5"]
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        make_client("anthropic")


def test_questions_schema_roundtrip_through_client(client):
    payload = {"questions": [{"defect_ids": ["D-001"], "owner": "analyst", "question": "Какой лимит нужен для карты?",
                              "options": [], "blocking": True}]}
    client.script = [_msg(text="```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```")]
    res = client.generate("s", "u", QuestionsResult, "вопросы")
    assert res.questions[0].blocking is True


def test_schema_rejected_by_api_switches_to_tool_mode(client):
    client.script = [BadRequest("output_config.format.schema: schema is too complex"),
                     _tool_msg({"rationale": "через инструмент", "top_fixes": []})]
    res = client.generate("система", "u", ConclusionText, "итог")
    assert res.rationale == "через инструмент"
    first, second = client.sent
    assert "format" in first["output_config"] and "tools" not in first
    # запасной режим: инструмент без принудительного выбора (forced tool_choice у новых моделей даёт 400)
    assert second["tool_choice"] == {"type": "auto"} and second["tools"][0]["name"] == "submit_result"
    assert "format" not in second.get("output_config", {}) and "submit_result" in second["system"]
    client.script = [_tool_msg({"rationale": "ещё раз", "top_fixes": []})]
    assert client.generate("s", "u", ConclusionText, "итог").rationale == "ещё раз"
    assert "tools" in client.sent[-1]                      # режим запомнен


def test_unrelated_bad_request_is_not_swallowed(client):
    client.script = [BadRequest("credit balance is too low")]
    with pytest.raises(LLMError, match="credit balance"):
        client.generate("s", "u", ConclusionText, "итог")
    assert client._mode == "json"


def test_refusal_retries_then_uses_next_model_of_the_chain(client):
    """Отказ (stop_reason=refusal) — повтор, затем другая модель цепочки только для этого запроса."""
    client.models = ["claude-sonnet-5-5", "claude-opus-5-5"]
    client.script = [_msg(text="", stop="refusal"), _msg(text="", stop="refusal"), _msg()]
    assert client.generate("s", "u", ConclusionText, "итог").rationale == "ок"
    assert [kw["model"] for kw in client.sent] == ["claude-sonnet-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]
    assert client.model == "claude-sonnet-5-5"                       # основная модель клиента не менялась
    assert client.calls_by_model == {"claude-opus-5-5": 1}


def test_refusal_by_all_models_is_a_clear_error(client):
    client.models = ["claude-sonnet-5-5", "claude-opus-5-5"]
    client.script = [_msg(text="", stop="refusal")]
    with pytest.raises(LLMError, match="Все модели отказались"):
        client.generate("s", "u", ConclusionText, "итог")


def test_conflict_scan_prompt_for_claude_does_not_ask_to_show_reasoning():
    from reqcheck import prompts

    reqs = [{"id": "R-01", "doc_id": "", "section": "A", "summary": "x"}]
    _, plain = prompts.conflict_scan_prompt("d", "1: текст", reqs)
    _, brief = prompts.conflict_scan_prompt("d", "1: текст", reqs, brief_notes=True)
    assert "пройди по шагам" in plain and "Покажи расчёт в analysis" in plain         # Gemini: прежний промпт
    assert "пройди по шагам" not in brief and "Покажи расчёт" not in brief


def test_error_text_shows_reason_from_response_body(client):
    class WithBody(anthropic.BadRequestError):
        def __init__(self):
            Exception.__init__(self, "Error code: 400")
            self.body = {"type": "error", "error": {"type": "invalid_request_error",
                                                   "message": "Your credit balance is too low to access the Anthropic API."}}

    client.script = [WithBody()]
    with pytest.raises(LLMError) as ei:
        client.generate("s", "u", ConclusionText, "итог")
    text = str(ei.value)
    assert "credit balance is too low" in text and "Billing" in text


def test_ping_without_key_reports_instead_of_crashing(monkeypatch, capsys):
    from reqcheck import ping

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert ping.main(["--provider", "anthropic"]) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().out
