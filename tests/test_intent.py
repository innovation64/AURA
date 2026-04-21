"""Unit tests for IntentInferrer (env-mediated ToM stage).

Scope: lock down the interface and the gap/alert/probe logic of both
the heuristic fallback and the LLM-backed inferrer *without* calling
a real API. LLMIntentInferrer is exercised against a scripted fake
client so we can assert deterministic parsing and fallback behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List

import pytest

from aura.intent import (
    HeuristicIntentInferrer,
    LLMIntentInferrer,
    _parse_intent_json,
    intent_frame_to_dict,
)
from aura.types import IntentFrame, MemoryItem, SceneState


def _scene(summary: str = "cafe, morning, 3 people", entities=None):
    return SceneState(summary=summary, entities=entities or ["Lin Wei", "Zhang Hao"])


def _mem(text: str) -> MemoryItem:
    return MemoryItem(content=text)


# ---------------------------------------------------------------------------
# HeuristicIntentInferrer
# ---------------------------------------------------------------------------

class TestHeuristic:
    def test_empty_query_gives_zero_gap(self):
        f = HeuristicIntentInferrer().infer("", _scene(), [])
        assert f.gap == 0.0
        assert f.implicit_need == []
        assert f.recommended_probes == []

    def test_literal_query_low_gap(self):
        f = HeuristicIntentInferrer().infer("what time is it?", _scene(), [])
        assert f.gap == 0.0
        assert f.recommended_probes == []
        assert f.should_alert is False

    def test_implicit_social_query_raises_gap(self):
        # Multiple implicit markers -> larger gap, probes recommended
        f = HeuristicIntentInferrer().infer(
            "is Lin Wei available to chat right now?", _scene(), []
        )
        assert f.gap > 0.0
        assert any("availability" in n or "social" in n for n in f.implicit_need)
        assert "get_nearby_agents" in f.recommended_probes

    def test_high_gap_triggers_alert(self):
        f = HeuristicIntentInferrer().infer(
            "is this a good time to chat with Lin Wei, is she busy or free?",
            _scene(),
            [],
        )
        # multiple implicit hits should push gap >= 0.4
        assert f.gap >= 0.4
        assert f.should_alert is True

    def test_literal_word_plus_implicit_still_marks_implicit(self):
        # "where" alone is literal, but adding "with whom" flips it
        f = HeuristicIntentInferrer().infer(
            "where is Lin Wei and with whom?", _scene(), []
        )
        assert f.gap > 0.0
        assert len(f.recommended_probes) > 0

    def test_confidence_is_in_unit_range(self):
        f = HeuristicIntentInferrer().infer("hello", _scene(), [])
        assert 0.0 <= f.confidence <= 1.0


# ---------------------------------------------------------------------------
# LLMIntentInferrer with a scripted fake client
# ---------------------------------------------------------------------------

@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResp:
    choices: List[_FakeChoice]


class _FakeChatCompletions:
    def __init__(self, scripted_contents: List[str]):
        self._scripted = list(scripted_contents)
        self.last_kwargs: dict = {}

    def create(self, **kwargs: Any) -> _FakeResp:
        self.last_kwargs = kwargs
        content = self._scripted.pop(0) if self._scripted else ""
        return _FakeResp(choices=[_FakeChoice(_FakeMessage(content=content))])


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, scripted_contents):
        self.chat = _FakeChat(_FakeChatCompletions(scripted_contents))


class TestLLMInferrerParseHappyPath:
    def test_valid_json_is_parsed_into_intent_frame(self):
        payload = json.dumps({
            "literal_need": "is Lin Wei free",
            "implicit_need": ["wants to chat with Lin Wei"],
            "gap": 0.6,
            "recommended_probes": ["get_nearby_agents", "get_recent_events"],
            "should_alert": True,
            "confidence": 0.8,
            "rationale": "social availability cue",
        })
        client = _FakeClient([payload])
        inferrer = LLMIntentInferrer(client=client)

        frame = inferrer.infer("is Lin Wei free?", _scene(), [_mem("saw Lin Wei earlier")])

        assert isinstance(frame, IntentFrame)
        assert frame.gap == 0.6
        assert "get_nearby_agents" in frame.recommended_probes
        assert frame.should_alert is True
        assert frame.confidence == 0.8
        # The LLM call carried our system prompt + user message
        kw = client.chat.completions.last_kwargs
        assert kw["response_format"] == {"type": "json_object"}
        assert any("USER QUERY" in m["content"] for m in kw["messages"] if m["role"] == "user")

    def test_json_fenced_output_is_still_parsed(self):
        payload = "```json\n" + json.dumps({"literal_need": "x"}) + "\n```"
        client = _FakeClient([payload])
        inferrer = LLMIntentInferrer(client=client)
        frame = inferrer.infer("x", _scene(), [])
        assert frame.literal_need == "x"


class TestLLMInferrerFallback:
    def test_invalid_json_falls_back_to_heuristic(self):
        client = _FakeClient(["not-json-at-all"])
        inferrer = LLMIntentInferrer(client=client)
        frame = inferrer.infer("is Lin Wei busy?", _scene(), [])
        # Heuristic should have fired; gap > 0 because "busy" is an implicit marker
        assert frame.gap > 0.0
        assert "get_nearby_agents" in frame.recommended_probes

    def test_missing_client_uses_heuristic_directly(self):
        inferrer = LLMIntentInferrer(client=None)
        frame = inferrer.infer("what time?", _scene(), [])
        assert frame.gap == 0.0

    def test_api_exception_falls_back(self):
        class _BoomChat:
            class completions:
                @staticmethod
                def create(**_kw):
                    raise RuntimeError("network down")
        class _BoomClient:
            chat = _BoomChat()
        inferrer = LLMIntentInferrer(client=_BoomClient())
        frame = inferrer.infer("available?", _scene(), [])
        # Heuristic should have run
        assert frame.gap > 0.0


class TestParseEdgeCases:
    def test_non_dict_root_returns_none(self):
        assert _parse_intent_json("[1,2,3]", "q") is None

    def test_unparseable_returns_none(self):
        assert _parse_intent_json("definitely not json", "q") is None

    def test_missing_literal_need_backfills_from_query(self):
        f = _parse_intent_json(json.dumps({"gap": 0.3}), "original query")
        assert f is not None
        assert f.literal_need == "original query"
        assert f.gap == 0.3

    def test_string_implicit_need_coerced_to_list(self):
        f = _parse_intent_json(
            json.dumps({"literal_need": "q", "implicit_need": "just a string"}),
            "q",
        )
        assert f is not None
        assert f.implicit_need == ["just a string"]

    def test_gap_clamped_to_unit_range(self):
        f = _parse_intent_json(
            json.dumps({"literal_need": "q", "gap": 2.5, "confidence": -0.3}), "q"
        )
        assert f is not None
        assert f.gap == 1.0
        assert f.confidence == 0.0

    def test_non_numeric_gap_defaults_to_zero(self):
        f = _parse_intent_json(
            json.dumps({"literal_need": "q", "gap": "high"}), "q"
        )
        assert f is not None
        assert f.gap == 0.0


class TestSerialization:
    def test_intent_frame_to_dict_roundtrip(self):
        frame = IntentFrame(
            literal_need="q",
            implicit_need=["a"],
            gap=0.4,
            recommended_probes=["t1"],
            should_alert=True,
            confidence=0.7,
            rationale="ok",
        )
        d = intent_frame_to_dict(frame)
        assert d["literal_need"] == "q"
        assert d["gap"] == 0.4
        assert d["recommended_probes"] == ["t1"]
        assert d["should_alert"] is True
