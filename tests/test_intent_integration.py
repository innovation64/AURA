"""Integration tests for IntentInferrer wired into the AURA pipeline.

Verifies:
1. intent_gap_to_budget maps gaps to the documented budget tiers.
2. With intent_enabled=False, pipeline behavior is byte-identical to
   the pre-retrofit path (no IntentFrame leaks into metadata).
3. With intent_enabled=True and a scripted IntentInferrer, the Explore
   budget is overridden dynamically (low gap -> 0 probes, high gap ->
   more probes) and the IntentFrame reaches reasoning.metadata and
   interaction.payload.
4. should_alert=True prepends a heads-up note to the user-facing
   message without losing the underlying content.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pytest

from aura.core import AURAAgent, AURAConfig, intent_gap_to_budget
from aura.explore import Explorer, HeuristicPlanner
from aura.types import IntentFrame, MemoryItem, SceneState


# ---------------------------------------------------------------------------
# intent_gap_to_budget
# ---------------------------------------------------------------------------

class TestIntentGapToBudget:
    @pytest.mark.parametrize("gap,expected", [
        (0.0, 0), (0.1, 0), (0.199, 0),
        (0.2, 1), (0.3, 1),
        (0.4, 2), (0.5, 2),
        (0.6, 3), (0.7, 3),
        (0.8, 5), (1.0, 5),
    ])
    def test_tiered_mapping_with_ample_max(self, gap, expected):
        assert intent_gap_to_budget(gap, max_steps=10) == expected

    def test_budget_is_clamped_by_max_steps(self):
        # Even high gap respects a low max_steps
        assert intent_gap_to_budget(1.0, max_steps=2) == 2
        assert intent_gap_to_budget(0.5, max_steps=1) == 1

    def test_negative_max_steps_gives_zero(self):
        assert intent_gap_to_budget(0.9, max_steps=-3) == 0

    def test_out_of_range_gap_is_clamped(self):
        assert intent_gap_to_budget(2.0, max_steps=10) == 5
        assert intent_gap_to_budget(-0.5, max_steps=10) == 0


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class _ScriptedIntent:
    """Deterministic IntentInferrer that returns a pre-built frame."""
    def __init__(self, frame: IntentFrame):
        self._frame = frame
        self.calls = 0
        self.last_available_tools = None

    def infer(self, user_query, scene, memories, user_profile=None, available_tools=None):
        self.calls += 1
        self.last_available_tools = available_tools
        return self._frame


class _CountingExplorer(Explorer):
    """Explorer that records the max_steps actually used on each call."""
    def __init__(self):
        # Minimal no-op setup: we only care about the call count
        from aura.tools import ToolRegistry
        super().__init__(
            planner=HeuristicPlanner(),
            registry=ToolRegistry(tools=[]),
            max_steps=5,
        )
        self.effective_max_steps_log: List[int] = []

    def explore(self, signals, user_query=None, raw_input=None, max_steps_override=None):
        effective = self.max_steps if max_steps_override is None else max_steps_override
        self.effective_max_steps_log.append(effective)
        return super().explore(
            signals, user_query=user_query, raw_input=raw_input,
            max_steps_override=max_steps_override,
        )


def _agent(intent_enabled: bool, explorer: Optional[Explorer] = None) -> AURAAgent:
    cfg = AURAConfig(
        proactive_enabled=False,   # keep the test deterministic
        guard_enabled=False,
        auditor_enabled=False,
        workflow_enabled=False,
        intent_enabled=intent_enabled,
        explore_max_steps=5,
    )
    return AURAAgent(explorer=explorer, config=cfg)


class TestIntentPipelineDisabled:
    def test_no_intent_metadata_when_flag_off(self):
        agent = _agent(intent_enabled=False)
        result = agent.run(raw_input="cafe, morning", user_query="what time is it?")
        # No intent keys should have leaked into the payload
        assert "intent" not in result.payload
        # And the message is the plain Interactor response
        assert "[heads-up]" not in result.message


class TestIntentPipelineEnabled:
    def test_low_gap_sets_explore_budget_to_zero(self):
        expl = _CountingExplorer()
        agent = _agent(intent_enabled=True, explorer=expl)
        agent.intent = _ScriptedIntent(IntentFrame(
            literal_need="what time",
            implicit_need=[],
            gap=0.1,
            recommended_probes=[],
            should_alert=False,
            confidence=0.9,
            rationale="literal",
        ))
        agent.run(raw_input="cafe, morning", user_query="what time?")
        # gap 0.1 -> 0 probes -> explorer.explore should NOT have been called
        assert expl.effective_max_steps_log == []

    def test_high_gap_allows_probes(self):
        expl = _CountingExplorer()
        agent = _agent(intent_enabled=True, explorer=expl)
        agent.intent = _ScriptedIntent(IntentFrame(
            literal_need="where is Lin Wei",
            implicit_need=["wants to chat"],
            gap=0.7,
            recommended_probes=["get_nearby_agents"],
            should_alert=True,
            confidence=0.8,
            rationale="implicit social",
        ))
        result = agent.run(raw_input="cafe", user_query="where is Lin Wei?")
        # gap 0.7 -> budget 3 (and max_steps=5 is enough), so explore WAS called
        assert len(expl.effective_max_steps_log) == 1
        assert expl.effective_max_steps_log[0] == 3

        # IntentFrame is in the interaction payload and message is alerted
        assert "intent" in result.payload
        assert result.payload["intent"]["gap"] == 0.7
        assert "[heads-up]" in result.message
        # Original Interactor content still present
        assert "Environment" in result.message or "environment" in result.message or "prepared" in result.message or "Based on" in result.message

    def test_medium_gap_maps_to_two_probes(self):
        expl = _CountingExplorer()
        agent = _agent(intent_enabled=True, explorer=expl)
        agent.intent = _ScriptedIntent(IntentFrame(
            literal_need="is it a good time",
            implicit_need=["user availability"],
            gap=0.5,
            recommended_probes=["get_nearby_agents"],
            should_alert=False,
            confidence=0.6,
            rationale="medium",
        ))
        agent.run(raw_input="cafe", user_query="good time to chat?")
        assert expl.effective_max_steps_log == [2]

    def test_available_tools_list_is_passed_through(self):
        """IntentInferrer should receive the registry tool names so the
        LLM backend can whitelist recommended_probes."""
        expl = _CountingExplorer()
        agent = _agent(intent_enabled=True, explorer=expl)
        scripted = _ScriptedIntent(IntentFrame(
            literal_need="q", implicit_need=[], gap=0.3, confidence=0.5,
        ))
        agent.intent = scripted
        agent.run(raw_input="cafe", user_query="where is Lin Wei?")
        # Tool registry is set by AURAConfig defaults; it must be forwarded
        assert scripted.last_available_tools is not None
        assert isinstance(scripted.last_available_tools, list)

    def test_intent_inference_exception_does_not_break_pipeline(self):
        class _BoomIntent:
            def infer(self, **_kw):
                raise RuntimeError("llm down")
        agent = _agent(intent_enabled=True)
        agent.intent = _BoomIntent()
        # Should not raise; intent is effectively absent on this call
        result = agent.run(raw_input="cafe", user_query="what time?")
        assert "intent" not in result.payload
        assert "[heads-up]" not in result.message

    def test_empty_query_skips_intent_and_does_not_emit_metadata(self):
        expl = _CountingExplorer()
        agent = _agent(intent_enabled=True, explorer=expl)
        sentinel = _ScriptedIntent(IntentFrame(literal_need="x", gap=0.9))
        agent.intent = sentinel
        agent.run(raw_input="cafe", user_query=None)
        # IntentInferrer should not even be called when user_query is falsy
        assert sentinel.calls == 0
