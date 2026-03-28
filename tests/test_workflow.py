"""Tests for the workflow engine — pipeline memory, background validation, tool forge."""

from __future__ import annotations

import time
from typing import Any, Dict

import pytest

from aura.feedback import ConditionalFeedbackStore, StatePattern
from aura.tools import Tool, ToolCall, ToolPolicy, ToolRegistry, ToolResult
from aura.workflow import (
    BackgroundValidator,
    ToolForge,
    ToolGapSignal,
    ValidationResult,
    Workflow,
    WorkflowEngine,
    WorkflowMemory,
    WorkflowStatus,
    WorkflowStep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pattern(**overrides: Any) -> StatePattern:
    defaults = dict(
        signal_types=frozenset({"cpu", "memory"}),
        anomaly_categories=frozenset(),
        active_entities=frozenset({"server1"}),
        error_signatures=frozenset(),
        resource_pressure="normal",
    )
    defaults.update(overrides)
    return StatePattern(**defaults)


def _echo_handler(**kwargs: Any) -> Dict[str, Any]:
    return {"echoed": kwargs}


def _fail_handler(**kwargs: Any) -> Dict[str, Any]:
    raise RuntimeError("tool broke")


def _make_registry(*names: str) -> ToolRegistry:
    tools = [Tool(name=n, description=f"test tool {n}", handler=_echo_handler) for n in names]
    return ToolRegistry(tools)


def _make_steps(*names: str) -> list:
    return [WorkflowStep(tool_name=n) for n in names]


# ===========================================================================
# WorkflowMemory
# ===========================================================================

class TestWorkflowMemory:

    def test_record_and_lookup(self):
        mem = WorkflowMemory()
        pattern = _make_pattern()
        steps = _make_steps("a", "b", "c")
        wf = mem.record(steps, pattern, name="test_wf")

        assert wf.status == WorkflowStatus.CANDIDATE
        assert len(wf.steps) == 3

        # Lookup with same pattern should find it
        found = mem.lookup(pattern)
        assert found is not None
        assert found.workflow_id == wf.workflow_id

    def test_lookup_returns_none_for_dissimilar_pattern(self):
        mem = WorkflowMemory(similarity_threshold=0.7)
        pattern = _make_pattern()
        mem.record(_make_steps("a"), pattern, name="wf")

        different = _make_pattern(
            signal_types=frozenset({"disk", "network"}),
            anomaly_categories=frozenset({"crash"}),
            active_entities=frozenset({"other"}),
            resource_pressure="critical",
        )
        assert mem.lookup(different) is None

    def test_promotion_after_successes(self):
        mem = WorkflowMemory(promotion_threshold=2)
        pattern = _make_pattern()
        wf = mem.record(_make_steps("a"), pattern)

        assert wf.status == WorkflowStatus.CANDIDATE
        mem.record_outcome(wf.workflow_id, succeeded=True)
        assert wf.status == WorkflowStatus.CANDIDATE  # 1 success, need 2
        mem.record_outcome(wf.workflow_id, succeeded=True)
        assert wf.status == WorkflowStatus.ACTIVE

    def test_demotion_after_failures(self):
        mem = WorkflowMemory(promotion_threshold=1, demotion_threshold=2)
        pattern = _make_pattern()
        wf = mem.record(_make_steps("a"), pattern)
        mem.record_outcome(wf.workflow_id, succeeded=True)  # promote
        assert wf.status == WorkflowStatus.ACTIVE

        mem.record_outcome(wf.workflow_id, succeeded=False)
        assert wf.status == WorkflowStatus.ACTIVE  # 1 failure, need 2
        mem.record_outcome(wf.workflow_id, succeeded=False)
        assert wf.status == WorkflowStatus.STALE

    def test_duplicate_detection(self):
        mem = WorkflowMemory()
        pattern = _make_pattern()
        wf1 = mem.record(_make_steps("a", "b"), pattern)
        wf2 = mem.record(_make_steps("a", "b"), pattern)
        assert wf1.workflow_id == wf2.workflow_id
        assert wf2.times_used == 1  # incremented once by duplicate record

    def test_eviction_when_full(self):
        mem = WorkflowMemory(max_workflows=2)
        p1 = _make_pattern(active_entities=frozenset({"e1"}))
        p2 = _make_pattern(active_entities=frozenset({"e2"}))
        p3 = _make_pattern(active_entities=frozenset({"e3"}))

        mem.record(_make_steps("a"), p1, name="w1")
        mem.record(_make_steps("b"), p2, name="w2")
        mem.record(_make_steps("c"), p3, name="w3")

        # Should have evicted weakest, keeping 2
        assert len(mem._workflows) == 2

    def test_validation_result(self):
        mem = WorkflowMemory()
        pattern = _make_pattern()
        wf = mem.record(_make_steps("a"), pattern)
        wf.status = WorkflowStatus.VALIDATING

        # Valid result restores to ACTIVE
        mem.record_validation(ValidationResult(
            workflow_id=wf.workflow_id, still_valid=True,
        ))
        assert wf.status == WorkflowStatus.ACTIVE

        # Invalid result demotes to STALE
        mem.record_validation(ValidationResult(
            workflow_id=wf.workflow_id, still_valid=False, issues=["broken"],
        ))
        assert wf.status == WorkflowStatus.STALE

    def test_list_active_and_stale(self):
        mem = WorkflowMemory(promotion_threshold=1, demotion_threshold=1)
        p = _make_pattern()
        wf1 = mem.record(_make_steps("a"), p, name="active_wf")
        mem.record_outcome(wf1.workflow_id, succeeded=True)

        p2 = _make_pattern(active_entities=frozenset({"x"}))
        wf2 = mem.record(_make_steps("b"), p2, name="stale_wf")
        mem.record_outcome(wf2.workflow_id, succeeded=True)  # promote
        mem.record_outcome(wf2.workflow_id, succeeded=False)  # demote

        assert len(mem.list_active()) >= 1
        assert len(mem.list_stale()) >= 1

    def test_retire(self):
        mem = WorkflowMemory()
        wf = mem.record(_make_steps("a"), _make_pattern())
        mem.retire(wf.workflow_id)
        assert wf.status == WorkflowStatus.RETIRED

    def test_get_stats(self):
        mem = WorkflowMemory()
        mem.record(_make_steps("a"), _make_pattern())
        stats = mem.get_stats()
        assert stats["total"] == 1
        assert "candidate" in stats["by_status"]


# ===========================================================================
# BackgroundValidator
# ===========================================================================

class TestBackgroundValidator:

    def test_validate_all_tools_present(self):
        reg = _make_registry("a", "b")
        mem = WorkflowMemory()
        wf = mem.record(_make_steps("a", "b"), _make_pattern())
        validator = BackgroundValidator(reg, mem)

        result = validator.validate(wf)
        assert result.still_valid
        assert len(result.issues) == 0

    def test_validate_missing_tool(self):
        reg = _make_registry("a")  # "b" missing
        mem = WorkflowMemory()
        wf = mem.record(_make_steps("a", "b"), _make_pattern())
        validator = BackgroundValidator(reg, mem)

        result = validator.validate(wf)
        assert not result.still_valid
        assert any("no longer available" in i for i in result.issues)

    def test_validate_low_success_rate(self):
        reg = _make_registry("a")
        mem = WorkflowMemory()
        wf = mem.record(_make_steps("a"), _make_pattern())
        # Simulate low success rate
        wf.times_used = 5
        wf.times_succeeded = 1
        wf.times_failed = 4

        validator = BackgroundValidator(reg, mem)
        result = validator.validate(wf)
        assert not result.still_valid
        assert any("success rate" in i for i in result.issues)

    def test_validate_denied_by_policy(self):
        reg = _make_registry("a")
        reg.policy = ToolPolicy(deny=["a"])
        mem = WorkflowMemory()
        wf = mem.record(_make_steps("a"), _make_pattern())
        validator = BackgroundValidator(reg, mem)

        result = validator.validate(wf)
        assert not result.still_valid
        assert any("denied" in i for i in result.issues)

    def test_should_validate_stale(self):
        reg = _make_registry("a")
        mem = WorkflowMemory()
        wf = mem.record(_make_steps("a"), _make_pattern())
        wf.status = WorkflowStatus.STALE
        validator = BackgroundValidator(reg, mem)
        assert validator.should_validate(wf)

    def test_should_validate_budget(self):
        reg = _make_registry("a")
        mem = WorkflowMemory()
        wf = mem.record(_make_steps("a"), _make_pattern())
        wf.status = WorkflowStatus.ACTIVE
        wf.last_validated = time.time()  # recently validated

        validator = BackgroundValidator(reg, mem, validation_rate=0.5)
        # validation_rate=0.5 means every 2nd step
        results = [validator.should_validate(wf) for _ in range(10)]
        assert any(results)  # should trigger at least sometimes

    def test_get_stats(self):
        reg = _make_registry("a")
        mem = WorkflowMemory()
        validator = BackgroundValidator(reg, mem)
        stats = validator.get_stats()
        assert "total_validations" in stats


# ===========================================================================
# ToolForge
# ===========================================================================

class TestToolForge:

    def test_compose_creates_composite_tool(self):
        reg = _make_registry("step1", "step2")
        forge = ToolForge(reg)

        tool = forge.compose(
            name="composed",
            description="test composite",
            tool_sequence=[("step1", {"x": 1}), ("step2", {"y": "$x"})],
        )
        assert tool is not None
        assert reg.has("composed")

        # Execute the composite tool
        result = reg.execute(ToolCall(name="composed"))
        assert result.ok

    def test_compose_fails_if_tool_missing(self):
        reg = _make_registry("step1")
        forge = ToolForge(reg)

        tool = forge.compose(
            name="bad_compose",
            description="missing step2",
            tool_sequence=[("step1", {}), ("missing_tool", {})],
        )
        assert tool is None

    def test_adapt_creates_adapted_tool(self):
        reg = _make_registry("base_tool")
        forge = ToolForge(reg)

        tool = forge.adapt(
            base_tool_name="base_tool",
            new_name="adapted",
            new_description="adapted version",
            fixed_args={"mode": "fast"},
        )
        assert tool is not None
        assert reg.has("adapted")

        result = reg.execute(ToolCall(name="adapted", arguments={"extra": "val"}))
        assert result.ok
        assert result.output["echoed"]["mode"] == "fast"
        assert result.output["echoed"]["extra"] == "val"

    def test_adapt_fails_if_base_missing(self):
        reg = _make_registry()
        forge = ToolForge(reg)
        assert forge.adapt("nonexistent", "new", "desc", {}) is None

    def test_retire_broken(self):
        reg = _make_registry("broken_tool")
        forge = ToolForge(reg)

        assert forge.retire_broken("broken_tool", "always fails")
        assert not reg.has("broken_tool")
        assert len(reg.list_retired()) == 1

    def test_report_gap_threshold(self):
        reg = _make_registry()
        forge = ToolForge(reg, forge_threshold=3)

        assert forge.report_gap("missing", "ctx") is None
        assert forge.report_gap("missing", "ctx") is None
        signal = forge.report_gap("missing", "ctx")
        assert signal is not None
        assert signal.times_observed == 3

    def test_get_stats(self):
        reg = _make_registry("a", "b")
        forge = ToolForge(reg)
        forge.compose("c", "desc", [("a", {}), ("b", {})])
        stats = forge.get_stats()
        assert stats["forged_count"] == 1
        assert "c" in stats["forged_tools"]


# ===========================================================================
# WorkflowEngine
# ===========================================================================

class TestWorkflowEngine:

    def test_before_action_no_workflow_returns_none(self):
        reg = _make_registry("a")
        engine = WorkflowEngine(reg)
        result = engine.before_action(_make_pattern())
        assert result is None

    def test_full_lifecycle_record_and_reuse(self):
        reg = _make_registry("tool_a", "tool_b")
        engine = WorkflowEngine(reg, reuse_rate=0.8)

        pattern = _make_pattern()

        # Step 1: explore (no known workflow)
        wf = engine.before_action(pattern)
        assert wf is None

        # Record exploration
        engine.start_recording()
        engine.record_step("tool_a", {}, ToolResult(name="tool_a", ok=True, output="ok"))
        engine.record_step("tool_b", {}, ToolResult(name="tool_b", ok=True, output="ok"))
        recorded = engine.finish_recording(pattern, succeeded=True, name="my_pipeline")
        assert recorded is not None
        assert recorded.status == WorkflowStatus.CANDIDATE

        # Promote it: record successes
        engine.memory.record_outcome(recorded.workflow_id, succeeded=True)
        engine.memory.record_outcome(recorded.workflow_id, succeeded=True)
        assert recorded.status == WorkflowStatus.ACTIVE

        # Step 2: reuse
        reused = engine.before_action(pattern)
        assert reused is not None
        assert reused.workflow_id == recorded.workflow_id

    def test_after_action_records_to_feedback(self):
        reg = _make_registry("a")
        feedback = ConditionalFeedbackStore()
        engine = WorkflowEngine(reg, feedback_store=feedback)

        pattern = _make_pattern()
        wf = engine.memory.record(_make_steps("a"), pattern, name="test")
        engine.after_action(wf.workflow_id, succeeded=True, scene_pattern=pattern)

        assert wf.times_succeeded == 1

    def test_handle_tool_failure_below_threshold(self):
        reg = _make_registry("flaky")
        engine = WorkflowEngine(reg, forge_threshold=5)
        result = engine.handle_tool_failure("flaky", "timeout", "test context")
        assert result is None  # below threshold

    def test_handle_tool_failure_retires_broken(self):
        reg = _make_registry("broken")
        engine = WorkflowEngine(reg, forge_threshold=1)
        engine.handle_tool_failure("broken", "always fails", "ctx")
        assert not reg.has("broken")

    def test_recording_not_stored_on_failure(self):
        reg = _make_registry("a")
        engine = WorkflowEngine(reg)

        engine.start_recording()
        engine.record_step("a", {}, ToolResult(name="a", ok=True, output="ok"))
        result = engine.finish_recording(_make_pattern(), succeeded=False)
        assert result is None
        assert len(engine.memory._workflows) == 0

    def test_get_stats(self):
        reg = _make_registry("a")
        engine = WorkflowEngine(reg)
        engine.before_action(_make_pattern())  # one explore step
        stats = engine.get_stats()
        assert stats["steps"] == 1
        assert stats["explore_count"] == 1
        assert "workflows" in stats
        assert "validation" in stats
        assert "forge" in stats

    def test_validation_triggers_during_reuse(self):
        reg = _make_registry("a")
        engine = WorkflowEngine(reg, reuse_rate=1.0, validation_rate=1.0)

        pattern = _make_pattern()
        wf = engine.memory.record(_make_steps("a"), pattern, name="val_test")
        # Promote to active
        wf.status = WorkflowStatus.ACTIVE
        wf.confidence = 0.9
        # Make it old so validation triggers
        wf.last_validated = time.time() - 7200

        reused = engine.before_action(pattern)
        # Validation should have run since last_validated is old
        assert reused is not None or wf.last_validated > time.time() - 10


# ===========================================================================
# Integration: WorkflowEngine + ToolRegistry dynamic management
# ===========================================================================

class TestToolRegistryDynamic:

    def test_register_and_retire(self):
        reg = _make_registry("a", "b")
        assert reg.has("a")

        retired = reg.retire("a")
        assert retired is not None
        assert not reg.has("a")
        assert len(reg.list_retired()) == 1

        assert reg.restore("a")
        assert reg.has("a")
        assert len(reg.list_retired()) == 0

    def test_restore_nonexistent(self):
        reg = _make_registry()
        assert not reg.restore("nonexistent")

    def test_retire_nonexistent(self):
        reg = _make_registry()
        assert reg.retire("nonexistent") is None
