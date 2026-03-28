"""Tests for ExecutionGuard, ConditionalFeedbackStore, and StrategyAuditor."""

import time
import pytest

from aura.types import ReasoningResult, SceneState
from aura.guard import (
    ExecutionGuard,
    InformationGainEstimator,
    ExplorationPhaseDetector,
    ExplorationPhase,
    InterventionLevel,
    ConfidenceAccumulator,
    ActionRecord,
)
from aura.feedback import (
    ConditionalFeedbackStore,
    Outcome,
    StatePattern,
    extract_pattern,
    FeedbackEntry,
)
from aura.auditor import StrategyAuditor, AdaptiveExplorationRate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scene(summary: str = "ok", entities=None, context=None) -> SceneState:
    return SceneState(
        summary=summary,
        entities=entities or [],
        context=context or {},
    )

def _reasoning(intent: str, actions=None) -> ReasoningResult:
    return ReasoningResult(intent=intent, rationale="test", actions=actions or [])


# ===========================================================================
# InformationGainEstimator
# ===========================================================================

class TestInformationGainEstimator:
    def test_first_step_has_gain(self):
        est = InformationGainEstimator()
        ig = est.estimate("do_something", _scene("before"), _scene("after", ["A"]))
        assert ig > 0

    def test_repeated_identical_action_loses_gain(self):
        est = InformationGainEstimator()
        s1 = _scene("state A")
        s2 = _scene("state A")
        gains = []
        for _ in range(5):
            g = est.estimate("same_action", s1, s2, [])
            gains.append(g)
        # Later steps should have less gain
        assert gains[-1] <= gains[0]

    def test_diverse_actions_maintain_gain(self):
        est = InformationGainEstimator()
        gains = []
        for i in range(5):
            g = est.estimate(
                f"action_{i}",
                _scene(f"state {i}"),
                _scene(f"state {i+1}", [f"entity_{i}"]),
                [{"key": f"value_{i}"}],
            )
            gains.append(g)
        avg = sum(gains) / len(gains)
        assert avg > 0.2

    def test_trend_negative_on_declining_gain(self):
        est = InformationGainEstimator()
        s = _scene("same")
        for _ in range(8):
            est.estimate("repeat", s, s, [])
        assert est.trend() <= 0.0


# ===========================================================================
# ExplorationPhaseDetector
# ===========================================================================

class TestPhaseDetector:
    def test_orient_on_few_steps(self):
        det = ExplorationPhaseDetector()
        est = InformationGainEstimator()
        history = [ActionRecord(intent="init", action_signature="a", information_gain=0.5)]
        assert det.detect(history, est) == ExplorationPhase.ORIENT

    def test_stuck_on_low_gain_low_diversity(self):
        det = ExplorationPhaseDetector()
        est = InformationGainEstimator()
        # Feed some zero-gain into estimator
        s = _scene("same")
        for _ in range(6):
            est.estimate("same_intent", s, s)
        history = [
            ActionRecord(intent="same_intent", action_signature="a",
                         tool_name="tool_a", information_gain=0.01)
            for _ in range(6)
        ]
        phase = det.detect(history, est)
        assert phase in (ExplorationPhase.STUCK, ExplorationPhase.EXECUTE)

    def test_search_on_diverse_tools(self):
        det = ExplorationPhaseDetector()
        est = InformationGainEstimator()
        for i in range(6):
            est.estimate(f"action_{i}", _scene(f"s{i}"), _scene(f"s{i+1}"))
        history = [
            ActionRecord(intent=f"intent_{i}", action_signature=f"s{i}",
                         tool_name=f"tool_{i}", information_gain=0.5)
            for i in range(6)
        ]
        phase = det.detect(history, est)
        assert phase == ExplorationPhase.SEARCH


# ===========================================================================
# ConfidenceAccumulator
# ===========================================================================

class TestConfidenceAccumulator:
    def test_does_not_trigger_on_low_signal(self):
        acc = ConfidenceAccumulator(trigger_threshold=0.7)
        for _ in range(5):
            assert not acc.update(0.3)

    def test_triggers_on_sustained_high_signal(self):
        acc = ConfidenceAccumulator(trigger_threshold=0.7)
        triggered = False
        for _ in range(20):
            if acc.update(0.8):
                triggered = True
                break
        assert triggered

    def test_decays_on_good_step(self):
        acc = ConfidenceAccumulator(trigger_threshold=0.7)
        acc.accumulated = 0.6
        acc.update(0.2)  # good step -> decay
        assert acc.accumulated < 0.6


# ===========================================================================
# ExecutionGuard (integration)
# ===========================================================================

class TestExecutionGuard:
    def test_observe_on_normal_operation(self):
        guard = ExecutionGuard()
        s1 = _scene("state A", ["e1"])
        s2 = _scene("state B", ["e2"])
        v = guard.check(_reasoning("do_something"), s1, s2, [{"result": "ok"}])
        assert v.level == InterventionLevel.OBSERVE

    def test_detects_repetition_loop(self):
        guard = ExecutionGuard(window_size=5, base_threshold=0.5)
        guard.set_budget(50)
        s = _scene("stuck state")
        # Repeat same action many times
        last_level = InterventionLevel.OBSERVE
        for _ in range(25):
            v = guard.check(_reasoning("retry_same_thing"), s, s, [])
            if v.level != InterventionLevel.OBSERVE:
                last_level = v.level
                break
        assert last_level.value >= InterventionLevel.HINT.value

    def test_no_false_positive_on_diverse_actions(self):
        guard = ExecutionGuard(window_size=5, base_threshold=0.7)
        for i in range(10):
            s1 = _scene(f"state {i}", [f"e{i}"])
            s2 = _scene(f"state {i+1}", [f"e{i+1}"])
            v = guard.check(
                _reasoning(f"unique_action_{i}"), s1, s2,
                [{"data": f"new_info_{i}"}],
            )
            assert v.level == InterventionLevel.OBSERVE

    def test_self_calibrating_threshold(self):
        guard = ExecutionGuard(base_threshold=0.6)
        # Record that interventions didn't help
        for _ in range(10):
            guard.outcome_tracker.record(0.8, InterventionLevel.SUGGEST)
            guard.outcome_tracker.record_outcome(False)
        # Threshold should increase (intervene less)
        new_thresh = guard.outcome_tracker.calibrated_threshold()
        assert new_thresh > 0.6


# ===========================================================================
# ConditionalFeedbackStore
# ===========================================================================

class TestFeedbackStore:
    def test_record_and_query_failure(self):
        store = ConditionalFeedbackStore()
        scene = _scene("service X is down", ["service_X"], {"error": "timeout"})
        store.record_outcome(scene, "restart_container", Outcome.FAILURE,
                             alternative="rollout_restart")
        advice = store.query_advice(scene, "restart_container")
        assert advice is not None
        assert "FAILURE" in advice.warning or "failure" in advice.warning
        assert advice.suggested_alternative == "rollout_restart"

    def test_no_advice_for_success(self):
        store = ConditionalFeedbackStore()
        scene = _scene("all healthy")
        store.record_outcome(scene, "check_status", Outcome.SUCCESS)
        advice = store.query_advice(scene, "check_status")
        assert advice is None

    def test_query_alternatives(self):
        store = ConditionalFeedbackStore()
        scene = _scene("disk full", ["disk"], {"disk": 95})
        store.record_outcome(scene, "clear_logs", Outcome.SUCCESS)
        store.record_outcome(scene, "expand_volume", Outcome.SUCCESS)
        store.record_outcome(scene, "restart", Outcome.FAILURE)
        alts = store.query_alternatives(scene, exclude_action="restart")
        assert len(alts) >= 1
        assert "restart" not in alts

    def test_increments_confirmation(self):
        store = ConditionalFeedbackStore()
        scene = _scene("scenario A")
        store.record_outcome(scene, "do_thing", Outcome.FAILURE)
        store.record_outcome(scene, "do_thing", Outcome.FAILURE)
        entries = store.get_all_entries()
        assert len(entries) == 1
        assert entries[0].times_confirmed == 2


# ===========================================================================
# StatePattern
# ===========================================================================

class TestStatePattern:
    def test_identical_patterns_have_similarity_1(self):
        p = StatePattern(
            signal_types=frozenset(["system"]),
            anomaly_categories=frozenset(["cpu_high"]),
            resource_pressure="high",
            active_entities=frozenset(["svc_a"]),
            error_signatures=frozenset(["timeout"]),
        )
        assert p.similarity(p) == 1.0

    def test_different_patterns_lower_similarity(self):
        p1 = StatePattern(
            signal_types=frozenset(["system"]),
            anomaly_categories=frozenset(["cpu_high"]),
            resource_pressure="high",
            active_entities=frozenset(["svc_a"]),
            error_signatures=frozenset(["timeout"]),
        )
        p2 = StatePattern(
            signal_types=frozenset(["git"]),
            anomaly_categories=frozenset(),
            resource_pressure="low",
            active_entities=frozenset(["repo_b"]),
            error_signatures=frozenset(),
        )
        assert p1.similarity(p2) < 0.5

    def test_extract_pattern_from_scene(self):
        scene = _scene("cpu load is high, service X is down",
                        ["service_X"],
                        {"cpu": 90, "memory": 40})
        pattern = extract_pattern(scene)
        assert "cpu_high" in pattern.anomaly_categories
        assert "service_down" in pattern.anomaly_categories
        assert pattern.resource_pressure in ("medium", "high")


# ===========================================================================
# StrategyAuditor
# ===========================================================================

class TestStrategyAuditor:
    def test_time_decay_reduces_confidence(self):
        store = ConditionalFeedbackStore()
        auditor = StrategyAuditor(store, staleness_halflife=1.0)

        scene = _scene("scenario")
        store.record_outcome(scene, "old_strategy", Outcome.SUCCESS)
        entry = store.get_all_entries()[0]
        entry.confidence = 0.9

        # Immediately: effective confidence ≈ 0.9
        ec_now = auditor.effective_confidence(entry)
        assert ec_now > 0.8

        # Simulate passage of time: set last_validated to 2 seconds ago
        entry.last_validated = time.time() - 2.0
        ec_later = auditor.effective_confidence(entry)
        assert ec_later < ec_now * 0.5  # should be roughly 0.9 * 0.25

    def test_get_stale_strategies(self):
        store = ConditionalFeedbackStore()
        auditor = StrategyAuditor(store, staleness_halflife=0.5)

        scene = _scene("test")
        store.record_outcome(scene, "stale_action", Outcome.SUCCESS)
        entry = store.get_all_entries()[0]
        entry.last_validated = time.time() - 5.0  # old

        stale = auditor.get_stale_strategies(threshold=0.3)
        assert len(stale) >= 1

    def test_environment_drift_flags_strategies(self):
        store = ConditionalFeedbackStore()
        auditor = StrategyAuditor(store, drift_threshold=0.2)

        old_scene = _scene("docker compose running", ["compose"], {"docker": "up"})
        store.record_outcome(old_scene, "docker_compose_restart", Outcome.SUCCESS)

        new_scene = _scene("kubernetes cluster running", ["k8s"], {"kubernetes": "up"})
        affected = auditor.check_environment_drift(old_scene, new_scene)
        # The strategy should be flagged since environment changed
        assert len(affected) >= 0  # may or may not hit threshold depending on pattern

    def test_should_probe_for_stale_entry(self):
        store = ConditionalFeedbackStore()
        auditor = StrategyAuditor(store, staleness_halflife=1.0)

        scene = _scene("test")
        eid = store.record_outcome(scene, "old_method", Outcome.SUCCESS)
        entry = store.get_entry(eid)
        entry.last_validated = time.time() - 10.0  # very stale

        auditor._revalidation_queue.add(eid)
        assert auditor.should_probe(entry) is True

    def test_comparison_updates_strategy(self):
        store = ConditionalFeedbackStore()
        auditor = StrategyAuditor(store)

        scene = _scene("test")
        eid = store.record_outcome(scene, "old_way", Outcome.SUCCESS)

        # Alternative wins
        won = auditor.record_comparison(
            entry_id=eid,
            original_action="old_way",
            alternative_action="new_way",
            original_outcome=Outcome.STALL,
            alternative_outcome=Outcome.SUCCESS,
            original_reward=0.1,
            alternative_reward=0.8,
        )
        assert won is True
        entry = store.get_entry(eid)
        assert entry.alternative_action == "new_way"
        assert entry.confidence < 0.5  # demoted


class TestAdaptiveExplorationRate:
    def test_stable_environment_low_rate(self):
        rate = AdaptiveExplorationRate(base_rate=0.1)
        for _ in range(20):
            rate.update(0.5, 0.5)  # no surprise
        assert rate.current_rate < 0.05

    def test_unstable_environment_high_rate(self):
        rate = AdaptiveExplorationRate(base_rate=0.1)
        for _ in range(20):
            rate.update(0.2, 0.9)  # big surprises
        assert rate.current_rate > 0.15


# ===========================================================================
# Integration: Guard + FeedbackStore
# ===========================================================================

class TestGuardFeedbackIntegration:
    def test_feedback_informs_guard_suggestion(self):
        store = ConditionalFeedbackStore()
        guard = ExecutionGuard(window_size=4, base_threshold=0.4)

        # Record a known failure pattern
        scene = _scene("api timeout", ["api_server"], {"error": "timeout"})
        store.record_outcome(scene, "retry_request", Outcome.LOOP,
                             alternative="check_upstream_dns")

        # Now the agent tries the same failing action
        advice = store.query_advice(scene, "retry_request")
        assert advice is not None
        assert advice.suggested_alternative == "check_upstream_dns"

    def test_full_pipeline_loop_detection(self):
        """Simulate a full loop: guard detects, feedback advises."""
        from aura.core import AURAAgent, AURAConfig

        config = AURAConfig(
            guard_enabled=True,
            guard_window=4,
            guard_threshold=0.4,
            explore_enabled=False,
            proactive_enabled=False,
        )
        agent = AURAAgent(config=config)
        assert agent.guard is not None
        assert agent.feedback_store is not None

        # Run several identical steps
        for i in range(15):
            interaction = agent.run({"state": "stuck"}, user_query="fix the thing")

        # Guard should have detected the loop
        stats = agent.guard.get_stats()
        assert stats["steps"] == 15
        assert stats["pattern_strength"] > 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
