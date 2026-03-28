"""Tests for proactive context engine."""
import time
from aura.proactive.change_detector import ChangeDetector, ChangeEvent
from aura.proactive.relevance_scorer import RelevanceScorer, TaskContext
from aura.proactive.context_assembler import ContextAssembler, EnvironmentContext
from aura.proactive.push_controller import PushController
from aura.proactive.attention_tracker import AttentionTracker
from aura.types import EnvironmentSignal


def _make_signal(source="test", payload=None, modality="system", confidence=1.0):
    return EnvironmentSignal(
        source=source,
        payload=payload or {},
        modality=modality,
        confidence=confidence,
    )


def _make_event(event_type="anomaly", source="probe.system", severity=0.8, description="Test", signals=None):
    return ChangeEvent(
        event_type=event_type,
        source=source,
        severity=severity,
        description=description,
        signals=signals or [],
        timestamp=time.time(),
    )


def _make_context(summary="test", critical_alerts=None, relevant_changes=None):
    return EnvironmentContext(
        summary=summary,
        critical_alerts=critical_alerts or [],
        relevant_changes=relevant_changes or [],
        environment_snapshot={},
        agent_hints=[],
        stale_after=time.time() + 60,
    )


class TestChangeDetector:
    def test_detect_threshold_anomaly(self):
        detector = ChangeDetector()
        signals = [_make_signal(
            source="probe.system",
            payload={"cpu_percent": 95, "type": "system_metrics"},
            modality="system",
        )]
        events = detector.detect(signals)
        assert isinstance(events, list)

    def test_detect_state_change(self):
        detector = ChangeDetector()
        signals = [_make_signal(
            source="probe.network",
            payload={"status": "down", "name": "api"},
            modality="network",
        )]
        events = detector.detect(signals)
        assert isinstance(events, list)

    def test_detect_empty_signals(self):
        detector = ChangeDetector()
        events = detector.detect([])
        assert events == []

    def test_event_has_fields(self):
        detector = ChangeDetector()
        signals = [_make_signal(
            source="probe.system",
            payload={"cpu_percent": 99},
        )]
        events = detector.detect(signals)
        for e in events:
            assert isinstance(e, ChangeEvent)
            assert hasattr(e, "event_type")
            assert hasattr(e, "severity")
            assert hasattr(e, "source")


class TestRelevanceScorer:
    def test_score_returns_float(self):
        scorer = RelevanceScorer()
        event = _make_event(
            event_type="anomaly",
            source="probe.system",
            severity=0.8,
            description="High CPU",
        )
        tc = TaskContext(agent_type="sysadmin")
        score = scorer.score(event, tc)
        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_sysadmin_cares_about_system(self):
        scorer = RelevanceScorer()
        event = _make_event(
            event_type="anomaly",
            source="probe.system",
            severity=0.8,
            description="High CPU usage",
        )
        sysadmin_score = scorer.score(event, TaskContext(agent_type="sysadmin"))
        coder_score = scorer.score(event, TaskContext(agent_type="coder"))
        assert sysadmin_score >= coder_score

    def test_coder_cares_about_git(self):
        scorer = RelevanceScorer()
        event = _make_event(
            event_type="state_change",
            source="probe.git",
            severity=0.5,
            description="New commits",
        )
        coder_score = scorer.score(event, TaskContext(agent_type="coder"))
        assert coder_score > 0

    def test_active_files_boost(self):
        scorer = RelevanceScorer()
        event = _make_event(
            event_type="state_change",
            source="probe.filesystem",
            severity=0.5,
            description="File modified: main.py",
        )
        with_files = scorer.score(event, TaskContext(active_files=["main.py"]))
        without_files = scorer.score(event, TaskContext(active_files=[]))
        assert with_files >= without_files


class TestContextAssembler:
    def test_assemble_empty(self):
        assembler = ContextAssembler()
        ctx = assembler.assemble([], {}, {})
        assert isinstance(ctx, EnvironmentContext)

    def test_assemble_with_events(self):
        assembler = ContextAssembler()
        events = [
            _make_event(event_type="anomaly", source="sys", severity=0.9, description="CPU critical"),
            _make_event(event_type="state_change", source="git", severity=0.4, description="New commit"),
        ]
        scores = {events[0].event_id: 0.9, events[1].event_id: 0.5}
        ctx = assembler.assemble(events, scores, {})
        assert len(ctx.critical_alerts) >= 1
        assert ctx.summary


class TestPushController:
    def test_critical_bypasses_throttle(self):
        controller = PushController(min_push_interval=9999)
        alert = _make_event(event_type="anomaly", source="sys", severity=0.95, description="Critical")
        ctx = _make_context(
            summary="test",
            critical_alerts=[alert],
        )
        assert controller.should_push(ctx) is True

    def test_respects_min_interval(self):
        controller = PushController(min_push_interval=60, critical_override=False)
        controller.record_push("normal", 1)
        ctx = _make_context(
            summary="test",
            relevant_changes=[_make_event(event_type="x", source="y", severity=0.5, description="z")],
        )
        assert controller.should_push(ctx) is False

    def test_allows_after_interval(self):
        controller = PushController(min_push_interval=0)
        ctx = _make_context(
            summary="test",
            relevant_changes=[_make_event(event_type="x", source="y", severity=0.5, description="z")],
        )
        assert controller.should_push(ctx) is True


class TestAttentionTracker:
    def test_learn_from_usage(self):
        tracker = AttentionTracker()
        ctx = _make_context(
            critical_alerts=[_make_event(event_type="x", source="probe.system", severity=0.5, description="")],
        )
        tracker.on_push(ctx)
        tracker.on_agent_action("check system", used_context=True)
        weight = tracker.get_source_weight("probe.system")
        assert weight > 1.0

    def test_learn_from_ignore(self):
        tracker = AttentionTracker()
        ctx = _make_context(
            critical_alerts=[_make_event(event_type="x", source="probe.git", severity=0.5, description="")],
        )
        tracker.on_push(ctx)
        tracker.on_agent_action("ignore", used_context=False)
        weight = tracker.get_source_weight("probe.git")
        assert weight < 1.0

    def test_keyword_learning(self):
        tracker = AttentionTracker()
        tracker.on_agent_query("docker container status check")
        boost = tracker.get_keyword_boost("docker container restarted")
        assert boost > 0

    def test_stats(self):
        tracker = AttentionTracker()
        stats = tracker.get_stats()
        assert "total_interactions" in stats
        assert stats["total_interactions"] == 0
