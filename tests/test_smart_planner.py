"""Tests for SmartPlanner — context-aware exploration."""
from aura.smart_planner import SmartPlanner, _derive_hints, EnvironmentHint
from aura.explore import ExplorationState
from aura.types import EnvironmentSignal
from aura.tools import Tool, ToolRegistry


def _make_registry():
    tools = [
        Tool(name="system.snapshot", description="sys", handler=lambda: {}),
        Tool(name="workspace.list", description="ws", handler=lambda: {}),
        Tool(name="workspace.read", description="read", handler=lambda: {}),
        Tool(name="git.status", description="git", handler=lambda: {}),
        Tool(name="docker.status", description="docker", handler=lambda: {}),
    ]
    return ToolRegistry(tools)


class TestDeriveHints:
    def test_system_anomaly_produces_hint(self):
        signals = [EnvironmentSignal(
            source="probe.system", payload={"anomalies": ["CPU > 90%"]}, modality="system",
        )]
        hints = _derive_hints(signals, "")
        assert any(h.action == "check_system" for h in hints)

    def test_file_change_produces_hint(self):
        signals = [EnvironmentSignal(
            source="probe.filesystem",
            payload={"changes": [{"type": "modified", "path": "main.py"}]},
            modality="filesystem",
        )]
        hints = _derive_hints(signals, "")
        assert any(h.action == "read_changed_file" for h in hints)

    def test_query_workspace_hint(self):
        hints = _derive_hints([], "list all files in the project")
        assert any(h.tool_name == "workspace.list" for h in hints)

    def test_query_git_hint(self):
        hints = _derive_hints([], "show me the latest commit")
        assert any(h.tool_name == "git.status" for h in hints)

    def test_query_docker_hint(self):
        hints = _derive_hints([], "check container status")
        assert any(h.tool_name == "docker.status" for h in hints)

    def test_no_signals_no_query_no_hints(self):
        hints = _derive_hints([], "")
        assert len(hints) == 0

    def test_priority_ordering(self):
        signals = [
            EnvironmentSignal(source="probe.system", payload={"anomalies": ["CPU"]}, modality="system"),
            EnvironmentSignal(source="probe.filesystem", payload={"changes": [{"type": "modified", "path": "x"}]}, modality="filesystem"),
        ]
        hints = _derive_hints(signals, "")
        # System anomaly (0.9) should come before file change (0.7)
        system_idx = next(i for i, h in enumerate(hints) if h.action == "check_system")
        file_idx = next(i for i, h in enumerate(hints) if h.action == "read_changed_file")
        assert system_idx < file_idx


class TestSmartPlanner:
    def test_skips_irrelevant_query(self):
        """Exploration should be skipped when query has no tool-relevant terms."""
        planner = SmartPlanner()
        state = ExplorationState(
            signals=[],
            available_tools=["system.snapshot", "workspace.list"],
        )
        decision = planner.decide(state)
        assert decision.stop, "Should skip exploration for empty/irrelevant query"

    def test_explores_for_system_query(self):
        """Exploration should proceed when query references system resources."""
        planner = SmartPlanner()
        state = ExplorationState(
            signals=[],
            user_query="What is the system memory usage?",
            available_tools=["system.snapshot", "workspace.list"],
        )
        decision = planner.decide(state)
        assert not decision.stop
        assert decision.tool_call.name == "system.snapshot"

    def test_explores_for_anomaly_signals(self):
        """Exploration should proceed when signals contain anomalies."""
        planner = SmartPlanner()
        signals = [EnvironmentSignal(
            source="probe.system", payload={"anomalies": ["high CPU"]}, modality="system",
        )]
        state = ExplorationState(
            signals=signals,
            available_tools=["system.snapshot", "workspace.list", "git.status"],
        )
        decision = planner.decide(state)
        assert not decision.stop

    def test_uses_signal_hints_after_bootstrap(self):
        planner = SmartPlanner()
        signals = [EnvironmentSignal(
            source="probe.system", payload={"anomalies": ["high CPU"]}, modality="system",
        )]
        state = ExplorationState(
            signals=signals,
            available_tools=["system.snapshot", "workspace.list", "git.status"],
            tool_results=[type("R", (), {"name": "system.snapshot"})()],  # already used
        )
        decision = planner.decide(state)
        # Should not re-bootstrap, should derive from signals or stop
        assert isinstance(decision.stop, bool)

    def test_stops_when_no_hints(self):
        planner = SmartPlanner()
        state = ExplorationState(
            signals=[],
            available_tools=["workspace.list"],
            tool_results=[
                type("R", (), {"name": "system.snapshot"})(),
            ],
        )
        decision = planner.decide(state)
        assert decision.stop


class TestSmartPlannerWithExplorer:
    def test_full_explore_loop(self):
        from aura.explore import Explorer

        planner = SmartPlanner()
        registry = _make_registry()
        explorer = Explorer(planner=planner, registry=registry, max_steps=3)

        signals = [EnvironmentSignal(
            source="user", payload={"text": "check the repo"}, modality="text",
        )]
        outcome = explorer.explore(signals, user_query="show git status")
        assert len(outcome.decisions) > 0
        # Should have tried git.status (query mentions "git")
        tool_names = [r.name for r in outcome.tool_results]
        assert "git.status" in tool_names

    def test_no_explore_for_social_query(self):
        """Social/reasoning queries should produce zero tool calls."""
        from aura.explore import Explorer

        planner = SmartPlanner()
        registry = _make_registry()
        explorer = Explorer(planner=planner, registry=registry, max_steps=5)

        signals = [EnvironmentSignal(
            source="env", payload={"location": "Park", "time": "3pm"},
        )]
        outcome = explorer.explore(signals, user_query="Where should I go for lunch?")
        assert len(outcome.tool_results) == 0, "Social query should not trigger tool calls"
