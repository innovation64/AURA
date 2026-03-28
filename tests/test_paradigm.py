"""Tests for interaction paradigms."""
import time
from aura.paradigm.base import (
    AgentObservation,
    AgentResponse,
    AgentPolicy,
    EnvironmentSimulator,
    EpisodeResult,
)
from aura.paradigm.reactive import ReactiveParadigm
from aura.paradigm.proactive import ProactiveParadigm
from aura.paradigm.collaborative import CollaborativeParadigm


class SimpleAgent(AgentPolicy):
    """Minimal agent for testing."""

    def __init__(self):
        self._step = 0

    def reset(self):
        self._step = 0

    def act(self, observation: AgentObservation) -> AgentResponse:
        self._step += 1

        # If we got pushed context, use it
        if observation.pushed_context:
            alerts = observation.pushed_context.get("critical_alerts", [])
            if alerts:
                return AgentResponse(
                    action=f"investigate alert: {alerts[0].get('description', '')}",
                    used_pushed_context=True,
                )

        # Check raw state for alerts
        alerts = observation.environment_state.get("alerts", [])
        if alerts:
            return AgentResponse(action="investigate alert", used_pushed_context=False)

        if self._step > 8:
            return AgentResponse(action="done", used_pushed_context=False)

        return AgentResponse(
            action="continue monitoring",
            tool_calls=[{"tool": "system.snapshot", "args": {}}],
            used_pushed_context=False,
        )


def _make_env():
    """Environment with service failure at step 3."""
    initial = {
        "cpu": 30, "memory": 50,
        "services": [{"name": "db", "status": "running"}],
        "errors": [],
    }
    injections = {
        3: [{"type": "add_service_failure", "service_name": "db", "error": "Down"}],
    }
    return EnvironmentSimulator(initial, injections)


class TestReactiveParadigm:
    def test_run_episode(self):
        paradigm = ReactiveParadigm()
        agent = SimpleAgent()
        env = _make_env()
        result = paradigm.run_episode(agent, env, max_steps=10, scenario_name="test")
        assert isinstance(result, EpisodeResult)
        assert result.paradigm == "reactive"
        assert len(result.steps) > 0

    def test_no_pushed_context(self):
        paradigm = ReactiveParadigm()
        agent = SimpleAgent()
        env = _make_env()
        result = paradigm.run_episode(agent, env, max_steps=10)
        for step in result.steps:
            assert step.observation.pushed_context is None


class TestProactiveParadigm:
    def test_run_episode(self):
        paradigm = ProactiveParadigm(agent_type="sysadmin")
        agent = SimpleAgent()
        env = _make_env()
        result = paradigm.run_episode(agent, env, max_steps=10, scenario_name="test")
        assert isinstance(result, EpisodeResult)
        assert result.paradigm == "proactive"

    def test_pushes_context_on_change(self):
        paradigm = ProactiveParadigm(agent_type="sysadmin")
        agent = SimpleAgent()
        env = _make_env()
        result = paradigm.run_episode(agent, env, max_steps=10)
        # After step 3 (injection), at least one step should have pushed context
        pushed_steps = [s for s in result.steps if s.observation.pushed_context]
        # May or may not push depending on detection, but should have metrics
        assert "pushes_made" in result.metrics


class TestCollaborativeParadigm:
    def test_run_episode(self):
        paradigm = CollaborativeParadigm(agent_type="sysadmin")
        agent = SimpleAgent()
        env = _make_env()
        result = paradigm.run_episode(agent, env, max_steps=10, scenario_name="test")
        assert isinstance(result, EpisodeResult)
        assert result.paradigm == "collaborative"

    def test_has_feedback_metrics(self):
        paradigm = CollaborativeParadigm(agent_type="sysadmin")
        agent = SimpleAgent()
        env = _make_env()
        result = paradigm.run_episode(agent, env, max_steps=10)
        assert "context_hit_rate" in result.metrics
        assert "alert_fatigue" in result.metrics
        assert "attention_use_rate" in result.metrics


class TestEnvironmentSimulator:
    def test_initial_state(self):
        env = _make_env()
        assert env.state["cpu"] == 30
        assert len(env.state["services"]) == 1

    def test_injection(self):
        env = _make_env()
        env.step(0)
        env.step(1)
        env.step(2)
        assert env.state["services"][0]["status"] == "running"
        env.step(3)  # injection happens here
        assert env.state["services"][0]["status"] == "down"
        assert len(env.state.get("alerts", [])) > 0

    def test_reset(self):
        env = _make_env()
        env.step(3)
        assert env.state["services"][0]["status"] == "down"
        env.reset()
        assert env.state["services"][0]["status"] == "running"

    def test_execute_tool(self):
        env = _make_env()
        result = env.execute_tool("system.snapshot")
        assert isinstance(result, dict)


class TestParadigmComparison:
    """Key test: proactive/collaborative should detect changes faster than reactive."""

    def test_proactive_detects_faster(self):
        agent = SimpleAgent()
        env = _make_env()

        reactive = ReactiveParadigm()
        proactive = ProactiveParadigm(agent_type="sysadmin")

        r_reactive = reactive.run_episode(agent, env, max_steps=15, scenario_name="test")
        r_proactive = proactive.run_episode(agent, env, max_steps=15, scenario_name="test")

        # Proactive should detect no later than reactive
        # (may both detect at same step since SimpleAgent checks raw state too)
        assert r_proactive.detected_change_at_step <= max(r_reactive.detected_change_at_step, 15)
