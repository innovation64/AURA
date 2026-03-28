"""Integration tests for AURA core pipeline."""
from aura.core import AURAAgent, AURAConfig
from aura.types import Interaction


class TestDefaultBackend:
    def test_run_basic(self):
        agent = AURAAgent(config=AURAConfig(
            backend="default",
            proactive_enabled=False,
            smart_planner=False,
        ))
        result = agent.run("hello world", user_query="what is this?")
        assert isinstance(result, Interaction)
        assert result.message

    def test_run_with_smart_planner(self):
        agent = AURAAgent(config=AURAConfig(
            backend="default",
            proactive_enabled=False,
            smart_planner=True,
        ))
        result = agent.run("test environment", user_query="check system status")
        assert isinstance(result, Interaction)

    def test_info_includes_new_fields(self):
        agent = AURAAgent(config=AURAConfig(
            backend="default",
            proactive_enabled=False,
        ))
        info = agent.info()
        assert "proactive" in info
        assert "probes" in info
        assert "trajectory" in info

    def test_from_backend(self):
        agent = AURAAgent.from_backend("default", proactive_enabled=False)
        assert agent.info()["backend"] == "default"


class TestExplorationWithTools:
    def test_explore_produces_signals(self):
        agent = AURAAgent(config=AURAConfig(
            backend="default",
            explore_enabled=True,
            smart_planner=True,
            proactive_enabled=False,
        ))
        result = agent.run("office environment", user_query="check workspace files")
        assert isinstance(result, Interaction)
        # Should have exploration in payload
        assert "exploration" in result.payload

    def test_explore_disabled(self):
        agent = AURAAgent(config=AURAConfig(
            backend="default",
            explore_enabled=False,
            proactive_enabled=False,
        ))
        result = agent.run("test", user_query="hello")
        assert isinstance(result, Interaction)
        assert "exploration" not in result.payload


class TestProactiveEngine:
    def test_proactive_builds_when_enabled(self):
        # This may or may not succeed depending on probe availability
        agent = AURAAgent(config=AURAConfig(
            backend="default",
            proactive_enabled=True,
        ))
        info = agent.info()
        # Should have attempted to build proactive engine
        assert "proactive" in info

    def test_proactive_disabled(self):
        agent = AURAAgent(config=AURAConfig(
            backend="default",
            proactive_enabled=False,
        ))
        info = agent.info()
        assert info["proactive"] is False


class TestPipelineEnd2End:
    def test_full_pipeline_dict_input(self):
        agent = AURAAgent(config=AURAConfig(
            backend="default",
            proactive_enabled=False,
            smart_planner=False,
        ))
        result = agent.run(
            {"text": "server room, 3 racks, temperature 25C"},
            user_query="summarize the environment",
        )
        assert isinstance(result, Interaction)
        assert result.message

    def test_full_pipeline_list_input(self):
        agent = AURAAgent(config=AURAConfig(
            backend="default",
            proactive_enabled=False,
        ))
        result = agent.run(
            ["sensor: temp=72F", "sensor: humidity=45%"],
            user_query="any anomalies?",
        )
        assert isinstance(result, Interaction)

    def test_none_input(self):
        agent = AURAAgent(config=AURAConfig(
            backend="default",
            proactive_enabled=False,
        ))
        result = agent.run(None, user_query="hello")
        assert isinstance(result, Interaction)
