"""Tests for agent-type views."""
from aura.views.base import ViewConfig, ViewRegistry
from aura.views.coder import CoderView
from aura.views.sysadmin import SysadminView
from aura.views.researcher import ResearcherView
from aura.types import EnvironmentSignal


def _signals():
    return [
        EnvironmentSignal(source="probe.system", payload={"cpu_pct": 45, "mem_pct": 60}, modality="system"),
        EnvironmentSignal(source="probe.filesystem", payload={"changes": [{"type": "modified", "path": "main.py"}]}, modality="filesystem"),
        EnvironmentSignal(source="probe.git", payload={"branch": "main", "uncommitted": 3}, modality="git"),
        EnvironmentSignal(source="probe.docker", payload={"containers": [{"name": "web", "status": "running"}]}, modality="docker"),
        EnvironmentSignal(source="probe.network", payload={"name": "api", "status": "up"}, modality="network"),
    ]


class TestCoderView:
    def test_filter_prioritizes_code(self):
        view = CoderView()
        config = ViewConfig(agent_type="coder")
        filtered = view.filter_signals(_signals(), config)
        modalities = [s.modality for s in filtered]
        # Should include filesystem and git, may filter system
        assert "filesystem" in modalities or "git" in modalities

    def test_render(self):
        view = CoderView()
        config = ViewConfig(agent_type="coder")
        result = view.render(_signals(), config)
        assert isinstance(result, dict)

    def test_summarize(self):
        view = CoderView()
        summary = view.summarize(_signals())
        assert isinstance(summary, str)
        assert len(summary) > 0


class TestSysadminView:
    def test_filter_prioritizes_system(self):
        view = SysadminView()
        config = ViewConfig(agent_type="sysadmin")
        filtered = view.filter_signals(_signals(), config)
        modalities = [s.modality for s in filtered]
        assert "system" in modalities or "docker" in modalities

    def test_render(self):
        view = SysadminView()
        config = ViewConfig(agent_type="sysadmin")
        result = view.render(_signals(), config)
        assert isinstance(result, dict)


class TestResearcherView:
    def test_render(self):
        view = ResearcherView()
        config = ViewConfig(agent_type="researcher")
        result = view.render(_signals(), config)
        assert isinstance(result, dict)


class TestViewRegistry:
    def test_register_and_get(self):
        registry = ViewRegistry()
        registry.register(CoderView())
        registry.register(SysadminView())
        assert registry.get_view("coder") is not None
        assert registry.get_view("sysadmin") is not None
        assert registry.get_view("unknown") is None

    def test_render_for_agent(self):
        registry = ViewRegistry()
        registry.register(CoderView())
        config = ViewConfig(agent_type="coder")
        result = registry.render_for_agent(_signals(), "coder", config)
        assert isinstance(result, dict)
