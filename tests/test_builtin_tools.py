"""Tests for builtin tools."""
import os
import tempfile

from aura.builtin_tools import default_tools


def _get_tool(name: str):
    for t in default_tools():
        if t.name == name:
            return t
    raise KeyError(name)


class TestSystemSnapshot:
    def test_returns_dict(self):
        tool = _get_tool("system.snapshot")
        result = tool.execute()
        assert isinstance(result, dict)
        assert "time_utc" in result
        assert "os" in result
        assert "cwd" in result

    def test_has_memory_info(self):
        tool = _get_tool("system.snapshot")
        result = tool.execute()
        if os.path.exists("/proc/meminfo"):
            assert "memory" in result
            assert "used_pct" in result["memory"]


class TestWorkspaceList:
    def test_list_cwd(self):
        tool = _get_tool("workspace.list")
        result = tool.execute(path=".")
        assert "entries" in result
        assert "total" in result
        assert result["total"] >= 0

    def test_list_nonexistent(self):
        tool = _get_tool("workspace.list")
        result = tool.execute(path="/nonexistent_path_12345")
        assert "error" in result


class TestWorkspaceRead:
    def test_read_file(self):
        tool = _get_tool("workspace.read")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("hello\nworld\n")
            f.flush()
            result = tool.execute(path=f.name)
        os.unlink(f.name)
        assert result["content"] == "hello\nworld\n"
        assert result["total_lines"] == 2

    def test_read_nonexistent(self):
        tool = _get_tool("workspace.read")
        result = tool.execute(path="/nonexistent_file_12345")
        assert "error" in result


class TestGitStatus:
    def test_non_repo(self):
        tool = _get_tool("git.status")
        result = tool.execute(repo_path="/tmp")
        assert result.get("is_repo") is False or "error" in result

    def test_returns_dict(self):
        tool = _get_tool("git.status")
        result = tool.execute()
        assert isinstance(result, dict)


class TestDockerStatus:
    def test_returns_dict(self):
        tool = _get_tool("docker.status")
        result = tool.execute()
        assert isinstance(result, dict)
        # Should either have containers or an error
        assert "containers" in result or "error" in result


class TestProcessList:
    def test_list_all(self):
        tool = _get_tool("process.list")
        result = tool.execute()
        assert "processes" in result
        assert len(result["processes"]) > 0

    def test_filter_by_name(self):
        tool = _get_tool("process.list")
        result = tool.execute(pattern="python")
        assert "processes" in result
        # Current process should match
        assert any("python" in p["name"].lower() for p in result["processes"])


class TestServiceCheck:
    def test_check_nonexistent(self):
        tool = _get_tool("service.check")
        result = tool.execute(url="http://127.0.0.1:19999", timeout=1.0)
        assert result["status"] == "down"
        assert "latency_ms" in result
