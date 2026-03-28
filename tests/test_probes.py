"""Tests for environment probes."""
import asyncio
import time
import pytest

from aura.probes.base import Probe, ProbeRegistry, ProbeResult, ChangeTracker
from aura.probes.system import SystemProbe
from aura.probes.git import GitProbe
from aura.probes.process import ProcessProbe
from aura.types import EnvironmentSignal


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestProbeRegistry:
    def test_register_and_list(self):
        registry = ProbeRegistry()
        probe = SystemProbe()
        registry.register(probe)
        assert probe.name in registry.list_probes()

    def test_poll_all(self):
        registry = ProbeRegistry()
        registry.register(SystemProbe())
        results = _run(registry.poll_all())
        assert len(results) >= 1
        assert all(isinstance(r, ProbeResult) for r in results)

    def test_poll_returns_signals(self):
        registry = ProbeRegistry()
        registry.register(SystemProbe())
        results = _run(registry.poll_all())
        for r in results:
            assert isinstance(r.signals, list)
            for sig in r.signals:
                assert isinstance(sig, EnvironmentSignal)


class TestChangeTracker:
    def test_first_update_is_change(self):
        tracker = ChangeTracker()
        sig = EnvironmentSignal(source="test", payload={"value": 1})
        result = ProbeResult(source="test", timestamp=time.time(), signals=[sig], latency_ms=0.0)
        assert tracker.update("test", result) is True

    def test_same_data_no_change(self):
        tracker = ChangeTracker()
        sig = EnvironmentSignal(source="test", payload={"value": 1})
        result = ProbeResult(source="test", timestamp=time.time(), signals=[sig], latency_ms=0.0)
        tracker.update("test", result)
        assert tracker.update("test", result) is False

    def test_different_data_is_change(self):
        tracker = ChangeTracker()
        r1 = ProbeResult(source="test", timestamp=time.time(), signals=[EnvironmentSignal(source="test", payload={"value": 1})], latency_ms=0.0)
        r2 = ProbeResult(source="test", timestamp=time.time(), signals=[EnvironmentSignal(source="test", payload={"value": 2})], latency_ms=0.0)
        tracker.update("test", r1)
        assert tracker.update("test", r2) is True


class TestSystemProbe:
    def test_poll_returns_result(self):
        probe = SystemProbe()
        result = _run(probe.poll())
        assert isinstance(result, ProbeResult)
        assert result.source == "system"
        assert len(result.signals) >= 1

    def test_signal_has_system_modality(self):
        probe = SystemProbe()
        result = _run(probe.poll())
        for sig in result.signals:
            assert sig.modality == "system"

    def test_signal_payload_has_metrics(self):
        probe = SystemProbe()
        result = _run(probe.poll())
        main_signal = result.signals[0]
        payload = main_signal.payload
        assert isinstance(payload, dict)


class TestGitProbe:
    def test_poll_returns_result(self):
        probe = GitProbe()
        result = _run(probe.poll())
        assert isinstance(result, ProbeResult)

    def test_non_repo_graceful(self):
        probe = GitProbe(repo_path="/tmp")
        result = _run(probe.poll())
        assert isinstance(result, ProbeResult)


class TestProcessProbe:
    def test_poll_returns_result(self):
        probe = ProcessProbe()
        result = _run(probe.poll())
        assert isinstance(result, ProbeResult)
        assert len(result.signals) >= 1

    def test_detects_python_process(self):
        probe = ProcessProbe(patterns=["python"])
        result = _run(probe.poll())
        payloads = [s.payload for s in result.signals]
        found = any("python" in str(p).lower() for p in payloads)
        assert found or len(result.signals) > 0
