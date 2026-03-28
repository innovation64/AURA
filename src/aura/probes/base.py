"""Base probe infrastructure for AURA environment sensing."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aura.types import EnvironmentSignal


@dataclass
class ProbeResult:
    """Result from a single probe poll cycle."""

    source: str
    timestamp: float
    signals: List[EnvironmentSignal]
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class Probe(ABC):
    """Abstract base class for all environment probes."""

    enabled: bool = True

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this probe."""
        ...

    @property
    def interval_seconds(self) -> float:
        """Polling interval in seconds. Override to customise."""
        return 30.0

    @abstractmethod
    async def poll(self) -> ProbeResult:
        """Execute one sensing cycle and return results."""
        ...

    async def setup(self) -> None:
        """Lifecycle hook called when the probe is registered."""

    async def teardown(self) -> None:
        """Lifecycle hook called when the probe is unregistered."""


class ChangeTracker:
    """Tracks previous probe states to detect changes via payload hashing."""

    def __init__(self, sensitivity_threshold: float = 0.0) -> None:
        self._hashes: Dict[str, str] = {}
        self.sensitivity_threshold = sensitivity_threshold

    @staticmethod
    def _hash_signals(signals: List[EnvironmentSignal]) -> str:
        """Produce a deterministic hash from a list of signals."""
        payloads = []
        for sig in signals:
            payloads.append(json.dumps(sig.payload, sort_keys=True, default=str))
        combined = "|".join(payloads)
        return hashlib.sha256(combined.encode()).hexdigest()

    def update(self, probe_name: str, result: ProbeResult) -> bool:
        """Update tracker state. Returns True if the result differs from the previous one."""
        new_hash = self._hash_signals(result.signals)
        previous = self._hashes.get(probe_name)
        self._hashes[probe_name] = new_hash
        if previous is None:
            # First poll is always considered a change.
            return True
        return new_hash != previous


class ProbeRegistry:
    """Central registry that manages probe lifecycle and orchestrates polling."""

    def __init__(self) -> None:
        self._probes: Dict[str, Probe] = {}
        self._change_tracker = ChangeTracker()

    def register(self, probe: Probe) -> None:
        """Register a probe instance."""
        self._probes[probe.name] = probe

    def unregister(self, name: str) -> None:
        """Remove a probe by name."""
        self._probes.pop(name, None)

    def get_probe(self, name: str) -> Optional[Probe]:
        """Look up a probe by name."""
        return self._probes.get(name)

    def list_probes(self) -> List[str]:
        """Return names of all registered probes."""
        return list(self._probes.keys())

    async def poll_all(self) -> List[ProbeResult]:
        """Poll every enabled probe in parallel and return all results."""
        enabled = [p for p in self._probes.values() if p.enabled]
        if not enabled:
            return []

        async def _safe_poll(probe: Probe) -> Optional[ProbeResult]:
            try:
                return await probe.poll()
            except Exception:
                return ProbeResult(
                    source=probe.name,
                    timestamp=time.time(),
                    signals=[],
                    latency_ms=0.0,
                    metadata={"error": "poll failed"},
                )

        results = await asyncio.gather(*[_safe_poll(p) for p in enabled])
        return [r for r in results if r is not None]

    async def poll_changed(self) -> List[ProbeResult]:
        """Poll all probes but only return those whose output changed."""
        all_results = await self.poll_all()
        changed: List[ProbeResult] = []
        for result in all_results:
            if self._change_tracker.update(result.source, result):
                changed.append(result)
        return changed
