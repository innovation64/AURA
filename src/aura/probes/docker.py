"""Docker probe -- monitors container lifecycle and resource usage."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Set

from aura.types import EnvironmentSignal

from .base import Probe, ProbeResult


async def _run_cmd(*args: str, timeout: float = 15.0) -> Optional[str]:
    """Run a subprocess and return stdout, or None on failure."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        if proc.returncode != 0:
            return None
        return stdout.decode(errors="replace").strip()
    except (FileNotFoundError, asyncio.TimeoutError, OSError):
        return None


def _parse_docker_ps(raw: str) -> List[Dict[str, str]]:
    """Parse `docker ps --format` output into container dicts."""
    containers: List[Dict[str, str]] = []
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        containers.append(
            {
                "id": parts[0],
                "name": parts[1],
                "image": parts[2],
                "status": parts[3],
                "health": parts[4] if len(parts) > 4 else "",
            }
        )
    return containers


def _parse_docker_stats(raw: str) -> Dict[str, Dict[str, str]]:
    """Parse `docker stats --no-stream --format` output into per-container resource dicts."""
    stats: Dict[str, Dict[str, str]] = {}
    for line in raw.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        stats[parts[0]] = {"cpu_percent": parts[1], "mem_percent": parts[2]}
    return stats


class DockerProbe(Probe):
    """Monitors Docker container state and resource usage."""

    def __init__(self) -> None:
        super().__init__()
        self._prev_containers: Dict[str, Dict[str, str]] = {}
        self._docker_available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "docker"

    @property
    def interval_seconds(self) -> float:
        return 20.0

    async def _check_docker(self) -> bool:
        result = await _run_cmd("docker", "info", timeout=5.0)
        return result is not None

    async def _list_containers(self) -> Optional[List[Dict[str, str]]]:
        raw = await _run_cmd(
            "docker",
            "ps",
            "-a",
            "--format",
            "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Label \"health\"}}",
        )
        if raw is None:
            return None
        return _parse_docker_ps(raw)

    async def _container_stats(self) -> Dict[str, Dict[str, str]]:
        raw = await _run_cmd(
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "{{.Name}}\t{{.CPUPerc}}\t{{.MemPerc}}",
            timeout=15.0,
        )
        if raw is None:
            return {}
        return _parse_docker_stats(raw)

    async def poll(self) -> ProbeResult:
        t0 = time.time()
        signals: List[EnvironmentSignal] = []

        try:
            # Lazy availability check.
            if self._docker_available is None:
                self._docker_available = await self._check_docker()
            if not self._docker_available:
                return ProbeResult(
                    source=self.name,
                    timestamp=t0,
                    signals=[],
                    latency_ms=round((time.time() - t0) * 1000, 2),
                    metadata={"docker_available": False},
                )

            containers_list, stats = await asyncio.gather(
                self._list_containers(),
                self._container_stats(),
            )

            if containers_list is None:
                # Docker became unavailable mid-run.
                self._docker_available = None
                return ProbeResult(
                    source=self.name,
                    timestamp=t0,
                    signals=[],
                    latency_ms=round((time.time() - t0) * 1000, 2),
                    metadata={"error": "docker ps failed"},
                )

            current: Dict[str, Dict[str, str]] = {}
            for c in containers_list:
                current[c["name"]] = c

            prev_names: Set[str] = set(self._prev_containers.keys())
            curr_names: Set[str] = set(current.keys())

            # Detect new containers.
            for cname in sorted(curr_names - prev_names):
                c = current[cname]
                signals.append(
                    EnvironmentSignal(
                        source="probe.docker",
                        modality="docker",
                        payload={
                            "event": "container_started",
                            "name": cname,
                            "image": c["image"],
                            "status": c["status"],
                        },
                        confidence=1.0,
                    )
                )

            # Detect removed containers.
            for cname in sorted(prev_names - curr_names):
                c = self._prev_containers[cname]
                signals.append(
                    EnvironmentSignal(
                        source="probe.docker",
                        modality="docker",
                        payload={
                            "event": "container_stopped",
                            "name": cname,
                            "image": c.get("image", ""),
                        },
                        confidence=1.0,
                    )
                )

            # Detect status/health changes for existing containers.
            for cname in sorted(prev_names & curr_names):
                prev_c = self._prev_containers[cname]
                curr_c = current[cname]
                if prev_c.get("status") != curr_c.get("status"):
                    signals.append(
                        EnvironmentSignal(
                            source="probe.docker",
                            modality="docker",
                            payload={
                                "event": "container_status_change",
                                "name": cname,
                                "from_status": prev_c.get("status"),
                                "to_status": curr_c.get("status"),
                            },
                            confidence=1.0,
                        )
                    )
                if prev_c.get("health") != curr_c.get("health") and curr_c.get("health"):
                    signals.append(
                        EnvironmentSignal(
                            source="probe.docker",
                            modality="docker",
                            payload={
                                "event": "container_health_change",
                                "name": cname,
                                "from_health": prev_c.get("health"),
                                "to_health": curr_c.get("health"),
                            },
                            confidence=1.0,
                        )
                    )

            # Attach resource usage summary (not a change signal, but useful metadata).
            if stats:
                resource_payload: Dict[str, Any] = {}
                for cname, s in stats.items():
                    resource_payload[cname] = {
                        "cpu_percent": s["cpu_percent"],
                        "mem_percent": s["mem_percent"],
                    }
                signals.append(
                    EnvironmentSignal(
                        source="probe.docker",
                        modality="docker",
                        payload={"event": "resource_snapshot", "containers": resource_payload},
                        confidence=1.0,
                    )
                )

            self._prev_containers = current

        except Exception as exc:
            signals.append(
                EnvironmentSignal(
                    source="probe.docker",
                    modality="docker",
                    payload={"error": str(exc)},
                    confidence=0.0,
                )
            )

        latency = (time.time() - t0) * 1000
        return ProbeResult(
            source=self.name,
            timestamp=t0,
            signals=signals,
            latency_ms=round(latency, 2),
        )
