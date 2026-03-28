"""System resource probe -- CPU, memory, disk, GPU, load average, uptime."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

from aura.types import EnvironmentSignal

from .base import Probe, ProbeResult


def _parse_proc_stat() -> Dict[str, Any]:
    """Read /proc/stat and return per-core + overall CPU times."""
    cpus: Dict[str, List[int]] = {}
    with open("/proc/stat", "r") as f:
        for line in f:
            if line.startswith("cpu"):
                parts = line.split()
                name = parts[0]
                values = [int(v) for v in parts[1:]]
                cpus[name] = values
    return cpus


def _cpu_usage(prev: Dict[str, List[int]], curr: Dict[str, List[int]]) -> Dict[str, float]:
    """Compute CPU usage percentages from two snapshots of /proc/stat."""
    usage: Dict[str, float] = {}
    for name in curr:
        if name not in prev:
            continue
        p, c = prev[name], curr[name]
        # Fields: user, nice, system, idle, iowait, irq, softirq, steal, ...
        prev_idle = p[3] + (p[4] if len(p) > 4 else 0)
        curr_idle = c[3] + (c[4] if len(c) > 4 else 0)
        prev_total = sum(p)
        curr_total = sum(c)
        diff_total = curr_total - prev_total
        diff_idle = curr_idle - prev_idle
        if diff_total == 0:
            usage[name] = 0.0
        else:
            usage[name] = round((1.0 - diff_idle / diff_total) * 100, 2)
    return usage


def _parse_meminfo() -> Dict[str, int]:
    """Parse /proc/meminfo into a dict of kB values."""
    info: Dict[str, int] = {}
    with open("/proc/meminfo", "r") as f:
        for line in f:
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip()
                val = parts[1].strip().split()[0]
                info[key] = int(val)
    return info


def _disk_usage(path: str = "/") -> Dict[str, Any]:
    """Return disk usage stats via os.statvfs."""
    st = os.statvfs(path)
    total = st.f_frsize * st.f_blocks
    free = st.f_frsize * st.f_bavail
    used = total - free
    pct = round((used / total) * 100, 2) if total > 0 else 0.0
    return {
        "total_bytes": total,
        "used_bytes": used,
        "free_bytes": free,
        "usage_percent": pct,
    }


async def _gpu_info() -> Optional[List[Dict[str, Any]]]:
    """Query nvidia-smi for GPU stats. Returns None if unavailable."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        if proc.returncode != 0:
            return None
        gpus: List[Dict[str, Any]] = []
        for line in stdout.decode().strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "temperature_c": int(parts[2]),
                    "utilization_percent": int(parts[3]),
                    "memory_used_mb": int(parts[4]),
                    "memory_total_mb": int(parts[5]),
                }
            )
        return gpus
    except (FileNotFoundError, asyncio.TimeoutError, Exception):
        return None


def _uptime() -> float:
    """Return system uptime in seconds from /proc/uptime."""
    with open("/proc/uptime", "r") as f:
        return float(f.read().split()[0])


class SystemProbe(Probe):
    """Polls system resources: CPU, memory, disk, GPU, load average, uptime."""

    def __init__(self) -> None:
        super().__init__()
        self._prev_stat: Optional[Dict[str, List[int]]] = None

    @property
    def name(self) -> str:
        return "system"

    @property
    def interval_seconds(self) -> float:
        return 15.0

    async def poll(self) -> ProbeResult:
        t0 = time.time()
        signals: List[EnvironmentSignal] = []
        anomalies: List[EnvironmentSignal] = []

        try:
            # --- CPU ---
            curr_stat = await asyncio.to_thread(_parse_proc_stat)
            cpu_usage: Dict[str, float] = {}
            if self._prev_stat is not None:
                cpu_usage = _cpu_usage(self._prev_stat, curr_stat)
            self._prev_stat = curr_stat

            # --- Memory ---
            mem = await asyncio.to_thread(_parse_meminfo)
            mem_total = mem.get("MemTotal", 1)
            mem_available = mem.get("MemAvailable", mem.get("MemFree", 0))
            mem_used = mem_total - mem_available
            mem_pct = round((mem_used / mem_total) * 100, 2) if mem_total > 0 else 0.0

            # --- Disk ---
            disk = await asyncio.to_thread(_disk_usage, "/")

            # --- GPU ---
            gpus = await _gpu_info()

            # --- Load average & uptime ---
            load1, load5, load15 = os.getloadavg()
            uptime_s = await asyncio.to_thread(_uptime)

            payload: Dict[str, Any] = {
                "cpu": cpu_usage,
                "memory": {
                    "total_kb": mem_total,
                    "used_kb": mem_used,
                    "available_kb": mem_available,
                    "usage_percent": mem_pct,
                },
                "disk": disk,
                "load_average": {"1m": load1, "5m": load5, "15m": load15},
                "uptime_seconds": uptime_s,
            }
            if gpus is not None:
                payload["gpu"] = gpus

            signals.append(
                EnvironmentSignal(
                    source="probe.system",
                    modality="system",
                    payload=payload,
                    confidence=1.0,
                )
            )

            # --- Anomaly detection ---
            overall_cpu = cpu_usage.get("cpu", 0.0)
            if overall_cpu > 90:
                anomalies.append(
                    EnvironmentSignal(
                        source="probe.system",
                        modality="system",
                        payload={"anomaly": "high_cpu", "value": overall_cpu, "threshold": 90, "severity": "warning"},
                        confidence=1.0,
                    )
                )

            if mem_pct > 85:
                anomalies.append(
                    EnvironmentSignal(
                        source="probe.system",
                        modality="system",
                        payload={"anomaly": "high_memory", "value": mem_pct, "threshold": 85, "severity": "warning"},
                        confidence=1.0,
                    )
                )

            if disk["usage_percent"] > 90:
                anomalies.append(
                    EnvironmentSignal(
                        source="probe.system",
                        modality="system",
                        payload={
                            "anomaly": "high_disk",
                            "value": disk["usage_percent"],
                            "threshold": 90,
                            "severity": "critical",
                        },
                        confidence=1.0,
                    )
                )

            if gpus:
                for gpu in gpus:
                    if gpu["temperature_c"] > 80:
                        anomalies.append(
                            EnvironmentSignal(
                                source="probe.system",
                                modality="system",
                                payload={
                                    "anomaly": "high_gpu_temp",
                                    "gpu_index": gpu["index"],
                                    "value": gpu["temperature_c"],
                                    "threshold": 80,
                                    "severity": "warning",
                                },
                                confidence=1.0,
                            )
                        )

        except Exception as exc:
            signals.append(
                EnvironmentSignal(
                    source="probe.system",
                    modality="system",
                    payload={"error": str(exc)},
                    confidence=0.0,
                )
            )

        latency = (time.time() - t0) * 1000
        return ProbeResult(
            source=self.name,
            timestamp=t0,
            signals=signals + anomalies,
            latency_ms=round(latency, 2),
        )
