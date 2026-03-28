"""Process probe -- monitors key processes via /proc filesystem."""

from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any, Dict, List, Optional, Set

from aura.types import EnvironmentSignal

from .base import Probe, ProbeResult

# Default process name patterns to monitor.
DEFAULT_PROCESS_PATTERNS: List[str] = [
    "python",
    "node",
    "java",
    "docker",
    "ollama",
    "vllm",
]

# Clock ticks per second (usually 100).
_CLK_TCK: int = os.sysconf("SC_CLK_TCK")


def _read_file(path: str) -> Optional[str]:
    """Read a small /proc file, returning None on error."""
    try:
        with open(path, "r") as f:
            return f.read()
    except (PermissionError, OSError, FileNotFoundError):
        return None


def _get_total_cpu_time() -> float:
    """Return total CPU time in seconds from /proc/stat."""
    content = _read_file("/proc/stat")
    if content is None:
        return 0.0
    first_line = content.splitlines()[0]
    parts = first_line.split()
    total = sum(int(v) for v in parts[1:])
    return total / _CLK_TCK


def _parse_proc_stat_pid(pid: int) -> Optional[Dict[str, Any]]:
    """Parse /proc/<pid>/stat for a single process."""
    content = _read_file(f"/proc/{pid}/stat")
    if content is None:
        return None
    # The comm field is wrapped in parens and may contain spaces / parens.
    match = re.match(r"(\d+)\s+\((.+)\)\s+(.+)", content)
    if match is None:
        return None
    fields = match.group(3).split()
    # field indices (0-based after the first 3 groups):
    # 0=state, 11=utime, 12=stime, 19=num_threads, 20=starttime, ...
    if len(fields) < 21:
        return None
    utime = int(fields[11])
    stime = int(fields[12])
    return {
        "pid": pid,
        "comm": match.group(2),
        "state": fields[0],
        "utime": utime,
        "stime": stime,
        "total_ticks": utime + stime,
    }


def _get_cmdline(pid: int) -> str:
    """Read the command line for a process."""
    content = _read_file(f"/proc/{pid}/cmdline")
    if content is None:
        return ""
    return content.replace("\x00", " ").strip()


def _get_mem_rss_kb(pid: int) -> int:
    """Read VmRSS from /proc/<pid>/status."""
    content = _read_file(f"/proc/{pid}/status")
    if content is None:
        return 0
    for line in content.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1])
    return 0


def _get_total_mem_kb() -> int:
    """Read total memory from /proc/meminfo."""
    content = _read_file("/proc/meminfo")
    if content is None:
        return 1
    for line in content.splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1])
    return 1


def _list_pids() -> List[int]:
    """List all numeric PID directories in /proc."""
    pids: List[int] = []
    try:
        for name in os.listdir("/proc"):
            if name.isdigit():
                pids.append(int(name))
    except OSError:
        pass
    return pids


def _matches_patterns(name: str, cmdline: str, patterns: List[str]) -> bool:
    """Check if a process name or cmdline matches any of the configured patterns."""
    lower_name = name.lower()
    lower_cmd = cmdline.lower()
    for pat in patterns:
        pat_lower = pat.lower()
        if pat_lower in lower_name or pat_lower in lower_cmd:
            return True
    return False


class ProcessProbe(Probe):
    """Monitors key processes using /proc filesystem (no psutil dependency)."""

    def __init__(self, patterns: Optional[List[str]] = None) -> None:
        super().__init__()
        self._patterns = patterns or list(DEFAULT_PROCESS_PATTERNS)
        # Tracked state: pid -> {comm, total_ticks, cpu_time_snapshot}
        self._prev_processes: Dict[int, Dict[str, Any]] = {}
        self._prev_total_cpu: float = 0.0
        self._total_mem_kb: int = 0

    @property
    def name(self) -> str:
        return "process"

    @property
    def interval_seconds(self) -> float:
        return 15.0

    def _scan_processes(self) -> Dict[int, Dict[str, Any]]:
        """Scan /proc for matching processes and gather stats."""
        if self._total_mem_kb == 0:
            self._total_mem_kb = _get_total_mem_kb()

        matched: Dict[int, Dict[str, Any]] = {}
        for pid in _list_pids():
            stat = _parse_proc_stat_pid(pid)
            if stat is None:
                continue
            cmdline = _get_cmdline(pid)
            if not _matches_patterns(stat["comm"], cmdline, self._patterns):
                continue
            rss_kb = _get_mem_rss_kb(pid)
            mem_pct = round((rss_kb / self._total_mem_kb) * 100, 2) if self._total_mem_kb > 0 else 0.0
            matched[pid] = {
                "pid": pid,
                "name": stat["comm"],
                "cmdline": cmdline[:256],  # Truncate long command lines.
                "total_ticks": stat["total_ticks"],
                "mem_rss_kb": rss_kb,
                "mem_percent": mem_pct,
            }
        return matched

    async def poll(self) -> ProbeResult:
        t0 = time.time()
        signals: List[EnvironmentSignal] = []

        try:
            current_total_cpu = await asyncio.to_thread(_get_total_cpu_time)
            current_procs = await asyncio.to_thread(self._scan_processes)

            cpu_delta = current_total_cpu - self._prev_total_cpu if self._prev_total_cpu > 0 else 0.0

            prev_pids: Set[int] = set(self._prev_processes.keys())
            curr_pids: Set[int] = set(current_procs.keys())

            # Detect process starts.
            for pid in sorted(curr_pids - prev_pids):
                p = current_procs[pid]
                signals.append(
                    EnvironmentSignal(
                        source="probe.process",
                        modality="process",
                        payload={
                            "event": "process_started",
                            "pid": pid,
                            "name": p["name"],
                            "cmdline": p["cmdline"],
                        },
                        confidence=1.0,
                    )
                )

            # Detect process stops.
            for pid in sorted(prev_pids - curr_pids):
                p = self._prev_processes[pid]
                signals.append(
                    EnvironmentSignal(
                        source="probe.process",
                        modality="process",
                        payload={
                            "event": "process_stopped",
                            "pid": pid,
                            "name": p["name"],
                            "cmdline": p["cmdline"],
                        },
                        confidence=1.0,
                    )
                )

            # Compute CPU% and check for anomalies on existing processes.
            for pid in sorted(curr_pids & prev_pids):
                curr_p = current_procs[pid]
                prev_p = self._prev_processes[pid]
                tick_delta = curr_p["total_ticks"] - prev_p["total_ticks"]
                cpu_pct = 0.0
                if cpu_delta > 0:
                    cpu_pct = round((tick_delta / (_CLK_TCK * cpu_delta / _CLK_TCK)) * 100, 2)
                    # Normalise: tick_delta is in ticks, cpu_delta is in seconds.
                    cpu_pct = round((tick_delta / _CLK_TCK) / (current_total_cpu - self._prev_total_cpu) * _CLK_TCK * 100, 2) if cpu_delta > 0 else 0.0

                curr_p["cpu_percent"] = cpu_pct

                # High CPU or memory alerts.
                if cpu_pct > 80:
                    signals.append(
                        EnvironmentSignal(
                            source="probe.process",
                            modality="process",
                            payload={
                                "event": "high_cpu",
                                "pid": pid,
                                "name": curr_p["name"],
                                "cpu_percent": cpu_pct,
                                "cmdline": curr_p["cmdline"],
                            },
                            confidence=1.0,
                        )
                    )
                if curr_p["mem_percent"] > 50:
                    signals.append(
                        EnvironmentSignal(
                            source="probe.process",
                            modality="process",
                            payload={
                                "event": "high_memory",
                                "pid": pid,
                                "name": curr_p["name"],
                                "mem_percent": curr_p["mem_percent"],
                                "mem_rss_kb": curr_p["mem_rss_kb"],
                                "cmdline": curr_p["cmdline"],
                            },
                            confidence=1.0,
                        )
                    )

            self._prev_processes = current_procs
            self._prev_total_cpu = current_total_cpu

        except Exception as exc:
            signals.append(
                EnvironmentSignal(
                    source="probe.process",
                    modality="process",
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
            metadata={"patterns": self._patterns},
        )
