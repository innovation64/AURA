from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .tools import Tool


# ---------------------------------------------------------------------------
# Original tools
# ---------------------------------------------------------------------------

def _system_snapshot() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.system(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
    }
    # Load average
    try:
        info["load_avg"] = list(os.getloadavg())
    except OSError:
        pass
    # Memory
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        mem = {}
        for line in lines:
            parts = line.split()
            if parts[0].rstrip(":") in ("MemTotal", "MemAvailable", "MemFree"):
                mem[parts[0].rstrip(":")] = int(parts[1])
        if "MemTotal" in mem and "MemAvailable" in mem:
            total = mem["MemTotal"]
            avail = mem["MemAvailable"]
            info["memory"] = {
                "total_mb": total // 1024,
                "available_mb": avail // 1024,
                "used_pct": round((1 - avail / total) * 100, 1) if total else 0,
            }
    except Exception:
        pass
    # GPU (nvidia-smi)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=5, text=True,
        )
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "name": parts[0],
                    "temp_c": int(parts[1]),
                    "utilization_pct": int(parts[2]),
                    "memory_used_mb": int(parts[3]),
                    "memory_total_mb": int(parts[4]),
                })
        if gpus:
            info["gpus"] = gpus
    except Exception:
        pass
    return info


def _workspace_list(path: str = ".", limit: int = 50) -> Dict[str, Any]:
    entries: List[str] = []
    total = 0
    try:
        entries = sorted(os.listdir(path))
        total = len(entries)
        entries = entries[: max(limit, 0)]
    except FileNotFoundError:
        return {"path": os.path.abspath(path), "entries": [], "total": 0, "error": "path not found"}

    return {
        "path": os.path.abspath(path),
        "entries": entries,
        "total": total,
    }


# ---------------------------------------------------------------------------
# New environment tools
# ---------------------------------------------------------------------------

def _workspace_read(path: str, max_lines: int = 200) -> Dict[str, Any]:
    """Read a file from the workspace."""
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        return {"path": abs_path, "error": "not a file or does not exist"}
    try:
        with open(abs_path) as f:
            lines = f.readlines()
        total = len(lines)
        content = "".join(lines[:max_lines])
        return {"path": abs_path, "content": content, "total_lines": total, "truncated": total > max_lines}
    except Exception as e:
        return {"path": abs_path, "error": str(e)}


def _git_status(repo_path: str = ".") -> Dict[str, Any]:
    """Get git repository status."""
    def _run(cmd: List[str]) -> Optional[str]:
        try:
            return subprocess.check_output(cmd, cwd=repo_path, timeout=10, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    # Check if in a git repo
    toplevel = _run(["git", "rev-parse", "--show-toplevel"])
    if not toplevel:
        return {"is_repo": False, "error": "not a git repository"}

    branch = _run(["git", "branch", "--show-current"]) or "detached"
    status_raw = _run(["git", "status", "--porcelain"]) or ""
    changed_files = [line[3:] for line in status_raw.splitlines() if line.strip()]

    # Ahead/behind
    upstream = _run(["git", "rev-parse", "--abbrev-ref", "@{upstream}"])
    ahead, behind = 0, 0
    if upstream:
        counts = _run(["git", "rev-list", "--left-right", "--count", f"HEAD...{upstream}"])
        if counts:
            parts = counts.split()
            ahead, behind = int(parts[0]), int(parts[1])

    # Recent commits
    log_raw = _run(["git", "log", "--oneline", "-5"]) or ""
    recent_commits = log_raw.splitlines()

    return {
        "is_repo": True,
        "root": toplevel,
        "branch": branch,
        "changed_files": changed_files,
        "uncommitted_count": len(changed_files),
        "ahead": ahead,
        "behind": behind,
        "recent_commits": recent_commits,
    }


def _docker_status() -> Dict[str, Any]:
    """List running docker containers with resource usage."""
    try:
        out = subprocess.check_output(
            ["docker", "ps", "--format", "{{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Ports}}"],
            timeout=10, text=True,
        )
    except Exception as e:
        return {"available": False, "error": str(e)}

    containers = []
    for line in out.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 4:
            containers.append({
                "id": parts[0][:12],
                "name": parts[1],
                "status": parts[2],
                "image": parts[3],
                "ports": parts[4] if len(parts) > 4 else "",
            })

    # Stats (non-blocking, quick snapshot)
    stats: Dict[str, Dict[str, str]] = {}
    try:
        stats_out = subprocess.check_output(
            ["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"],
            timeout=15, text=True,
        )
        for line in stats_out.strip().splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                stats[parts[0]] = {"cpu": parts[1], "memory": parts[2]}
    except Exception:
        pass

    for c in containers:
        s = stats.get(c["name"], {})
        c["cpu"] = s.get("cpu", "N/A")
        c["memory"] = s.get("memory", "N/A")

    return {"available": True, "containers": containers, "count": len(containers)}


def _process_list(pattern: str = "") -> Dict[str, Any]:
    """List processes, optionally filtered by name pattern."""
    processes = []
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                with open(f"/proc/{pid}/comm") as f:
                    name = f.read().strip()
                if pattern and pattern.lower() not in name.lower():
                    continue
                with open(f"/proc/{pid}/stat") as f:
                    stat_parts = f.read().split()
                # Fields: pid, comm, state, ...
                state = stat_parts[2] if len(stat_parts) > 2 else "?"
                processes.append({"pid": pid, "name": name, "state": state})
            except (PermissionError, FileNotFoundError, ProcessLookupError):
                continue
    except Exception as e:
        return {"error": str(e), "processes": []}

    return {"processes": processes[:100], "total": len(processes)}


def _service_check(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    """HTTP health check for a URL."""
    import urllib.request
    import urllib.error
    import time as _time

    start = _time.monotonic()
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            latency_ms = (_time.monotonic() - start) * 1000
            body = resp.read(1024).decode("utf-8", errors="replace")
            return {
                "url": url,
                "status": "up",
                "status_code": resp.status,
                "latency_ms": round(latency_ms, 1),
                "body_preview": body[:200],
            }
    except Exception as e:
        latency_ms = (_time.monotonic() - start) * 1000
        return {
            "url": url,
            "status": "down",
            "error": str(e),
            "latency_ms": round(latency_ms, 1),
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def default_tools() -> List[Tool]:
    return [
        Tool(
            name="system.snapshot",
            description="Collect host context: OS, CPU, memory, GPU, load average",
            handler=_system_snapshot,
        ),
        Tool(
            name="workspace.list",
            description="List directory entries for the local workspace",
            handler=_workspace_list,
        ),
        Tool(
            name="workspace.read",
            description="Read a file from the workspace (up to 200 lines)",
            handler=_workspace_read,
        ),
        Tool(
            name="git.status",
            description="Get git repository status: branch, changes, ahead/behind",
            handler=_git_status,
        ),
        Tool(
            name="docker.status",
            description="List running Docker containers with resource usage",
            handler=_docker_status,
        ),
        Tool(
            name="process.list",
            description="List running processes, optionally filtered by name",
            handler=_process_list,
        ),
        Tool(
            name="service.check",
            description="HTTP health check for a URL endpoint",
            handler=_service_check,
        ),
    ]
