"""Sysadmin agent environment view.

Prioritises system metrics, docker containers, process health, and
network services.  Filters out filesystem details (unless config files)
and git information (unless ops repos).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from aura.types import EnvironmentSignal
from aura.views.base import EnvironmentView, ViewConfig

logger = logging.getLogger(__name__)

_SYSTEM_SOURCES = {"cpu", "memory", "disk", "gpu", "system", "network", "docker", "process", "service"}
_SYSTEM_MODALITIES = {"metric", "container", "process", "service_status", "alert", "health"}

_CONFIG_PREFIXES = ("/etc/", "/var/", "/opt/", "/usr/local/etc/")
_OPS_KEYWORDS = {"ansible", "terraform", "helm", "k8s", "docker", "compose", "ops", "deploy", "infra"}


class SysadminView(EnvironmentView):
    """Environment view tailored for a sysadmin / operations agent."""

    @property
    def agent_type(self) -> str:
        return "sysadmin"

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_signals(
        self, signals: List[EnvironmentSignal], config: ViewConfig
    ) -> List[EnvironmentSignal]:
        filtered: List[EnvironmentSignal] = []
        for sig in signals:
            source_lower = sig.source.lower()
            modality_lower = sig.modality.lower()

            # Always keep critical signals
            if sig.payload.get("severity") == "critical":
                filtered.append(sig)
                continue

            # Keep system-related sources / modalities
            if source_lower in _SYSTEM_SOURCES or modality_lower in _SYSTEM_MODALITIES:
                filtered.append(sig)
                continue

            # Filter OUT filesystem unless it's a config path
            if source_lower == "filesystem" or modality_lower == "file_change":
                sig_path = sig.payload.get("path", "")
                if any(sig_path.startswith(p) for p in _CONFIG_PREFIXES):
                    filtered.append(sig)
                elif config.focus_paths and any(
                    sig_path.startswith(fp) for fp in config.focus_paths
                ):
                    filtered.append(sig)
                # else drop
                continue

            # Filter OUT git unless it's an ops repo
            if source_lower == "git" or modality_lower == "git_status":
                repo = sig.payload.get("repo", "").lower()
                branch = sig.payload.get("branch", "").lower()
                combined = f"{repo} {branch}"
                if any(kw in combined for kw in _OPS_KEYWORDS):
                    filtered.append(sig)
                elif config.focus_paths and any(
                    repo.startswith(fp) for fp in config.focus_paths
                ):
                    filtered.append(sig)
                # else drop
                continue

            # Keep signals matching focus services
            service_name = sig.payload.get("service", sig.payload.get("name", ""))
            if config.focus_services and service_name in config.focus_services:
                filtered.append(sig)
                continue

            # Keep signals matching keywords
            if config.keywords:
                payload_str = str(sig.payload).lower()
                if any(kw.lower() in payload_str for kw in config.keywords):
                    filtered.append(sig)
                    continue

            # Default: keep (inclusive)
            filtered.append(sig)

        return filtered

    # ------------------------------------------------------------------
    # Prioritisation
    # ------------------------------------------------------------------

    _PRIORITY_ORDER: Dict[str, int] = {
        "alert": 0,
        "health": 1,
        "service_status": 2,
        "container": 3,
        "process": 4,
        "metric": 5,
    }

    def prioritize(
        self, signals: List[EnvironmentSignal]
    ) -> List[EnvironmentSignal]:
        def _key(sig: EnvironmentSignal) -> tuple:
            modality_rank = self._PRIORITY_ORDER.get(sig.modality.lower(), 10)
            severity_rank = 0 if sig.payload.get("severity") == "critical" else 1
            return (severity_rank, modality_rank, -sig.timestamp.timestamp())

        return sorted(signals, key=_key)

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    def summarize(self, signals: List[EnvironmentSignal]) -> str:
        cpu = "N/A"
        mem = "N/A"
        disk = "N/A"
        services_info: List[str] = []
        alerts: List[str] = []

        for sig in signals:
            sl = sig.source.lower()
            ml = sig.modality.lower()

            if sl == "cpu" or (sl == "system" and "cpu" in sig.payload):
                cpu = f"{sig.payload.get('usage', sig.payload.get('cpu', 'N/A'))}%"
            elif sl == "memory" or (sl == "system" and "memory" in sig.payload):
                mem = f"{sig.payload.get('usage', sig.payload.get('memory', 'N/A'))}%"
            elif sl == "disk" or (sl == "system" and "disk" in sig.payload):
                disk = f"{sig.payload.get('usage', sig.payload.get('disk', 'N/A'))}%"

            if ml in ("service_status", "container"):
                name = sig.payload.get("name", sig.source)
                status = sig.payload.get("status", "unknown")
                services_info.append(f"{name}={status}")

            if ml == "alert" or sig.payload.get("severity") == "critical":
                alerts.append(sig.payload.get("message", str(sig.payload)))

        parts = ["## System Health"]
        parts.append(f"- CPU: {cpu} | Mem: {mem} | Disk: {disk}")
        parts.append(
            f"- Services: {', '.join(services_info) if services_info else 'all nominal'}"
        )
        parts.append(
            f"- Alerts: {'; '.join(alerts) if alerts else 'none'}"
        )
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(
        self, signals: List[EnvironmentSignal], config: ViewConfig
    ) -> dict:
        filtered = self.filter_signals(signals, config)
        prioritized = self.prioritize(filtered)
        capped = prioritized[: config.max_context_items]

        system_health: Dict[str, Any] = {
            "cpu": None,
            "memory": None,
            "disk": None,
            "gpu": None,
        }
        containers: List[Dict[str, Any]] = []
        services: List[Dict[str, Any]] = []
        processes: List[Dict[str, Any]] = []
        alerts: List[Dict[str, Any]] = []
        suggested_actions: List[str] = []

        for sig in capped:
            sl = sig.source.lower()
            ml = sig.modality.lower()
            p = sig.payload

            # System metrics
            if sl in ("cpu", "system") and "cpu" in str(p):
                system_health["cpu"] = p.get("usage", p.get("cpu"))
            if sl in ("memory", "system") and "memory" in str(p):
                system_health["memory"] = p.get("usage", p.get("memory"))
            if sl in ("disk", "system") and "disk" in str(p):
                system_health["disk"] = p.get("usage", p.get("disk"))
            if sl in ("gpu", "system") and "gpu" in str(p):
                system_health["gpu"] = p.get("usage", p.get("gpu"))

            # Containers
            if ml == "container" or sl == "docker":
                containers.append(
                    {
                        "name": p.get("name", sig.source),
                        "status": p.get("status", "unknown"),
                        "cpu": p.get("cpu", None),
                        "mem": p.get("mem", p.get("memory", None)),
                    }
                )

            # Services
            if ml == "service_status":
                services.append(
                    {
                        "name": p.get("name", sig.source),
                        "status": p.get("status", "unknown"),
                        "latency_ms": p.get("latency_ms", None),
                    }
                )

            # Processes
            if ml == "process":
                processes.append(
                    {
                        "name": p.get("name", sig.source),
                        "pid": p.get("pid", None),
                        "status": p.get("status", "running"),
                        "cpu": p.get("cpu", None),
                        "mem": p.get("mem", None),
                    }
                )

            # Alerts
            if ml == "alert" or p.get("severity") == "critical":
                alerts.append(
                    {
                        "source": sig.source,
                        "message": p.get("message", str(p)),
                        "severity": p.get("severity", "warning"),
                    }
                )

        # Suggested actions
        if alerts:
            suggested_actions.append(
                f"Investigate {len(alerts)} active alert(s) immediately."
            )
        cpu_val = system_health.get("cpu")
        if cpu_val is not None and isinstance(cpu_val, (int, float)) and cpu_val > 90:
            suggested_actions.append("CPU usage is critically high; consider scaling or killing processes.")
        mem_val = system_health.get("memory")
        if mem_val is not None and isinstance(mem_val, (int, float)) and mem_val > 90:
            suggested_actions.append("Memory usage is critically high; check for leaks or OOM risk.")
        disk_val = system_health.get("disk")
        if disk_val is not None and isinstance(disk_val, (int, float)) and disk_val > 90:
            suggested_actions.append("Disk usage is critically high; free space or expand volume.")
        down_services = [s for s in services if s.get("status") not in ("running", "healthy", "up")]
        if down_services:
            names = ", ".join(s["name"] for s in down_services)
            suggested_actions.append(f"Restore degraded services: {names}.")

        return {
            "system_health": system_health,
            "containers": containers,
            "services": services,
            "processes": processes,
            "alerts": alerts,
            "suggested_actions": suggested_actions,
        }
