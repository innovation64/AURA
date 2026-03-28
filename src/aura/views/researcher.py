"""Researcher agent environment view.

Prioritises network/API availability, data files, notebook changes,
and service endpoints.  Filters out low-level system metrics unless
they are critical.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from aura.types import EnvironmentSignal
from aura.views.base import EnvironmentView, ViewConfig

logger = logging.getLogger(__name__)

_DATA_EXTENSIONS = {
    ".csv", ".json", ".jsonl", ".parquet", ".tsv", ".h5", ".hdf5",
    ".npy", ".npz", ".pkl", ".pickle", ".feather", ".arrow",
    ".xlsx", ".xls", ".db", ".sqlite",
}

_NOTEBOOK_EXTENSIONS = {".ipynb"}

_RESEARCH_SOURCES = {
    "api", "service", "endpoint", "network", "notebook", "data",
    "gpu", "compute",
}
_RESEARCH_MODALITIES = {
    "service_status", "api_status", "file_change", "notebook_change",
    "data_update", "metric",
}

_LOW_LEVEL_SOURCES = {"cpu", "memory", "disk", "process"}


class ResearcherView(EnvironmentView):
    """Environment view tailored for a research / data-science agent."""

    @property
    def agent_type(self) -> str:
        return "researcher"

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

            # Always keep critical
            if sig.payload.get("severity") == "critical":
                filtered.append(sig)
                continue

            # Filter OUT low-level system metrics
            if source_lower in _LOW_LEVEL_SOURCES and modality_lower == "metric":
                continue

            # Keep research-relevant sources
            if source_lower in _RESEARCH_SOURCES or modality_lower in _RESEARCH_MODALITIES:
                filtered.append(sig)
                continue

            # Keep data / notebook file changes
            sig_path = sig.payload.get("path", "")
            if sig_path:
                ext = _get_extension(sig_path)
                if ext in _DATA_EXTENSIONS or ext in _NOTEBOOK_EXTENSIONS:
                    filtered.append(sig)
                    continue

            # Keep signals matching focus paths
            if sig_path and config.focus_paths:
                if any(sig_path.startswith(fp) for fp in config.focus_paths):
                    filtered.append(sig)
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

            # Default: keep
            filtered.append(sig)

        return filtered

    # ------------------------------------------------------------------
    # Prioritisation
    # ------------------------------------------------------------------

    _PRIORITY_ORDER: Dict[str, int] = {
        "api_status": 0,
        "service_status": 1,
        "data_update": 2,
        "notebook_change": 3,
        "file_change": 4,
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
        apis: List[str] = []
        data_info: List[str] = []
        compute: List[str] = []

        for sig in signals:
            sl = sig.source.lower()
            ml = sig.modality.lower()

            if ml in ("api_status", "service_status") or sl in ("api", "service", "endpoint"):
                name = sig.payload.get("name", sig.source)
                status = sig.payload.get("status", "unknown")
                apis.append(f"{name}={status}")

            sig_path = sig.payload.get("path", "")
            if sig_path:
                ext = _get_extension(sig_path)
                if ext in _DATA_EXTENSIONS:
                    data_info.append(sig_path)

            if sl in ("gpu", "compute") or "gpu" in str(sig.payload):
                gpu_avail = sig.payload.get("gpu_available", sig.payload.get("available"))
                if gpu_avail is not None:
                    compute.append(f"GPU={'available' if gpu_avail else 'unavailable'}")
                gpu_mem = sig.payload.get("gpu_memory")
                if gpu_mem is not None:
                    compute.append(f"GPU mem={gpu_mem}")

        parts = ["## Research Environment"]
        parts.append(
            f"- Available APIs: {', '.join(apis) if apis else 'none detected'}"
        )
        parts.append(
            f"- Data status: {', '.join(data_info) if data_info else 'no data file changes'}"
        )
        parts.append(
            f"- Compute: {', '.join(compute) if compute else 'unknown'}"
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

        available_services: List[Dict[str, Any]] = []
        data_files: List[Dict[str, Any]] = []
        compute_resources: Dict[str, Any] = {
            "gpu_available": None,
            "gpu_memory": None,
            "cpu_cores": None,
        }
        recent_changes: List[Dict[str, Any]] = []
        suggested_actions: List[str] = []

        seen_services: set = set()

        for sig in capped:
            sl = sig.source.lower()
            ml = sig.modality.lower()
            p = sig.payload

            # Services / APIs
            if ml in ("api_status", "service_status") or sl in ("api", "service", "endpoint"):
                svc_name = p.get("name", sig.source)
                if svc_name not in seen_services:
                    seen_services.add(svc_name)
                    available_services.append(
                        {
                            "name": svc_name,
                            "status": p.get("status", "unknown"),
                            "url": p.get("url", None),
                            "latency_ms": p.get("latency_ms", None),
                        }
                    )

            # Data files
            sig_path = p.get("path", "")
            if sig_path:
                ext = _get_extension(sig_path)
                if ext in _DATA_EXTENSIONS:
                    data_files.append(
                        {
                            "path": sig_path,
                            "size": p.get("size", None),
                            "modified": p.get("modified", None),
                        }
                    )
                # Notebook or other file changes -> recent_changes
                if ext in _NOTEBOOK_EXTENSIONS or ml == "file_change":
                    recent_changes.append(
                        {
                            "path": sig_path,
                            "change_type": p.get("change_type", "modified"),
                            "timestamp": sig.timestamp.isoformat(),
                        }
                    )

            # Compute resources
            if sl in ("gpu", "compute") or "gpu" in str(p):
                if "gpu_available" in p or "available" in p:
                    compute_resources["gpu_available"] = p.get(
                        "gpu_available", p.get("available")
                    )
                if "gpu_memory" in p:
                    compute_resources["gpu_memory"] = p["gpu_memory"]
                if "cpu_cores" in p:
                    compute_resources["cpu_cores"] = p["cpu_cores"]

        # Suggested actions
        down_svcs = [s for s in available_services if s["status"] not in ("up", "healthy", "running", "available")]
        if down_svcs:
            names = ", ".join(s["name"] for s in down_svcs)
            suggested_actions.append(f"Service(s) may be unavailable: {names}. Check connectivity.")
        if compute_resources["gpu_available"] is False:
            suggested_actions.append("No GPU available; consider switching to CPU or waiting for resources.")
        if data_files:
            suggested_actions.append(f"{len(data_files)} data file(s) detected; verify data freshness.")

        return {
            "available_services": available_services,
            "data_files": data_files,
            "compute_resources": compute_resources,
            "recent_changes": recent_changes,
            "suggested_actions": suggested_actions,
        }


def _get_extension(path: str) -> str:
    """Return the lowercase file extension including the dot."""
    dot_idx = path.rfind(".")
    if dot_idx == -1:
        return ""
    return path[dot_idx:].lower()
