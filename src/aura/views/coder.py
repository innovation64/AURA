"""Coder agent environment view.

Prioritises filesystem changes, git status, test results, and code errors.
Filters out system metrics and unrelated docker status.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from aura.types import EnvironmentSignal
from aura.views.base import EnvironmentView, ViewConfig

logger = logging.getLogger(__name__)

# Signal sources / modalities that a coder cares about
_CODER_SOURCES = {"filesystem", "git", "test", "lint", "compiler", "editor", "code"}
_CODER_MODALITIES = {"file_change", "git_status", "test_result", "error", "warning", "code"}

# Sources the coder usually does not need
_SYSTEM_SOURCES = {"cpu", "memory", "disk", "gpu", "system", "network"}
_DOCKER_SOURCE = "docker"


class CoderView(EnvironmentView):
    """Environment view tailored for a coding agent."""

    @property
    def agent_type(self) -> str:
        return "coder"

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

            # Always keep critical signals regardless of source
            if sig.payload.get("severity") == "critical":
                filtered.append(sig)
                continue

            # Filter OUT system metrics unless critical
            if source_lower in _SYSTEM_SOURCES:
                continue

            # Filter OUT docker unless it concerns a focus service
            if source_lower == _DOCKER_SOURCE:
                service_name = sig.payload.get("service", "")
                if config.focus_services and service_name in config.focus_services:
                    filtered.append(sig)
                continue

            # Keep anything that matches coder-relevant sources/modalities
            if source_lower in _CODER_SOURCES or modality_lower in _CODER_MODALITIES:
                filtered.append(sig)
                continue

            # Keep signals whose path matches a focus path
            sig_path = sig.payload.get("path", "")
            if sig_path and config.focus_paths:
                if any(sig_path.startswith(fp) for fp in config.focus_paths):
                    filtered.append(sig)
                    continue

            # Keep signals that mention any keyword
            if config.keywords:
                payload_str = str(sig.payload).lower()
                if any(kw.lower() in payload_str for kw in config.keywords):
                    filtered.append(sig)
                    continue

            # Default: keep (be inclusive rather than lose context)
            filtered.append(sig)

        return filtered

    # ------------------------------------------------------------------
    # Prioritisation
    # ------------------------------------------------------------------

    _PRIORITY_ORDER: Dict[str, int] = {
        "error": 0,
        "test_result": 1,
        "warning": 2,
        "git_status": 3,
        "file_change": 4,
        "code": 5,
    }

    def prioritize(
        self, signals: List[EnvironmentSignal]
    ) -> List[EnvironmentSignal]:
        def _key(sig: EnvironmentSignal) -> tuple:
            modality_rank = self._PRIORITY_ORDER.get(sig.modality.lower(), 10)
            # Within same rank, newer first
            return (modality_rank, -sig.timestamp.timestamp())

        return sorted(signals, key=_key)

    # ------------------------------------------------------------------
    # Summarisation
    # ------------------------------------------------------------------

    def summarize(self, signals: List[EnvironmentSignal]) -> str:
        modified_files: List[str] = []
        git_lines: List[str] = []
        errors: List[str] = []

        for sig in signals:
            ml = sig.modality.lower()
            sl = sig.source.lower()
            if ml == "file_change" or sl == "filesystem":
                path = sig.payload.get("path", str(sig.payload))
                modified_files.append(path)
            elif sl == "git" or ml == "git_status":
                branch = sig.payload.get("branch", "unknown")
                uncommitted = sig.payload.get("uncommitted", "?")
                git_lines.append(f"branch={branch}, uncommitted={uncommitted}")
            elif ml in ("error", "warning"):
                msg = sig.payload.get("message", str(sig.payload))
                errors.append(msg)

        parts = ["## Code Environment"]
        parts.append(
            f"- Modified files: {', '.join(modified_files) if modified_files else 'none'}"
        )
        parts.append(
            f"- Git status: {'; '.join(git_lines) if git_lines else 'clean / unknown'}"
        )
        parts.append(
            f"- Active errors: {'; '.join(errors) if errors else 'none'}"
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

        modified_files: List[str] = []
        git_status: Dict[str, Any] = {
            "branch": "unknown",
            "uncommitted": 0,
            "ahead": 0,
        }
        active_errors: List[Dict[str, Any]] = []
        dependencies_status: Dict[str, str] = {}
        suggested_actions: List[str] = []

        for sig in capped:
            ml = sig.modality.lower()
            sl = sig.source.lower()

            if ml == "file_change" or sl == "filesystem":
                path = sig.payload.get("path", "")
                if path and path not in modified_files:
                    modified_files.append(path)

            elif sl == "git" or ml == "git_status":
                git_status["branch"] = sig.payload.get(
                    "branch", git_status["branch"]
                )
                git_status["uncommitted"] = sig.payload.get(
                    "uncommitted", git_status["uncommitted"]
                )
                git_status["ahead"] = sig.payload.get(
                    "ahead", git_status["ahead"]
                )

            elif ml == "error":
                active_errors.append(
                    {
                        "source": sig.source,
                        "message": sig.payload.get("message", str(sig.payload)),
                        "file": sig.payload.get("file", ""),
                        "line": sig.payload.get("line", None),
                    }
                )

            elif sl == "dependency" or ml == "dependency":
                dep_name = sig.payload.get("name", sig.source)
                dep_status = sig.payload.get("status", "unknown")
                dependencies_status[dep_name] = dep_status

        # Generate suggested actions
        if active_errors:
            suggested_actions.append(
                f"Fix {len(active_errors)} active error(s) before proceeding."
            )
        uncommitted = git_status.get("uncommitted", 0)
        if isinstance(uncommitted, int) and uncommitted > 5:
            suggested_actions.append(
                "Consider committing; there are many uncommitted changes."
            )
        if modified_files:
            suggested_actions.append(
                f"Review {len(modified_files)} recently modified file(s)."
            )

        return {
            "modified_files": modified_files,
            "git_status": git_status,
            "active_errors": active_errors,
            "dependencies_status": dependencies_status,
            "suggested_actions": suggested_actions,
        }
