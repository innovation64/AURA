"""Assembles structured, prioritised context for agent consumption.

Takes scored :class:`ChangeEvent` objects, probe snapshots, and task
context, then produces a single :class:`EnvironmentContext` ready for
the agent to consume.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aura.proactive.change_detector import ChangeEvent
from aura.proactive.relevance_scorer import TaskContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentContext:
    """Structured, prioritised snapshot of the environment pushed to agents."""

    summary: str
    critical_alerts: List[ChangeEvent]
    relevant_changes: List[ChangeEvent]
    environment_snapshot: Dict[str, Any]
    agent_hints: List[str]
    stale_after: float  # epoch timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hint templates
# ---------------------------------------------------------------------------

_HINT_TEMPLATES: Dict[str, str] = {
    # event_type / context key -> template
    "file_changed": "File '{file}' was modified externally -- you may want to re-read it.",
    "test_failed": "Tests are failing ({detail}) -- consider checking test output.",
    "service_down": "Service '{service}' is unreachable -- check if it needs a restart.",
    "high_cpu": "System under heavy CPU load ({value:.0f}%) -- consider deferring intensive tasks.",
    "high_memory": "Memory usage is elevated ({value:.0f}%) -- watch for OOM conditions.",
    "high_disk": "Disk usage is critical ({value:.0f}%) -- free space may be needed.",
    "git_diverged": "Remote has new commits -- consider pulling to stay in sync.",
    "spike": "Sudden change detected in {field} ({change_pct:.0f}% swing) -- verify stability.",
    "repeated_errors": "Repeated errors from {source} ({error_count} occurrences) -- investigate root cause.",
    "state_recovery": "'{field}' recovered to '{new}' (was '{old}').",
    "state_degraded": "'{field}' transitioned to '{new}' (was '{old}') -- may need attention.",
}


# ---------------------------------------------------------------------------
# ContextAssembler
# ---------------------------------------------------------------------------

class ContextAssembler:
    """Builds an :class:`EnvironmentContext` from scored change events and
    raw probe snapshots.

    Parameters
    ----------
    max_alerts:
        Maximum number of critical alerts to include.
    max_changes:
        Maximum number of relevant (non-critical) changes.
    context_ttl:
        Seconds until the assembled context is considered stale.
    """

    def __init__(
        self,
        max_alerts: int = 5,
        max_changes: int = 10,
        context_ttl: float = 60.0,
    ) -> None:
        self.max_alerts = max_alerts
        self.max_changes = max_changes
        self.context_ttl = context_ttl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble(
        self,
        change_events: List[ChangeEvent],
        relevance_scores: Dict[str, float],
        probe_snapshots: Dict[str, Any],
        task_context: Optional[TaskContext] = None,
    ) -> EnvironmentContext:
        """Assemble a complete :class:`EnvironmentContext`.

        Parameters
        ----------
        change_events:
            All detected change events from the latest cycle.
        relevance_scores:
            Mapping of ``event_id`` -> relevance score ``[0, 1]``.
        probe_snapshots:
            Latest raw results from each probe (``probe_name -> data``).
        task_context:
            Optional current task context used for hint generation.
        """
        # Sort by combined relevance * severity (descending)
        scored = sorted(
            change_events,
            key=lambda e: relevance_scores.get(e.event_id, 0.0) * e.severity,
            reverse=True,
        )

        critical_alerts = [
            e for e in scored if e.severity > 0.8
        ][: self.max_alerts]

        relevant_changes = [
            e for e in scored
            if relevance_scores.get(e.event_id, 0.0) > 0.5 and e.severity <= 0.8
        ][: self.max_changes]

        snapshot = self._build_snapshot(probe_snapshots)
        hints = self._generate_hints(critical_alerts + relevant_changes, task_context)
        summary = self._generate_summary(critical_alerts, relevant_changes)

        return EnvironmentContext(
            summary=summary,
            critical_alerts=critical_alerts,
            relevant_changes=relevant_changes,
            environment_snapshot=snapshot,
            agent_hints=hints,
            stale_after=time.time() + self.context_ttl,
            metadata={
                "assembled_at": time.time(),
                "total_events": len(change_events),
                "critical_count": len(critical_alerts),
                "relevant_count": len(relevant_changes),
            },
        )

    # ------------------------------------------------------------------
    # Summary generation (template-based, no LLM)
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_summary(
        critical: List[ChangeEvent],
        relevant: List[ChangeEvent],
    ) -> str:
        """Build a natural-language summary from the top events."""
        parts: List[str] = []

        if not critical and not relevant:
            return "Environment is stable -- no significant changes detected."

        if critical:
            parts.append(
                f"{len(critical)} critical alert{'s' if len(critical) != 1 else ''}: "
                + "; ".join(e.description for e in critical[:3])
            )
        if relevant:
            parts.append(
                f"{len(relevant)} notable change{'s' if len(relevant) != 1 else ''}: "
                + "; ".join(e.description for e in relevant[:3])
            )
        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Hint generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_hints(
        events: List[ChangeEvent],
        task_context: Optional[TaskContext] = None,
    ) -> List[str]:
        """Produce actionable hints based on change types."""
        hints: List[str] = []
        seen_types: set = set()

        for ev in events:
            ctx = ev.context
            etype = ev.event_type
            source_lower = ev.source.lower()
            desc_lower = ev.description.lower()

            # -- File changed --
            if etype == "state_change" and ("file" in source_lower or "filesystem" in source_lower):
                fname = ctx.get("new", ctx.get("field", "unknown"))
                hint = _HINT_TEMPLATES["file_changed"].format(file=fname)
                if hint not in hints:
                    hints.append(hint)
                continue

            # -- Test failed --
            if "test" in source_lower and ("fail" in desc_lower or "error" in desc_lower):
                hint = _HINT_TEMPLATES["test_failed"].format(detail=ev.description[:80])
                if "test_failed" not in seen_types:
                    hints.append(hint)
                    seen_types.add("test_failed")
                continue

            # -- Service down --
            if etype == "state_change":
                new_state = str(ctx.get("new", "")).lower()
                if new_state in {"down", "stopped", "error", "failed", "unreachable", "crashed", "unhealthy"}:
                    svc = ctx.get("field", ev.source)
                    hints.append(_HINT_TEMPLATES["service_down"].format(service=svc))
                    continue
                # Recovery
                old_state = str(ctx.get("old", "")).lower()
                if old_state in {"down", "stopped", "error", "failed", "unreachable"}:
                    hints.append(_HINT_TEMPLATES["state_recovery"].format(**ctx))
                    continue
                # Generic degradation
                hints.append(_HINT_TEMPLATES["state_degraded"].format(**ctx))
                continue

            # -- High CPU --
            if ctx.get("field") == "cpu_percent" and etype in {"anomaly", "spike"}:
                if "high_cpu" not in seen_types:
                    hints.append(_HINT_TEMPLATES["high_cpu"].format(value=ctx.get("value", 0)))
                    seen_types.add("high_cpu")
                continue

            # -- High Memory --
            if ctx.get("field") == "memory_percent" and etype in {"anomaly", "spike"}:
                if "high_memory" not in seen_types:
                    hints.append(_HINT_TEMPLATES["high_memory"].format(value=ctx.get("value", 0)))
                    seen_types.add("high_memory")
                continue

            # -- High Disk --
            if ctx.get("field") == "disk_percent":
                if "high_disk" not in seen_types:
                    hints.append(_HINT_TEMPLATES["high_disk"].format(value=ctx.get("value", 0)))
                    seen_types.add("high_disk")
                continue

            # -- Git diverged --
            if "git" in source_lower and ("diverge" in desc_lower or "behind" in desc_lower or "ahead" in desc_lower):
                if "git" not in seen_types:
                    hints.append(_HINT_TEMPLATES["git_diverged"])
                    seen_types.add("git")
                continue

            # -- Spike --
            if etype == "spike":
                hints.append(_HINT_TEMPLATES["spike"].format(
                    field=ctx.get("field", "metric"),
                    change_pct=ctx.get("change_pct", 0),
                ))
                continue

            # -- Pattern (repeated errors) --
            if etype == "pattern":
                hints.append(_HINT_TEMPLATES["repeated_errors"].format(
                    source=ev.source,
                    error_count=ctx.get("error_count", 0),
                ))
                continue

        return hints

    # ------------------------------------------------------------------
    # Snapshot builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_snapshot(probe_snapshots: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten latest probe results into a structured dict."""
        snapshot: Dict[str, Any] = {}
        for probe_name, data in probe_snapshots.items():
            if isinstance(data, dict):
                snapshot[probe_name] = data
            elif hasattr(data, "__dict__"):
                snapshot[probe_name] = {
                    k: v for k, v in data.__dict__.items() if not k.startswith("_")
                }
            else:
                snapshot[probe_name] = {"value": data}
        return snapshot
