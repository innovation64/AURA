"""Context-aware exploration planner — replaces HeuristicPlanner with
probe-driven, relevance-scored planning decisions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .explore import ExplorationDecision, ExplorationState, Planner
from .tools import ToolCall
from .types import EnvironmentSignal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal-derived hints — what the environment is telling us
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentHint:
    """A concrete suggestion derived from environment signals."""
    action: str          # e.g., "check_service", "read_file", "run_tests"
    tool_name: str       # tool to call
    tool_args: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    priority: float = 0.5   # 0-1, higher = more urgent
    source_signal: Optional[str] = None


def _derive_hints(signals: Sequence[EnvironmentSignal], query: str) -> List[EnvironmentHint]:
    """Analyze environment signals and query to derive actionable hints."""
    hints: List[EnvironmentHint] = []
    query_lower = query.lower() if query else ""

    for sig in signals:
        payload = sig.payload or {}
        source = sig.source or ""

        # System anomaly → check system
        if sig.modality == "system" and payload.get("anomalies"):
            hints.append(EnvironmentHint(
                action="check_system",
                tool_name="system.snapshot",
                reason=f"System anomaly detected: {payload.get('anomalies')}",
                priority=0.9,
                source_signal=source,
            ))

        # File changes → read changed files
        if sig.modality == "filesystem":
            for change in payload.get("changes", []):
                if change.get("type") in ("modified", "created"):
                    path = change.get("path", "")
                    hints.append(EnvironmentHint(
                        action="read_changed_file",
                        tool_name="workspace.read",
                        tool_args={"path": path},
                        reason=f"File {change.get('type')}: {path}",
                        priority=0.7,
                        source_signal=source,
                    ))

        # Service down → check service
        if sig.modality == "network" and payload.get("status") == "down":
            hints.append(EnvironmentHint(
                action="check_service",
                tool_name="system.snapshot",
                reason=f"Service unreachable: {payload.get('name', source)}",
                priority=0.85,
                source_signal=source,
            ))

        # Docker container issue
        if sig.modality == "docker" and payload.get("event") in ("stop", "die", "unhealthy"):
            hints.append(EnvironmentHint(
                action="check_docker",
                tool_name="docker.status",
                reason=f"Container event: {payload.get('name', '?')} {payload.get('event')}",
                priority=0.8,
                source_signal=source,
            ))

        # Git diverged
        if sig.modality == "git" and payload.get("diverged"):
            hints.append(EnvironmentHint(
                action="check_git",
                tool_name="git.status",
                reason="Git remote has diverged from local",
                priority=0.6,
                source_signal=source,
            ))

    # Query-driven hints
    if query_lower:
        _query_hints(query_lower, hints)

    # Deduplicate by tool_name (keep highest priority)
    seen_tools: Dict[str, EnvironmentHint] = {}
    for h in hints:
        if h.tool_name not in seen_tools or h.priority > seen_tools[h.tool_name].priority:
            seen_tools[h.tool_name] = h
    hints = list(seen_tools.values())

    # Sort by priority
    hints.sort(key=lambda h: h.priority, reverse=True)
    return hints


def _query_hints(query: str, hints: List[EnvironmentHint]) -> None:
    """Add hints based on user query content."""
    workspace_terms = ("file", "repo", "workspace", "project", "directory",
                       "folder", "code", "source", "module")
    system_terms = ("cpu", "memory", "disk", "gpu", "load", "process",
                    "system", "resource", "health", "status")
    docker_terms = ("container", "docker", "service", "pod", "deploy")
    git_terms = ("commit", "branch", "merge", "pull", "push", "diff", "git")

    if any(t in query for t in workspace_terms):
        hints.append(EnvironmentHint(
            action="list_workspace",
            tool_name="workspace.list",
            tool_args={"path": ".", "limit": 50},
            reason="Query references workspace context",
            priority=0.65,
        ))
    if any(t in query for t in system_terms):
        hints.append(EnvironmentHint(
            action="check_system",
            tool_name="system.snapshot",
            reason="Query references system resources",
            priority=0.7,
        ))
    if any(t in query for t in docker_terms):
        hints.append(EnvironmentHint(
            action="check_docker",
            tool_name="docker.status",
            reason="Query references containers/services",
            priority=0.7,
        ))
    if any(t in query for t in git_terms):
        hints.append(EnvironmentHint(
            action="check_git",
            tool_name="git.status",
            reason="Query references git operations",
            priority=0.7,
        ))


# ---------------------------------------------------------------------------
# SmartPlanner — the upgraded planner
# ---------------------------------------------------------------------------

class SmartPlanner(Planner):
    """Context-aware planner that uses environment signals and query analysis
    to decide which tools to invoke proactively.

    Unlike HeuristicPlanner which only has 2 hardcoded rules, SmartPlanner:
    1. Checks whether exploration is relevant to the query at all
    2. Derives hints from environment signals (anomalies, changes, events)
    3. Analyzes query semantics for relevant tools
    4. Prioritizes by urgency and relevance
    5. Avoids redundant tool calls
    """

    def __init__(self, max_hints_per_step: int = 1):
        self._max_hints = max_hints_per_step
        self.reset()

    def reset(self) -> None:
        """Clear per-exploration state.

        SmartPlanner is constructed once and reused across many agent turns.
        Without an explicit reset, relevance gates and hint queues leak from
        one exploration episode into the next, contaminating later decisions.
        """
        self._hint_queue: List[EnvironmentHint] = []
        self._hints_derived: bool = False
        self._relevance_checked: bool = False
        self._is_relevant: bool = False

    def decide(self, state: ExplorationState) -> ExplorationDecision:
        available = set(state.available_tools)
        used = set(state.tools_used)

        # Gate: check if exploration is relevant at all before doing anything.
        # If the query/signals don't match any tool category, skip entirely.
        if not self._relevance_checked:
            self._relevance_checked = True
            self._is_relevant = self._should_explore(state)
            if not self._is_relevant:
                return ExplorationDecision(
                    stop=True,
                    rationale="Query does not require environment probing; skipping exploration.",
                )

        # Only bootstrap with system snapshot for system-related queries
        if "system.snapshot" in available and "system.snapshot" not in used:
            if self._needs_system_context(state):
                return ExplorationDecision(
                    stop=False,
                    rationale="Query requires system context; taking snapshot.",
                    tool_call=ToolCall(name="system.snapshot"),
                )

        # Derive hints once from current signals + query
        if not self._hints_derived:
            self._hints_derived = True
            self._hint_queue = _derive_hints(
                state.signals, state.user_query or ""
            )

        # Try each hint in priority order
        while self._hint_queue:
            hint = self._hint_queue.pop(0)

            # Skip if tool not available or already used
            if hint.tool_name not in available:
                continue
            if hint.tool_name in used:
                continue

            return ExplorationDecision(
                stop=False,
                rationale=hint.reason,
                tool_call=ToolCall(name=hint.tool_name, arguments=hint.tool_args),
            )

        return ExplorationDecision(
            stop=True,
            rationale="No further environment probing required.",
        )

    @staticmethod
    def _should_explore(state: ExplorationState) -> bool:
        """Determine whether exploration is warranted for this query/signal set.

        Returns False for queries that are purely about reasoning, social
        interaction, or memory recall — where system/workspace probing adds noise.
        """
        query = (state.user_query or "").lower()

        # If there are already anomaly signals, exploration is warranted
        for sig in state.signals:
            payload = sig.payload or {}
            if payload.get("anomalies") or sig.modality in ("filesystem", "network", "docker"):
                return True

        # Check if query matches any tool-relevant category
        relevant_terms = (
            # System (require explicit technical context)
            "cpu usage", "memory usage", "disk space", "gpu util", "load average",
            "process list", "system status", "system health", "performance metric",
            # Workspace
            "file system", "repo", "workspace", "directory listing",
            "source code", "module import",
            # Docker/services
            "container", "docker", "pod", "deploy",
            # Git
            "commit", "branch", "merge", "pull request", "push", "git diff",
        )
        if any(term in query for term in relevant_terms):
            return True

        # No match — exploration would just add noise
        return False

    @staticmethod
    def _needs_system_context(state: ExplorationState) -> bool:
        """Check if system snapshot is specifically needed."""
        query = (state.user_query or "").lower()
        system_terms = ("cpu", "memory", "disk", "gpu", "load", "system",
                        "resource", "health", "status", "performance")
        if any(t in query for t in system_terms):
            return True
        # Also trigger for system anomaly signals
        for sig in state.signals:
            if sig.modality == "system" and (sig.payload or {}).get("anomalies"):
                return True
        return False
