from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from .tools import ToolCall, ToolRegistry, ToolResult
from .types import EnvironmentSignal

logger = logging.getLogger(__name__)

# Tool names whose output is system-level and rarely task-relevant
_SYSTEM_TOOLS = frozenset({
    "system.snapshot", "docker.status", "process.list", "service.check",
})

# Keys that indicate task-relevant grounding content
_GROUNDING_KEYS = frozenset({
    "location", "time", "hour", "action", "agent", "agents", "name",
    "nearby_agents", "agents_here", "plan", "memories", "events",
    "current_action", "current_location",
})


def _signal_confidence(result: "ToolResult", query: Optional[str] = None) -> float:
    """Score how relevant a tool result is to the agent's task.

    System monitoring tools get low confidence (filtered by Scene) unless
    the query explicitly asks for system info. Task-grounding tools get
    high confidence.
    """
    if result.name in _SYSTEM_TOOLS:
        # Only high confidence if query mentions system terms
        q = (query or "").lower()
        system_terms = ("cpu", "memory", "disk", "gpu", "system", "status",
                        "container", "docker", "process", "health")
        if any(t in q for t in system_terms):
            return 0.8
        return 0.2  # Low confidence — Scene will filter this out

    # Check if output contains grounding info
    output = result.output
    if isinstance(output, dict):
        if set(output.keys()) & _GROUNDING_KEYS:
            return 1.0

    return 0.7  # Default: moderately relevant


@dataclass
class ExplorationDecision:
    stop: bool
    rationale: str
    tool_call: Optional[ToolCall] = None


@dataclass
class ExplorationState:
    signals: List[EnvironmentSignal]
    user_query: Optional[str] = None
    raw_input: Any = None
    tool_results: List[ToolResult] = field(default_factory=list)
    available_tools: Sequence[str] = field(default_factory=list)

    @property
    def tools_used(self) -> List[str]:
        return [result.name for result in self.tool_results]


@dataclass
class ExplorationOutcome:
    extra_signals: List[EnvironmentSignal]
    tool_results: List[ToolResult]
    decisions: List[ExplorationDecision]

    def summary(self) -> dict:
        return {
            "tools": [
                {
                    "name": result.name,
                    "ok": result.ok,
                    "error": result.error,
                    "duration_ms": result.duration_ms,
                }
                for result in self.tool_results
            ],
            "decisions": [
                {
                    "stop": decision.stop,
                    "rationale": decision.rationale,
                    "tool": decision.tool_call.name if decision.tool_call else None,
                }
                for decision in self.decisions
            ],
        }


class Planner:
    def reset(self) -> None:
        """Reset any per-exploration state before a new exploration episode."""
        return None

    def decide(self, state: ExplorationState) -> ExplorationDecision:
        raise NotImplementedError


class HeuristicPlanner(Planner):
    def decide(self, state: ExplorationState) -> ExplorationDecision:
        available = set(state.available_tools)
        used = set(state.tools_used)
        query = (state.user_query or "").lower()

        # Only take system snapshot when query is about system resources
        system_terms = ("cpu", "memory", "disk", "gpu", "system", "status",
                        "health", "performance", "resource", "load")
        if "system.snapshot" in available and "system.snapshot" not in used:
            if any(term in query for term in system_terms):
                return ExplorationDecision(
                    stop=False,
                    rationale="Query references system resources; taking snapshot.",
                    tool_call=ToolCall(name="system.snapshot"),
                )

        if any(term in query for term in ("file", "repo", "workspace", "project", "directory", "folder")):
            if "workspace.list" in available and "workspace.list" not in used:
                return ExplorationDecision(
                    stop=False,
                    rationale="Query references workspace context; list entries.",
                    tool_call=ToolCall(name="workspace.list", arguments={"path": ".", "limit": 50}),
                )

        return ExplorationDecision(stop=True, rationale="No further environment probing required.")


class Explorer:
    def __init__(self, planner: Planner, registry: ToolRegistry, max_steps: int = 3) -> None:
        self.planner = planner
        self.registry = registry
        self.max_steps = max_steps

    def explore(
        self,
        signals: Sequence[EnvironmentSignal],
        user_query: Optional[str] = None,
        raw_input: Any = None,
        max_steps_override: Optional[int] = None,
    ) -> ExplorationOutcome:
        self.planner.reset()
        collected_signals: List[EnvironmentSignal] = list(signals)
        tool_results: List[ToolResult] = []
        decisions: List[ExplorationDecision] = []

        effective_max_steps = self.max_steps if max_steps_override is None else max_steps_override
        for step in range(max(0, effective_max_steps)):
            # FIX: Build state with ALL collected signals including latest tool results
            # Previously, planner could see stale state because tool_results were
            # added after the planner decision in the same iteration.
            state = ExplorationState(
                signals=collected_signals,
                user_query=user_query,
                raw_input=raw_input,
                tool_results=tool_results,
                available_tools=[tool.name for tool in self.registry.list() if self.registry.is_allowed(tool.name)],
            )
            decision = self.planner.decide(state)
            decisions.append(decision)

            if decision.stop or not decision.tool_call:
                break

            result = self.registry.execute(decision.tool_call)
            tool_results.append(result)

            if result.ok:
                # Score relevance: tool outputs with grounding info get high
                # confidence, pure system metrics get lower confidence
                conf = _signal_confidence(result, user_query)
                sig = EnvironmentSignal(
                    source=result.name,
                    payload={"output": result.output},
                    modality="tool",
                    confidence=conf,
                )
            else:
                sig = EnvironmentSignal(
                    source=result.name,
                    payload={"error": result.error},
                    modality="tool",
                    confidence=0.0,
                )
                logger.warning("Probe tool '%s' failed: %s", result.name, result.error)

            # FIX: Immediately append signal so next iteration's state is up-to-date
            collected_signals.append(sig)

        extra = [s for s in collected_signals[len(signals):] if s.confidence >= 0.5]
        return ExplorationOutcome(extra_signals=extra, tool_results=tool_results, decisions=decisions)
