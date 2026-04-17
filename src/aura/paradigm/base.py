"""Base classes for interaction paradigms.

Formal Definitions (Information-Theoretic)
==========================================

We formalize agent-environment interaction through the lens of information
flow between environment state E_t and agent belief B_t at time t.

Let I(·;·) denote mutual information, H(·) entropy, and H(·|·) conditional
entropy.

**Reactive Paradigm** (On-Demand Information Acquisition):
    At each step, the agent issues a query q_t and receives response r_t:
        I_reactive(E_t ; B_t) = I(E_t ; r_t | q_t)
    Information flow is agent-initiated and bounded by query quality.
    The agent's belief update depends entirely on what it asks for.

**Proactive Paradigm** (Unsolicited Context Injection):
    The environment monitors its own state, detects changes via a relevance
    function ρ(·), and pushes context c_t when ρ(ΔE_t) > τ:
        I_proactive(E_t ; B_t) = I(E_t ; r_t | q_t) + I(E_t ; c_t | ρ)
    The second term is "free" information the agent never asked for.
    This reduces H(E_t | B_t) — the agent's environmental uncertainty —
    without consuming the agent's action budget.

**Collaborative Paradigm** (Bilateral Information Flow with Attention):
    Extends Proactive with a feedback channel: the agent's behavior updates
    the relevance function ρ via an attention tracker A_t:
        ρ_{t+1} = ρ_t + η · ∇_ρ L(A_t, c_t, used_t)
    where used_t ∈ {0,1} indicates whether the agent utilized the pushed
    context, and η is the learning rate.
    Over time, the environment learns to push *what the agent actually needs*:
        I_collab(E_t ; B_t) → max_{ρ} I(E_t ; c_t | ρ)
    This is a self-improving system that converges to optimal context delivery.

Key Insight: The progression Reactive → Proactive → Collaborative
monotonically increases I(E_t ; B_t) while the Collaborative paradigm
additionally minimizes "alert fatigue" — i.e., H(c_t | used_t) → 0.
"""

from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class AgentObservation:
    """What the agent sees at each step."""

    environment_state: Dict[str, Any]
    pushed_context: Optional[Dict[str, Any]] = None  # only in proactive/collaborative
    available_tools: List[str] = field(default_factory=list)
    step_number: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResponse:
    """What the agent outputs at each step."""

    action: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    used_pushed_context: bool = False
    reasoning: str = ""


@dataclass
class StepRecord:
    """Full record of one interaction step."""

    step_number: int
    observation: AgentObservation
    response: AgentResponse
    env_state_before: Dict[str, Any]
    env_state_after: Dict[str, Any]
    reward: float = 0.0
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class EpisodeResult:
    """Complete result of running one episode under a paradigm."""

    paradigm: str
    scenario_name: str
    steps: List[StepRecord]
    total_reward: float = 0.0
    detected_change_at_step: int = -1  # step when agent first noticed injected change
    task_completed: bool = False
    metrics: Dict[str, float] = field(default_factory=dict)


class AgentPolicy(ABC):
    """Abstract agent policy — decides actions given observations."""

    @abstractmethod
    def act(self, observation: AgentObservation) -> AgentResponse:
        ...

    def reset(self) -> None:
        """Reset internal state for a new episode."""


class EnvironmentSimulator:
    """Simulates an environment with state, tool results, and injected changes.

    The simulator models a realistic agent workspace:
    - Maintains mutable environment state
    - Supports tool calls that read state
    - Injects changes at specified steps (simulating real environment events)
    """

    def __init__(
        self,
        initial_state: Dict[str, Any],
        injected_changes: Optional[Dict[int, List[Dict[str, Any]]]] = None,
        tool_handlers: Optional[Dict[str, Callable]] = None,
    ):
        self.state: Dict[str, Any] = copy.deepcopy(initial_state)
        self._initial_state = copy.deepcopy(initial_state)
        self._injections = injected_changes or {}
        self._tool_handlers = tool_handlers or {}
        self._current_step = 0

    def reset(self) -> None:
        self.state = copy.deepcopy(self._initial_state)
        self._current_step = 0

    def step(self, step_num: int) -> Dict[str, Any]:
        """Advance environment state, applying any injections for this step."""
        self._current_step = step_num
        if step_num in self._injections:
            for change in self._injections[step_num]:
                self._apply_change(change)
        return dict(self.state)

    def execute_tool(self, tool_name: str, args: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a tool call and return results from current state."""
        handler = self._tool_handlers.get(tool_name)
        if handler:
            return handler(self.state, args or {})
        # Default: return relevant subset of state
        if tool_name == "system.snapshot":
            return {k: v for k, v in self.state.items() if k in ("cpu", "memory", "disk", "load", "errors", "services")}
        if tool_name == "git.status":
            return {k: v for k, v in self.state.items() if k in ("branch", "uncommitted", "ahead", "behind", "conflicts")}
        if tool_name == "docker.status":
            return {"containers": self.state.get("containers", [])}
        if tool_name == "process.list":
            return {"processes": self.state.get("processes", [])}
        if tool_name == "service.check":
            return {"services": self.state.get("services", [])}
        return {"error": f"Unknown tool: {tool_name}"}

    def _apply_change(self, change: Dict[str, Any]) -> None:
        """Apply an injected change to the environment state."""
        change_type = change.get("type", "update")
        if change_type == "update":
            for k, v in change.get("updates", {}).items():
                self.state[k] = v
        elif change_type == "add_service_failure":
            services = self.state.setdefault("services", [])
            target = change.get("service_name", "unknown")
            for svc in services:
                if isinstance(svc, dict) and svc.get("name") == target:
                    svc["status"] = "down"
                    svc["error"] = change.get("error", "Connection refused")
                    break
            self.state.setdefault("alerts", []).append({
                "type": "service_down",
                "service": target,
                "severity": "critical",
                "message": change.get("error", "Service down"),
                "step": self._current_step,
            })
        elif change_type == "add_file_conflict":
            self.state.setdefault("conflicts", []).append(change.get("file", "unknown"))
            self.state.setdefault("alerts", []).append({
                "type": "file_conflict",
                "file": change.get("file"),
                "severity": "warning",
                "step": self._current_step,
            })
        elif change_type == "resource_spike":
            for k, v in change.get("metrics", {}).items():
                self.state[k] = v
            self.state.setdefault("alerts", []).append({
                "type": "resource_spike",
                "severity": "critical",
                "metrics": change.get("metrics", {}),
                "step": self._current_step,
            })
        elif change_type == "security_alert":
            self.state.setdefault("processes", []).append(change.get("process", {}))
            self.state.setdefault("alerts", []).append({
                "type": "security",
                "severity": "critical",
                "process": change.get("process", {}),
                "step": self._current_step,
            })


class InteractionParadigm(ABC):
    """Base class for interaction paradigms."""

    name: str = "base"

    @abstractmethod
    def run_episode(
        self,
        agent: AgentPolicy,
        env: EnvironmentSimulator,
        max_steps: int = 20,
        scenario_name: str = "",
    ) -> EpisodeResult:
        ...
