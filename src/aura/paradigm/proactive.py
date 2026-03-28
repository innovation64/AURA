"""Proactive paradigm: environment pushes relevant context to agent.

AURA's core innovation. The agent receives proactive context — alerts,
summaries, and hints — before it acts. It doesn't need to query for
information that the environment already knows is relevant.

Key difference from reactive:
    Reactive:    Agent → (tool call) → Environment → (result) → Agent
    Proactive:   Environment → (detect change) → (score relevance) → Agent
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..proactive.change_detector import ChangeDetector, ChangeEvent
from ..proactive.relevance_scorer import RelevanceScorer, TaskContext
from ..proactive.context_assembler import ContextAssembler, EnvironmentContext
from ..proactive.push_controller import PushController
from ..types import EnvironmentSignal

from .base import (
    AgentObservation,
    AgentPolicy,
    AgentResponse,
    EnvironmentSimulator,
    EpisodeResult,
    InteractionParadigm,
    StepRecord,
)


class ProactiveParadigm(InteractionParadigm):
    """Proactive interaction: environment pushes context to the agent.

    At each step, the environment:
    1. Converts state into EnvironmentSignals
    2. Detects changes via ChangeDetector (5 strategies)
    3. Scores relevance via RelevanceScorer
    4. Assembles context via ContextAssembler
    5. Decides whether to push via PushController
    6. Delivers context to agent's observation

    The agent sees: environment state + pushed context (alerts, hints, summary)
    """

    name = "proactive"

    def __init__(
        self,
        agent_type: str = "sysadmin",
        relevance_threshold: float = 0.3,
        min_push_interval: float = 0.0,  # no throttle in simulation
    ):
        self.agent_type = agent_type
        self.detector = ChangeDetector()
        self.scorer = RelevanceScorer()
        self.assembler = ContextAssembler()
        self.push_controller = PushController(
            min_push_interval=min_push_interval,
            critical_override=True,
            use_logical_time=True,  # simulation mode
        )
        self.relevance_threshold = relevance_threshold

    def run_episode(
        self,
        agent: AgentPolicy,
        env: EnvironmentSimulator,
        max_steps: int = 20,
        scenario_name: str = "",
    ) -> EpisodeResult:
        agent.reset()
        env.reset()

        steps: List[StepRecord] = []
        total_reward = 0.0
        detected_at = -1
        pushes_made = 0
        pushes_used = 0

        task_ctx = TaskContext(agent_type=self.agent_type)

        for step_num in range(max_steps):
            state_before = dict(env.state)
            env_state = env.step(step_num)
            state_after = dict(env.state)

            # Advance logical clock for simulation-mode push controller
            self.push_controller.tick()

            # --- Proactive pipeline ---
            signals = self._state_to_signals(env_state, step_num)
            events = self.detector.detect(signals)

            # Score relevance
            relevance_scores: Dict[str, float] = {}
            for evt in events:
                relevance_scores[evt.event_id] = self.scorer.score(evt, task_ctx)

            # Assemble context
            assembled_ctx = self.assembler.assemble(events, relevance_scores, {})

            # Decide whether to push
            pushed_context: Optional[Dict[str, Any]] = None
            if self.push_controller.should_push(assembled_ctx):
                pushed_context = {
                    "summary": assembled_ctx.summary,
                    "critical_alerts": [
                        {"type": e.event_type, "source": e.source,
                         "severity": e.severity, "description": e.description}
                        for e in assembled_ctx.critical_alerts
                    ],
                    "relevant_changes": [
                        {"type": e.event_type, "source": e.source,
                         "severity": e.severity, "description": e.description}
                        for e in assembled_ctx.relevant_changes
                    ],
                    "hints": assembled_ctx.agent_hints,
                }
                self.push_controller.record_push(
                    self.push_controller.classify_priority(assembled_ctx),
                    len(assembled_ctx.critical_alerts),
                )
                pushes_made += 1

            # Build observation WITH pushed context
            observation = AgentObservation(
                environment_state=env_state,
                pushed_context=pushed_context,
                available_tools=["system.snapshot", "git.status", "docker.status",
                                 "process.list", "service.check"],
                step_number=step_num,
            )

            t0 = time.time()
            response = agent.act(observation)
            latency = (time.time() - t0) * 1000

            # Execute tool calls
            for tc in response.tool_calls:
                tool_name = tc.get("tool", "")
                tool_args = tc.get("args", {})
                env.execute_tool(tool_name, tool_args)

            # Track context usage
            if pushed_context and response.used_pushed_context:
                pushes_used += 1

            # Detection tracking
            if detected_at < 0 and _agent_aware_of_change(response, pushed_context):
                detected_at = step_num

            reward = _compute_proactive_reward(response, env.state, pushed_context, detected_at >= 0)
            total_reward += reward

            steps.append(StepRecord(
                step_number=step_num,
                observation=observation,
                response=response,
                env_state_before=state_before,
                env_state_after=state_after,
                reward=reward,
                latency_ms=latency,
            ))

            if response.action.lower().strip() in ("done", "exit", "complete"):
                break

        return EpisodeResult(
            paradigm=self.name,
            scenario_name=scenario_name,
            steps=steps,
            total_reward=total_reward,
            detected_change_at_step=detected_at,
            task_completed=any(s.response.action.lower().strip() in ("done", "complete") for s in steps),
            metrics={
                "pushes_made": pushes_made,
                "pushes_used": pushes_used,
                "context_hit_rate": pushes_used / max(pushes_made, 1),
            },
        )

    def _state_to_signals(self, state: Dict[str, Any], step_num: int) -> List[EnvironmentSignal]:
        """Convert environment state dict into EnvironmentSignals for detection."""
        signals: List[EnvironmentSignal] = []

        # System metrics
        sys_payload = {}
        for key in ("cpu", "memory", "disk", "load", "gpu_memory"):
            if key in state:
                val = state[key]
                if isinstance(val, (int, float)):
                    sys_payload[f"{key}_percent" if key in ("cpu", "memory", "disk") else key] = val
        if sys_payload:
            signals.append(EnvironmentSignal(
                source="probe.system", modality="system", payload=sys_payload,
            ))

        # Service states
        for svc in state.get("services", []):
            if isinstance(svc, dict):
                signals.append(EnvironmentSignal(
                    source="probe.network", modality="network",
                    payload={"name": svc.get("name", ""), "status": svc.get("status", "")},
                ))

        # Alerts (injected changes)
        for alert in state.get("alerts", []):
            if isinstance(alert, dict):
                signals.append(EnvironmentSignal(
                    source=f"probe.{alert.get('type', 'system')}",
                    modality="alert",
                    payload=alert,
                ))

        # File conflicts
        for conflict in state.get("conflicts", []):
            signals.append(EnvironmentSignal(
                source="probe.filesystem", modality="filesystem",
                payload={"file": conflict, "type": "conflict"},
            ))

        # Processes
        for proc in state.get("processes", []):
            if isinstance(proc, dict):
                signals.append(EnvironmentSignal(
                    source="probe.process", modality="process", payload=proc,
                ))

        return signals


def _agent_aware_of_change(response: AgentResponse, pushed_context: Optional[Dict]) -> bool:
    """Check if agent is aware of the injected change."""
    action_lower = response.action.lower()

    # Agent explicitly used pushed context
    if response.used_pushed_context:
        return True

    # Check action for change-related keywords
    for keyword in ("alert", "down", "failure", "conflict", "spike", "suspicious",
                     "critical", "restart", "fix", "resolve", "investigate"):
        if keyword in action_lower:
            return True

    return False


def _compute_proactive_reward(
    response: AgentResponse,
    env_state: dict,
    pushed_context: Optional[Dict],
    detected: bool,
) -> float:
    reward = 0.3
    alerts = env_state.get("alerts", [])
    if alerts and detected:
        reward += 0.4
    if pushed_context and response.used_pushed_context:
        reward += 0.2  # bonus for using proactive context
    if response.tool_calls:
        reward += 0.05
    if response.action.lower().strip() in ("idle", "wait", "pass"):
        reward = 0.1
    return min(1.0, reward)
