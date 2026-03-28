"""Collaborative paradigm: proactive + online feedback adaptation.

Extends the proactive paradigm with a feedback loop that learns from the
agent's behavior. Over time, the system adapts what it pushes based on
what the agent actually uses vs ignores.

    Environment → Detect → Score → Push → Agent → Act
         ↑                                          ↓
         └──── AttentionTracker ← (used_context?) ──┘

This is AURA's full paradigm — proactive context delivery with
self-improving relevance through online learning.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..proactive.change_detector import ChangeDetector
from ..proactive.relevance_scorer import RelevanceScorer, TaskContext
from ..proactive.context_assembler import ContextAssembler, EnvironmentContext
from ..proactive.push_controller import PushController
from ..proactive.attention_tracker import AttentionTracker
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
from .proactive import ProactiveParadigm


class CollaborativeParadigm(InteractionParadigm):
    """Collaborative interaction: proactive + attention feedback loop.

    Extends ProactiveParadigm with:
    - AttentionTracker: learns source/keyword weights from agent usage
    - Dynamic relevance: scorer weights updated each step from tracker
    - Fatigue adaptation: push frequency adapts to agent's ack rate
    """

    name = "collaborative"

    def __init__(
        self,
        agent_type: str = "sysadmin",
        learning_rate: float = 0.15,
        relevance_threshold: float = 0.3,
    ):
        self.agent_type = agent_type
        self.detector = ChangeDetector()
        self.scorer = RelevanceScorer()
        self.assembler = ContextAssembler()
        self.push_controller = PushController(
            min_push_interval=0.0,
            critical_override=True,
            use_logical_time=True,  # simulation mode — avoid wall-clock rate limiting
        )
        self.attention_tracker = AttentionTracker(learning_rate=learning_rate)
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
        # Reset discrete states so first-occurrence detection fires each episode
        # (numeric rolling windows preserved for cross-episode learning)
        self.detector.reset_discrete_states()

        steps: List[StepRecord] = []
        total_reward = 0.0
        detected_at = -1
        pushes_made = 0
        pushes_used = 0
        pushes_ignored = 0

        task_ctx = TaskContext(agent_type=self.agent_type)

        for step_num in range(max_steps):
            state_before = dict(env.state)
            env_state = env.step(step_num)
            state_after = dict(env.state)

            # Advance logical clock for simulation-mode push controller
            self.push_controller.tick()

            # --- Proactive pipeline with feedback ---
            signals = _state_to_signals(env_state)

            events = self.detector.detect(signals)

            # Score relevance with learned source weights from attention tracker
            self.scorer.update_source_weights(
                self.attention_tracker.get_attention_weights().source_weights
            )

            relevance_scores: Dict[str, float] = {}
            for evt in events:
                base_score = self.scorer.score(evt, task_ctx)
                # Apply keyword boost — meaningful multiplier, not negligible additive
                keyword_boost = self.attention_tracker.get_keyword_boost(evt.description)
                boosted = base_score * (1.0 + keyword_boost * 0.3)
                relevance_scores[evt.event_id] = min(1.0, boosted)

            assembled_ctx = self.assembler.assemble(events, relevance_scores, {})

            # Push decision
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

                # Record push for attention tracking
                self.attention_tracker.on_push(assembled_ctx)

            # Build observation
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
                env.execute_tool(tc.get("tool", ""), tc.get("args", {}))

            # --- Feedback loop ---
            if pushed_context:
                self.attention_tracker.on_agent_action(
                    response.action,
                    used_context=response.used_pushed_context,
                )
                if response.used_pushed_context:
                    pushes_used += 1
                    self.push_controller.record_acknowledgement()
                else:
                    pushes_ignored += 1
                    self.push_controller.record_ignore()

            # Learn from agent's action keywords
            if response.action and response.action.lower().strip() not in ("idle", "wait", "pass"):
                self.attention_tracker.on_agent_query(response.action)

            # Detection tracking
            if detected_at < 0 and _agent_aware(response, pushed_context):
                detected_at = step_num

            reward = _compute_collaborative_reward(
                response, env.state, pushed_context, detected_at >= 0, step_num,
            )
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

        attention_stats = self.attention_tracker.get_stats()

        return EpisodeResult(
            paradigm=self.name,
            scenario_name=scenario_name,
            steps=steps,
            total_reward=total_reward,
            detected_change_at_step=detected_at,
            task_completed=any(
                s.response.action.lower().strip() in ("done", "complete") for s in steps
            ),
            metrics={
                "pushes_made": pushes_made,
                "pushes_used": pushes_used,
                "pushes_ignored": pushes_ignored,
                "context_hit_rate": pushes_used / max(pushes_made, 1),
                "alert_fatigue": pushes_ignored / max(pushes_made, 1),
                "attention_use_rate": attention_stats.get("use_rate", 0.0),
                "tracked_sources": attention_stats.get("tracked_sources", 0),
                "tracked_keywords": attention_stats.get("tracked_keywords", 0),
            },
        )


def _state_to_signals(state: Dict[str, Any]) -> List[EnvironmentSignal]:
    """Convert environment state to signals."""
    signals: List[EnvironmentSignal] = []

    sys_payload = {}
    for key in ("cpu", "memory", "disk", "load", "gpu_memory"):
        if key in state and isinstance(state[key], (int, float)):
            sys_payload[f"{key}_percent" if key in ("cpu", "memory", "disk") else key] = state[key]
    if sys_payload:
        signals.append(EnvironmentSignal(source="probe.system", modality="system", payload=sys_payload))

    for svc in state.get("services", []):
        if isinstance(svc, dict):
            signals.append(EnvironmentSignal(
                source="probe.network", modality="network",
                payload={"name": svc.get("name", ""), "status": svc.get("status", "")},
            ))

    for alert in state.get("alerts", []):
        if isinstance(alert, dict):
            signals.append(EnvironmentSignal(
                source=f"probe.{alert.get('type', 'system')}",
                modality="alert", payload=alert,
            ))

    for conflict in state.get("conflicts", []):
        signals.append(EnvironmentSignal(
            source="probe.filesystem", modality="filesystem",
            payload={"file": conflict, "type": "conflict"},
        ))

    for proc in state.get("processes", []):
        if isinstance(proc, dict):
            signals.append(EnvironmentSignal(
                source="probe.process", modality="process", payload=proc,
            ))

    return signals


def _agent_aware(response: AgentResponse, pushed_context: Optional[Dict]) -> bool:
    if response.used_pushed_context:
        return True
    action_lower = response.action.lower()
    for kw in ("alert", "down", "failure", "conflict", "spike", "suspicious",
                "critical", "restart", "fix", "resolve", "investigate"):
        if kw in action_lower:
            return True
    return False


def _compute_collaborative_reward(
    response: AgentResponse,
    env_state: dict,
    pushed_context: Optional[Dict],
    detected: bool,
    step_num: int,
) -> float:
    reward = 0.3
    alerts = env_state.get("alerts", [])
    if alerts and detected:
        reward += 0.35
    if pushed_context and response.used_pushed_context:
        reward += 0.2
    if response.tool_calls:
        reward += 0.05
    # Early detection bonus — faster is better
    if detected and step_num <= 1:
        reward += 0.1
    if response.action.lower().strip() in ("idle", "wait", "pass"):
        reward = 0.1
    return min(1.0, reward)
