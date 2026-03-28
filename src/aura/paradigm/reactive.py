"""Reactive paradigm: agent must explicitly query for environment info.

This is the traditional agent interaction model. The agent receives a minimal
observation (step number, task description) and must use tool calls to discover
environment state. It has NO awareness of changes unless it checks.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from .base import (
    AgentObservation,
    AgentPolicy,
    AgentResponse,
    EnvironmentSimulator,
    EpisodeResult,
    InteractionParadigm,
    StepRecord,
)


class ReactiveParadigm(InteractionParadigm):
    """Reactive interaction: agent only sees what it explicitly queries.

    The agent receives:
    - step_number
    - available_tools list
    - results of any tool calls it made in the previous step

    It does NOT receive:
    - proactive alerts
    - environment change notifications
    - context pushes
    """

    name = "reactive"

    def __init__(self, tool_latency_ms: float = 50.0):
        self.tool_latency_ms = tool_latency_ms

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
        prev_tool_results: Dict = {}

        for step_num in range(max_steps):
            # Environment advances (changes may be injected)
            state_before = dict(env.state)
            env_state = env.step(step_num)
            state_after = dict(env.state)

            # Agent sees ONLY: step number, tool list, previous tool results
            observation = AgentObservation(
                environment_state=prev_tool_results,  # only what agent asked for
                pushed_context=None,  # reactive: no push
                available_tools=["system.snapshot", "git.status", "docker.status",
                                 "process.list", "service.check"],
                step_number=step_num,
            )

            t0 = time.time()
            response = agent.act(observation)
            latency = (time.time() - t0) * 1000

            # Execute any tool calls the agent made
            prev_tool_results = {}
            for tc in response.tool_calls:
                tool_name = tc.get("tool", "")
                tool_args = tc.get("args", {})
                result = env.execute_tool(tool_name, tool_args)
                prev_tool_results[tool_name] = result

            # Check if agent detected the injected change
            if detected_at < 0 and _action_references_alert(response.action, env.state):
                detected_at = step_num

            # Compute reward
            reward = _compute_step_reward(response, env.state, detected_at >= 0)
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
        )


def _action_references_alert(action: str, state: dict) -> bool:
    """Check if the agent's action indicates awareness of an alert."""
    action_lower = action.lower()
    alerts = state.get("alerts", [])
    for alert in alerts:
        alert_type = alert.get("type", "")
        if alert_type and alert_type.replace("_", " ") in action_lower:
            return True
        svc = alert.get("service", "")
        if svc and svc.lower() in action_lower:
            return True
        f = alert.get("file", "")
        if f and f.lower() in action_lower:
            return True
    # Also check keywords
    for keyword in ("alert", "down", "failure", "conflict", "spike", "suspicious"):
        if keyword in action_lower:
            return True
    return False


def _compute_step_reward(response: AgentResponse, env_state: dict, detected: bool) -> float:
    """Simple reward: bonus for detection, penalty for idle."""
    reward = 0.3  # baseline for acting
    alerts = env_state.get("alerts", [])
    if alerts and detected:
        reward += 0.4  # detected the problem
    if response.tool_calls:
        reward += 0.1  # gathering info is good
    if response.action.lower().strip() in ("idle", "wait", "pass"):
        reward = 0.1
    return min(1.0, reward)
