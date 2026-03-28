"""Benchmark framework for running agents through controlled scenarios."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aura.trajectory.collector import TrajectoryCollector, TrajectoryStep

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkScenario:
    """A controlled environment scenario for benchmarking agents."""

    name: str
    description: str
    initial_signals: List[Dict[str, Any]] = field(default_factory=list)
    injected_changes: List[Dict[str, Any]] = field(default_factory=list)
    inject_at_step: List[int] = field(default_factory=list)
    expected_agent_actions: List[str] = field(default_factory=list)
    max_steps: int = 20


class BenchmarkRunner:
    """Runs agents through controlled environment scenarios.

    The *agent* object must expose a callable interface::

        agent(environment_state: dict) -> str   # returns the action taken

    If the agent has a ``reset()`` method it will be called before each
    scenario.
    """

    def __init__(
        self,
        agent: Any,
        scenarios: Optional[List[BenchmarkScenario]] = None,
    ) -> None:
        self._agent = agent
        self._scenarios = scenarios if scenarios is not None else self.default_scenarios()

    # ------------------------------------------------------------------
    # Running scenarios
    # ------------------------------------------------------------------

    def run_scenario(self, scenario: BenchmarkScenario) -> Dict[str, Any]:
        """Simulate a single scenario step-by-step and score the agent."""
        logger.info("Running scenario: %s", scenario.name)

        # Reset agent if possible
        if hasattr(self._agent, "reset") and callable(self._agent.reset):
            self._agent.reset()

        collector = TrajectoryCollector()
        episode_id = collector.start_episode(task_description=scenario.description)

        # Build initial environment state from signals
        env_state: Dict[str, Any] = {
            "signals": list(scenario.initial_signals),
            "step": 0,
        }

        actions_taken: List[str] = []
        trajectory: List[TrajectoryStep] = []

        # Map step numbers to injected changes for fast lookup
        inject_map: Dict[int, List[Dict[str, Any]]] = {}
        for idx, step_num in enumerate(scenario.inject_at_step):
            if idx < len(scenario.injected_changes):
                inject_map.setdefault(step_num, []).append(
                    scenario.injected_changes[idx]
                )

        for step_num in range(scenario.max_steps):
            # Inject changes scheduled for this step
            if step_num in inject_map:
                for change in inject_map[step_num]:
                    env_state["signals"].append(change)
                    env_state.setdefault("injected", []).append(change)
                logger.debug(
                    "Step %d: injected %d change(s).",
                    step_num,
                    len(inject_map[step_num]),
                )

            env_state["step"] = step_num

            # Ask the agent for an action
            try:
                action = self._agent(env_state)
            except Exception as exc:
                logger.error(
                    "Agent raised an exception at step %d: %s", step_num, exc
                )
                action = f"ERROR: {exc}"

            actions_taken.append(action)

            # Determine whether the agent used context (heuristic: action
            # references something from the latest signals)
            context_used = _action_references_signals(
                action, env_state.get("signals", [])
            )

            # Compute a simple reward based on action-to-expected overlap
            reward = _score_action(action, scenario.expected_agent_actions)

            ts = collector.record_step(
                environment_state=dict(env_state),
                agent_action=action,
                result=None,
                reward=reward,
                context_was_used=context_used,
            )
            trajectory.append(ts)

            # Check if agent signalled completion
            if action.lower().strip() in ("done", "exit", "complete"):
                break

        collector.end_episode()

        # Score: fraction of expected actions that were approximately matched
        matches = _count_matches(actions_taken, scenario.expected_agent_actions)
        expected_total = max(len(scenario.expected_agent_actions), 1)
        score = matches / expected_total

        return {
            "scenario_name": scenario.name,
            "steps": len(trajectory),
            "actions_taken": actions_taken,
            "matches": matches,
            "score": round(score, 4),
            "trajectory": trajectory,
        }

    def run_all(self) -> Dict[str, Any]:
        """Run every registered scenario and aggregate results."""
        results: List[Dict[str, Any]] = []
        for scenario in self._scenarios:
            result = self.run_scenario(scenario)
            results.append(result)

        scores = [r["score"] for r in results]
        mean_score = sum(scores) / max(len(scores), 1)

        return {
            "scenarios": results,
            "aggregate": {
                "mean_score": round(mean_score, 4),
                "total_scenarios": len(results),
                "total_steps": sum(r["steps"] for r in results),
            },
        }

    # ------------------------------------------------------------------
    # Default scenarios
    # ------------------------------------------------------------------

    @staticmethod
    def default_scenarios() -> List[BenchmarkScenario]:
        """Create five default benchmark scenarios."""
        return [
            BenchmarkScenario(
                name="service_failure",
                description=(
                    "A dependent service goes down mid-task. "
                    "The agent should detect the outage and take corrective action."
                ),
                initial_signals=[
                    {
                        "source": "service",
                        "modality": "service_status",
                        "payload": {"name": "database", "status": "running", "latency_ms": 12},
                    },
                    {
                        "source": "service",
                        "modality": "service_status",
                        "payload": {"name": "cache", "status": "running", "latency_ms": 2},
                    },
                ],
                injected_changes=[
                    {
                        "source": "service",
                        "modality": "alert",
                        "payload": {
                            "name": "database",
                            "status": "down",
                            "message": "Connection refused on port 5432",
                            "severity": "critical",
                        },
                    },
                ],
                inject_at_step=[3],
                expected_agent_actions=[
                    "check database service status",
                    "attempt to restart database service",
                    "verify database connectivity restored",
                ],
                max_steps=10,
            ),
            BenchmarkScenario(
                name="file_conflict",
                description=(
                    "Another process modifies a file the agent is editing. "
                    "The agent should detect the conflict and resolve it."
                ),
                initial_signals=[
                    {
                        "source": "filesystem",
                        "modality": "file_change",
                        "payload": {"path": "/src/main.py", "change_type": "modified"},
                    },
                ],
                injected_changes=[
                    {
                        "source": "filesystem",
                        "modality": "file_change",
                        "payload": {
                            "path": "/src/main.py",
                            "change_type": "external_modification",
                            "message": "File modified by another process",
                            "severity": "warning",
                        },
                    },
                ],
                inject_at_step=[2],
                expected_agent_actions=[
                    "detect file conflict on /src/main.py",
                    "compare local changes with external changes",
                    "merge or resolve the conflict",
                ],
                max_steps=10,
            ),
            BenchmarkScenario(
                name="resource_exhaustion",
                description=(
                    "GPU memory fills up during computation. "
                    "The agent should detect the resource constraint and adapt."
                ),
                initial_signals=[
                    {
                        "source": "gpu",
                        "modality": "metric",
                        "payload": {"gpu_available": True, "gpu_memory": 40, "usage": 40},
                    },
                ],
                injected_changes=[
                    {
                        "source": "gpu",
                        "modality": "alert",
                        "payload": {
                            "gpu_available": True,
                            "gpu_memory": 98,
                            "usage": 98,
                            "message": "GPU memory nearly exhausted",
                            "severity": "critical",
                        },
                    },
                ],
                inject_at_step=[4],
                expected_agent_actions=[
                    "detect GPU memory exhaustion",
                    "reduce batch size or free GPU memory",
                    "resume computation with adjusted parameters",
                ],
                max_steps=12,
            ),
            BenchmarkScenario(
                name="security_alert",
                description=(
                    "A suspicious process is detected. "
                    "The agent should investigate and mitigate the threat."
                ),
                initial_signals=[
                    {
                        "source": "system",
                        "modality": "metric",
                        "payload": {"cpu": 25, "memory": 40, "disk": 55},
                    },
                ],
                injected_changes=[
                    {
                        "source": "process",
                        "modality": "alert",
                        "payload": {
                            "name": "unknown_miner",
                            "pid": 66612,
                            "cpu": 95,
                            "message": "Suspicious process consuming excessive CPU",
                            "severity": "critical",
                            "security_alert": True,
                        },
                    },
                ],
                inject_at_step=[2],
                expected_agent_actions=[
                    "investigate suspicious process unknown_miner",
                    "terminate or isolate the suspicious process",
                    "scan for additional compromise indicators",
                ],
                max_steps=10,
            ),
            BenchmarkScenario(
                name="dependency_update",
                description=(
                    "A dependency version changes unexpectedly. "
                    "The agent should detect breaking changes and adapt."
                ),
                initial_signals=[
                    {
                        "source": "dependency",
                        "modality": "dependency",
                        "payload": {"name": "requests", "version": "2.28.0", "status": "ok"},
                    },
                ],
                injected_changes=[
                    {
                        "source": "dependency",
                        "modality": "warning",
                        "payload": {
                            "name": "requests",
                            "old_version": "2.28.0",
                            "new_version": "3.0.0",
                            "message": "Major version bump detected; breaking changes possible",
                            "severity": "warning",
                        },
                    },
                    {
                        "source": "test",
                        "modality": "error",
                        "payload": {
                            "message": "ImportError: cannot import name 'compat' from 'requests'",
                            "file": "test_api.py",
                        },
                    },
                ],
                inject_at_step=[3, 4],
                expected_agent_actions=[
                    "detect dependency version change for requests",
                    "review breaking changes in requests 3.0.0",
                    "update code to be compatible with new version",
                    "run tests to verify fix",
                ],
                max_steps=15,
            ),
        ]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _action_references_signals(action: str, signals: List[Dict[str, Any]]) -> bool:
    """Heuristic: does the action text reference content from signals?"""
    action_lower = action.lower()
    for sig in signals:
        payload = sig.get("payload", {})
        # Check a few key fields
        for key in ("name", "path", "message", "service"):
            val = payload.get(key, "")
            if val and len(val) > 2 and val.lower() in action_lower:
                return True
    return False


def _score_action(action: str, expected_actions: List[str]) -> float:
    """Score a single action against the list of expected actions.

    Uses token overlap as a simple heuristic.  Returns max overlap
    across all expected actions, normalised to [0, 1].
    """
    if not expected_actions:
        return 0.5

    action_tokens = set(action.lower().split())
    best = 0.0
    for expected in expected_actions:
        expected_tokens = set(expected.lower().split())
        if not expected_tokens:
            continue
        overlap = len(action_tokens & expected_tokens) / len(expected_tokens)
        best = max(best, overlap)
    return best


def _count_matches(
    actions_taken: List[str], expected_actions: List[str]
) -> int:
    """Count how many expected actions were approximately matched.

    An expected action is considered matched if at least 50% of its
    tokens appear in any taken action.
    """
    matched = 0
    for expected in expected_actions:
        expected_tokens = set(expected.lower().split())
        if not expected_tokens:
            matched += 1
            continue
        for action in actions_taken:
            action_tokens = set(action.lower().split())
            overlap = len(action_tokens & expected_tokens) / len(expected_tokens)
            if overlap >= 0.5:
                matched += 1
                break
    return matched
