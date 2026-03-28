"""Evaluation metrics for AURA's proactive context system."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from aura.trajectory.collector import TrajectoryStep

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Container for a single evaluation metric result."""

    metric_name: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class AURAMetrics:
    """Metrics for evaluating AURA's proactive context system."""

    @staticmethod
    def context_hit_rate(trajectory: List[TrajectoryStep]) -> EvalResult:
        """What fraction of pushed contexts were actually used by the agent?"""
        total = len(trajectory)
        used = sum(1 for s in trajectory if s.context_was_used)
        rate = used / max(total, 1)
        return EvalResult(
            metric_name="context_hit_rate",
            value=rate,
            details={"used": used, "total": total},
        )

    @staticmethod
    def proactive_precision(
        pushed_events: List[Any], actually_relevant: List[Any]
    ) -> EvalResult:
        """Of the changes we pushed, how many were actually relevant?

        Both lists are compared by identity / equality.  Items in
        *pushed_events* that also appear in *actually_relevant* count
        as true positives.
        """
        if not pushed_events:
            return EvalResult(
                metric_name="proactive_precision",
                value=0.0,
                details={"true_positives": 0, "total_pushed": 0},
            )

        relevant_set = set(_hashable(e) for e in actually_relevant)
        true_positives = sum(
            1 for e in pushed_events if _hashable(e) in relevant_set
        )
        precision = true_positives / len(pushed_events)
        return EvalResult(
            metric_name="proactive_precision",
            value=precision,
            details={
                "true_positives": true_positives,
                "total_pushed": len(pushed_events),
            },
        )

    @staticmethod
    def proactive_recall(
        pushed_events: List[Any], actually_relevant: List[Any]
    ) -> EvalResult:
        """Of the relevant changes, how many did we push?"""
        if not actually_relevant:
            return EvalResult(
                metric_name="proactive_recall",
                value=1.0,  # vacuously true
                details={"true_positives": 0, "total_relevant": 0},
            )

        pushed_set = set(_hashable(e) for e in pushed_events)
        true_positives = sum(
            1 for e in actually_relevant if _hashable(e) in pushed_set
        )
        recall = true_positives / len(actually_relevant)
        return EvalResult(
            metric_name="proactive_recall",
            value=recall,
            details={
                "true_positives": true_positives,
                "total_relevant": len(actually_relevant),
            },
        )

    @staticmethod
    def mean_time_to_awareness(
        events_with_timestamps: List[Dict[str, Any]],
    ) -> EvalResult:
        """Average time between environment change and agent becoming aware.

        Each dict is expected to have ``"change_time"`` and ``"aware_time"``
        keys (both floats, epoch seconds).
        """
        delays: List[float] = []
        for evt in events_with_timestamps:
            change_t = evt.get("change_time")
            aware_t = evt.get("aware_time")
            if change_t is not None and aware_t is not None:
                delay = float(aware_t) - float(change_t)
                if delay >= 0:
                    delays.append(delay)

        if not delays:
            return EvalResult(
                metric_name="mean_time_to_awareness",
                value=0.0,
                details={"event_count": 0},
            )

        mean_delay = sum(delays) / len(delays)
        return EvalResult(
            metric_name="mean_time_to_awareness",
            value=mean_delay,
            details={
                "event_count": len(delays),
                "min_delay": min(delays),
                "max_delay": max(delays),
                "total_delay": sum(delays),
            },
        )

    @staticmethod
    def context_freshness(contexts: List[Dict[str, Any]]) -> EvalResult:
        """How fresh is the context when the agent uses it?

        Each dict should have ``"created_at"`` and ``"used_at"`` (epoch
        seconds).  Returns the average age in seconds.
        """
        ages: List[float] = []
        for ctx in contexts:
            created = ctx.get("created_at")
            used = ctx.get("used_at")
            if created is not None and used is not None:
                age = float(used) - float(created)
                if age >= 0:
                    ages.append(age)

        if not ages:
            return EvalResult(
                metric_name="context_freshness",
                value=0.0,
                details={"context_count": 0},
            )

        mean_age = sum(ages) / len(ages)
        return EvalResult(
            metric_name="context_freshness",
            value=mean_age,
            details={
                "context_count": len(ages),
                "min_age": min(ages),
                "max_age": max(ages),
            },
        )

    @staticmethod
    def alert_fatigue_score(push_history: List[Dict[str, Any]]) -> EvalResult:
        """Ratio of ignored pushes to total pushes.

        Each dict should have a boolean ``"was_used"`` field.
        A high score means too many alerts are being ignored.
        """
        if not push_history:
            return EvalResult(
                metric_name="alert_fatigue_score",
                value=0.0,
                details={"ignored": 0, "total": 0},
            )

        total = len(push_history)
        ignored = sum(1 for p in push_history if not p.get("was_used", False))
        score = ignored / total

        return EvalResult(
            metric_name="alert_fatigue_score",
            value=score,
            details={"ignored": ignored, "total": total},
        )

    @staticmethod
    def task_success_rate(trajectory: List[TrajectoryStep]) -> EvalResult:
        """Fraction of episodes whose final step has reward > 0.7."""
        if not trajectory:
            return EvalResult(
                metric_name="task_success_rate",
                value=0.0,
                details={"successful": 0, "total_episodes": 0},
            )

        # Group by episode
        episodes: Dict[str, List[TrajectoryStep]] = {}
        for step in trajectory:
            episodes.setdefault(step.episode_id, []).append(step)

        successful = 0
        for ep_id, steps in episodes.items():
            # Sort by timestamp to find the final step
            steps.sort(key=lambda s: s.timestamp)
            final_reward = steps[-1].reward
            if final_reward > 0.7:
                successful += 1

        total_episodes = len(episodes)
        rate = successful / max(total_episodes, 1)
        return EvalResult(
            metric_name="task_success_rate",
            value=rate,
            details={
                "successful": successful,
                "total_episodes": total_episodes,
            },
        )

    @staticmethod
    def environment_stability(
        pre_states: List[Dict[str, Any]],
        post_states: List[Dict[str, Any]],
    ) -> EvalResult:
        """Did the agent's actions improve or degrade the environment?

        Compares error counts and service health across paired
        (pre, post) state snapshots.  Returns a score where:
        - 1.0 = all transitions improved the environment
        - 0.5 = neutral
        - 0.0 = all transitions degraded the environment
        """
        if not pre_states or not post_states:
            return EvalResult(
                metric_name="environment_stability",
                value=0.5,
                details={"pairs": 0},
            )

        pairs = min(len(pre_states), len(post_states))
        improvements = 0
        degradations = 0

        for pre, post in zip(pre_states[:pairs], post_states[:pairs]):
            pre_errors = _error_count(pre)
            post_errors = _error_count(post)
            pre_healthy = _healthy_count(pre)
            post_healthy = _healthy_count(post)

            delta = (pre_errors - post_errors) + (post_healthy - pre_healthy)
            if delta > 0:
                improvements += 1
            elif delta < 0:
                degradations += 1

        if improvements + degradations == 0:
            score = 0.5
        else:
            score = improvements / (improvements + degradations)

        return EvalResult(
            metric_name="environment_stability",
            value=score,
            details={
                "pairs": pairs,
                "improvements": improvements,
                "degradations": degradations,
                "neutral": pairs - improvements - degradations,
            },
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _hashable(item: Any) -> Any:
    """Convert *item* to a hashable form for set operations."""
    if isinstance(item, dict):
        return tuple(sorted(item.items()))
    if isinstance(item, list):
        return tuple(item)
    return item


def _error_count(state: dict) -> int:
    errors = state.get("errors") or state.get("active_errors")
    if isinstance(errors, list):
        return len(errors)
    if isinstance(errors, (int, float)):
        return int(errors)
    return state.get("error_count", 0)


def _healthy_count(state: dict) -> int:
    services = state.get("services") or state.get("containers") or []
    if not isinstance(services, list):
        return 0
    healthy = {"running", "healthy", "up", "available"}
    return sum(
        1
        for s in services
        if isinstance(s, dict) and s.get("status", "").lower() in healthy
    )
