"""Reward signal computation from environment feedback."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "task_completion": 0.4,
    "env_improvement": 0.2,
    "efficiency": 0.2,
    "safety": 0.2,
}


class RewardSignal:
    """Computes reward signals from environment feedback.

    The reward is a weighted combination of four components:
    - task_completion:  Did the action move toward the goal?
    - env_improvement:  Did metrics improve (fewer errors, services restored)?
    - efficiency:       Was the action direct / minimal?
    - safety:           Did the action avoid making things worse?
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = dict(_DEFAULT_WEIGHTS)
        if weights is not None:
            for k, v in weights.items():
                if k in self.weights:
                    self.weights[k] = v
                else:
                    logger.warning("Unknown reward component '%s'; ignoring.", k)
        # Normalise so weights sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        pre_state: dict,
        post_state: dict,
        action: str,
        task_goal: str = "",
    ) -> float:
        """Compute a scalar reward in [0, 1]."""
        tc = self._task_completion(action, task_goal, post_state)
        ei = self._env_improvement(pre_state, post_state)
        ef = self._efficiency(action)
        sa = self._safety(pre_state, post_state)

        reward = (
            self.weights["task_completion"] * tc
            + self.weights["env_improvement"] * ei
            + self.weights["efficiency"] * ef
            + self.weights["safety"] * sa
        )
        return max(0.0, min(1.0, reward))

    # ------------------------------------------------------------------
    # Component calculations
    # ------------------------------------------------------------------

    def _task_completion(
        self, action: str, task_goal: str, post_state: dict
    ) -> float:
        """Keyword-overlap heuristic: how much does the action relate to the goal?"""
        if not task_goal:
            return 0.5  # neutral when no goal specified

        goal_tokens = _tokenize(task_goal)
        action_tokens = _tokenize(action)

        if not goal_tokens:
            return 0.5

        overlap = goal_tokens & action_tokens
        score = len(overlap) / len(goal_tokens)

        # Bonus if post_state indicates completion
        if post_state.get("task_complete") or post_state.get("done"):
            score = min(1.0, score + 0.3)

        return min(1.0, score)

    @staticmethod
    def _env_improvement(pre_state: dict, post_state: dict) -> float:
        """Did error counts decrease and service health improve?"""
        score = 0.5  # neutral baseline

        pre_errors = _count_errors(pre_state)
        post_errors = _count_errors(post_state)
        if pre_errors > 0:
            if post_errors < pre_errors:
                score += 0.25 * ((pre_errors - post_errors) / pre_errors)
            elif post_errors > pre_errors:
                score -= 0.25 * ((post_errors - pre_errors) / max(post_errors, 1))

        pre_services_up = _count_healthy_services(pre_state)
        post_services_up = _count_healthy_services(post_state)
        if post_services_up > pre_services_up:
            score += 0.2
        elif post_services_up < pre_services_up:
            score -= 0.2

        return max(0.0, min(1.0, score))

    @staticmethod
    def _efficiency(action: str) -> float:
        """Shorter, more direct actions score higher."""
        length = len(action)
        if length == 0:
            return 0.0
        # Actions under 100 chars get full marks; linear decay up to 500
        if length <= 100:
            return 1.0
        if length >= 500:
            return 0.3
        return 1.0 - 0.7 * ((length - 100) / 400)

    @staticmethod
    def _safety(pre_state: dict, post_state: dict) -> float:
        """Penalise if new errors or service disruptions appeared."""
        score = 1.0  # start safe

        pre_errors = _count_errors(pre_state)
        post_errors = _count_errors(post_state)
        new_errors = max(0, post_errors - pre_errors)
        if new_errors > 0:
            score -= min(0.5, 0.1 * new_errors)

        pre_services = _count_healthy_services(pre_state)
        post_services = _count_healthy_services(post_state)
        lost = max(0, pre_services - post_services)
        if lost > 0:
            score -= min(0.4, 0.2 * lost)

        # Check for explicit danger signals
        if post_state.get("security_alert") and not pre_state.get("security_alert"):
            score -= 0.3

        return max(0.0, min(1.0, score))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_STOP_WORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "and", "but", "or", "nor", "not", "so", "yet",
    "it", "its", "this", "that", "these", "those",
}


def _tokenize(text: str) -> Set[str]:
    """Extract meaningful lowercase tokens from *text*."""
    tokens = set(re.findall(r"[a-z0-9_]+", text.lower()))
    return tokens - _STOP_WORDS


def _count_errors(state: dict) -> int:
    """Heuristic count of errors in a state dictionary."""
    count = 0
    errors = state.get("errors") or state.get("active_errors")
    if isinstance(errors, list):
        count += len(errors)
    elif isinstance(errors, (int, float)):
        count += int(errors)
    error_count = state.get("error_count")
    if isinstance(error_count, (int, float)):
        count += int(error_count)
    return count


def _count_healthy_services(state: dict) -> int:
    """Heuristic count of healthy services in a state dictionary."""
    services = state.get("services") or state.get("containers") or []
    if not isinstance(services, list):
        return 0
    healthy_statuses = {"running", "healthy", "up", "available"}
    return sum(
        1
        for svc in services
        if isinstance(svc, dict)
        and svc.get("status", "").lower() in healthy_statuses
    )
