"""Scores how relevant a change event is to the current agent task.

Combines multiple relevance dimensions (source match, keyword overlap,
agent-type affinity, severity boost, recency) into a single ``[0, 1]``
score that downstream components use for filtering and prioritisation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from aura.proactive.change_detector import ChangeEvent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TaskContext:
    """Describes what the agent is currently doing so we can gauge relevance."""

    task_description: str = ""
    agent_type: str = "default"  # "coder" | "sysadmin" | "researcher" | "default"
    active_files: List[str] = field(default_factory=list)
    active_services: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    history: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent-type affinity maps: probe source substrings -> weight [0,1]
# ---------------------------------------------------------------------------

_AGENT_AFFINITIES: Dict[str, Dict[str, float]] = {
    "sysadmin": {
        "system": 1.0, "docker": 1.0, "process": 0.9, "cpu": 0.9,
        "memory": 0.9, "disk": 0.9, "network": 0.8, "service": 0.8,
        "container": 1.0, "load": 0.9, "uptime": 0.7,
    },
    "coder": {
        "git": 1.0, "filesystem": 1.0, "file": 1.0, "test": 0.9,
        "lint": 0.9, "build": 0.9, "compile": 0.9, "editor": 0.8,
        "diff": 0.8, "process": 0.5,
    },
    "researcher": {
        "network": 1.0, "service": 1.0, "http": 0.9, "api": 0.9,
        "database": 0.8, "search": 0.8, "crawl": 0.7, "dns": 0.7,
    },
    "default": {},  # no special affinity
}

# Default dimension weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "source_relevance": 0.30,
    "keyword_overlap": 0.20,
    "agent_type": 0.20,
    "severity_boost": 0.15,
    "recency": 0.15,
}


# ---------------------------------------------------------------------------
# RelevanceScorer
# ---------------------------------------------------------------------------

class RelevanceScorer:
    """Scores the relevance of a :class:`ChangeEvent` to a :class:`TaskContext`.

    Parameters
    ----------
    weights:
        Override default per-dimension weights.  Keys must be from
        ``{"source_relevance", "keyword_overlap", "agent_type",
        "severity_boost", "recency"}``.
    recency_halflife:
        Seconds after which a change's recency score decays to 0.5.
        Default is 120 seconds (2 minutes).
    extra_source_weights:
        Optional external weights (e.g. from :class:`AttentionTracker`)
        that are blended into source relevance.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        recency_halflife: float = 120.0,
        extra_source_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.weights: Dict[str, float] = {**_DEFAULT_WEIGHTS}
        if weights:
            self.weights.update(weights)
        self.recency_halflife = recency_halflife
        self.extra_source_weights: Dict[str, float] = extra_source_weights or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, event: ChangeEvent, task_context: TaskContext) -> float:
        """Return a relevance score in ``[0, 1]``."""
        dimensions: Dict[str, float] = {
            "source_relevance": self._score_source(event, task_context),
            "keyword_overlap": self._score_keywords(event, task_context),
            "agent_type": self._score_agent_type(event, task_context),
            "severity_boost": self._score_severity(event),
            "recency": self._score_recency(event),
        }
        total = sum(self.weights[dim] * dimensions[dim] for dim in dimensions)
        # Normalise by sum of weights (allows partial weight overrides)
        weight_sum = sum(self.weights[dim] for dim in dimensions)
        if weight_sum <= 0:
            return 0.0
        return round(min(1.0, max(0.0, total / weight_sum)), 4)

    def update_source_weights(self, weights: Dict[str, float]) -> None:
        """Hot-update external source weights (e.g. from attention tracker)."""
        self.extra_source_weights = weights

    # ------------------------------------------------------------------
    # Dimension scorers
    # ------------------------------------------------------------------

    def _score_source(self, event: ChangeEvent, ctx: TaskContext) -> float:
        """Source relevance: does the change match the agent's active area?

        Learned source weights from AttentionTracker are applied as
        multipliers (not max), so low-trust sources are actively
        suppressed and high-trust sources are amplified.
        """
        source_lower = event.source.lower()
        desc_lower = event.description.lower()

        score = 0.3  # baseline for system-level events

        # Check active files
        for fp in ctx.active_files:
            if fp in event.description or fp in source_lower:
                score = 1.0
                break
            parts = fp.rsplit("/", 1)
            fname = parts[-1] if parts else fp
            if fname and fname in desc_lower:
                score = max(score, 0.85)

        # Check active services
        for svc in ctx.active_services:
            svc_lower = svc.lower()
            if svc_lower in source_lower or svc_lower in desc_lower:
                score = max(score, 0.95)

        # Apply learned source weights as multipliers (not max)
        # Weight > 1.0 = agent trusts this source → amplify
        # Weight < 1.0 = agent ignores this source → suppress
        if self.extra_source_weights:
            for key, w in self.extra_source_weights.items():
                if key.lower() in source_lower:
                    score = score * w
                    break

        return min(1.0, max(0.0, score))

    def _score_keywords(self, event: ChangeEvent, ctx: TaskContext) -> float:
        """Keyword overlap between event description and task keywords."""
        if not ctx.keywords:
            return 0.0
        desc_lower = event.description.lower()
        source_lower = event.source.lower()
        combined = f"{desc_lower} {source_lower}"
        # Also include context values
        for v in event.context.values():
            combined += f" {str(v).lower()}"

        matches = sum(1 for kw in ctx.keywords if kw.lower() in combined)
        if not matches:
            return 0.0
        return min(1.0, matches / max(len(ctx.keywords), 1))

    def _score_agent_type(self, event: ChangeEvent, ctx: TaskContext) -> float:
        """Agent-type affinity: does this probe matter to this kind of agent?"""
        affinities = _AGENT_AFFINITIES.get(ctx.agent_type, {})
        if not affinities:
            return 0.5  # neutral for unknown types
        source_lower = event.source.lower()
        desc_lower = event.description.lower()
        best = 0.0
        for keyword, weight in affinities.items():
            if keyword in source_lower or keyword in desc_lower:
                best = max(best, weight)
        return best

    @staticmethod
    def _score_severity(event: ChangeEvent) -> float:
        """High severity events get a direct relevance boost."""
        return event.severity

    def _score_recency(self, event: ChangeEvent) -> float:
        """Exponential decay based on event age."""
        age = max(0.0, time.time() - event.timestamp)
        if self.recency_halflife <= 0:
            return 1.0 if age == 0 else 0.0
        # Exponential decay: score = 2^(-age / halflife)
        return 2.0 ** (-age / self.recency_halflife)
