"""ConditionalFeedbackStore — stores and retrieves experience-based advice.

Records (condition, action, outcome) triples so that the ExecutionGuard
can query "in this situation, has this action worked before?" and suggest
alternatives when it detects the agent is stuck.

Also provides the foundation for StrategyAuditor to detect stale strategies.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, FrozenSet, List, Optional, Set, Tuple

from .types import SceneState

logger = logging.getLogger(__name__)

_STOP = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in on at to for "
    "with by from as into about and or but not no nor so yet it its "
    "i me my we our you your he him she her they them their".split()
)


def _tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in _STOP and len(w) > 1]


# ---------------------------------------------------------------------------
# Outcome enum
# ---------------------------------------------------------------------------

class Outcome(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    STALL = "stall"
    LOOP = "loop"
    PARTIAL = "partial"


# ---------------------------------------------------------------------------
# StatePattern — structured environment fingerprint for matching
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StatePattern:
    signal_types: FrozenSet[str]
    anomaly_categories: FrozenSet[str]
    resource_pressure: str              # "low" | "medium" | "high"
    active_entities: FrozenSet[str]
    error_signatures: FrozenSet[str]

    def similarity(self, other: "StatePattern") -> float:
        scores: List[float] = []
        for attr in ("signal_types", "anomaly_categories", "active_entities", "error_signatures"):
            a: frozenset = getattr(self, attr)
            b: frozenset = getattr(other, attr)
            if not a and not b:
                scores.append(1.0)
            else:
                scores.append(len(a & b) / max(len(a | b), 1))
        scores.append(1.0 if self.resource_pressure == other.resource_pressure else 0.3)
        return sum(scores) / len(scores)


def extract_pattern(scene: SceneState) -> StatePattern:
    """Build a StatePattern from a SceneState."""
    ctx = scene.context or {}
    summary_lower = scene.summary.lower()

    # signal types
    signal_types: Set[str] = set()
    for key in ctx:
        signal_types.add(key.split(".")[0] if "." in key else key)
    if not signal_types:
        for kw, st in [("system", "system"), ("docker", "docker"),
                        ("git", "git"), ("file", "filesystem"),
                        ("service", "network"), ("process", "process")]:
            if kw in summary_lower:
                signal_types.add(st)

    # anomalies
    anomalies: Set[str] = set()
    anomaly_kws = {
        "cpu_high": ("cpu", "load"),
        "memory_high": ("memory", "oom"),
        "disk_full": ("disk", "storage"),
        "service_down": ("down", "unreachable", "refused"),
        "error": ("error", "exception", "fail", "crash"),
    }
    for cat, keywords in anomaly_kws.items():
        if any(k in summary_lower for k in keywords):
            anomalies.add(cat)

    # resource pressure
    cpu = ctx.get("cpu", ctx.get("cpu_percent", 0))
    mem = ctx.get("memory", ctx.get("memory_percent", 0))
    if isinstance(cpu, (int, float)) and isinstance(mem, (int, float)):
        avg = (cpu + mem) / 2
        pressure = "high" if avg > 70 else ("medium" if avg > 40 else "low")
    else:
        pressure = "low"

    # error signatures
    errors: Set[str] = set()
    for val in ctx.values():
        s = str(val).lower()
        for kw in ("error", "exception", "timeout", "refused", "denied"):
            if kw in s:
                errors.add(kw)

    return StatePattern(
        signal_types=frozenset(signal_types),
        anomaly_categories=frozenset(anomalies),
        resource_pressure=pressure,
        active_entities=frozenset(scene.entities[:10]),
        error_signatures=frozenset(errors),
    )


# ---------------------------------------------------------------------------
# FeedbackEntry & FeedbackAdvice
# ---------------------------------------------------------------------------

@dataclass
class FeedbackEntry:
    entry_id: str
    condition: StatePattern
    attempted_action: str
    outcome: Outcome
    alternative_action: str = ""
    confidence: float = 0.5
    times_confirmed: int = 1
    total_reward: float = 0.0
    learned_at: float = field(default_factory=time.time)
    last_validated: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    environment_version: str = ""


@dataclass
class FeedbackAdvice:
    warning: str
    confidence: float
    suggested_alternative: str
    times_observed: int
    entry_id: str = ""


# ---------------------------------------------------------------------------
# ConditionalFeedbackStore
# ---------------------------------------------------------------------------

class ConditionalFeedbackStore:
    """Stores (condition, action, outcome) triples for experience-based advice."""

    def __init__(self, max_entries: int = 500, similarity_threshold: float = 0.6):
        self._entries: List[FeedbackEntry] = []
        self._max_entries = max_entries
        self._similarity_threshold = similarity_threshold
        self._id_counter = 0

    def record_outcome(
        self,
        scene_state: SceneState,
        action: str,
        outcome: Outcome,
        reward: float = 0.0,
        alternative: str = "",
        env_version: str = "",
    ) -> str:
        """Record the outcome of an action in a given state."""
        pattern = extract_pattern(scene_state)

        # Look for existing entry with similar condition + same action
        existing = self._find_matching(pattern, action)
        if existing is not None:
            existing.times_confirmed += 1
            existing.total_reward += reward
            existing.last_validated = time.time()
            # Update outcome if new evidence is stronger
            if outcome in (Outcome.FAILURE, Outcome.LOOP):
                existing.outcome = outcome
            if alternative and not existing.alternative_action:
                existing.alternative_action = alternative
            existing.confidence = min(1.0, existing.confidence + 0.1)
            return existing.entry_id

        # Create new entry
        self._id_counter += 1
        eid = f"fb_{self._id_counter}"
        entry = FeedbackEntry(
            entry_id=eid,
            condition=pattern,
            attempted_action=action,
            outcome=outcome,
            alternative_action=alternative,
            confidence=0.5,
            total_reward=reward,
            environment_version=env_version,
        )
        self._entries.append(entry)

        # Evict oldest low-confidence entries if over capacity
        if len(self._entries) > self._max_entries:
            self._entries.sort(key=lambda e: e.confidence * e.times_confirmed, reverse=True)
            self._entries = self._entries[: self._max_entries]

        return eid

    def query_advice(
        self,
        scene_state: SceneState,
        proposed_action: str,
    ) -> Optional[FeedbackAdvice]:
        """Check if the proposed action has historically failed in similar states."""
        pattern = extract_pattern(scene_state)
        match = self._find_matching(pattern, proposed_action)

        if match is None:
            return None

        if match.outcome in (Outcome.FAILURE, Outcome.LOOP, Outcome.STALL):
            return FeedbackAdvice(
                warning=f"Action '{proposed_action}' previously resulted in {match.outcome.value} "
                        f"in similar conditions ({match.times_confirmed} observations)",
                confidence=match.confidence,
                suggested_alternative=match.alternative_action,
                times_observed=match.times_confirmed,
                entry_id=match.entry_id,
            )
        return None

    def query_alternatives(
        self,
        scene_state: SceneState,
        exclude_action: str = "",
        limit: int = 3,
    ) -> List[str]:
        """Find actions that succeeded in similar states."""
        pattern = extract_pattern(scene_state)
        candidates: List[Tuple[float, str]] = []

        for entry in self._entries:
            if entry.outcome != Outcome.SUCCESS:
                continue
            if entry.attempted_action == exclude_action:
                continue
            sim = pattern.similarity(entry.condition)
            if sim >= self._similarity_threshold:
                score = sim * entry.confidence * entry.times_confirmed
                candidates.append((score, entry.attempted_action))

        candidates.sort(reverse=True)
        seen: Set[str] = set()
        result: List[str] = []
        for _, action in candidates:
            if action not in seen:
                seen.add(action)
                result.append(action)
            if len(result) >= limit:
                break
        return result

    def query_entry(
        self,
        scene_state: SceneState,
        action: str,
    ) -> Optional[FeedbackEntry]:
        """Find the best-matching entry for a (state, action) pair."""
        pattern = extract_pattern(scene_state)
        return self._find_matching(pattern, action)

    def get_entry(self, entry_id: str) -> Optional[FeedbackEntry]:
        for e in self._entries:
            if e.entry_id == entry_id:
                return e
        return None

    def get_all_entries(self) -> List[FeedbackEntry]:
        return list(self._entries)

    def get_stats(self) -> Dict[str, Any]:
        outcomes = defaultdict(int)
        for e in self._entries:
            outcomes[e.outcome.value] += 1
        return {
            "total_entries": len(self._entries),
            "outcome_distribution": dict(outcomes),
            "avg_confidence": (
                sum(e.confidence for e in self._entries) / max(len(self._entries), 1)
            ),
        }

    # -- internal -----------------------------------------------------------

    def _find_matching(self, pattern: StatePattern, action: str) -> Optional[FeedbackEntry]:
        best: Optional[FeedbackEntry] = None
        best_sim = 0.0
        action_tokens = set(_tokenize(action))

        for entry in self._entries:
            # Quick action similarity check
            entry_tokens = set(_tokenize(entry.attempted_action))
            if action_tokens and entry_tokens:
                token_overlap = len(action_tokens & entry_tokens) / max(len(action_tokens | entry_tokens), 1)
                if token_overlap < 0.3:
                    continue

            sim = pattern.similarity(entry.condition)
            if sim >= self._similarity_threshold and sim > best_sim:
                best_sim = sim
                best = entry

        return best
