"""StrategyAuditor — proactively detects stale strategies and explores alternatives.

Three detection mechanisms:
1. Time decay — confidence decays naturally, forcing periodic revalidation
2. Environment drift — marks strategies when the world changes
3. Counterfactual probing — spends a budget fraction testing alternatives

Integrates with ConditionalFeedbackStore for strategy records and with
ExecutionGuard for runtime intervention signals.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from .feedback import (
    ConditionalFeedbackStore,
    FeedbackEntry,
    Outcome,
    StatePattern,
    extract_pattern,
)
from .types import SceneState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AdaptiveExplorationRate
# ---------------------------------------------------------------------------

class AdaptiveExplorationRate:
    """Adjusts exploration rate based on environment stability."""

    def __init__(self, base_rate: float = 0.1):
        self.base_rate = base_rate
        self._surprises: Deque[float] = deque(maxlen=50)
        self._stability: float = 0.5

    def update(self, expected: float, actual: float) -> None:
        surprise = abs(expected - actual)
        self._surprises.append(surprise)
        avg = sum(self._surprises) / len(self._surprises)
        self._stability = max(0.0, min(1.0, 1.0 - avg))

    @property
    def stability(self) -> float:
        return self._stability

    @property
    def current_rate(self) -> float:
        # stability 1.0 -> rate * 0.2  (very stable, explore little)
        # stability 0.0 -> rate * 3.0  (unstable, explore a lot)
        multiplier = 0.2 + 2.8 * (1.0 - self._stability)
        return min(0.5, self.base_rate * multiplier)


# ---------------------------------------------------------------------------
# StrategyRecord — augmented view over FeedbackEntry
# ---------------------------------------------------------------------------

@dataclass
class StrategyRecord:
    entry_id: str
    condition: StatePattern
    action: str
    learned_at: float
    last_validated: float
    last_used: float
    success_count: int
    failure_count: int
    total_reward: float
    environment_version: str
    confidence: float
    effective_confidence: float = 0.0

    @staticmethod
    def from_feedback_entry(entry: FeedbackEntry) -> "StrategyRecord":
        return StrategyRecord(
            entry_id=entry.entry_id,
            condition=entry.condition,
            action=entry.attempted_action,
            learned_at=entry.learned_at,
            last_validated=entry.last_validated,
            last_used=entry.last_used,
            success_count=entry.times_confirmed if entry.outcome == Outcome.SUCCESS else 0,
            failure_count=entry.times_confirmed if entry.outcome != Outcome.SUCCESS else 0,
            total_reward=entry.total_reward,
            environment_version=entry.environment_version,
            confidence=entry.confidence,
        )


# ---------------------------------------------------------------------------
# ComparisonRecord — A/B test result
# ---------------------------------------------------------------------------

@dataclass
class ComparisonRecord:
    strategy_id: str
    original_action: str
    alternative_action: str
    original_outcome: Outcome
    alternative_outcome: Outcome
    original_reward: float
    alternative_reward: float
    timestamp: float = field(default_factory=time.time)

    @property
    def alternative_won(self) -> bool:
        if self.alternative_outcome == Outcome.SUCCESS and self.original_outcome != Outcome.SUCCESS:
            return True
        return self.alternative_reward > self.original_reward * 1.1  # 10% margin


# ---------------------------------------------------------------------------
# StrategyAuditor
# ---------------------------------------------------------------------------

class StrategyAuditor:
    """Proactively detects stale strategies and tests alternatives.

    Parameters
    ----------
    feedback_store : ConditionalFeedbackStore
        The backing store for strategy records.
    staleness_halflife : float
        Seconds for confidence to halve.  Default 86400 (1 day).
    revalidation_budget : float
        Fraction of steps to reserve for counterfactual probing.  Default 0.1.
    drift_threshold : float
        Minimum environment drift score to trigger revalidation.  Default 0.3.
    """

    def __init__(
        self,
        feedback_store: ConditionalFeedbackStore,
        staleness_halflife: float = 86400.0,
        revalidation_budget: float = 0.1,
        drift_threshold: float = 0.3,
    ):
        self.store = feedback_store
        self.staleness_halflife = staleness_halflife
        self.revalidation_budget = revalidation_budget
        self.drift_threshold = drift_threshold

        self.exploration_rate = AdaptiveExplorationRate(base_rate=revalidation_budget)
        self._revalidation_queue: Set[str] = set()
        self._comparison_history: Deque[ComparisonRecord] = deque(maxlen=100)
        self._env_fingerprint: str = ""
        self._step_count: int = 0
        self._explore_count: int = 0

    # ------------------------------------------------------------------
    # 1. Time decay
    # ------------------------------------------------------------------

    def effective_confidence(self, entry: FeedbackEntry) -> float:
        """Confidence decayed by time since last validation."""
        age = time.time() - entry.last_validated
        if self.staleness_halflife <= 0:
            return entry.confidence
        decay = 2.0 ** (-age / self.staleness_halflife)
        return round(entry.confidence * decay, 4)

    def get_stale_strategies(self, threshold: float = 0.3) -> List[FeedbackEntry]:
        """Return strategies whose effective confidence has dropped below threshold."""
        stale: List[FeedbackEntry] = []
        for entry in self.store.get_all_entries():
            if entry.outcome != Outcome.SUCCESS:
                continue
            ec = self.effective_confidence(entry)
            if ec < threshold:
                stale.append(entry)
                self._revalidation_queue.add(entry.entry_id)
        return stale

    # ------------------------------------------------------------------
    # 2. Environment drift detection
    # ------------------------------------------------------------------

    def check_environment_drift(
        self,
        old_state: Optional[SceneState],
        new_state: SceneState,
    ) -> Dict[str, float]:
        """Compare two environment states and flag affected strategies.

        Returns a dict of entry_id -> drift_score for entries that exceed
        the drift threshold.
        """
        if old_state is None:
            return {}

        old_pattern = extract_pattern(old_state)
        new_pattern = extract_pattern(new_state)

        # Overall drift: 1 - similarity
        drift = 1.0 - old_pattern.similarity(new_pattern)
        if drift < self.drift_threshold:
            return {}

        # Find which strategies are affected
        affected: Dict[str, float] = {}
        for entry in self.store.get_all_entries():
            if entry.outcome != Outcome.SUCCESS:
                continue
            # How much does this strategy's condition overlap with the changed domains?
            condition_sim_old = entry.condition.similarity(old_pattern)
            condition_sim_new = entry.condition.similarity(new_pattern)
            impact = abs(condition_sim_old - condition_sim_new)

            if impact > 0.1:
                affected[entry.entry_id] = round(impact, 4)
                self._revalidation_queue.add(entry.entry_id)
                # Reduce confidence proportionally
                entry.confidence = max(0.1, entry.confidence * (1.0 - impact * 0.5))

        # Update exploration rate with drift as "surprise"
        self.exploration_rate.update(0.0, drift)

        return affected

    # ------------------------------------------------------------------
    # 3. Counterfactual probing
    # ------------------------------------------------------------------

    def should_probe(self, current_entry: Optional[FeedbackEntry] = None) -> bool:
        """Should we spend this step testing an alternative?"""
        self._step_count += 1
        rate = self.exploration_rate.current_rate

        # Force probe if entry is in revalidation queue
        if current_entry and current_entry.entry_id in self._revalidation_queue:
            return True

        # UCB-inspired decision
        if current_entry:
            n_uses = current_entry.times_confirmed
            ucb_bonus = math.sqrt(
                math.log(self._step_count + 1) / max(n_uses, 1)
            )
            ec = self.effective_confidence(current_entry)
            explore_value = 0.5 + ucb_bonus
            if explore_value > ec:
                return True

        # Budget-based: explore at the adaptive rate
        if self._step_count > 0:
            actual_rate = self._explore_count / self._step_count
            if actual_rate < rate:
                return True

        return False

    def select_alternative(
        self,
        scene_state: SceneState,
        current_action: str,
    ) -> Optional[str]:
        """Pick the most informative alternative to test."""
        alternatives = self.store.query_alternatives(
            scene_state, exclude_action=current_action, limit=5,
        )

        # Prefer alternatives that haven't been tested recently
        if alternatives:
            return alternatives[0]

        # If nothing in store, check revalidation queue for entries
        # whose action differs from current
        for eid in list(self._revalidation_queue):
            entry = self.store.get_entry(eid)
            if entry and entry.attempted_action != current_action:
                return entry.attempted_action

        return None

    def record_comparison(
        self,
        entry_id: str,
        original_action: str,
        alternative_action: str,
        original_outcome: Outcome,
        alternative_outcome: Outcome,
        original_reward: float = 0.0,
        alternative_reward: float = 0.0,
    ) -> bool:
        """Record A/B comparison result.  Returns True if alternative won."""
        self._explore_count += 1
        rec = ComparisonRecord(
            strategy_id=entry_id,
            original_action=original_action,
            alternative_action=alternative_action,
            original_outcome=original_outcome,
            alternative_outcome=alternative_outcome,
            original_reward=original_reward,
            alternative_reward=alternative_reward,
        )
        self._comparison_history.append(rec)

        # Remove from revalidation queue
        self._revalidation_queue.discard(entry_id)

        if rec.alternative_won:
            # Update the store: demote original, promote alternative
            original_entry = self.store.get_entry(entry_id)
            if original_entry:
                original_entry.confidence = max(0.1, original_entry.confidence * 0.5)
                original_entry.alternative_action = alternative_action

            logger.info(
                "Strategy %s superseded: '%s' -> '%s'",
                entry_id, original_action, alternative_action,
            )
            return True

        # Original held up — refresh its validation timestamp
        original_entry = self.store.get_entry(entry_id)
        if original_entry:
            original_entry.last_validated = time.time()
            original_entry.confidence = min(1.0, original_entry.confidence + 0.05)

        return False

    # ------------------------------------------------------------------
    # Audit sweep — run periodically
    # ------------------------------------------------------------------

    def audit(self, current_state: SceneState) -> Dict[str, Any]:
        """Run a full audit sweep.  Returns summary of findings."""
        stale = self.get_stale_strategies()
        revalidation_count = len(self._revalidation_queue)

        # Check for strategies that have never been validated
        # (learned long ago but never re-tested)
        never_validated: List[str] = []
        now = time.time()
        for entry in self.store.get_all_entries():
            if entry.outcome == Outcome.SUCCESS:
                age = now - entry.learned_at
                since_validated = now - entry.last_validated
                if age > self.staleness_halflife and since_validated > self.staleness_halflife * 0.5:
                    never_validated.append(entry.entry_id)
                    self._revalidation_queue.add(entry.entry_id)

        return {
            "stale_count": len(stale),
            "revalidation_queue_size": revalidation_count + len(never_validated),
            "never_revalidated": len(never_validated),
            "exploration_rate": self.exploration_rate.current_rate,
            "environment_stability": self.exploration_rate.stability,
            "comparisons_total": len(self._comparison_history),
            "comparisons_alternative_won": sum(
                1 for c in self._comparison_history if c.alternative_won
            ),
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "step_count": self._step_count,
            "explore_count": self._explore_count,
            "actual_explore_rate": (
                self._explore_count / max(self._step_count, 1)
            ),
            "target_explore_rate": self.exploration_rate.current_rate,
            "stability": self.exploration_rate.stability,
            "revalidation_queue": len(self._revalidation_queue),
            "staleness_halflife": self.staleness_halflife,
        }
