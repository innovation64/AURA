"""ExecutionGuard — runtime detection of ineffective agent behavior.

Sits between Reason and Act in the AURA pipeline.  Monitors whether
the agent is making progress and intervenes at calibrated levels when
it detects loops, stagnation, or goal drift.

Pipeline position:
    Sense -> Explore -> Scene -> Memory -> Reason -> [ExecutionGuard] -> Act -> Interact
                                                          ^                     |
                                                    FeedbackStore <-- outcome --+
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, FrozenSet, List, Optional, Set, Tuple

from .types import ReasoningResult, SceneState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class InterventionLevel(Enum):
    OBSERVE = 0       # silent, only log
    HINT = 1          # inject informational signal
    SUGGEST = 2       # propose alternative from feedback store
    CONSTRAIN = 3     # prune action space
    REDIRECT = 4      # force replan


class ExplorationPhase(Enum):
    ORIENT = "orient"
    SEARCH = "search"
    CONVERGE = "converge"
    EXECUTE = "execute"
    VERIFY = "verify"
    STUCK = "stuck"


# Per-phase tolerance: higher = more lenient before intervening
PHASE_TOLERANCE: Dict[ExplorationPhase, float] = {
    ExplorationPhase.ORIENT: 0.9,
    ExplorationPhase.SEARCH: 0.7,
    ExplorationPhase.CONVERGE: 0.5,
    ExplorationPhase.EXECUTE: 0.3,
    ExplorationPhase.VERIFY: 0.6,
    ExplorationPhase.STUCK: 0.1,
}


@dataclass
class ActionRecord:
    intent: str
    action_signature: str
    timestamp: float = field(default_factory=time.time)
    tool_name: Optional[str] = None
    outcome_signature: Optional[str] = None
    information_gain: float = 0.0


@dataclass
class GuardVerdict:
    level: InterventionLevel
    urgency: float                        # [0, 1]
    detail: str = ""
    suggested_constraint: str = ""        # for CONSTRAIN / SUGGEST
    suggested_alternative: str = ""       # from feedback store
    confidence: float = 0.0


@dataclass
class StateFingerprint:
    """Compact representation of environment state for change tracking."""
    digest: str
    entities: FrozenSet[str]
    key_values: Dict[str, str]
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# InformationGainEstimator
# ---------------------------------------------------------------------------

class InformationGainEstimator:
    """Estimates the marginal information contributed by each agent step."""

    def __init__(self, window: int = 10):
        self._observations: Set[str] = set()
        self._action_outcome_pairs: Set[Tuple[str, str]] = set()
        self._action_history: Deque[str] = deque(maxlen=window)
        self._state_history: Deque[str] = deque(maxlen=window)
        self._gain_history: Deque[float] = deque(maxlen=window)
        self._window = window

    def estimate(
        self,
        action: str,
        state_before: Optional[SceneState],
        state_after: Optional[SceneState],
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        tool_results = tool_results or []
        gains: List[Tuple[str, float, float]] = []

        # Dimension 1: state change
        state_delta = self._state_divergence(state_before, state_after)
        gains.append(("state_change", state_delta, 0.3))

        # Dimension 2: observation novelty
        new_obs = self._extract_new_observations(tool_results)
        novelty = len(new_obs) / max(len(self._observations) + len(new_obs), 1)
        self._observations.update(new_obs)
        gains.append(("novelty", min(1.0, novelty), 0.3))

        # Dimension 3: hypothesis elimination (failure is also info)
        elim = self._hypothesis_elimination(action, tool_results)
        gains.append(("elimination", elim, 0.2))

        # Dimension 4: action diversity
        act_nov = self._action_novelty(action)
        gains.append(("action_diversity", act_nov, 0.2))

        total = sum(w * v for _, v, w in gains)
        self._action_history.append(self._sig(action))
        fp = self._fingerprint_state(state_after) if state_after else ""
        self._state_history.append(fp)
        self._gain_history.append(total)
        return round(total, 4)

    def trend(self, last_n: int = 5) -> float:
        """Return slope of recent information gain (negative = declining)."""
        values = list(self._gain_history)[-last_n:]
        if len(values) < 2:
            return 0.0
        n = len(values)
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den else 0.0

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _sig(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:8]

    def _state_divergence(
        self, before: Optional[SceneState], after: Optional[SceneState]
    ) -> float:
        if before is None or after is None:
            return 0.5
        fp_a = self._fingerprint_state(before)
        fp_b = self._fingerprint_state(after)
        if fp_a == fp_b:
            return 0.0
        # rough jaccard on entities
        e_a = set(before.entities)
        e_b = set(after.entities)
        if not e_a and not e_b:
            return 0.3 if before.summary != after.summary else 0.0
        jaccard = len(e_a & e_b) / max(len(e_a | e_b), 1)
        return round(1.0 - jaccard, 4)

    @staticmethod
    def _fingerprint_state(state: SceneState) -> str:
        raw = state.summary + "|".join(sorted(state.entities))
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _extract_new_observations(self, results: List[Dict[str, Any]]) -> Set[str]:
        new: Set[str] = set()
        for r in results:
            for k, v in r.items():
                token = f"{k}={str(v)[:60]}"
                if token not in self._observations:
                    new.add(token)
        return new

    def _hypothesis_elimination(self, action: str, results: List[Dict[str, Any]]) -> float:
        outcome_sig = self._sig(str(results)[:200])
        pair = (self._sig(action), outcome_sig)
        if pair not in self._action_outcome_pairs:
            self._action_outcome_pairs.add(pair)
            return 0.7
        return 0.0

    def _action_novelty(self, action: str) -> float:
        sig = self._sig(action)
        recent = list(self._action_history)
        if not recent:
            return 1.0
        repeats = sum(1 for a in recent if a == sig)
        return max(0.0, 1.0 - repeats / len(recent))


# ---------------------------------------------------------------------------
# ExplorationPhaseDetector
# ---------------------------------------------------------------------------

class ExplorationPhaseDetector:
    """Infer which exploration phase the agent is currently in."""

    def detect(
        self,
        action_history: List[ActionRecord],
        gain_estimator: InformationGainEstimator,
    ) -> ExplorationPhase:
        if len(action_history) <= 2:
            return ExplorationPhase.ORIENT

        recent = action_history[-5:]
        tool_names = [a.tool_name for a in recent if a.tool_name]
        tool_diversity = len(set(tool_names))
        intents = [a.intent for a in recent]
        intent_diversity = len(set(intents)) / max(len(intents), 1)

        gains = [a.information_gain for a in recent]
        avg_gain = sum(gains) / len(gains) if gains else 0.0
        trend = gain_estimator.trend(len(recent))

        # verification pattern: look, check, confirm type intents
        verify_kws = ("verify", "check", "confirm", "test", "validate", "assert")
        verify_count = sum(
            1 for i in intents if any(k in i.lower() for k in verify_kws)
        )
        if verify_count >= len(recent) * 0.6:
            return ExplorationPhase.VERIFY

        if tool_diversity > 3 and intent_diversity > 0.6:
            return ExplorationPhase.SEARCH

        if tool_diversity <= 2 and intent_diversity < 0.4:
            if avg_gain > 0.2:
                return ExplorationPhase.EXECUTE
            else:
                return ExplorationPhase.STUCK

        if trend < -0.05 and avg_gain < 0.3:
            return ExplorationPhase.STUCK

        return ExplorationPhase.CONVERGE


# ---------------------------------------------------------------------------
# ConfidenceAccumulator
# ---------------------------------------------------------------------------

class ConfidenceAccumulator:
    """Accumulates intervention confidence; only fires when threshold met."""

    def __init__(self, trigger_threshold: float = 0.7, decay_rate: float = 0.15):
        self.trigger_threshold = trigger_threshold
        self.decay_rate = decay_rate
        self.accumulated: float = 0.0

    def update(self, signal: float) -> bool:
        if signal > 0.5:
            self.accumulated = min(1.0, self.accumulated + signal * 0.3)
        else:
            self.accumulated = max(0.0, self.accumulated - self.decay_rate)
        return self.accumulated >= self.trigger_threshold

    def reset(self) -> None:
        self.accumulated = 0.0


# ---------------------------------------------------------------------------
# InterventionOutcomeTracker — self-calibrating thresholds
# ---------------------------------------------------------------------------

@dataclass
class InterventionRecord:
    urgency: float
    level: InterventionLevel
    outcome_improved: bool = False
    timestamp: float = field(default_factory=time.time)


class InterventionOutcomeTracker:
    """Tracks whether past interventions actually helped, and calibrates."""

    def __init__(self, lookback: int = 20):
        self._records: Deque[InterventionRecord] = deque(maxlen=lookback)
        self._lookback = lookback

    def record(self, urgency: float, level: InterventionLevel) -> int:
        """Record an intervention, return its index for later outcome update."""
        self._records.append(InterventionRecord(urgency=urgency, level=level))
        return len(self._records) - 1

    def record_outcome(self, improved: bool) -> None:
        """Mark the most recent intervention's outcome."""
        if self._records:
            self._records[-1].outcome_improved = improved

    def calibrated_threshold(self, default: float = 0.7) -> float:
        if len(self._records) < 5:
            return default
        recent = list(self._records)[-self._lookback:]
        success_rate = sum(1 for r in recent if r.outcome_improved) / len(recent)
        # high success → lower threshold (intervene more freely)
        # low success  → higher threshold (intervene less)
        return round(0.5 + (1.0 - success_rate) * 0.4, 3)


# ---------------------------------------------------------------------------
# ExecutionGuard — the main class
# ---------------------------------------------------------------------------

class ExecutionGuard:
    """Runtime monitor that detects loops, stagnation, and drift.

    Computes an urgency score each step and maps it to an intervention
    level on a continuous spectrum from OBSERVE to REDIRECT.
    """

    def __init__(
        self,
        window_size: int = 8,
        base_threshold: float = 0.7,
        max_history: int = 50,
    ):
        self.window_size = window_size
        self.gain_estimator = InformationGainEstimator(window=window_size)
        self.phase_detector = ExplorationPhaseDetector()
        self.confidence_acc = ConfidenceAccumulator(trigger_threshold=base_threshold)
        self.outcome_tracker = InterventionOutcomeTracker()

        self._action_history: Deque[ActionRecord] = deque(maxlen=max_history)
        self._state_fingerprints: Deque[str] = deque(maxlen=max_history)
        self._step_count: int = 0
        self._total_budget: int = 100  # default, can be overridden
        self._original_goal: str = ""
        self._interventions_this_episode: int = 0

    def set_budget(self, total_steps: int) -> None:
        self._total_budget = max(1, total_steps)

    def set_goal(self, goal: str) -> None:
        self._original_goal = goal

    def reset(self) -> None:
        self._action_history.clear()
        self._state_fingerprints.clear()
        self._step_count = 0
        self._interventions_this_episode = 0
        self.gain_estimator = InformationGainEstimator(window=self.window_size)
        self.confidence_acc.reset()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def check(
        self,
        reasoning: ReasoningResult,
        state_before: Optional[SceneState],
        state_after: Optional[SceneState],
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> GuardVerdict:
        """Evaluate whether the agent should proceed, be nudged, or redirected."""
        self._step_count += 1
        tool_results = tool_results or []

        # 1. Compute information gain — focus on the *reasoning intent*,
        #    not on ambient exploration noise from probes/tools
        ig = self.gain_estimator.estimate(
            reasoning.intent, state_before, state_after,
            [],  # exclude tool_results to avoid exploration masking loops
        )

        # 2. Record action
        record = ActionRecord(
            intent=reasoning.intent,
            action_signature=hashlib.md5(reasoning.intent.encode()).hexdigest()[:8],
            tool_name=reasoning.actions[0] if reasoning.actions else None,
            information_gain=ig,
        )
        self._action_history.append(record)

        # 3. Detect phase
        phase = self.phase_detector.detect(
            list(self._action_history), self.gain_estimator,
        )

        # 4. Compute urgency
        urgency = self._compute_urgency(ig, phase)

        # 5. Accumulate confidence
        triggered = self.confidence_acc.update(urgency)

        # 6. Determine intervention level
        if not triggered:
            return GuardVerdict(
                level=InterventionLevel.OBSERVE,
                urgency=urgency,
                confidence=self.confidence_acc.accumulated,
            )

        level = self._urgency_to_level(self.confidence_acc.accumulated)
        detail = self._build_detail(ig, phase, level)

        self._interventions_this_episode += 1
        self.outcome_tracker.record(urgency, level)

        # Dynamically adjust threshold
        new_thresh = self.outcome_tracker.calibrated_threshold()
        self.confidence_acc.trigger_threshold = new_thresh

        return GuardVerdict(
            level=level,
            urgency=urgency,
            detail=detail,
            confidence=self.confidence_acc.accumulated,
        )

    def record_outcome(self, improved: bool) -> None:
        """After an intervention, record whether it helped."""
        self.outcome_tracker.record_outcome(improved)
        if improved:
            self.confidence_acc.reset()

    # ------------------------------------------------------------------
    # Urgency computation
    # ------------------------------------------------------------------

    def _compute_urgency(self, ig: float, phase: ExplorationPhase) -> float:
        # Factor 1: information gain trend (declining is bad)
        ig_trend = self.gain_estimator.trend()
        ig_factor = max(0.0, -ig_trend) * 2.0

        # Factor 2: resource burn rate
        budget_used = self._step_count / max(self._total_budget, 1)
        progress_est = self._estimate_progress()
        resource_factor = max(0.0, budget_used - progress_est)

        # Factor 3: repetition pattern strength
        pattern_strength = self._detect_pattern_strength()

        # Factor 4: phase tolerance
        tolerance = PHASE_TOLERANCE.get(phase, 0.5)

        # Factor 5: absolute information gain (near-zero is suspicious)
        ig_abs_factor = max(0.0, 0.3 - ig) / 0.3 if ig < 0.3 else 0.0

        raw = (
            0.25 * ig_factor
            + 0.20 * resource_factor
            + 0.25 * pattern_strength
            + 0.10 * (1.0 - tolerance)
            + 0.20 * ig_abs_factor
        )
        return round(min(1.0, max(0.0, raw)), 4)

    def _estimate_progress(self) -> float:
        """Rough progress estimate from cumulative information gain."""
        if not self._action_history:
            return 0.0
        total_ig = sum(a.information_gain for a in self._action_history)
        # Normalize: assume "complete" is ~0.5 gain per step on average
        expected_total = self._step_count * 0.5
        if expected_total <= 0:
            return 0.0
        return min(1.0, total_ig / expected_total)

    def _detect_pattern_strength(self) -> float:
        """Detect how strongly the recent actions form a repeating pattern."""
        recent = [a.action_signature for a in list(self._action_history)[-self.window_size:]]
        if len(recent) < 3:
            return 0.0

        # Exact repetition: A-A-A
        if len(set(recent)) == 1:
            return 1.0

        # Periodic repetition: A-B-A-B
        n = len(recent)
        for period in range(2, n // 2 + 1):
            pattern = recent[:period]
            matches = sum(
                1 for i in range(n) if recent[i] == pattern[i % period]
            )
            if matches == n:
                return 0.9

        # High repetition ratio
        counts = Counter(recent)
        most_common_ratio = counts.most_common(1)[0][1] / n
        if most_common_ratio > 0.7:
            return 0.7

        return max(0.0, most_common_ratio - 0.3)

    # ------------------------------------------------------------------
    # Level mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _urgency_to_level(confidence: float) -> InterventionLevel:
        if confidence < 0.5:
            return InterventionLevel.HINT
        if confidence < 0.7:
            return InterventionLevel.SUGGEST
        if confidence < 0.9:
            return InterventionLevel.CONSTRAIN
        return InterventionLevel.REDIRECT

    @staticmethod
    def _build_detail(ig: float, phase: ExplorationPhase, level: InterventionLevel) -> str:
        parts = [
            f"phase={phase.value}",
            f"info_gain={ig:.3f}",
            f"intervention={level.name}",
        ]
        return "; ".join(parts)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        gains = [a.information_gain for a in self._action_history]
        return {
            "steps": self._step_count,
            "interventions": self._interventions_this_episode,
            "avg_information_gain": sum(gains) / max(len(gains), 1),
            "ig_trend": self.gain_estimator.trend(),
            "pattern_strength": self._detect_pattern_strength(),
            "accumulated_confidence": self.confidence_acc.accumulated,
            "calibrated_threshold": self.outcome_tracker.calibrated_threshold(),
        }
