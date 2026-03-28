"""Plasticity engine — learnable memory dynamics (Hebbian, contrastive, forgetting)."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plasticity Rule interfaces
# ---------------------------------------------------------------------------

class PlasticityRule(Protocol):
    """Interface for a single plasticity rule."""

    def apply(self, memory_state: Dict[str, Any], signal: Dict[str, Any]) -> Dict[str, Any]:
        """Apply this rule to a memory state given a learning signal. Returns updated state."""
        ...


@dataclass
class HebbianRule:
    """
    Hebbian learning: 'neurons that fire together wire together'.

    Strengthens associations between co-occurring memory items.
    When item A and item B are retrieved together, their association weight increases.
    """

    learning_rate: float = 0.01
    decay_rate: float = 0.001

    def apply(self, memory_state: Dict[str, Any], signal: Dict[str, Any]) -> Dict[str, Any]:
        associations = memory_state.get("associations", {})
        co_activated = signal.get("co_activated_ids", [])

        # Strengthen pairwise associations
        for i, id_a in enumerate(co_activated):
            for id_b in co_activated[i + 1:]:
                key = tuple(sorted([id_a, id_b]))
                key_str = f"{key[0]}:{key[1]}"
                current = associations.get(key_str, 0.0)
                associations[key_str] = min(1.0, current + self.learning_rate)

        # Apply decay to all associations
        for key in list(associations.keys()):
            associations[key] = max(0.0, associations[key] - self.decay_rate)
            if associations[key] <= 0.0:
                del associations[key]

        memory_state["associations"] = associations
        return memory_state


@dataclass
class ContrastiveShaping:
    """
    Contrastive learning for memory representations.

    Pulls together representations of related memories,
    pushes apart representations of unrelated ones.
    """

    positive_lr: float = 0.05
    negative_lr: float = 0.02

    def apply(self, memory_state: Dict[str, Any], signal: Dict[str, Any]) -> Dict[str, Any]:
        weights = memory_state.get("retrieval_weights", {})
        positive_ids = signal.get("relevant_ids", [])
        negative_ids = signal.get("irrelevant_ids", [])

        for mid in positive_ids:
            weights[mid] = min(2.0, weights.get(mid, 1.0) + self.positive_lr)

        for mid in negative_ids:
            weights[mid] = max(0.1, weights.get(mid, 1.0) - self.negative_lr)

        memory_state["retrieval_weights"] = weights
        return memory_state


@dataclass
class ForgettingCurve:
    """
    Ebbinghaus forgetting curve: memory strength decays exponentially.

    Each successful recall resets the decay, simulating spaced repetition.
    """

    half_life_seconds: float = 86400.0  # 1 day default
    recall_boost: float = 1.5  # multiplier on half_life after recall

    def apply(self, memory_state: Dict[str, Any], signal: Dict[str, Any]) -> Dict[str, Any]:
        strengths = memory_state.get("strengths", {})
        last_access = memory_state.get("last_access", {})
        half_lives = memory_state.get("half_lives", {})
        now = signal.get("timestamp", time.time())

        for mid, ts in last_access.items():
            hl = half_lives.get(mid, self.half_life_seconds)
            elapsed = now - ts
            # Exponential decay: S(t) = S0 * 2^(-t/half_life)
            decay = math.pow(2, -elapsed / hl)
            strengths[mid] = strengths.get(mid, 1.0) * decay

        # Boost recalled items
        recalled_ids = signal.get("recalled_ids", [])
        for mid in recalled_ids:
            strengths[mid] = min(1.0, strengths.get(mid, 0.5) + 0.1)
            last_access[mid] = now
            half_lives[mid] = half_lives.get(mid, self.half_life_seconds) * self.recall_boost

        memory_state["strengths"] = strengths
        memory_state["last_access"] = last_access
        memory_state["half_lives"] = half_lives
        return memory_state


# ---------------------------------------------------------------------------
# Plasticity Engine
# ---------------------------------------------------------------------------

class PlasticityEngine:
    """
    Orchestrates multiple plasticity rules to dynamically shape memory.

    This is the cross-cutting concern that makes AURA's memory adaptive:
    - Applied after every store/recall operation
    - Tracks learning signals from retrieval outcomes
    - Updates memory weights, associations, and strengths

    Can be used standalone (with any MemoryStore) or with BMAM/model backends.
    """

    def __init__(
        self,
        rules: Optional[List[Any]] = None,
    ) -> None:
        self.rules: List[Any] = rules or [
            HebbianRule(),
            ContrastiveShaping(),
            ForgettingCurve(),
        ]
        self._state: Dict[str, Any] = {
            "associations": {},
            "retrieval_weights": {},
            "strengths": {},
            "last_access": {},
            "half_lives": {},
            "learning_events": 0,
        }

    def on_store(self, memory_id: str, content: str, metadata: Dict[str, Any]) -> None:
        """Hook called after a memory is stored."""
        self._state["strengths"][memory_id] = 1.0
        self._state["last_access"][memory_id] = time.time()
        self._state["learning_events"] += 1

    def on_recall(self, query: str, retrieved_ids: List[str], relevance_scores: List[float]) -> None:
        """Hook called after a memory recall — drives learning."""
        signal = {
            "co_activated_ids": retrieved_ids,
            "recalled_ids": retrieved_ids,
            "relevant_ids": [mid for mid, score in zip(retrieved_ids, relevance_scores) if score > 0.5],
            "irrelevant_ids": [mid for mid, score in zip(retrieved_ids, relevance_scores) if score <= 0.3],
            "timestamp": time.time(),
        }

        for rule in self.rules:
            self._state = rule.apply(self._state, signal)

        self._state["learning_events"] += 1

    def get_weight(self, memory_id: str) -> float:
        """Get the current retrieval weight for a memory."""
        base = self._state.get("retrieval_weights", {}).get(memory_id, 1.0)
        strength = self._state.get("strengths", {}).get(memory_id, 1.0)
        return base * strength

    def get_associations(self, memory_id: str) -> Dict[str, float]:
        """Get associated memories for a given memory."""
        associations = self._state.get("associations", {})
        result = {}
        for key, weight in associations.items():
            parts = key.split(":")
            if memory_id in parts:
                other = parts[0] if parts[1] == memory_id else parts[1]
                result[other] = weight
        return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            "learning_events": self._state["learning_events"],
            "tracked_memories": len(self._state.get("strengths", {})),
            "active_associations": len(self._state.get("associations", {})),
            "rules": [type(r).__name__ for r in self.rules],
        }

    @property
    def state(self) -> Dict[str, Any]:
        return dict(self._state)


def build_model_plasticity(**kwargs: Any) -> PlasticityEngine:
    return PlasticityEngine()
