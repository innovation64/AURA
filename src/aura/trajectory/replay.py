"""Prioritised experience replay buffer for trajectory training."""

from __future__ import annotations

import logging
import random
from typing import List, Optional

from aura.trajectory.collector import TrajectoryStep

logger = logging.getLogger(__name__)

_EPSILON = 1e-6  # small constant to avoid zero priority


class ExperienceBuffer:
    """Prioritised circular experience replay buffer.

    Steps with surprising outcomes (reward far from 0.5) are sampled
    more frequently, controlled by *priority_alpha*.
    """

    def __init__(
        self,
        capacity: int = 50000,
        priority_alpha: float = 0.6,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._alpha = priority_alpha

        # Circular buffer storage
        self._steps: List[Optional[TrajectoryStep]] = [None] * capacity
        self._priorities: List[float] = [0.0] * capacity
        self._write_idx: int = 0
        self._size: int = 0

        # step_id -> buffer index (for fast priority updates)
        self._id_to_idx: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        step: TrajectoryStep,
        priority: Optional[float] = None,
    ) -> None:
        """Add a step to the buffer.

        If *priority* is not given, it is computed as
        ``|reward - 0.5| + epsilon`` so that surprising outcomes get
        higher priority.
        """
        if priority is None:
            priority = abs(step.reward - 0.5) + _EPSILON

        idx = self._write_idx

        # Evict old mapping if overwriting
        old_step = self._steps[idx]
        if old_step is not None:
            self._id_to_idx.pop(old_step.step_id, None)

        self._steps[idx] = step
        self._priorities[idx] = priority
        self._id_to_idx[step.step_id] = idx

        self._write_idx = (self._write_idx + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int = 32) -> List[TrajectoryStep]:
        """Sample a batch weighted by priority raised to *alpha*.

        Returns up to *batch_size* steps (fewer if the buffer is smaller).
        """
        if self._size == 0:
            return []

        n = min(batch_size, self._size)

        # Compute sampling weights
        raw_priorities = self._priorities[: self._size]
        powered = [p ** self._alpha for p in raw_priorities]
        total = sum(powered)
        if total <= 0:
            # Uniform fallback
            indices = random.sample(range(self._size), n)
        else:
            weights = [p / total for p in powered]
            indices = random.choices(range(self._size), weights=weights, k=n)

        result: List[TrajectoryStep] = []
        seen: set = set()
        for idx in indices:
            step = self._steps[idx]
            if step is not None and step.step_id not in seen:
                seen.add(step.step_id)
                result.append(step)
        return result

    def update_priority(self, step_id: str, new_priority: float) -> None:
        """Update the priority of an existing step by its id."""
        idx = self._id_to_idx.get(step_id)
        if idx is None:
            logger.warning(
                "Cannot update priority: step_id %s not found in buffer.",
                step_id,
            )
            return
        self._priorities[idx] = new_priority

    def __len__(self) -> int:
        return self._size
