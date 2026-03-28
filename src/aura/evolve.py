"""EnvironmentEvolver — abstract interface for world-level evolution.

Evolve is a world-level cross-cutting concern (like PlasticityEngine for memory).
It runs once per tick after all agents have acted, analyzing aggregate activity
to produce structured mutations.
"""

from __future__ import annotations

import logging
from typing import Sequence

from .evolve_types import (
    ActivitySignal,
    EvolutionResult,
    WorldState,
)

logger = logging.getLogger(__name__)


class EnvironmentEvolver:
    """Base class for world-level evolution strategies.

    Subclasses implement ``evolve()`` to produce mutations and
    ``should_evolve()`` as a lightweight gate to skip unnecessary work.
    """

    def should_evolve(
        self,
        world_state: WorldState,
        activity_signals: Sequence[ActivitySignal],
        tick_index: int,
    ) -> bool:
        """Return True if evolution should run this tick."""
        return False

    def evolve(
        self,
        world_state: WorldState,
        activity_signals: Sequence[ActivitySignal],
        tick_index: int,
    ) -> EvolutionResult:
        """Analyze world state + activity and return mutations."""
        return EvolutionResult()


class NoOpEvolver(EnvironmentEvolver):
    """Stub evolver that never produces mutations (like StubActor)."""

    def should_evolve(
        self,
        world_state: WorldState,
        activity_signals: Sequence[ActivitySignal],
        tick_index: int,
    ) -> bool:
        return False

    def evolve(
        self,
        world_state: WorldState,
        activity_signals: Sequence[ActivitySignal],
        tick_index: int,
    ) -> EvolutionResult:
        return EvolutionResult()
