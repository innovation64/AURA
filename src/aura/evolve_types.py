"""Core types for the AURA Evolve system — adaptive, self-evolving environments.

Defines mutation types, world state protocol, activity signals, and evolution
results used by EnvironmentEvolver implementations.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple


class MutationType(enum.Enum):
    """All supported world mutation operations."""

    ADD_LOCATION = "add_location"
    MODIFY_LOCATION = "modify_location"
    REMOVE_LOCATION = "remove_location"
    EXPAND_GRID = "expand_grid"
    ADD_PROPERTY = "add_property"
    MODIFY_PROPERTY = "modify_property"
    REMOVE_PROPERTY = "remove_property"
    EVOLVE_AGENT = "evolve_agent"
    EVOLVE_RELATIONSHIP = "evolve_relationship"
    WORLD_EVENT = "world_event"


@dataclass(frozen=True)
class WorldMutation:
    """A single structured mutation to apply to the world."""

    type: MutationType
    target: str  # location name, property key, agent name, etc.
    payload: Dict[str, Any]  # type-specific data
    reason: str = ""
    priority: int = 0  # higher = apply first
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class EvolutionTrigger:
    """What caused an evolution check to fire."""

    source: str  # "interval", "edge_detection", "manual", "threshold"
    description: str
    agent_signals: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionResult:
    """Output of an evolution cycle."""

    mutations: List[WorldMutation] = field(default_factory=list)
    trigger: Optional[EvolutionTrigger] = None
    applied: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        if not self.mutations:
            return "No mutations produced."
        types = [m.type.value for m in self.mutations]
        return f"{len(self.mutations)} mutation(s): {', '.join(types)}"


class WorldState(Protocol):
    """Protocol that world objects must satisfy for the evolver."""

    @property
    def grid_width(self) -> int: ...

    @property
    def grid_height(self) -> int: ...

    def get_locations(self) -> List[Dict[str, Any]]: ...

    def get_agents(self) -> List[Dict[str, Any]]: ...

    def get_world_properties(self) -> Dict[str, Any]: ...

    def get_utilization(self) -> Dict[str, Any]: ...


@dataclass(frozen=True)
class ActivitySignal:
    """Aggregated activity info from one agent during a tick."""

    agent_name: str
    location: str
    action: str
    nearby_agents: Tuple[str, ...] = ()
    memory_patterns: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)
