"""LLM-driven environment evolver — generates world mutations via LLM.

Accumulates ActivitySignals across ticks, checks time-interval gates,
and formats world state + activity into an LLM prompt to produce
structured JSON mutations.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from ..evolve import EnvironmentEvolver
from ..evolve_types import (
    ActivitySignal,
    EvolutionResult,
    EvolutionTrigger,
    MutationType,
    WorldMutation,
    WorldState,
)

logger = logging.getLogger(__name__)

EVOLUTION_SYSTEM_PROMPT = """\
You are the World Evolution Engine for a living, evolving town simulation.
Analyze the current world state and recent agent activity to decide if the world should change.

You may produce mutations of these types:
- add_location: Add a new building/area. Payload: {{"name", "type", "x", "y", "width", "height", "emoji", "capacity", "description"}}
- modify_location: Change an existing location. Payload: {{"field": "description"|"capacity"|"type"|"emoji", "value": ...}}
- remove_location: Remove a non-home location. Payload: {{}}
- expand_grid: Grow the map. Payload: {{"direction": "north"|"south"|"east"|"west", "amount": int}}
- add_property: Add a world property. Payload: {{"value": ...}}
- modify_property: Change a world property. Payload: {{"value": ...}}
- world_event: Announce a town-wide event. Payload: {{"description": str, "duration_ticks": int}}
- evolve_agent: Change an agent's traits. Payload: {{"field": "personality"|"occupation"|"routine", "value": ...}}
- evolve_relationship: Update relationship between agents. Payload: {{"other_agent": str, "relationship": str}}

Rules:
1. Only produce mutations that are justified by agent activity patterns.
2. New locations should fit the cultural context (Chinese/Asian-inspired small town).
3. Keep names consistent with existing locations.
4. Limit to at most {max_mutations} mutations per cycle.
5. Return valid JSON: {{"mutations": [...]}} where each mutation has "type", "target", "payload", "reason".
6. If no changes are warranted, return {{"mutations": []}}.
"""

EVOLUTION_USER_PROMPT = """\
World State:
- Grid: {grid_width}x{grid_height}
- Locations ({location_count}): {locations}
- World Properties: {properties}
- Utilization: {utilization}

Recent Activity ({signal_count} signals over {tick_span} ticks):
{activity_summary}

Tick: {tick_index}

Analyze and produce mutations (or empty list if none needed):"""


class LLMEvolver(EnvironmentEvolver):
    """LLM-driven evolver that accumulates signals and periodically evolves."""

    def __init__(
        self,
        llm: Any = None,
        evolve_interval: int = 5,
        min_activity: int = 3,
        max_mutations: int = 3,
    ) -> None:
        self._llm = llm
        self._evolve_interval = evolve_interval
        self._min_activity = min_activity
        self._max_mutations = max_mutations
        self._accumulated_signals: List[ActivitySignal] = []
        self._last_evolve_tick: int = -1

    def should_evolve(
        self,
        world_state: WorldState,
        activity_signals: Sequence[ActivitySignal],
        tick_index: int,
    ) -> bool:
        # Accumulate signals
        self._accumulated_signals.extend(activity_signals)

        # Time-interval gate
        if tick_index - self._last_evolve_tick < self._evolve_interval:
            return False

        # Minimum activity threshold
        if len(self._accumulated_signals) < self._min_activity:
            return False

        return True

    def evolve(
        self,
        world_state: WorldState,
        activity_signals: Sequence[ActivitySignal],
        tick_index: int,
    ) -> EvolutionResult:
        # Accumulate any new signals not yet added
        if activity_signals and (
            not self._accumulated_signals
            or self._accumulated_signals[-1] is not activity_signals[-1]
        ):
            self._accumulated_signals.extend(activity_signals)

        trigger = EvolutionTrigger(
            source="interval",
            description=f"Periodic evolution at tick {tick_index}",
            agent_signals=len(self._accumulated_signals),
        )

        if self._llm is None:
            self._last_evolve_tick = tick_index
            self._accumulated_signals.clear()
            return EvolutionResult(trigger=trigger)

        # Format activity summary
        activity_lines = []
        for sig in self._accumulated_signals[-20:]:  # last 20 signals
            nearby = ", ".join(sig.nearby_agents) if sig.nearby_agents else "none"
            activity_lines.append(
                f"  - {sig.agent_name} at {sig.location}: {sig.action} (nearby: {nearby})"
            )
        activity_summary = "\n".join(activity_lines) if activity_lines else "  (no activity)"

        # Format locations
        locations = world_state.get_locations()
        loc_names = ", ".join(loc.get("name", "?") for loc in locations)

        system_msg = EVOLUTION_SYSTEM_PROMPT.format(max_mutations=self._max_mutations)
        user_msg = EVOLUTION_USER_PROMPT.format(
            grid_width=world_state.grid_width,
            grid_height=world_state.grid_height,
            location_count=len(locations),
            locations=loc_names,
            properties=json.dumps(world_state.get_world_properties(), default=str),
            utilization=json.dumps(world_state.get_utilization(), default=str),
            signal_count=len(self._accumulated_signals),
            tick_span=tick_index - self._last_evolve_tick,
            activity_summary=activity_summary,
            tick_index=tick_index,
        )

        try:
            result = self._llm.chat_json(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )

            mutations = self._parse_mutations(result)
            self._last_evolve_tick = tick_index
            self._accumulated_signals.clear()

            return EvolutionResult(
                mutations=mutations[: self._max_mutations],
                trigger=trigger,
            )

        except Exception as e:
            logger.info("LLM evolution failed, returning empty result: %s", e)
            self._last_evolve_tick = tick_index
            self._accumulated_signals.clear()
            return EvolutionResult(trigger=trigger)

    def _parse_mutations(self, raw: Any) -> List[WorldMutation]:
        """Parse LLM JSON output into WorldMutation objects."""
        mutations: List[WorldMutation] = []

        if isinstance(raw, dict):
            items = raw.get("mutations", [])
        elif isinstance(raw, list):
            items = raw
        else:
            return mutations

        type_map = {t.value: t for t in MutationType}

        for item in items:
            if not isinstance(item, dict):
                continue
            type_str = item.get("type", "")
            mutation_type = type_map.get(type_str)
            if mutation_type is None:
                continue
            target = item.get("target", "")
            payload = item.get("payload", {})
            reason = item.get("reason", "")
            if not isinstance(payload, dict):
                continue
            mutations.append(
                WorldMutation(
                    type=mutation_type,
                    target=target,
                    payload=payload,
                    reason=reason,
                )
            )

        return mutations


def build_llm_evolver(**kwargs: Any) -> LLMEvolver:
    """Factory function for backend registration."""
    return LLMEvolver(
        llm=kwargs.get("llm"),
        evolve_interval=kwargs.get("evolve_interval", 5),
        max_mutations=kwargs.get("evolve_max_mutations", 3),
    )
