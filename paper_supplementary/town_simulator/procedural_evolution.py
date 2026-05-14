"""Procedural (non-LLM) evolution system for AURA Town.

Runs every tick to provide continuous world changes:
- Micro-events (ambient seasonal flavor text)
- Weather transitions (Markov chain)
- Season rotation
- Popularity-based location upgrades
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .chunks import _seeded_random

logger = logging.getLogger(__name__)


# Weather Markov chain: current -> [(next_state, weight), ...]
WEATHER_TRANSITIONS: Dict[str, List[Tuple[str, float]]] = {
    "clear": [("clear", 0.5), ("partly_cloudy", 0.3), ("windy", 0.2)],
    "partly_cloudy": [("partly_cloudy", 0.3), ("clear", 0.2), ("cloudy", 0.3), ("windy", 0.2)],
    "cloudy": [("cloudy", 0.3), ("partly_cloudy", 0.2), ("rain", 0.3), ("fog", 0.2)],
    "rain": [("rain", 0.3), ("storm", 0.2), ("cloudy", 0.3), ("partly_cloudy", 0.2)],
    "storm": [("storm", 0.2), ("rain", 0.4), ("cloudy", 0.3), ("clear", 0.1)],
    "windy": [("windy", 0.3), ("clear", 0.3), ("partly_cloudy", 0.2), ("cloudy", 0.2)],
    "fog": [("fog", 0.3), ("cloudy", 0.3), ("partly_cloudy", 0.2), ("clear", 0.2)],
    # Winter variants
    "snow": [("snow", 0.4), ("blizzard", 0.2), ("cloudy", 0.2), ("clear", 0.2)],
    "blizzard": [("blizzard", 0.2), ("snow", 0.4), ("cloudy", 0.3), ("clear", 0.1)],
}

SEASONS = ["spring", "summer", "autumn", "winter"]

# Seasonal micro-events
MICRO_EVENTS: Dict[str, List[str]] = {
    "spring": [
        "Cherry blossoms scatter in the breeze",
        "Fresh buds appear on the old plum tree",
        "A pair of swallows nest under the temple eaves",
        "Morning dew glistens on the bamboo leaves",
        "The scent of osmanthus drifts through the streets",
        "Peach blossoms line the riverbank path",
        "Spring rain nourishes the young rice shoots",
        "A rainbow appears after the morning shower",
    ],
    "summer": [
        "Cicadas hum in the afternoon heat",
        "Lotus flowers bloom in the pond",
        "Fireflies dance by the riverside at dusk",
        "The evening breeze carries the scent of jasmine",
        "Children splash in the cool stream",
        "A thunderhead builds over the western mountains",
        "Dragonflies hover over the still water",
        "The old banyan tree offers welcome shade",
    ],
    "autumn": [
        "Golden ginkgo leaves carpet the stone path",
        "The harvest moon rises over the town",
        "Persimmons ripen on the courtyard trees",
        "Migrating geese fly in formation overhead",
        "Chrysanthemums bloom in every garden",
        "The maple forest blazes with crimson and gold",
        "Cool winds carry the scent of roasted chestnuts",
        "Morning mist lingers in the mountain valleys",
    ],
    "winter": [
        "Snowflakes drift past the red lanterns",
        "Hot steam rises from the dumpling stall",
        "Icicles hang from the curved temple roof",
        "The plum tree blooms despite the cold",
        "Smoke curls from every chimney in town",
        "Frost paints delicate patterns on the windows",
        "The frozen pond reflects the pale winter sky",
        "Bundled villagers hurry through the crisp air",
    ],
}

# Weather-specific micro-events (appended to seasonal ones)
WEATHER_EVENTS: Dict[str, List[str]] = {
    "rain": [
        "Rain drums on the tiled rooftops",
        "Puddles form along the cobblestone streets",
        "Umbrellas bloom like flowers across the town",
    ],
    "storm": [
        "Thunder echoes through the valley",
        "Lightning illuminates the distant peaks",
        "The wind howls through the narrow alleys",
    ],
    "fog": [
        "A thick fog rolls in from the river",
        "Lanterns glow softly through the mist",
        "Distant buildings fade into the haze",
    ],
    "snow": [
        "Fresh snow blankets the rooftops",
        "Footprints mark the fresh powder",
        "Snow-laden branches bow gracefully",
    ],
    "blizzard": [
        "A fierce blizzard sweeps through the town",
        "Visibility drops as snow swirls wildly",
        "Everyone seeks shelter from the bitter storm",
    ],
}


class ProceduralEvolver:
    """Lightweight evolution system that runs every tick without LLM calls."""

    def __init__(
        self,
        weather_interval: int = 3,
        season_length: int = 40,
        micro_event_prob: float = 0.3,
    ) -> None:
        self.weather_interval = weather_interval
        self.season_length = season_length
        self.micro_event_prob = micro_event_prob
        self._season_index = 0

    def tick(
        self,
        tick_index: int,
        world_properties: Dict[str, Any],
        visit_counts: Dict[str, int],
        locations: list,
    ) -> List[Dict[str, Any]]:
        """Run one tick of procedural evolution.

        Returns list of event dicts: [{"type": ..., "description": ...}, ...]
        """
        events: List[Dict[str, Any]] = []

        current_season = world_properties.get("season", "spring")
        current_weather = world_properties.get("weather", "clear")

        # Season rotation
        season_event = self._check_season_rotation(tick_index, world_properties)
        if season_event:
            events.append(season_event)
            current_season = world_properties["season"]

        # Weather transition
        weather_event = self._check_weather_transition(
            tick_index, world_properties, current_season
        )
        if weather_event:
            events.append(weather_event)
            current_weather = world_properties["weather"]

        # Micro-events
        micro = self._generate_micro_event(
            tick_index, current_season, current_weather
        )
        if micro:
            events.append(micro)

        # Popularity upgrades
        upgrade = self._check_popularity_upgrades(visit_counts, locations)
        if upgrade:
            events.append(upgrade)

        return events

    def _check_season_rotation(
        self, tick_index: int, world_properties: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Rotate seasons every season_length ticks."""
        if tick_index > 0 and tick_index % self.season_length == 0:
            self._season_index = (self._season_index + 1) % len(SEASONS)
            new_season = SEASONS[self._season_index]
            old_season = world_properties.get("season", "spring")
            world_properties["season"] = new_season

            # Reset weather for the new season
            if new_season == "winter":
                world_properties["weather"] = "snow"
            elif new_season == "spring":
                world_properties["weather"] = "partly_cloudy"
            else:
                world_properties["weather"] = "clear"

            return {
                "type": "season_change",
                "description": f"The season changes from {old_season} to {new_season}",
            }
        return None

    def _check_weather_transition(
        self,
        tick_index: int,
        world_properties: Dict[str, Any],
        season: str,
    ) -> Optional[Dict[str, Any]]:
        """Transition weather using Markov chain every weather_interval ticks."""
        if tick_index % self.weather_interval != 0:
            return None

        current = world_properties.get("weather", "clear")

        # In winter, bias toward snow/blizzard
        if season == "winter" and current in ("rain", "storm"):
            current = "snow"

        transitions = WEATHER_TRANSITIONS.get(current, WEATHER_TRANSITIONS["clear"])

        # Weighted random selection
        r = _seeded_random(tick_index * 7919, 0)
        cumulative = 0.0
        new_weather = current
        for state, weight in transitions:
            cumulative += weight
            if r < cumulative:
                new_weather = state
                break

        if new_weather != current:
            world_properties["weather"] = new_weather
            return {
                "type": "weather_change",
                "description": f"Weather changes: {current.replace('_', ' ')} \u2192 {new_weather.replace('_', ' ')}",
            }
        return None

    def _generate_micro_event(
        self, tick_index: int, season: str, weather: str
    ) -> Optional[Dict[str, Any]]:
        """Generate a seasonal/weather micro-event with configured probability."""
        r = _seeded_random(tick_index * 6271, 1)
        if r >= self.micro_event_prob:
            return None

        # Combine seasonal + weather-specific events
        pool = list(MICRO_EVENTS.get(season, MICRO_EVENTS["spring"]))
        pool.extend(WEATHER_EVENTS.get(weather, []))

        if not pool:
            return None

        idx = int(_seeded_random(tick_index * 3571, 2) * len(pool))
        return {
            "type": "micro_event",
            "description": pool[idx],
        }

    def _check_popularity_upgrades(
        self, visit_counts: Dict[str, int], locations: list
    ) -> Optional[Dict[str, Any]]:
        """Upgrade locations that have >50 visits."""
        for loc in locations:
            name = loc.name if hasattr(loc, "name") else loc.get("name", "")
            count = visit_counts.get(name, 0)
            capacity = loc.capacity if hasattr(loc, "capacity") else loc.get("capacity", 0)

            if count > 50 and capacity < 30:
                # Upgrade capacity
                if hasattr(loc, "capacity"):
                    loc.capacity = min(loc.capacity + 5, 30)
                return {
                    "type": "popularity_upgrade",
                    "description": f"{name} is thriving! Capacity increased due to popularity ({count} visits)",
                }
        return None
