"""Town world: grid map, locations, time system, and movement."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .chunks import CHUNK_SIZE, ChunkManager, assign_biome
from .procedural_buildings import ProceduralBuildingGenerator

logger = logging.getLogger(__name__)


@dataclass
class Location:
    name: str
    type: str
    x: int
    y: int
    width: int
    height: int
    emoji: str
    capacity: int
    owner: Optional[str]
    description: str
    interior_objects: List[Dict[str, Any]] = field(default_factory=list)
    items: List[str] = field(default_factory=list)
    atmosphere: str = ""
    region_id: Optional[str] = None

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.width and self.y <= py < self.y + self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class TimeState:
    day: int = 1
    hour: int = 6
    minute: int = 0

    @property
    def total_minutes(self) -> int:
        return self.hour * 60 + self.minute

    @property
    def display(self) -> str:
        period = "AM" if self.hour < 12 else "PM"
        h = self.hour % 12 or 12
        return f"{h}:{self.minute:02d} {period} Day {self.day}"

    @property
    def time_24h(self) -> str:
        return f"{self.hour:02d}:{self.minute:02d}"

    def copy(self) -> TimeState:
        return TimeState(day=self.day, hour=self.hour, minute=self.minute)


class TownWorld:
    """Grid-based town map with locations and time system."""

    def __init__(self, width: int = 20, height: int = 20, world_seed: int = 42) -> None:
        self.width = width
        self.height = height
        self.locations: List[Location] = []
        self.time = TimeState()
        # Evolution support
        self.properties: Dict[str, Any] = {
            "season": "spring",
            "weather": "clear",
            "economy": "stable",
            "events": [],
        }
        self._visit_counts: Dict[str, int] = {}
        self._total_ticks: int = 0
        self.evolution_log: List[Dict[str, Any]] = []
        # Chunk system for infinite world
        self.chunk_manager = ChunkManager(world_seed=world_seed)
        self._building_generator = ProceduralBuildingGenerator()
        # Track explored bounds (dynamically expand as agents explore)
        self._explored_min_x: int = 0
        self._explored_min_y: int = 0
        self._explored_max_x: int = width
        self._explored_max_y: int = height
        self._generated_chunks: set = set()  # chunk coords that have had buildings placed

    def load_map(self, map_path: Optional[str] = None) -> None:
        """Load town layout from JSON file."""
        if map_path is None:
            map_path = os.path.join(
                os.path.dirname(__file__), "assets", "town_map.json"
            )
        with open(map_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.width = data.get("width", self.width)
        self.height = data.get("height", self.height)
        self.locations = []
        for loc_data in data.get("locations", []):
            self.locations.append(Location(**loc_data))

    def get_location_at(self, x: int, y: int) -> Optional[Location]:
        """Return the location at the given coordinates, or None."""
        for loc in self.locations:
            if loc.contains(x, y):
                return loc
        return None

    def get_location_by_name(self, name: str) -> Optional[Location]:
        """Find a location by name (case-insensitive partial match)."""
        lower = name.lower()
        for loc in self.locations:
            if lower in loc.name.lower() or lower in loc.type.lower():
                return loc
        return None

    def get_home_for(self, agent_name: str) -> Optional[Location]:
        """Find the home location for a given agent."""
        for loc in self.locations:
            if loc.type == "home" and loc.owner == agent_name:
                return loc
        return None

    def distance(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Manhattan distance between two points."""
        return abs(x1 - x2) + abs(y1 - y2)

    def move_toward(
        self, x: int, y: int, target_x: int, target_y: int, speed: int = 2
    ) -> Tuple[int, int]:
        """Move from (x, y) toward (target_x, target_y) by up to `speed` steps."""
        dx = target_x - x
        dy = target_y - y
        total = abs(dx) + abs(dy)
        if total == 0:
            return (x, y)

        steps_left = speed
        new_x, new_y = x, y

        # Move in x first, then y
        if dx != 0:
            step_x = min(abs(dx), steps_left) * (1 if dx > 0 else -1)
            new_x += step_x
            steps_left -= abs(step_x)

        if dy != 0 and steps_left > 0:
            step_y = min(abs(dy), steps_left) * (1 if dy > 0 else -1)
            new_y += step_y

        # Update explored bounds instead of clamping
        self._explored_min_x = min(self._explored_min_x, new_x)
        self._explored_min_y = min(self._explored_min_y, new_y)
        self._explored_max_x = max(self._explored_max_x, new_x + 1)
        self._explored_max_y = max(self._explored_max_y, new_y + 1)
        return (new_x, new_y)

    def advance_time(self, tick_minutes: int = 30) -> bool:
        """Advance simulation time. Returns False if the day is over."""
        self.time.minute += tick_minutes
        while self.time.minute >= 60:
            self.time.hour += 1
            self.time.minute -= 60
        if self.time.hour >= 23:
            return False  # Day is over
        return True

    def new_day(self) -> None:
        """Start a new day."""
        self.time.day += 1
        self.time.hour = 6
        self.time.minute = 0

    def is_daytime(self) -> bool:
        return 6 <= self.time.hour < 23

    def get_all_location_names(self) -> List[str]:
        return [loc.name for loc in self.locations]

    def get_public_locations(self) -> List[Location]:
        """Return non-home locations."""
        return [loc for loc in self.locations if loc.type != "home"]

    def get_location_detail(self, name: str) -> Optional[Dict[str, Any]]:
        """Return full info for a location including visit count."""
        loc = self.get_location_by_name(name)
        if loc is None:
            return None
        return {
            "name": loc.name,
            "type": loc.type,
            "emoji": loc.emoji,
            "x": loc.x,
            "y": loc.y,
            "width": loc.width,
            "height": loc.height,
            "capacity": loc.capacity,
            "owner": loc.owner,
            "description": loc.description,
            "interior_objects": loc.interior_objects,
            "items": loc.items,
            "atmosphere": loc.atmosphere,
            "visit_count": self._visit_counts.get(loc.name, 0),
        }

    # ── WorldState protocol conformance ──────────────────────────

    @property
    def grid_width(self) -> int:
        return self.width

    @property
    def grid_height(self) -> int:
        return self.height

    def get_locations(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": loc.name,
                "type": loc.type,
                "x": loc.x,
                "y": loc.y,
                "width": loc.width,
                "height": loc.height,
                "emoji": loc.emoji,
                "capacity": loc.capacity,
                "description": loc.description,
            }
            for loc in self.locations
        ]

    def get_agents(self) -> List[Dict[str, Any]]:
        # Agents are managed externally; return empty from world
        return []

    def get_world_properties(self) -> Dict[str, Any]:
        return dict(self.properties)

    def get_utilization(self) -> Dict[str, Any]:
        return {
            "total_ticks": self._total_ticks,
            "visit_counts": dict(self._visit_counts),
        }

    # ── Utilization tracking ─────────────────────────────────────

    def record_visit(self, location_name: str) -> None:
        self._visit_counts[location_name] = self._visit_counts.get(location_name, 0) + 1

    def tick_utilization(self) -> None:
        self._total_ticks += 1

    # ── Edge detection ───────────────────────────────────────────

    def is_at_edge(self, x: int, y: int, margin: int = 2) -> Optional[str]:
        """Return edge direction if (x, y) is within margin of explored boundary."""
        if x <= self._explored_min_x + margin:
            return "west"
        if x >= self._explored_max_x - 1 - margin:
            return "east"
        if y <= self._explored_min_y + margin:
            return "north"
        if y >= self._explored_max_y - 1 - margin:
            return "south"
        return None

    # ── Mutation dispatch ────────────────────────────────────────

    def apply_mutation(self, mutation: Any) -> bool:
        """Apply a WorldMutation. Returns True if successful."""
        from aura.evolve_types import MutationType

        handlers = {
            MutationType.ADD_LOCATION: self._apply_add_location,
            MutationType.MODIFY_LOCATION: self._apply_modify_location,
            MutationType.REMOVE_LOCATION: self._apply_remove_location,
            MutationType.EXPAND_GRID: self._apply_expand_grid,
            MutationType.ADD_PROPERTY: self._apply_add_property,
            MutationType.MODIFY_PROPERTY: self._apply_modify_property,
            MutationType.WORLD_EVENT: self._apply_world_event,
        }

        handler = handlers.get(mutation.type)
        if handler is None:
            return False

        try:
            result = handler(mutation)
            if result:
                self.evolution_log.append({
                    "type": mutation.type.value,
                    "target": mutation.target,
                    "reason": mutation.reason,
                    "tick": self._total_ticks,
                    "time": self.time.display,
                })
            return result
        except Exception as e:
            logger.warning("Mutation %s failed: %s", mutation.type.value, e)
            return False

    def _apply_add_location(self, mutation: Any) -> bool:
        """Add a new location to the world."""
        p = mutation.payload
        x = int(p.get("x", 0))
        y = int(p.get("y", 0))
        w = int(p.get("width", 4))
        h = int(p.get("height", 4))

        # Auto-expand grid if needed
        self._grow_to_fit(x + w, y + h)

        # Check area is free
        if not self._area_free(x, y, w, h):
            return False

        # Check name doesn't already exist
        if self.get_location_by_name(p.get("name", "")) is not None:
            return False

        loc = Location(
            name=p.get("name", mutation.target),
            type=p.get("type", "building"),
            x=x,
            y=y,
            width=w,
            height=h,
            emoji=p.get("emoji", "🏠"),
            capacity=int(p.get("capacity", 10)),
            owner=p.get("owner"),
            description=p.get("description", "A new location."),
        )
        self.locations.append(loc)
        logger.info("Added location: %s at (%d, %d)", loc.name, x, y)
        return True

    def _apply_modify_location(self, mutation: Any) -> bool:
        """Modify an existing location's properties."""
        loc = self.get_location_by_name(mutation.target)
        if loc is None:
            return False

        p = mutation.payload
        field_name = p.get("field", "")
        value = p.get("value")

        if field_name == "description" and isinstance(value, str):
            loc.description = value
        elif field_name == "capacity" and isinstance(value, int):
            loc.capacity = value
        elif field_name == "type" and isinstance(value, str):
            loc.type = value
        elif field_name == "emoji" and isinstance(value, str):
            loc.emoji = value
        else:
            return False

        return True

    def _apply_remove_location(self, mutation: Any) -> bool:
        """Remove a non-home location."""
        loc = self.get_location_by_name(mutation.target)
        if loc is None or loc.type == "home":
            return False
        self.locations.remove(loc)
        logger.info("Removed location: %s", loc.name)
        return True

    def _apply_expand_grid(self, mutation: Any) -> bool:
        """Expand the grid in a direction, shifting locations if needed."""
        p = mutation.payload
        direction = p.get("direction", "")
        amount = int(p.get("amount", 10))
        if amount <= 0:
            return False

        if direction == "east":
            self.width += amount
        elif direction == "south":
            self.height += amount
        elif direction == "west":
            self.width += amount
            for loc in self.locations:
                loc.x += amount
        elif direction == "north":
            self.height += amount
            for loc in self.locations:
                loc.y += amount
        else:
            return False

        # Place new locations from payload if provided
        new_locations = p.get("locations", [])
        for loc_data in new_locations:
            if isinstance(loc_data, dict):
                self._apply_add_location(type(
                    "_FakeMutation", (), {
                        "payload": loc_data,
                        "target": loc_data.get("name", ""),
                    }
                )())

        logger.info("Expanded grid %s by %d → now %dx%d", direction, amount, self.width, self.height)
        return True

    def _apply_add_property(self, mutation: Any) -> bool:
        """Add a world property."""
        self.properties[mutation.target] = mutation.payload.get("value")
        return True

    def _apply_modify_property(self, mutation: Any) -> bool:
        """Modify an existing world property."""
        self.properties[mutation.target] = mutation.payload.get("value")
        return True

    def _apply_world_event(self, mutation: Any) -> bool:
        """Record a world-wide event."""
        events = self.properties.get("events", [])
        events.append({
            "description": mutation.payload.get("description", mutation.target),
            "duration_ticks": mutation.payload.get("duration_ticks", 1),
            "started_tick": self._total_ticks,
        })
        self.properties["events"] = events
        return True

    # ── Chunk / infinite world ─────────────────────────────────────

    def ensure_chunk_locations(self, cx: int, cy: int) -> List[Location]:
        """Generate procedural buildings when a chunk is first accessed.

        Returns newly created locations (empty if chunk was already processed).
        """
        key = (cx, cy)
        if key in self._generated_chunks:
            return []

        self._generated_chunks.add(key)
        chunk = self.chunk_manager.get_chunk(cx, cy)

        new_locs = self._building_generator.generate_for_chunk(
            cx, cy, chunk.biome, chunk.seed
        )
        created = []
        for loc_data in new_locs:
            # Check area is free
            x, y = loc_data["x"], loc_data["y"]
            w, h = loc_data["width"], loc_data["height"]
            if self._area_free(x, y, w, h):
                loc = Location(**loc_data)
                self.locations.append(loc)
                created.append(loc)
                # Expand grid dimensions to cover new locations
                self._grow_to_fit(x + w, y + h)

        return created

    def get_visible_data(
        self, cam_x: int, cam_y: int, vw: int, vh: int
    ) -> Dict[str, Any]:
        """Get chunk biome data for the visible viewport area."""
        biomes = self.chunk_manager.get_biome_map(cam_x, cam_y, vw, vh)
        return {"chunk_biomes": biomes}

    def init_building_generator(self) -> None:
        """Register existing location names with the building generator."""
        self._building_generator.register_existing(self.get_all_location_names())

    # ── Helpers ───────────────────────────────────────────────────

    def _area_free(self, x: int, y: int, w: int, h: int) -> bool:
        """Check that a rectangular area doesn't overlap existing locations."""
        for loc in self.locations:
            if (
                x < loc.x + loc.width
                and x + w > loc.x
                and y < loc.y + loc.height
                and y + h > loc.y
            ):
                return False
        return True

    def _grow_to_fit(self, needed_w: int, needed_h: int) -> None:
        """Silently expand grid to fit the requested dimensions."""
        if needed_w > self.width:
            self.width = needed_w
        if needed_h > self.height:
            self.height = needed_h
