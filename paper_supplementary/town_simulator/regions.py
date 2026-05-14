"""Region management and world map data for AURA Town."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .chunks import CHUNK_SIZE, assign_biome

logger = logging.getLogger(__name__)


@dataclass
class RegionInfo:
    """Metadata about a distinct region in the world."""

    id: str
    name: str
    biome: str
    world_x: int  # top-left x in world coordinates
    world_y: int  # top-left y in world coordinates
    width: int  # region width in cells
    height: int  # region height in cells
    description: str = ""
    emoji: str = "\U0001F5FA"
    discovered: bool = True
    location_count: int = 0
    population: int = 0  # number of agents currently in region


@dataclass
class RegionConnection:
    """A connection/path between two regions."""

    region_a: str
    region_b: str
    description: str = ""


@dataclass
class WorldMapData:
    """Aggregated data for the world map overview."""

    regions: List[Dict[str, Any]] = field(default_factory=list)
    agent_positions: List[Dict[str, Any]] = field(default_factory=list)
    connections: List[Dict[str, Any]] = field(default_factory=list)
    world_bounds: Dict[str, int] = field(default_factory=dict)


# Biome display info
BIOME_INFO = {
    "town_center": {"emoji": "\U0001F3D8", "color": "#e8d44d", "label": "Town Center"},
    "farmland": {"emoji": "\U0001F33E", "color": "#7cb342", "label": "Farmland"},
    "riverside": {"emoji": "\U0001F30A", "color": "#42a5f5", "label": "Riverside"},
    "forest": {"emoji": "\U0001F332", "color": "#2e7d32", "label": "Forest"},
    "mountain": {"emoji": "\u26F0", "color": "#78909c", "label": "Mountain"},
}


class RegionManager:
    """Manages regions and provides world map data."""

    def __init__(self, world: Any) -> None:
        self._world = world
        self._regions: Dict[str, RegionInfo] = {}
        self._connections: List[RegionConnection] = []

        # Auto-register the original town center
        self._register_town_center()

    def _register_town_center(self) -> None:
        """Register the initial 60x60 area as the town_center region."""
        loc_count = sum(
            1 for loc in self._world.locations
            if 0 <= loc.x < self._world.width and 0 <= loc.y < self._world.height
        )
        self._regions["town_center"] = RegionInfo(
            id="town_center",
            name="Town Center",
            biome="town_center",
            world_x=0,
            world_y=0,
            width=self._world.width,
            height=self._world.height,
            description="The heart of the town, where the original five agents live and work.",
            emoji="\U0001F3D8",
            discovered=True,
            location_count=loc_count,
        )

    def register_region(self, info: RegionInfo) -> None:
        """Register a new region (e.g., when MapGenerator creates an area)."""
        self._regions[info.id] = info
        logger.info("Registered region: %s (%s)", info.name, info.id)

        # Auto-create connections to nearby regions
        for rid, region in self._regions.items():
            if rid == info.id:
                continue
            # Check adjacency (within 32 cells of each other)
            dist_x = abs((info.world_x + info.width // 2) - (region.world_x + region.width // 2))
            dist_y = abs((info.world_y + info.height // 2) - (region.world_y + region.height // 2))
            if dist_x < info.width + region.width and dist_y < info.height + region.height:
                self._connections.append(RegionConnection(
                    region_a=rid,
                    region_b=info.id,
                    description=f"Path from {region.name} to {info.name}",
                ))

    def get_region_at(self, x: int, y: int) -> Optional[RegionInfo]:
        """Find which region contains the given world coordinates.

        When regions overlap, prefer the smallest (most specific) region.
        """
        candidates = []
        for region in self._regions.values():
            if (region.world_x <= x < region.world_x + region.width and
                    region.world_y <= y < region.world_y + region.height):
                candidates.append(region)
        if not candidates:
            return None
        # Return smallest region (most specific match)
        return min(candidates, key=lambda r: r.width * r.height)

    def get_region_by_id(self, region_id: str) -> Optional[RegionInfo]:
        """Get a region by its ID."""
        return self._regions.get(region_id)

    def auto_detect_regions(self, explored_chunks: Set[Tuple[int, int]]) -> None:
        """Detect new regions from explored biome clusters.

        Groups explored chunks by biome and creates region entries for
        significant clusters that don't already have a registered region.
        """
        # Group chunks by biome
        biome_clusters: Dict[str, List[Tuple[int, int]]] = {}
        for cx, cy in explored_chunks:
            biome = assign_biome(cx, cy)
            if biome == "town_center":
                continue  # Already registered
            biome_clusters.setdefault(biome, []).append((cx, cy))

        for biome, chunks in biome_clusters.items():
            if len(chunks) < 2:
                continue

            # Check if we already have a region covering this area
            min_x = min(c[0] for c in chunks) * CHUNK_SIZE
            min_y = min(c[1] for c in chunks) * CHUNK_SIZE
            max_x = (max(c[0] for c in chunks) + 1) * CHUNK_SIZE
            max_y = (max(c[1] for c in chunks) + 1) * CHUNK_SIZE

            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2

            existing = self.get_region_at(center_x, center_y)
            if existing and existing.id != "town_center":
                continue

            region_id = f"{biome}_{min_x}_{min_y}"
            if region_id in self._regions:
                continue

            # Count locations in this area
            loc_count = sum(
                1 for loc in self._world.locations
                if min_x <= loc.x < max_x and min_y <= loc.y < max_y
            )

            biome_meta = BIOME_INFO.get(biome, {})
            self.register_region(RegionInfo(
                id=region_id,
                name=f"{biome_meta.get('label', biome.title())} Area",
                biome=biome,
                world_x=min_x,
                world_y=min_y,
                width=max_x - min_x,
                height=max_y - min_y,
                description=f"A {biome.replace('_', ' ')} region discovered through exploration.",
                emoji=biome_meta.get("emoji", "\U0001F5FA"),
                discovered=True,
                location_count=loc_count,
            ))

    def get_world_map_data(self, agents: list) -> WorldMapData:
        """Aggregate all region, agent, and connection data for the world map.

        Args:
            agents: List of TownAgent objects
        """
        # Update population counts
        for region in self._regions.values():
            region.population = 0

        agent_positions = []
        for agent in agents:
            ax, ay = agent.state.x, agent.state.y
            region = self.get_region_at(ax, ay)
            if region:
                region.population += 1

            agent_positions.append({
                "name": agent.name,
                "emoji": agent.profile.emoji,
                "x": ax,
                "y": ay,
                "region_id": region.id if region else None,
            })

        # Update location counts
        for region in self._regions.values():
            region.location_count = sum(
                1 for loc in self._world.locations
                if (region.world_x <= loc.x < region.world_x + region.width and
                    region.world_y <= loc.y < region.world_y + region.height)
            )

        regions_data = []
        for r in self._regions.values():
            biome_meta = BIOME_INFO.get(r.biome, {})
            regions_data.append({
                "id": r.id,
                "name": r.name,
                "biome": r.biome,
                "world_x": r.world_x,
                "world_y": r.world_y,
                "width": r.width,
                "height": r.height,
                "description": r.description,
                "emoji": r.emoji,
                "discovered": r.discovered,
                "location_count": r.location_count,
                "population": r.population,
                "color": biome_meta.get("color", "#888"),
            })

        connections_data = [
            {"from": c.region_a, "to": c.region_b, "description": c.description}
            for c in self._connections
        ]

        return WorldMapData(
            regions=regions_data,
            agent_positions=agent_positions,
            connections=connections_data,
            world_bounds={
                "min_x": self._world._explored_min_x,
                "min_y": self._world._explored_min_y,
                "max_x": self._world._explored_max_x,
                "max_y": self._world._explored_max_y,
            },
        )

    @property
    def regions(self) -> Dict[str, RegionInfo]:
        return dict(self._regions)
