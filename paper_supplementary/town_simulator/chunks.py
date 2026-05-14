"""Chunk-based infinite world system for AURA Town."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


CHUNK_SIZE = 16  # 16x16 cells per chunk


@dataclass
class Chunk:
    """A single 16x16 chunk of the world."""
    cx: int  # chunk coordinate x
    cy: int  # chunk coordinate y
    biome: str
    seed: int
    terrain_grid: List[List[str]] = field(default_factory=list)
    decorations: List[Dict[str, Any]] = field(default_factory=list)
    generated: bool = False


def _chunk_seed(world_seed: int, cx: int, cy: int) -> int:
    """Deterministic seed for a chunk based on world seed and chunk coords."""
    h = (world_seed * 374761393 + cx * 668265263 + cy * 1013904223) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return (h ^ (h >> 16)) & 0xFFFFFFFF


def _seeded_random(seed: int, index: int = 0) -> float:
    """Simple seeded pseudo-random float in [0, 1)."""
    h = (seed + index * 374761393 + 1013904223) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFFFF) / 4294967296.0


BIOMES = ["town_center", "farmland", "riverside", "forest", "mountain"]

# Terrain palette per biome (tile name references for frontend)
BIOME_TERRAIN = {
    "town_center": ["grass_1", "grass_1", "grass_1", "grass_2"],
    "farmland": ["dirt", "grass_1", "grass_2", "dirt"],
    "riverside": ["grass_1", "grass_edge_1", "grass_edge_2", "grass_1"],
    "forest": ["grass_1", "grass_1", "grass_2", "grass_2"],
    "mountain": ["dirt", "dirt", "grass_edge_3", "dirt"],
}


def assign_biome(cx: int, cy: int) -> str:
    """Assign biome based on chunk distance from origin.

    - (0,0)-(3,3): town_center (covers the original 60x60 area)
    - Distance 3-5: farmland
    - East of center (cx > 3, cy 0-3): riverside
    - Distance 5-7: forest
    - Distance >7: mountain
    """
    # Town center covers original area
    if 0 <= cx <= 3 and 0 <= cy <= 3:
        return "town_center"

    dist = math.sqrt(cx * cx + cy * cy)

    # Riverside: east of center
    if cx > 3 and 0 <= cy <= 3:
        return "riverside"

    if dist <= 5:
        return "farmland"
    if dist <= 7:
        return "forest"
    return "mountain"


class ChunkManager:
    """Manages lazy chunk generation with LRU eviction.

    Chunks are generated deterministically from seed so evicted chunks
    can be regenerated identically.
    """

    def __init__(self, world_seed: int = 42, max_chunks: int = 64) -> None:
        self.world_seed = world_seed
        self.max_chunks = max_chunks
        self._chunks: OrderedDict[Tuple[int, int], Chunk] = OrderedDict()

    def world_to_chunk(self, wx: int, wy: int) -> Tuple[int, int]:
        """Convert world coordinates to chunk coordinates."""
        return (wx // CHUNK_SIZE, wy // CHUNK_SIZE)

    def chunk_to_world(self, cx: int, cy: int) -> Tuple[int, int]:
        """Convert chunk coordinates to world coordinates (top-left corner)."""
        return (cx * CHUNK_SIZE, cy * CHUNK_SIZE)

    def get_chunk(self, cx: int, cy: int) -> Chunk:
        """Get or generate a chunk at the given chunk coordinates."""
        key = (cx, cy)
        if key in self._chunks:
            # Move to end (most recently used)
            self._chunks.move_to_end(key)
            return self._chunks[key]

        # Generate new chunk
        chunk = self._generate_chunk(cx, cy)
        self._chunks[key] = chunk

        # LRU eviction
        while len(self._chunks) > self.max_chunks:
            self._chunks.popitem(last=False)

        return chunk

    def _generate_chunk(self, cx: int, cy: int) -> Chunk:
        """Generate a chunk with terrain and decorations."""
        seed = _chunk_seed(self.world_seed, cx, cy)
        biome = assign_biome(cx, cy)

        # Generate terrain grid
        palette = BIOME_TERRAIN.get(biome, BIOME_TERRAIN["town_center"])
        terrain_grid = []
        for row in range(CHUNK_SIZE):
            terrain_row = []
            for col in range(CHUNK_SIZE):
                r = _seeded_random(seed, row * CHUNK_SIZE + col)
                terrain_row.append(palette[int(r * len(palette))])
            terrain_grid.append(terrain_row)

        # Generate decorations based on biome
        decorations = []
        deco_density = {
            "town_center": 0.08,
            "farmland": 0.05,
            "riverside": 0.06,
            "forest": 0.15,
            "mountain": 0.04,
        }
        density = deco_density.get(biome, 0.08)

        deco_palettes = {
            "town_center": ["flower", "bush", "tree_small", "hedge_1"],
            "farmland": ["flower", "hedge_1", "hedge_2"],
            "riverside": ["flower", "bush", "tree_round"],
            "forest": ["tree_round", "tree_pine", "tree_large", "bush", "tree_tall"],
            "mountain": ["tree_pine", "bush"],
        }
        deco_choices = deco_palettes.get(biome, ["flower", "bush"])

        for row in range(CHUNK_SIZE):
            for col in range(CHUNK_SIZE):
                r = _seeded_random(seed + 500, row * CHUNK_SIZE + col)
                if r < density:
                    r2 = _seeded_random(seed + 600, row * CHUNK_SIZE + col)
                    decorations.append({
                        "x": col,
                        "y": row,
                        "tile": deco_choices[int(r2 * len(deco_choices))],
                    })

        return Chunk(
            cx=cx, cy=cy, biome=biome, seed=seed,
            terrain_grid=terrain_grid, decorations=decorations,
            generated=True,
        )

    def get_chunks_in_rect(
        self, world_x: int, world_y: int, width: int, height: int
    ) -> List[Chunk]:
        """Get all chunks that overlap with a world-space rectangle."""
        cx0, cy0 = self.world_to_chunk(world_x, world_y)
        cx1, cy1 = self.world_to_chunk(world_x + width - 1, world_y + height - 1)

        chunks = []
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                chunks.append(self.get_chunk(cx, cy))
        return chunks

    def get_biome_at(self, wx: int, wy: int) -> str:
        """Get the biome at a world coordinate."""
        cx, cy = self.world_to_chunk(wx, wy)
        return assign_biome(cx, cy)

    def get_biome_map(
        self, world_x: int, world_y: int, width: int, height: int
    ) -> Dict[str, str]:
        """Get a dict of 'cx,cy' -> biome for chunks in the given area."""
        cx0, cy0 = self.world_to_chunk(world_x, world_y)
        cx1, cy1 = self.world_to_chunk(world_x + width - 1, world_y + height - 1)

        biomes = {}
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                biomes[f"{cx},{cy}"] = assign_biome(cx, cy)
        return biomes

    @property
    def loaded_chunk_keys(self) -> List[Tuple[int, int]]:
        """Return list of currently loaded chunk coordinates."""
        return list(self._chunks.keys())
