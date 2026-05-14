"""Procedural building generator for AURA Town infinite world."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from .chunks import CHUNK_SIZE, _seeded_random


# Chinese/Asian-inspired name pools per biome
BIOME_NAME_POOLS: Dict[str, List[Dict[str, Any]]] = {
    "town_center": [
        {"name": "Jade Lantern Inn", "type": "cafe", "emoji": "\u2615", "capacity": 15},
        {"name": "Golden Pavilion Hall", "type": "townhall", "emoji": "\U0001F3DB", "capacity": 20},
        {"name": "Morning Dew Teahouse", "type": "teahouse", "emoji": "\U0001F375", "capacity": 12},
        {"name": "Silk Road Market", "type": "shop", "emoji": "\U0001F3EA", "capacity": 10},
        {"name": "Bamboo Grove Academy", "type": "school", "emoji": "\U0001F3EB", "capacity": 15},
        {"name": "Lotus Pond Gallery", "type": "gallery", "emoji": "\U0001F3A8", "capacity": 10},
        {"name": "Red Lantern Bakery", "type": "bakery", "emoji": "\U0001F950", "capacity": 8},
        {"name": "Harmony Pharmacy", "type": "pharmacy", "emoji": "\U0001F48A", "capacity": 8},
    ],
    "farmland": [
        {"name": "Rice Paddy Cottage", "type": "home", "emoji": "\U0001F33E", "capacity": 4},
        {"name": "Harvest Moon Barn", "type": "shop", "emoji": "\U0001F33D", "capacity": 8},
        {"name": "Five Grain Mill", "type": "bakery", "emoji": "\U0001F33E", "capacity": 6},
        {"name": "Peach Blossom Farm", "type": "home", "emoji": "\U0001F351", "capacity": 4},
        {"name": "Silkworm House", "type": "shop", "emoji": "\U0001F41B", "capacity": 6},
        {"name": "Water Buffalo Rest", "type": "cafe", "emoji": "\U0001F403", "capacity": 8},
        {"name": "Terrace Garden Home", "type": "home", "emoji": "\U0001F331", "capacity": 4},
        {"name": "Sunflower Meadow", "type": "park", "emoji": "\U0001F33B", "capacity": 20},
    ],
    "riverside": [
        {"name": "Willow Bank Pavilion", "type": "teahouse", "emoji": "\U0001F343", "capacity": 10},
        {"name": "Moonlight Bridge", "type": "square", "emoji": "\U0001F309", "capacity": 15},
        {"name": "Carp Leap Lodge", "type": "cafe", "emoji": "\U0001F41F", "capacity": 10},
        {"name": "Fisherman's Wharf", "type": "shop", "emoji": "\U0001F3A3", "capacity": 8},
        {"name": "Lotus Ferry Dock", "type": "square", "emoji": "\U0001F6A2", "capacity": 12},
        {"name": "Misty River Inn", "type": "home", "emoji": "\U0001F32B", "capacity": 6},
        {"name": "Dragon Boat Rest", "type": "cafe", "emoji": "\U0001F409", "capacity": 10},
        {"name": "Reed Marsh Cottage", "type": "home", "emoji": "\U0001F33F", "capacity": 4},
    ],
    "forest": [
        {"name": "Hidden Grove Shrine", "type": "temple", "emoji": "\u26E9", "capacity": 8},
        {"name": "Pine Wind Hermitage", "type": "home", "emoji": "\U0001F332", "capacity": 4},
        {"name": "Deer Trail Cabin", "type": "home", "emoji": "\U0001F98C", "capacity": 4},
        {"name": "Mushroom Glen Market", "type": "shop", "emoji": "\U0001F344", "capacity": 6},
        {"name": "Owl's Watch Tower", "type": "library", "emoji": "\U0001F989", "capacity": 6},
        {"name": "Maple Leaf Sanctuary", "type": "temple", "emoji": "\U0001F341", "capacity": 8},
        {"name": "Bamboo Shade Rest", "type": "cafe", "emoji": "\U0001F38B", "capacity": 6},
        {"name": "Firefly Clearing", "type": "park", "emoji": "\u2728", "capacity": 15},
    ],
    "mountain": [
        {"name": "Cloud Peak Monastery", "type": "temple", "emoji": "\u26F0", "capacity": 10},
        {"name": "Stone Gate Fortress", "type": "townhall", "emoji": "\U0001F3EF", "capacity": 12},
        {"name": "Eagle Nest Lookout", "type": "library", "emoji": "\U0001F985", "capacity": 6},
        {"name": "Hot Spring Retreat", "type": "cafe", "emoji": "\u2668", "capacity": 8},
        {"name": "Iron Forge Smithy", "type": "shop", "emoji": "\U0001F528", "capacity": 6},
        {"name": "Celestial Pavilion", "type": "temple", "emoji": "\u2B50", "capacity": 8},
        {"name": "Snow Plum Hermitage", "type": "home", "emoji": "\U0001F338", "capacity": 4},
        {"name": "Wind Cave Dwelling", "type": "home", "emoji": "\U0001F32C", "capacity": 4},
    ],
}

# Interior templates per building type for procedurally generated buildings
INTERIOR_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "cafe": {
        "objects": [
            {"name": "Counter", "description": "A wooden counter with a coffee machine and pastry display"},
            {"name": "Seating Area", "description": "Small tables and chairs for patrons"},
        ],
        "items": ["coffee", "tea", "pastries", "cups"],
        "atmosphere": "Warm and inviting, the aroma of fresh brew fills the air",
    },
    "shop": {
        "objects": [
            {"name": "Shelves", "description": "Neatly organized shelves stocked with various goods"},
            {"name": "Checkout", "description": "A simple wooden counter for transactions"},
        ],
        "items": ["groceries", "tools", "cloth", "pottery"],
        "atmosphere": "Busy and colorful, the clink of coins and friendly haggling",
    },
    "library": {
        "objects": [
            {"name": "Bookshelves", "description": "Tall shelves packed with books and scrolls"},
            {"name": "Reading Table", "description": "A quiet table with a lamp for focused study"},
        ],
        "items": ["books", "scrolls", "maps", "ink"],
        "atmosphere": "Hushed and scholarly, dust motes floating in shafts of light",
    },
    "temple": {
        "objects": [
            {"name": "Altar", "description": "A sacred altar with offerings and incense"},
            {"name": "Meditation Space", "description": "A quiet area with cushions for meditation"},
        ],
        "items": ["incense", "candles", "prayer beads", "scripture"],
        "atmosphere": "Serene and reverent, the gentle curl of incense smoke",
    },
    "teahouse": {
        "objects": [
            {"name": "Tea Bar", "description": "A long bar with tea canisters and brewing sets"},
            {"name": "Private Room", "description": "A cozy room with floor cushions"},
        ],
        "items": ["green tea", "oolong", "jasmine tea", "teapots"],
        "atmosphere": "Calm and fragrant, the gentle pour of hot water over tea leaves",
    },
    "bakery": {
        "objects": [
            {"name": "Oven", "description": "A large stone oven radiating warmth"},
            {"name": "Display Case", "description": "Glass case with fresh breads and pastries"},
        ],
        "items": ["bread", "buns", "flour", "sugar"],
        "atmosphere": "Warm and yeasty, the golden glow of freshly baked goods",
    },
    "home": {
        "objects": [
            {"name": "Living Area", "description": "A simple but comfortable living space"},
            {"name": "Kitchen", "description": "A small kitchen with basic cooking supplies"},
        ],
        "items": ["furniture", "cooking pots", "blankets"],
        "atmosphere": "Quiet and personal, a lived-in comfort",
    },
    "school": {
        "objects": [
            {"name": "Classroom", "description": "Desks and a blackboard for teaching"},
            {"name": "Courtyard", "description": "An open area where students gather"},
        ],
        "items": ["textbooks", "chalk", "globes", "notebooks"],
        "atmosphere": "Energetic, the buzz of young minds learning",
    },
    "gallery": {
        "objects": [
            {"name": "Exhibition Wall", "description": "Walls hung with paintings and prints"},
            {"name": "Sculpture Stand", "description": "Pedestals displaying carved works"},
        ],
        "items": ["paintings", "sculptures", "catalogs"],
        "atmosphere": "Contemplative, spotlit art against white walls",
    },
    "townhall": {
        "objects": [
            {"name": "Council Chamber", "description": "A round table for community meetings"},
            {"name": "Notice Board", "description": "Official announcements and town decrees"},
        ],
        "items": ["records", "meeting minutes", "town seal"],
        "atmosphere": "Formal and civic, echoing footsteps on stone floors",
    },
    "pharmacy": {
        "objects": [
            {"name": "Medicine Counter", "description": "Shelves of remedies and supplements"},
            {"name": "Herb Cabinet", "description": "Drawers of dried herbs and ingredients"},
        ],
        "items": ["medicines", "herbs", "bandages", "tonics"],
        "atmosphere": "Clean and medicinal, the earthy scent of dried herbs",
    },
    "park": {
        "objects": [
            {"name": "Bench", "description": "A wooden bench under a shady tree"},
            {"name": "Path", "description": "A winding path through greenery"},
        ],
        "items": ["flowers", "bird feeders", "lanterns"],
        "atmosphere": "Fresh and open, birdsong and rustling leaves",
    },
    "square": {
        "objects": [
            {"name": "Central Feature", "description": "A gathering point at the center of the square"},
            {"name": "Market Stall", "description": "A small stall selling local goods"},
        ],
        "items": ["produce", "crafts", "street food"],
        "atmosphere": "Bustling with community life and chatter",
    },
}

# Building sizes per type
BUILDING_TEMPLATES: Dict[str, Dict[str, int]] = {
    "home": {"width": 3, "height": 3},
    "cafe": {"width": 3, "height": 3},
    "bakery": {"width": 3, "height": 2},
    "library": {"width": 4, "height": 3},
    "shop": {"width": 3, "height": 2},
    "townhall": {"width": 5, "height": 4},
    "pharmacy": {"width": 3, "height": 2},
    "school": {"width": 4, "height": 3},
    "gallery": {"width": 3, "height": 3},
    "teahouse": {"width": 3, "height": 3},
    "temple": {"width": 3, "height": 3},
    "park": {"width": 4, "height": 4},
    "square": {"width": 4, "height": 4},
}


class ProceduralBuildingGenerator:
    """Generates buildings procedurally when new chunks are explored."""

    def __init__(self) -> None:
        self._used_names: Set[str] = set()

    def register_existing(self, names: List[str]) -> None:
        """Register existing location names to avoid duplicates."""
        for name in names:
            self._used_names.add(name)

    def generate_for_chunk(
        self, cx: int, cy: int, biome: str, chunk_seed: int
    ) -> List[Dict[str, Any]]:
        """Generate procedural buildings for a chunk.

        Building probability decreases with distance from center.
        Returns list of location dicts ready for world.locations.
        """
        # Don't generate for town_center chunks - those have the original map
        if biome == "town_center":
            return []

        dist = math.sqrt(cx * cx + cy * cy)
        # Probability of building in this chunk (0-2 buildings)
        base_prob = max(0.05, 0.6 - dist * 0.05)

        buildings = []
        name_pool = BIOME_NAME_POOLS.get(biome, BIOME_NAME_POOLS["farmland"])

        # Try to place 0-2 buildings per chunk
        for attempt in range(2):
            r = _seeded_random(chunk_seed + 1000, attempt)
            if r > base_prob:
                continue

            # Pick a name from the pool
            name_idx = int(_seeded_random(chunk_seed + 2000, attempt) * len(name_pool))
            template = name_pool[name_idx]

            # Deduplicate - try other names if this one is taken
            chosen = None
            for offset in range(len(name_pool)):
                candidate = name_pool[(name_idx + offset) % len(name_pool)]
                if candidate["name"] not in self._used_names:
                    chosen = candidate
                    break

            if chosen is None:
                continue

            self._used_names.add(chosen["name"])

            # Get building dimensions
            btype = chosen["type"]
            dims = BUILDING_TEMPLATES.get(btype, {"width": 3, "height": 3})
            bw = dims["width"]
            bh = dims["height"]

            # Place within chunk bounds with margin
            world_x = cx * CHUNK_SIZE
            world_y = cy * CHUNK_SIZE
            margin = 2
            max_x = CHUNK_SIZE - bw - margin
            max_y = CHUNK_SIZE - bh - margin
            if max_x < margin or max_y < margin:
                continue

            local_x = margin + int(_seeded_random(chunk_seed + 3000, attempt) * (max_x - margin))
            local_y = margin + int(_seeded_random(chunk_seed + 4000, attempt) * (max_y - margin))

            # Separate buildings vertically for second attempt
            if attempt == 1:
                local_y = min(local_y + 8, max_y)

            # Get interior template for this building type
            interior = INTERIOR_TEMPLATES.get(btype, {})

            buildings.append({
                "name": chosen["name"],
                "type": btype,
                "x": world_x + local_x,
                "y": world_y + local_y,
                "width": bw,
                "height": bh,
                "emoji": chosen["emoji"],
                "capacity": chosen["capacity"],
                "owner": None,
                "description": f"A {btype} in the {biome.replace('_', ' ')} area.",
                "interior_objects": list(interior.get("objects", [])),
                "items": list(interior.get("items", [])),
                "atmosphere": interior.get("atmosphere", ""),
            })

        return buildings
