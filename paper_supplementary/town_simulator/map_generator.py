"""Natural language + template-based map generation for AURA Town."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .procedural_buildings import BUILDING_TEMPLATES, INTERIOR_TEMPLATES

logger = logging.getLogger(__name__)


@dataclass
class MapSpec:
    """Specification for a generated map area."""

    name: str
    width: int
    height: int
    origin_x: int
    origin_y: int
    biome: str
    locations: List[Dict[str, Any]] = field(default_factory=list)
    description: str = ""


# ── Pre-built map templates ──────────────────────────────────────────

MAP_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "fishing_village": {
        "name": "Fishing Village",
        "width": 48,
        "height": 32,
        "biome": "riverside",
        "description": "A tranquil fishing village along the riverbank.",
        "locations": [
            {"name": "Fisher's Wharf", "type": "square", "emoji": "\U0001F3A3",
             "capacity": 15, "dx": 4, "dy": 4, "description": "A busy dock where fishermen unload their daily catch."},
            {"name": "Tide Pool Tavern", "type": "cafe", "emoji": "\u2615",
             "capacity": 12, "dx": 12, "dy": 6, "description": "A cozy tavern overlooking the water."},
            {"name": "Net Mender's Cottage", "type": "home", "emoji": "\U0001F3E0",
             "capacity": 4, "dx": 20, "dy": 4, "description": "A small cottage where fishing nets are repaired."},
            {"name": "Coral Shrine", "type": "temple", "emoji": "\u26E9",
             "capacity": 8, "dx": 28, "dy": 8, "description": "A seaside shrine decorated with coral and shells."},
            {"name": "Driftwood Market", "type": "shop", "emoji": "\U0001F3EA",
             "capacity": 10, "dx": 8, "dy": 16, "description": "A market selling fresh fish and coastal goods."},
            {"name": "Pearl Diver's Hut", "type": "home", "emoji": "\U0001F41A",
             "capacity": 4, "dx": 18, "dy": 18, "description": "Home of the village's most skilled pearl diver."},
            {"name": "Anchor Rest Inn", "type": "cafe", "emoji": "\u2693",
             "capacity": 10, "dx": 30, "dy": 16, "description": "An inn where travelers rest before journeys."},
            {"name": "Seabreeze Park", "type": "park", "emoji": "\U0001F332",
             "capacity": 20, "dx": 36, "dy": 4, "description": "A small park with a view of the water."},
        ],
    },
    "mountain_village": {
        "name": "Mountain Village",
        "width": 48,
        "height": 32,
        "biome": "mountain",
        "description": "A remote village nestled among misty mountain peaks.",
        "locations": [
            {"name": "Summit Monastery", "type": "temple", "emoji": "\u26F0",
             "capacity": 10, "dx": 8, "dy": 2, "description": "An ancient monastery atop the highest accessible peak."},
            {"name": "Ironvein Forge", "type": "shop", "emoji": "\U0001F528",
             "capacity": 6, "dx": 4, "dy": 14, "description": "A forge powered by mountain streams."},
            {"name": "Cloud Rest Tea Pavilion", "type": "teahouse", "emoji": "\U0001F375",
             "capacity": 8, "dx": 18, "dy": 6, "description": "A peaceful teahouse above the clouds."},
            {"name": "Stone Hearth Lodge", "type": "cafe", "emoji": "\u2615",
             "capacity": 10, "dx": 28, "dy": 4, "description": "A warm lodge serving hot drinks to weary climbers."},
            {"name": "Eagle's Perch Lookout", "type": "library", "emoji": "\U0001F985",
             "capacity": 6, "dx": 36, "dy": 8, "description": "A watchtower converted into a mountain library."},
            {"name": "Hermit's Cave Dwelling", "type": "home", "emoji": "\U0001F3D4",
             "capacity": 4, "dx": 14, "dy": 18, "description": "A cave dwelling of a mountain hermit."},
            {"name": "Alpine Herb Garden", "type": "pharmacy", "emoji": "\U0001F33F",
             "capacity": 6, "dx": 26, "dy": 18, "description": "A garden growing rare medicinal mountain herbs."},
            {"name": "Wind Chime Square", "type": "square", "emoji": "\U0001F390",
             "capacity": 15, "dx": 38, "dy": 18, "description": "A gathering place where wind chimes sing."},
        ],
    },
    "urban_district": {
        "name": "Urban District",
        "width": 48,
        "height": 48,
        "biome": "town_center",
        "description": "A bustling urban area with markets, schools, and cultural venues.",
        "locations": [
            {"name": "Grand Bazaar", "type": "shop", "emoji": "\U0001F3EA",
             "capacity": 20, "dx": 4, "dy": 4, "description": "The largest marketplace in the region."},
            {"name": "Scholars' Academy", "type": "school", "emoji": "\U0001F3EB",
             "capacity": 15, "dx": 16, "dy": 4, "description": "A prestigious school of arts and sciences."},
            {"name": "Moonlight Theater", "type": "gallery", "emoji": "\U0001F3AD",
             "capacity": 12, "dx": 28, "dy": 4, "description": "An open-air theater for performances."},
            {"name": "Jade Dragon Bakery", "type": "bakery", "emoji": "\U0001F950",
             "capacity": 8, "dx": 4, "dy": 16, "description": "Famous for its dragon-shaped pastries."},
            {"name": "Silk Weaver's Studio", "type": "gallery", "emoji": "\U0001F3A8",
             "capacity": 8, "dx": 14, "dy": 18, "description": "A studio showcasing silk artistry."},
            {"name": "City Hall", "type": "townhall", "emoji": "\U0001F3DB",
             "capacity": 20, "dx": 26, "dy": 16, "description": "The administrative center of the district."},
            {"name": "Nightingale Cafe", "type": "cafe", "emoji": "\u2615",
             "capacity": 12, "dx": 38, "dy": 12, "description": "An upscale cafe with live music."},
            {"name": "Central Fountain Plaza", "type": "square", "emoji": "\u26F2",
             "capacity": 25, "dx": 16, "dy": 30, "description": "A grand plaza with a tiered fountain."},
        ],
    },
    "forest_settlement": {
        "name": "Forest Settlement",
        "width": 48,
        "height": 32,
        "biome": "forest",
        "description": "A secluded settlement deep within an ancient forest.",
        "locations": [
            {"name": "Elder Oak Shrine", "type": "temple", "emoji": "\U0001F333",
             "capacity": 8, "dx": 6, "dy": 4, "description": "A shrine built around a thousand-year-old oak."},
            {"name": "Mushroom Apothecary", "type": "pharmacy", "emoji": "\U0001F344",
             "capacity": 6, "dx": 18, "dy": 4, "description": "A healer's hut stocked with forest remedies."},
            {"name": "Treehouse Library", "type": "library", "emoji": "\U0001F4DA",
             "capacity": 8, "dx": 30, "dy": 2, "description": "A library suspended in the treetops."},
            {"name": "Woodcutter's Lodge", "type": "home", "emoji": "\U0001FA93",
             "capacity": 4, "dx": 4, "dy": 16, "description": "A sturdy lodge built from local timber."},
            {"name": "Fern Glade Cafe", "type": "cafe", "emoji": "\U0001F33F",
             "capacity": 8, "dx": 16, "dy": 16, "description": "A cafe nestled in a sunlit clearing."},
            {"name": "Owl's Nest Watchtower", "type": "school", "emoji": "\U0001F989",
             "capacity": 6, "dx": 28, "dy": 14, "description": "A tower for observing forest wildlife."},
            {"name": "Firefly Meadow", "type": "park", "emoji": "\u2728",
             "capacity": 15, "dx": 38, "dy": 10, "description": "A magical meadow that glows at dusk."},
            {"name": "Hunter's Market", "type": "shop", "emoji": "\U0001F3F9",
             "capacity": 8, "dx": 36, "dy": 22, "description": "A small market for forest goods and furs."},
        ],
    },
    "riverside_town": {
        "name": "Riverside Town",
        "width": 48,
        "height": 32,
        "biome": "riverside",
        "description": "A prosperous town along a great river, known for trade and culture.",
        "locations": [
            {"name": "Dragon Boat Pier", "type": "square", "emoji": "\U0001F6A2",
             "capacity": 15, "dx": 4, "dy": 2, "description": "The main pier for river trading boats."},
            {"name": "Lotus Bridge Teahouse", "type": "teahouse", "emoji": "\U0001F375",
             "capacity": 10, "dx": 16, "dy": 4, "description": "A teahouse on a bridge over lotus ponds."},
            {"name": "River Silk Emporium", "type": "shop", "emoji": "\U0001F3EA",
             "capacity": 10, "dx": 28, "dy": 4, "description": "A trade hub for silk and fine goods."},
            {"name": "Willow Moon Inn", "type": "cafe", "emoji": "\U0001F343",
             "capacity": 10, "dx": 38, "dy": 6, "description": "An inn beneath ancient weeping willows."},
            {"name": "River Guard Post", "type": "townhall", "emoji": "\U0001F3EF",
             "capacity": 8, "dx": 4, "dy": 18, "description": "The river patrol headquarters."},
            {"name": "Calligrapher's Study", "type": "library", "emoji": "\u270D",
             "capacity": 6, "dx": 16, "dy": 18, "description": "A scholar's studio for calligraphy arts."},
            {"name": "Koi Garden", "type": "park", "emoji": "\U0001F41F",
             "capacity": 15, "dx": 28, "dy": 16, "description": "A serene garden with colorful koi ponds."},
            {"name": "Waterside Bakery", "type": "bakery", "emoji": "\U0001F950",
             "capacity": 8, "dx": 38, "dy": 20, "description": "A bakery famous for steamed river buns."},
        ],
    },
}

# ── LLM prompt for natural-language map generation ───────────────────

MAP_GEN_SYSTEM = """You are a map designer for a Chinese-inspired town simulation game.
Given a user description, generate a list of 4-8 locations for a new map area.

Each location must use one of these building types: {building_types}

Return JSON format:
{{
  "name": "Area Name",
  "biome": "town_center|farmland|riverside|forest|mountain",
  "description": "Brief area description",
  "locations": [
    {{
      "name": "Location Name (Chinese/Asian-inspired)",
      "type": "building_type",
      "emoji": "single_emoji",
      "capacity": 4-20,
      "dx": 0-40,
      "dy": 0-28,
      "description": "Brief description of this place"
    }}
  ]
}}

Rules:
- Names should be culturally coherent (Chinese/Asian-inspired)
- Spread locations across the area (dx: 2-40, dy: 2-28)
- Leave at least 6 cells gap between locations
- Vary building types for interesting gameplay
- Include at least one public gathering space (park, square, cafe, or teahouse)"""

MAP_GEN_USER = """Generate a map area based on this description:
"{prompt}"

The area will be placed at world origin ({origin_x}, {origin_y}).
Maximum {max_locations} locations."""


class MapGenerator:
    """Generates map areas from templates or natural language descriptions."""

    def __init__(self, llm_engine: Any = None) -> None:
        self._llm = llm_engine

    def from_template(
        self,
        key: str,
        origin: Tuple[int, int],
        customizations: Optional[Dict[str, Any]] = None,
    ) -> Optional[MapSpec]:
        """Create a MapSpec from a pre-defined template.

        Args:
            key: Template key (e.g., "fishing_village")
            origin: (x, y) world coordinates for the area's top-left corner
            customizations: Optional overrides (name, description, etc.)
        """
        tmpl = MAP_TEMPLATES.get(key)
        if tmpl is None:
            return None

        name = tmpl["name"]
        width = tmpl["width"]
        height = tmpl["height"]
        biome = tmpl["biome"]
        description = tmpl["description"]

        if customizations:
            name = customizations.get("name", name)
            description = customizations.get("description", description)
            biome = customizations.get("biome", biome)

        ox, oy = origin

        # Convert template locations to absolute world coordinates
        locations = []
        for loc_tmpl in tmpl["locations"]:
            btype = loc_tmpl["type"]
            dims = BUILDING_TEMPLATES.get(btype, {"width": 3, "height": 3})
            interior = INTERIOR_TEMPLATES.get(btype, {})

            locations.append({
                "name": loc_tmpl["name"],
                "type": btype,
                "x": ox + loc_tmpl["dx"],
                "y": oy + loc_tmpl["dy"],
                "width": dims["width"],
                "height": dims["height"],
                "emoji": loc_tmpl["emoji"],
                "capacity": loc_tmpl["capacity"],
                "owner": None,
                "description": loc_tmpl.get("description", f"A {btype} in {name}."),
                "interior_objects": list(interior.get("objects", [])),
                "items": list(interior.get("items", [])),
                "atmosphere": interior.get("atmosphere", ""),
            })

        return MapSpec(
            name=name,
            width=width,
            height=height,
            origin_x=ox,
            origin_y=oy,
            biome=biome,
            locations=locations,
            description=description,
        )

    def from_natural_language(
        self,
        prompt: str,
        origin: Tuple[int, int],
        max_locations: int = 8,
    ) -> Optional[MapSpec]:
        """Use LLM to generate a map area from a natural language description."""
        if self._llm is None:
            logger.warning("No LLM engine available for NL map generation")
            return None

        building_types = ", ".join(BUILDING_TEMPLATES.keys())
        ox, oy = origin

        system_msg = MAP_GEN_SYSTEM.format(building_types=building_types)
        user_msg = MAP_GEN_USER.format(
            prompt=prompt,
            origin_x=ox,
            origin_y=oy,
            max_locations=max_locations,
        )

        try:
            result = self._llm.chat_json([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ], max_tokens=1024)

            if "raw" in result:
                logger.warning("LLM returned unparsable map generation response")
                return None

            area_name = result.get("name", "Generated Area")
            biome = result.get("biome", "town_center")
            description = result.get("description", "")
            raw_locs = result.get("locations", [])

            locations = []
            for loc_data in raw_locs[:max_locations]:
                if not isinstance(loc_data, dict):
                    continue

                btype = loc_data.get("type", "home")
                # Validate building type
                if btype not in BUILDING_TEMPLATES:
                    btype = "home"

                dims = BUILDING_TEMPLATES[btype]
                interior = INTERIOR_TEMPLATES.get(btype, {})

                locations.append({
                    "name": loc_data.get("name", "Unnamed Place"),
                    "type": btype,
                    "x": ox + int(loc_data.get("dx", 0)),
                    "y": oy + int(loc_data.get("dy", 0)),
                    "width": dims["width"],
                    "height": dims["height"],
                    "emoji": loc_data.get("emoji", "\U0001F3E0"),
                    "capacity": int(loc_data.get("capacity", 8)),
                    "owner": None,
                    "description": loc_data.get("description", f"A {btype}."),
                    "interior_objects": list(interior.get("objects", [])),
                    "items": list(interior.get("items", [])),
                    "atmosphere": interior.get("atmosphere", ""),
                })

            return MapSpec(
                name=area_name,
                width=48,
                height=32,
                origin_x=ox,
                origin_y=oy,
                biome=biome,
                locations=locations,
                description=description,
            )

        except Exception as e:
            logger.error("NL map generation failed: %s", e)
            return None

    def apply_to_world(self, world: Any, spec: MapSpec) -> list:
        """Validate and place MapSpec locations into the TownWorld.

        Returns list of successfully placed Location objects.
        """
        from .world import Location

        placed = []
        for loc_data in spec.locations:
            x = loc_data["x"]
            y = loc_data["y"]
            w = loc_data["width"]
            h = loc_data["height"]

            # Ensure grid is big enough
            world._grow_to_fit(x + w, y + h)

            # Check area is free
            if not world._area_free(x, y, w, h):
                logger.info("Skipping %s: area not free at (%d,%d)", loc_data["name"], x, y)
                continue

            # Check name uniqueness
            if world.get_location_by_name(loc_data["name"]) is not None:
                logger.info("Skipping %s: name already exists", loc_data["name"])
                continue

            loc = Location(
                name=loc_data["name"],
                type=loc_data["type"],
                x=x,
                y=y,
                width=w,
                height=h,
                emoji=loc_data["emoji"],
                capacity=loc_data["capacity"],
                owner=loc_data.get("owner"),
                description=loc_data["description"],
                interior_objects=loc_data.get("interior_objects", []),
                items=loc_data.get("items", []),
                atmosphere=loc_data.get("atmosphere", ""),
            )
            world.locations.append(loc)
            placed.append(loc)
            logger.info("Placed location: %s at (%d, %d)", loc.name, x, y)

        return placed

    @staticmethod
    def list_templates() -> List[Dict[str, Any]]:
        """Return summary info for all available templates."""
        result = []
        for key, tmpl in MAP_TEMPLATES.items():
            result.append({
                "key": key,
                "name": tmpl["name"],
                "biome": tmpl["biome"],
                "description": tmpl["description"],
                "location_count": len(tmpl["locations"]),
                "width": tmpl["width"],
                "height": tmpl["height"],
            })
        return result
