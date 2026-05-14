"""Status-driven agent movement: match actions to appropriate locations."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Action keywords → preferred location types (ordered by preference)
ACTION_LOCATION_AFFINITY: Dict[str, List[str]] = {
    # Intellectual activities
    "writing": ["library", "cafe", "home", "teahouse", "park"],
    "reading": ["library", "cafe", "home", "teahouse", "park"],
    "studying": ["library", "school", "cafe", "home"],
    "researching": ["library", "school"],
    "reflecting": ["temple", "park", "teahouse", "home"],
    "meditating": ["temple", "park"],
    "thinking": ["park", "library", "cafe", "teahouse"],

    # Social activities
    "chatting": ["cafe", "teahouse", "square", "park"],
    "talking": ["cafe", "teahouse", "square", "park"],
    "socializing": ["cafe", "square", "teahouse", "park"],
    "meeting": ["cafe", "townhall", "teahouse", "square"],
    "discussing": ["cafe", "teahouse", "library", "townhall"],

    # Food / drink
    "eating": ["cafe", "bakery", "home", "teahouse"],
    "cooking": ["home", "bakery", "cafe"],
    "baking": ["bakery", "home", "cafe"],
    "drinking": ["cafe", "teahouse", "home"],
    "brewing": ["cafe", "teahouse"],
    "lunch": ["cafe", "home", "bakery", "teahouse"],
    "breakfast": ["home", "cafe", "bakery"],
    "dinner": ["home", "cafe"],

    # Shopping / commerce
    "shopping": ["shop", "square", "bakery"],
    "buying": ["shop", "square", "pharmacy", "bakery"],
    "selling": ["shop", "square"],
    "browsing": ["shop", "gallery", "square"],
    "trading": ["shop", "square"],

    # Arts / culture
    "painting": ["gallery", "park", "home"],
    "drawing": ["gallery", "park", "home", "cafe"],
    "performing": ["gallery", "square", "park"],
    "observing": ["gallery", "park", "square"],
    "appreciating": ["gallery", "temple", "park"],

    # Physical / outdoor
    "walking": ["park", "square"],
    "exercising": ["park", "square"],
    "tai chi": ["park", "square", "temple"],
    "strolling": ["park", "square"],
    "exploring": ["park", "square"],
    "gardening": ["park", "home"],

    # Rest / personal
    "sleeping": ["home"],
    "resting": ["home", "park", "cafe"],
    "napping": ["home", "park"],
    "relaxing": ["home", "park", "cafe", "teahouse"],

    # Work-related
    "working": ["shop", "cafe", "library", "townhall"],
    "cleaning": ["home", "cafe", "shop"],
    "organizing": ["home", "shop", "library"],
    "teaching": ["school", "library"],
    "learning": ["school", "library"],
    "tutoring": ["school", "library", "cafe"],
    "stocking": ["shop"],

    # Health / wellbeing
    "healing": ["pharmacy", "temple", "home"],
    "praying": ["temple"],
    "worshipping": ["temple"],

    # Community
    "volunteering": ["townhall", "school", "temple"],
    "attending": ["townhall", "school", "temple", "gallery"],
    "announcing": ["townhall", "square"],
}


class StatusMovementResolver:
    """Resolves the best location for an agent based on their current action."""

    def __init__(self, world: Any) -> None:
        self._world = world

    def resolve_location(
        self,
        action: str,
        agent_x: int,
        agent_y: int,
        agent_name: str,
        current_location: Optional[Any] = None,
    ) -> Optional[Any]:
        """Find the best location matching the agent's action.

        Priority order:
        1. If already at a matching location type, stay there
        2. Prefer owned locations (e.g., agent's home for "cooking")
        3. Find nearest matching location type

        Args:
            action: Current action string (e.g., "writing a novel")
            agent_x, agent_y: Agent's current position
            agent_name: Agent's name (for ownership checks)
            current_location: Agent's current Location object (or None)

        Returns:
            Best matching Location, or None if no match found
        """
        # Find matching location types from action keywords
        matching_types = self._get_matching_types(action)
        if not matching_types:
            return None

        # 1. Already at a matching location?
        if current_location and current_location.type in matching_types:
            return current_location

        # 2. Find candidates
        candidates = []
        for loc in self._world.locations:
            if loc.type not in matching_types:
                continue

            dist = abs(loc.center[0] - agent_x) + abs(loc.center[1] - agent_y)
            type_priority = matching_types.index(loc.type)

            # Bonus for owned locations
            owned_bonus = -100 if loc.owner == agent_name else 0

            # Score: lower is better
            score = type_priority * 50 + dist + owned_bonus
            candidates.append((score, loc))

        if not candidates:
            return None

        candidates.sort(key=lambda c: c[0])
        return candidates[0][1]

    @staticmethod
    def _get_matching_types(action: str) -> List[str]:
        """Extract location type preferences from an action string."""
        action_lower = action.lower()
        all_matches: List[str] = []
        best_match_len = 0

        for keyword, types in ACTION_LOCATION_AFFINITY.items():
            if keyword in action_lower:
                if len(keyword) > best_match_len:
                    best_match_len = len(keyword)
                    all_matches = list(types)
                elif len(keyword) == best_match_len:
                    # Merge types preserving order
                    for t in types:
                        if t not in all_matches:
                            all_matches.append(t)

        return all_matches
