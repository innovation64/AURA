"""Rule-based fallback engine for AURA Town.

When the LLM is unavailable (no API key, network error, etc.) every agent
behaviour falls back to deterministic, locally-computed actions so the
simulation stays alive and the demo remains interactive.
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agents import AgentProfile, TownAgent

# ── Location keyword → canonical map name ──────────────────────────────

_LOCATION_KEYWORDS: Dict[str, str] = {
    "cafe": "Sunrise Cafe",
    "coffee": "Sunrise Cafe",
    "pastries": "Sunrise Cafe",
    "brew": "Sunrise Cafe",
    "library": "Town Library",
    "read": "Town Library",
    "research": "Town Library",
    "book": "Town Library",
    "store": "Chen's General Store",
    "shop": "Chen's General Store",
    "snack": "Chen's General Store",
    "shelves": "Chen's General Store",
    "restock": "Chen's General Store",
    "groceries": "Chen's General Store",
    "park": "Town Park",
    "walk": "Town Park",
    "nature": "Town Park",
    "tai chi": "Town Park",
    "pond": "Town Park",
    "stargaze": "Town Park",
    "square": "Town Square",
    "town square": "Town Square",
    "fountain": "Town Square",
    "community": "Town Square",
    "home": "__HOME__",
    "breakfast": "__HOME__",
    "bed": "__HOME__",
    "rest": "__HOME__",
    "sleep": "__HOME__",
    "relax": "__HOME__",
}


def _resolve_location(text: str, agent_name: str) -> str:
    """Map a routine description to a canonical location name."""
    lower = text.lower()
    for keyword, loc in _LOCATION_KEYWORDS.items():
        if keyword in lower:
            if loc == "__HOME__":
                return f"{agent_name}'s Home"
            return loc
    # Default: stay at home
    return f"{agent_name}'s Home"


def _parse_hour(time_str: str) -> Optional[int]:
    """Extract hour from strings like '07:00', '7:00 AM', '14:30'."""
    m = re.match(r"(\d{1,2}):(\d{2})", time_str.strip())
    if m:
        return int(m.group(1))
    return None


# ── Public fallback functions ──────────────────────────────────────────


def fallback_daily_plan(profile: AgentProfile) -> List[str]:
    """Convert the agent's static daily_routine into a plan list."""
    return list(profile.daily_routine)


def fallback_action(
    agent: TownAgent,
    world_hour: int,
    world_minute: int,
) -> Dict[str, Any]:
    """Pick an action + location from the agent's routine based on current time.

    Returns a dict compatible with the LLM action format:
      {"action": str, "location": str, "thought": str, "emoji": str}
    """
    routine = agent.profile.daily_routine
    current_minutes = world_hour * 60 + world_minute

    # Find the routine entry whose time is <= current time (latest match)
    best_entry: Optional[str] = None
    for entry in routine:
        parts = entry.split(" - ", 1)
        if len(parts) < 2:
            continue
        hour = _parse_hour(parts[0])
        if hour is None:
            continue
        entry_minutes = hour * 60
        if entry_minutes <= current_minutes:
            best_entry = entry

    if best_entry is None:
        best_entry = routine[0] if routine else "resting at home"

    # Extract activity text and target location
    parts = best_entry.split(" - ", 1)
    activity = parts[1] if len(parts) == 2 else best_entry
    location = _resolve_location(activity, agent.name)

    # Trim activity to a short action phrase
    action = activity.split(",")[0].strip().lower()

    return {
        "action": action,
        "location": location,
        "thought": f"Following my routine: {activity}",
        "emoji": agent.profile.emoji,
    }


# ── Conversation templates ────────────────────────────────────────────

_GREETINGS = {
    "friend": [
        '{a1}: "Hey {a2_name}! How\'s it going today?"',
        '{a2}: "Pretty good! Just {a2_action}. You?"',
        '{a1}: "Same old, {a1_action}. Nice running into you here at {location}."',
    ],
    "mentor": [
        '{a1}: "Good to see you, {a2_name}."',
        '{a2}: "Likewise! I was hoping to run into you."',
        '{a1}: "How have things been? I heard you\'ve been busy."',
        '{a2}: "Yes, but it\'s the good kind of busy. Thanks for asking."',
    ],
    "acquaintance": [
        '{a1}: "Oh hi {a2_name}, didn\'t expect to see you here."',
        '{a2}: "Hey! I was just {a2_action}. Nice day, isn\'t it?"',
        '{a1}: "It really is. Enjoy the rest of your day!"',
    ],
}

_TIME_TOPICS = {
    "morning": [
        '{a1}: "Beautiful morning, isn\'t it?"',
        '{a2}: "Absolutely. I love this time of day."',
    ],
    "afternoon": [
        '{a1}: "The afternoon is flying by."',
        '{a2}: "Tell me about it — I still have so much to do!"',
    ],
    "evening": [
        '{a1}: "Wrapping up for the day?"',
        '{a2}: "Almost. It\'s been a long one."',
    ],
}


def _time_of_day(hour: int) -> str:
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    return "evening"


def _relationship_bucket(rel_text: str) -> str:
    lower = rel_text.lower()
    if any(w in lower for w in ("friend", "close", "best")):
        return "friend"
    if any(w in lower for w in ("mentor", "professor", "guide", "teach")):
        return "mentor"
    return "acquaintance"


def fallback_conversation(
    agent1: TownAgent,
    agent2: TownAgent,
    location: str,
    hour: int,
) -> List[Dict[str, str]]:
    """Generate a short scripted conversation between two agents.

    Returns a list of {"speaker": ..., "text": ...} dicts.
    """
    rel = agent1.profile.relationships.get(agent2.name, "acquaintance")
    bucket = _relationship_bucket(rel)

    template = _GREETINGS.get(bucket, _GREETINGS["acquaintance"])
    tod = _time_of_day(hour)
    extra = _TIME_TOPICS.get(tod, _TIME_TOPICS["afternoon"])

    # Build lines
    fmt = {
        "a1": agent1.name,
        "a2": agent2.name,
        "a1_action": agent1.state.current_action,
        "a2_action": agent2.state.current_action,
        "a2_name": agent2.name.split()[0],
        "location": location,
    }

    lines = template + random.choice([extra, []])  # sometimes add time topic

    exchanges: List[Dict[str, str]] = []
    for line in lines:
        formatted = line.format(**fmt)
        # Parse "Speaker: \"text\""
        m = re.match(r'^(.+?):\s*"(.+)"$', formatted)
        if m:
            exchanges.append({"speaker": m.group(1), "text": m.group(2)})
        else:
            exchanges.append({"speaker": agent1.name, "text": formatted})

    return exchanges


# ── Reflection ────────────────────────────────────────────────────────

_REFLECTION_TEMPLATES = {
    "social": "Spending time with people in town is important to me.",
    "routine": "Sticking to my routine helps me feel grounded.",
    "place": "I appreciate the places in this town — each one has its own character.",
}


def fallback_reflection(agent_name: str, personality: str) -> List[str]:
    """Return 2-3 deterministic insights based on personality keywords."""
    insights: List[str] = []
    lower = personality.lower()

    if any(w in lower for w in ("social", "warm", "community", "nurturing")):
        insights.append(
            f"{agent_name} values the social bonds in town."
        )
    if any(w in lower for w in ("thoughtful", "reflective", "observer", "wise")):
        insights.append(
            f"{agent_name} finds meaning in quiet observation and reflection."
        )
    if any(w in lower for w in ("curious", "energetic", "idealistic", "exploring")):
        insights.append(
            f"{agent_name} is driven by curiosity and a desire to learn."
        )
    if any(w in lower for w in ("practical", "dependable", "organized")):
        insights.append(
            f"{agent_name} appreciates structure and reliability."
        )

    # Always include at least one generic insight
    if not insights:
        insights.append(f"{agent_name} is reflecting on the day's events.")
    insights.append(
        f"{agent_name} feels grateful for the familiar rhythm of life in town."
    )

    return insights[:3]


# ── Chat response ─────────────────────────────────────────────────────


def fallback_chat_response(
    agent: TownAgent,
    message: str,
    env_context: Dict[str, Any],
) -> str:
    """Build a descriptive response without calling LLM."""
    loc = env_context.get("location", "somewhere in town")
    time_str = env_context.get("time", "")
    action = env_context.get("current_action", "going about my day")

    nearby = env_context.get("nearby_agents") or []
    nearby_names = [a["name"] for a in nearby] if nearby else []

    parts = [
        f"[{agent.name} — {agent.profile.occupation}]",
        f"Right now I'm at {loc}, {action}.",
    ]
    if time_str:
        parts.append(f"The time is {time_str}.")
    if nearby_names:
        parts.append(f"I can see {', '.join(nearby_names)} nearby.")

    parts.append(
        f'You asked: "{message}" — '
        "I don't have a detailed answer at the moment, "
        "but I'm happy to chat more as the day goes on!"
    )

    return " ".join(parts)


# ── Evolution fallback ─────────────────────────────────────────────

_SEASONS = ["spring", "summer", "autumn", "winter"]


def fallback_evolution_check(
    world_properties: Dict[str, Any],
    tick_index: int,
    utilization: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Deterministic evolution: seasonal changes every 20 ticks.

    Returns a list of mutation dicts (compatible with WorldMutation parsing).
    """
    mutations: List[Dict[str, Any]] = []

    # Seasonal rotation every 20 ticks
    if tick_index > 0 and tick_index % 20 == 0:
        current = world_properties.get("season", "spring")
        try:
            idx = _SEASONS.index(current)
            next_season = _SEASONS[(idx + 1) % len(_SEASONS)]
        except ValueError:
            next_season = "summer"

        mutations.append({
            "type": "modify_property",
            "target": "season",
            "payload": {"value": next_season},
            "reason": f"Seasonal change: {current} → {next_season}",
        })

        # Weather follows season
        season_weather = {
            "spring": "mild and breezy",
            "summer": "warm and sunny",
            "autumn": "cool and crisp",
            "winter": "cold and overcast",
        }
        mutations.append({
            "type": "modify_property",
            "target": "weather",
            "payload": {"value": season_weather.get(next_season, "clear")},
            "reason": f"Weather changes with {next_season}",
        })

    return mutations
