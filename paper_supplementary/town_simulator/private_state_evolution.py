"""Event-driven evolution of agent private state.

The private fields on `AgentState` (`availability`, `emotional_state`,
`unspoken_goal`, `beliefs_about_others`) used to be inert default values that
tests / queries had to monkey-patch. That made the AURATown setup look like
"a frozen SQL table" to reviewers.

This module turns those fields into deterministic functions of:
  - the agent's current action keyword,
  - co-located peer count,
  - workplace context (cafe, shop, library, etc.),
  - recent simulation events,
  - time-of-day windows.

Rules are intentionally simple, transparent, and cheap (zero LLM calls) so
the evolution itself is reproducible and not a confound when ablating other
mechanisms. Beliefs-about-others lag actual state and only refresh on
co-location, so AURA's second-order ToM probe still has work to do.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

# ────────────────────────────────────────────────────────────────────
# Action / location keyword patterns
# ────────────────────────────────────────────────────────────────────

_BUSY_ACTION_PATTERNS = [
    r"\bserv(?:e|ing)\b", r"\bbrew", r"\bbak(?:e|ing)", r"\brestock",
    r"\bclos(?:e|ing) (?:up|down|out)", r"\brush\b",
]
_DEEP_FOCUS_PATTERNS = [
    r"\bwrit(?:e|ing)\b", r"\bdraft(?:ing)?\b", r"\bchapter\b",
    r"\bstud(?:y|ying)\b", r"\bresearch", r"\bmeditat",
]
_RELAXED_PATTERNS = [
    r"\bwalk", r"\brest", r"\bread", r"\btai chi", r"\bstargaz",
    r"\bsleep", r"\beat", r"\blunch", r"\bbreakfast", r"\bdinner",
]

_ACTION_KEYWORDS = {
    "busy": _BUSY_ACTION_PATTERNS,
    "focus": _DEEP_FOCUS_PATTERNS,
    "relax": _RELAXED_PATTERNS,
}


def _match_any(text: str, patterns: Sequence[str]) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in patterns)


# Workplace map: which location is each agent's primary work site, plus
# how a "loaded" workplace is detected. Loaded = busy_threshold peers in
# the same location (customers/visitors) AND the agent's action is
# work-flavored.
WORKPLACE_OF: Dict[str, str] = {
    "Lin Wei": "Sunrise Cafe",
    "Chen Mei": "Chen's General Store",
    "Zhang Hao": "Zhang Hao's Home",  # writer works at home/library
    "Liu Yang": "Library",  # student
    "Wang Jun": "Library",  # retired prof reads here
}


# ────────────────────────────────────────────────────────────────────
# Rule outcome
# ────────────────────────────────────────────────────────────────────

@dataclass
class PrivateStateUpdate:
    """The outcome of one tick of private-state evolution for one agent.

    Returned (rather than mutated in place) so callers can log a transition
    trace for the paper / debug UI without a second pass.
    """
    availability: str
    emotional_state: str
    unspoken_goal: Optional[str]
    rule_fired: str  # which case in the rule table fired (for transparency)


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────

def evolve_private_state(
    *,
    agent_name: str,
    current_action: str,
    location_name: str,
    nearby_count: int,
    hour_24: int,
    minute: int,
    recent_event_descriptions: Sequence[str] = (),
    prev_availability: str = "available",
    prev_emotional_state: str = "neutral",
) -> PrivateStateUpdate:
    """Compute one tick of private-state evolution for one agent.

    The rule table is intentionally short and transparent. Order matters:
    earlier rules win. Every code path returns through one named rule so
    we can show "which rule fired this tick" in logs.
    """

    # ── 1. Sleep / off-hours dominates everything
    if _match_any(current_action, [r"\bsleep"]):
        return PrivateStateUpdate(
            availability="do_not_disturb",
            emotional_state="resting",
            unspoken_goal=None,
            rule_fired="sleep",
        )

    workplace = WORKPLACE_OF.get(agent_name)
    at_workplace = workplace and (location_name == workplace)
    loaded = at_workplace and nearby_count >= 3 and _match_any(
        current_action, _BUSY_ACTION_PATTERNS,
    )

    # ── 2. Workplace under heavy load → busy + tired-focused
    if loaded:
        next_break_window = _next_break_minutes(hour_24, minute)
        return PrivateStateUpdate(
            availability="busy",
            emotional_state="tired-focused",
            unspoken_goal=(
                f"wants to close out the current rush before the next break "
                f"in {next_break_window} min" if next_break_window is not None
                else "wants to clear the queue"
            ),
            rule_fired="workplace_loaded",
        )

    # ── 3. Deep-focus action (writing/studying/research) → DND
    if _match_any(current_action, _DEEP_FOCUS_PATTERNS):
        return PrivateStateUpdate(
            availability="do_not_disturb",
            emotional_state="creatively flowing",
            unspoken_goal=_focus_goal(agent_name, current_action),
            rule_fired="deep_focus",
        )

    # ── 4. Empty workplace / slow shift → lonely (Chen Mei pattern)
    if at_workplace and nearby_count == 0 and _is_open_hour(agent_name, hour_24):
        return PrivateStateUpdate(
            availability="available",
            emotional_state="lonely",
            unspoken_goal="hoping a regular drops by for conversation",
            rule_fired="workplace_empty",
        )

    # ── 5. Relaxed activities → calm-available
    if _match_any(current_action, _RELAXED_PATTERNS):
        # Carry over prior emotional shading slightly; relaxed neutral default
        emo = "calm"
        if prev_emotional_state in {"tired-focused", "stressed"}:
            emo = "recovering"
        elif prev_emotional_state == "lonely":
            emo = "lonely"  # walking around alone doesn't fix loneliness
        return PrivateStateUpdate(
            availability="available",
            emotional_state=emo,
            unspoken_goal=None,
            rule_fired="relaxed",
        )

    # ── 6. Recent stressful event override
    stressful = any(
        _match_any(d, [r"\bargument\b", r"\bemergency\b", r"\bfailed\b",
                       r"\bbroke\b", r"\bworried\b"])
        for d in recent_event_descriptions
    )
    if stressful:
        return PrivateStateUpdate(
            availability="available",
            emotional_state="stressed",
            unspoken_goal=None,
            rule_fired="recent_stress",
        )

    # ── 7. Default
    return PrivateStateUpdate(
        availability="available",
        emotional_state="neutral",
        unspoken_goal=None,
        rule_fired="default",
    )


def update_beliefs_about_others(
    *,
    self_agent_name: str,
    self_beliefs: Dict[str, Dict[str, Any]],
    co_located_others: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Refresh second-order ToM beliefs on co-location.

    Beliefs lag actual state on purpose: only when this agent shares a
    location with another agent does this agent's belief about that other
    refresh. Otherwise the belief sits stale, which is what the
    `second_order` query subcategory probes (e.g. "does Lin Wei think
    Zhang Hao is at home? — actually he's at the cafe right now, but she
    last saw him at home").

    Args:
        self_beliefs: existing dict of {other_name: {availability, mood, location}}.
        co_located_others: list of dicts with at least
            {name, location, availability, emotional_state} for agents
            currently in the same location as this agent.

    Returns:
        Updated beliefs dict (new dict; caller assigns).
    """
    out = dict(self_beliefs)
    for other in co_located_others:
        oname = other.get("name")
        if not oname or oname == self_agent_name:
            continue
        out[oname] = {
            "location": other.get("location"),
            "availability": other.get("availability", "available"),
            "emotional_state": other.get("emotional_state", "neutral"),
        }
    return out


# ────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────

def _next_break_minutes(hour: int, minute: int) -> Optional[int]:
    """Minutes to the next conventional break (lunch noon, dinner 6pm)."""
    breaks = [(12, 0), (18, 0)]
    now = hour * 60 + minute
    candidates = [h * 60 + m - now for h, m in breaks if h * 60 + m >= now]
    return min(candidates) if candidates else None


def _is_open_hour(agent_name: str, hour: int) -> bool:
    """Conventional opening hours per agent role."""
    if agent_name == "Lin Wei":
        return 7 <= hour < 17  # cafe
    if agent_name == "Chen Mei":
        return 8 <= hour < 18  # shop
    if agent_name in ("Liu Yang", "Wang Jun"):
        return 8 <= hour < 19  # library hours
    return 8 <= hour < 18


def _focus_goal(agent_name: str, action: str) -> Optional[str]:
    """Best-guess unspoken goal for a deep-focus action."""
    a = (action or "").lower()
    if "writ" in a or "draft" in a or "chapter" in a:
        return "wants to finish a writing milestone before sunset"
    if "stud" in a:
        return "trying to make progress on the current syllabus before the next class"
    if "research" in a:
        return "tracking down a specific reference"
    if "meditat" in a:
        return "wants undisturbed quiet to settle the mind"
    return None
