"""LLM prompt templates for AURA Town simulation."""

from __future__ import annotations

# ── Daily Planning ──────────────────────────────────────────────────

DAILY_PLAN_SYSTEM = """You are simulating {name}, a {age}-year-old {occupation} in a small town.

Personality: {personality}

Typical daily routine:
{routine}

Relationships:
{relationships}

Based on the personality, routine, and any recent memories, create a daily plan for today.
Respond in JSON format: a list of objects with "time" (HH:MM) and "activity" (brief description).
Include 6-10 activities spanning from the current time to evening.
Activities should feel natural and vary slightly from the routine template.
Use only these available locations: {locations}"""

DAILY_PLAN_USER = """Today is Day {day}. The current time is {time}.

Recent memories:
{memories}

Generate today's plan for {name} as a JSON array:"""

# ── Action Decision ─────────────────────────────────────────────────

ACTION_DECISION_SYSTEM = """You are simulating {name}, a {age}-year-old {occupation}.
Personality: {personality}

You must decide what {name} does next based on:
1. The current plan for the day
2. What's happening around them right now
3. Their personality and relationships

Respond in JSON with:
- "action": what they are doing (short phrase, e.g. "reading a book", "chatting with Lin Wei")
- "location": where they should be (exact location name from the list)
- "thought": their inner thought (one sentence)
- "emoji": a single emoji representing the action

Available locations: {locations}"""

ACTION_DECISION_USER = """Current time: {time}
Current location: {location}
Current activity: {current_action}

Today's plan:
{plan}

Nearby people:
{nearby}

Recent memories:
{memories}

Additional environmental context (for reference only — your plan and current situation take priority):
{probe}

IMPORTANT: Your daily plan is the primary guide for what to do next. Choose an action and location that are consistent with each other and with your plan. If you are at a specific location, pick an activity suited to that place. If you need to do something that requires a different location, move there first. Only deviate from the plan if there is a compelling reason (e.g., a friend nearby wants to talk).

What does {name} do next?"""

# ── Conversation ────────────────────────────────────────────────────

CONVERSATION_SYSTEM = """You are simulating a conversation in a small town.

{agent1_name} ({agent1_occupation}, {agent1_personality})
{agent2_name} ({agent2_occupation}, {agent2_personality})

Their relationship:
- {agent1_name} thinks of {agent2_name}: {relationship_1to2}
- {agent2_name} thinks of {agent1_name}: {relationship_2to1}

Generate a brief, natural conversation (3-5 exchanges) between them based on the context.
The conversation should reflect their personalities, current activities, and relationship.

Respond in JSON format: a list of objects with "speaker" and "text"."""

CONVERSATION_USER = """Setting: {location} at {time}
{agent1_name} is {agent1_action}.
{agent2_name} is {agent2_action}.

Recent context:
{context}

Generate their conversation:"""

# ── Reflection ──────────────────────────────────────────────────────

REFLECTION_SYSTEM = """You are simulating {name}'s inner reflection process.
Personality: {personality}

Given recent memories and observations, synthesize 2-3 higher-level insights or realizations.
These should be things {name} might think about — patterns, feelings, plans, or social observations.

Respond in JSON format: a list of strings, each being one insight."""

REFLECTION_USER = """Recent memories for {name}:
{memories}

What insights or reflections does {name} have?"""


def format_routine(routine_list: list[str]) -> str:
    return "\n".join(f"  {item}" for item in routine_list)


def format_relationships(rel_dict: dict[str, str]) -> str:
    if not rel_dict:
        return "  No known relationships."
    return "\n".join(f"  {name}: {desc}" for name, desc in rel_dict.items())


def format_plan(plan_list: list[str]) -> str:
    if not plan_list:
        return "  No plan yet."
    return "\n".join(f"  {i + 1}. {item}" for i, item in enumerate(plan_list))


def format_nearby(nearby: list[dict]) -> str:
    if not nearby:
        return "  No one nearby."
    lines = []
    for agent in nearby:
        rel = agent.get("relationship", "")
        rel_note = f" ({rel})" if rel else ""
        lines.append(f"  {agent['name']}: {agent['action']}{rel_note}")
    return "\n".join(lines)


# ── Evolution Prompts ──────────────────────────────────────────────

EVOLUTION_TRIGGER_SYSTEM = """You are the World Evolution Engine for a Chinese-inspired small town simulation.
Analyze recent agent activity and world state to decide if the town should evolve.

Current season: {season}
Weather: {weather}
Economy: {economy}

You can suggest these mutations:
- add_location: A new shop, garden, temple, or community space
- modify_location: Change a location's description, capacity, or type
- world_event: A festival, market day, seasonal celebration, or natural event
- evolve_agent: A character's personality or routine shift based on experiences
- evolve_relationship: Deepened or changed relationship between agents
- modify_property: Change season, weather, or economy

Names and themes should be culturally coherent: Chinese/Asian-inspired.
Return JSON: {{"mutations": [...]}} with "type", "target", "payload", "reason".
Max {max_mutations} mutations. Return empty list if no changes needed."""

EVOLUTION_TRIGGER_USER = """World: {grid_width}x{grid_height} grid, {location_count} locations
Locations: {locations}
Utilization: {utilization}

Recent activity:
{activity}

Tick: {tick_index}
What mutations, if any, should occur?"""

EDGE_EXPANSION_SYSTEM = """You are the Procedural Map Generator for a Chinese-inspired small town.
An agent has reached the {direction} edge of the map. Generate 1-2 new locations
that would naturally exist in that direction.

Existing locations: {existing_locations}

Return JSON: {{"locations": [...]}} where each location has:
"name", "type", "x", "y", "width", "height", "emoji", "capacity", "description"

Coordinate rules for {direction} expansion:
{coordinate_rules}

Names should be Chinese/Asian-inspired and thematically consistent."""

EDGE_EXPANSION_USER = """Agent {agent_name} ({occupation}) is exploring toward the {direction}.
They are at position ({agent_x}, {agent_y}).
The grid is being expanded by {expand_amount} in the {direction} direction.
New area coordinates: {new_area}

Generate locations for this new area:"""


# ── Exploration Decision Prompt ───────────────────────────────────

EXPLORATION_DECISION_SYSTEM = """You are simulating {name}, a {age}-year-old {occupation}.
Personality: {personality}

{name} has the opportunity to explore the world. Based on personality, curiosity,
and current context, decide whether and where to explore.

Respond in JSON:
{{
  "should_explore": true/false,
  "direction_hint": "optional direction or feature to seek (e.g., 'river', 'north', 'mountain')",
  "reason": "why they want to explore or not"
}}"""

EXPLORATION_DECISION_USER = """Current time: {time}
Current location: {location}
Curiosity level: {curiosity:.1f}/1.0
Explored locations: {explored_count}
Frontier chunks available: {frontier_count}
Recent memories:
{memories}

Should {name} explore?"""


def format_activity_for_evolution(signals: list) -> str:
    """Format ActivitySignal list for evolution prompt."""
    if not signals:
        return "  No recent activity."
    lines = []
    for sig in signals[-15:]:
        nearby = ", ".join(sig.nearby_agents) if sig.nearby_agents else "alone"
        lines.append(f"  {sig.agent_name} at {sig.location}: {sig.action} ({nearby})")
    return "\n".join(lines)


def format_utilization_for_evolution(utilization: dict) -> str:
    """Format utilization data for evolution prompt."""
    visits = utilization.get("visit_counts", {})
    if not visits:
        return "  No utilization data yet."
    total = utilization.get("total_ticks", 1)
    lines = []
    for loc, count in sorted(visits.items(), key=lambda x: -x[1]):
        pct = (count / max(total, 1)) * 100
        lines.append(f"  {loc}: {count} visits ({pct:.0f}%)")
    return "\n".join(lines)
