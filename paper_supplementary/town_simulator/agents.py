"""Agent profiles, state management, and perception for AURA Town."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .world import Location, TownWorld


@dataclass
class AgentProfile:
    name: str
    age: int
    occupation: str
    personality: str
    daily_routine: List[str]
    emoji: str
    relationships: Dict[str, str] = field(default_factory=dict)
    # Evolution history
    personality_history: List[str] = field(default_factory=list)
    routine_history: List[List[str]] = field(default_factory=list)
    evolved_traits: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Mutable runtime state for an agent."""

    x: int = 0
    y: int = 0
    current_action: str = "sleeping"
    current_location_name: str = ""
    destination: Optional[str] = None
    daily_plan: List[str] = field(default_factory=list)
    plan_index: int = 0
    conversation_cooldown: int = 0  # ticks until next conversation allowed
    last_probe_tick: int = -1000
    last_probe_summary: str = ""
    last_probe_steps: List[Dict[str, Any]] = field(default_factory=list)
    # Curiosity / exploration
    curiosity: float = 0.0
    explored_locations: Set[str] = field(default_factory=set)
    exploration_target: Optional[Tuple[int, int]] = None
    # Frontier-based exploration (Phase 1A)
    exploration_goal: Optional[Any] = None  # ExplorationGoal from spatial_explorer
    exploration_history: List[Tuple[int, int]] = field(default_factory=list)  # visited chunk coords
    preferred_biomes: List[str] = field(default_factory=list)
    # Player control
    player_controlled: bool = False
    pending_action: Optional[Dict[str, Any]] = None
    # Visual bubbles (set by simulation, cleared after N ticks)
    speech_bubble: Optional[str] = None
    speech_bubble_tick: int = 0
    thought_bubble: Optional[str] = None
    thought_bubble_tick: int = 0
    mood: str = "neutral"  # neutral/happy/thinking/tired/excited
    interaction_partner: Optional[str] = None
    explored_chunks: Set[Tuple[int, int]] = field(default_factory=set)
    # Private state — fields NOT visible in the passive scene snapshot.
    # These drive the implicit-intent query set: a user's surface question
    # ("where is Lin Wei?") may actually be asking about an agent's
    # availability/mood/unspoken goals, which only probing can reveal.
    # Simulation owns updates; scene.build() deliberately omits these
    # so that AURA's IntentInferrer / Explore stage must request them.
    availability: str = "available"       # available/busy/do_not_disturb
    unspoken_goal: Optional[str] = None   # what the agent is privately trying to do
    secrets: List[str] = field(default_factory=list)  # known only to the agent
    emotional_state: str = "neutral"      # finer-grained than mood; drives ToM alerts
    # Second-order ToM: this agent's beliefs about OTHER agents' private state.
    # Keyed by the other agent's name; each entry is a dict with (optionally)
    # the same schema as the private fields above (e.g. {'availability':
    # 'busy', 'emotional_state': 'stressed'}). These are beliefs, not facts;
    # they may be stale or wrong, and second-order ToM queries ("does Lin Wei
    # think Zhang Hao is available?") probe precisely this layer.
    beliefs_about_others: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# ── Default agent profiles ──────────────────────────────────────────

AGENT_PROFILES = [
    AgentProfile(
        name="Lin Wei",
        age=32,
        occupation="Cafe owner",
        personality="Warm, social, and nurturing. Loves bringing people together over coffee. A good listener who remembers everyone's favorite drinks.",
        daily_routine=[
            "06:00 - Wake up and prepare for the day",
            "07:00 - Open Sunrise Cafe, brew coffee, bake pastries",
            "10:00 - Chat with morning regulars at the cafe",
            "12:00 - Lunch break, sometimes walk to the park",
            "14:00 - Afternoon cafe service",
            "17:00 - Close the cafe, clean up",
            "18:00 - Visit the shop or town square",
            "20:00 - Go home, read a book, rest",
        ],
        emoji="👩‍🍳",
        relationships={
            "Zhang Hao": "Regular customer and friend. He writes at the cafe every morning.",
            "Chen Mei": "Close friend and neighbor. They often share recipes.",
            "Liu Yang": "Young student who drops by for afternoon coffee. Like a little sister.",
            "Wang Jun": "Respected elder. He gives great life advice over tea.",
        },
    ),
    AgentProfile(
        name="Zhang Hao",
        age=28,
        occupation="Writer",
        personality="Introverted and thoughtful. A keen observer of human nature. Struggles with writer's block but finds inspiration in everyday conversations.",
        daily_routine=[
            "07:00 - Wake up, morning routine",
            "08:00 - Go to Sunrise Cafe, write while drinking coffee",
            "11:00 - Walk to the library for research",
            "13:00 - Lunch at the cafe or park",
            "14:00 - Continue writing at the library",
            "17:00 - Take a walk in the park for inspiration",
            "19:00 - Go home, review the day's writing",
            "21:00 - Read before bed",
        ],
        emoji="✍️",
        relationships={
            "Lin Wei": "The cafe owner who always saves his favorite spot. A warm friend.",
            "Chen Mei": "Acquaintance from the shop. She tells interesting stories about customers.",
            "Liu Yang": "A curious student who sometimes asks about writing. Reminds him of his younger self.",
            "Wang Jun": "A mentor figure. The professor's wisdom inspires his writing.",
        },
    ),
    AgentProfile(
        name="Chen Mei",
        age=45,
        occupation="Shop owner",
        personality="Practical, community-oriented, and observant. Knows everyone's business (in a caring way). The unofficial town connector.",
        daily_routine=[
            "06:30 - Wake up, tend the herb garden",
            "08:00 - Open Chen's General Store",
            "10:00 - Morning customers, restock shelves",
            "12:30 - Quick lunch in the back of the shop",
            "13:00 - Afternoon shift at the store",
            "16:00 - Walk to the town square for fresh air",
            "17:30 - Close the shop",
            "18:00 - Visit the cafe or chat with neighbors",
            "20:30 - Go home, relax",
        ],
        emoji="🏪",
        relationships={
            "Lin Wei": "Best friend. They share everything and look out for each other.",
            "Zhang Hao": "Quiet young man who buys notebooks and pens. She worries he doesn't eat enough.",
            "Liu Yang": "Student who helps stock shelves part-time on weekends. Energetic kid.",
            "Wang Jun": "Wise neighbor. They discuss town affairs and community events.",
        },
    ),
    AgentProfile(
        name="Liu Yang",
        age=20,
        occupation="University student",
        personality="Curious, energetic, and idealistic. Studies environmental science. Loves exploring nature and asking big questions.",
        daily_routine=[
            "07:30 - Wake up, quick breakfast",
            "08:30 - Study at home or library",
            "10:30 - Take a break, walk to the park",
            "12:00 - Lunch, maybe at the cafe",
            "13:00 - Afternoon study session at the library",
            "15:30 - Explore the park, observe nature",
            "17:00 - Visit the shop for snacks",
            "18:30 - Go to the town square, hang out",
            "20:00 - Go home, study or stargaze",
        ],
        emoji="🎒",
        relationships={
            "Lin Wei": "The kind cafe owner. She gives him free cookies sometimes.",
            "Zhang Hao": "A cool writer who gives him book recommendations.",
            "Chen Mei": "The shop owner he helps on weekends. She's like an aunt.",
            "Wang Jun": "His unofficial mentor. The professor teaches him about philosophy and life.",
        },
    ),
    AgentProfile(
        name="Wang Jun",
        age=68,
        occupation="Retired philosophy professor",
        personality="Wise, calm, and reflective. Enjoys mentoring the younger generation. Takes daily walks and reads extensively. Has a dry sense of humor.",
        daily_routine=[
            "06:00 - Wake up, morning tai chi in the park",
            "07:30 - Breakfast at home, read the news",
            "09:00 - Walk to the library, read or discuss ideas",
            "11:30 - Visit the cafe for tea",
            "13:00 - Lunch at home",
            "14:30 - Walk in the park, perhaps meet someone to talk to",
            "16:00 - Visit the town square, observe town life",
            "18:00 - Go home, evening reading",
            "21:00 - Rest",
        ],
        emoji="🎓",
        relationships={
            "Lin Wei": "A kind young woman. Her cafe is a pleasant place for tea and thought.",
            "Zhang Hao": "A talented but struggling writer. He tries to guide him gently.",
            "Chen Mei": "Practical and dependable. They discuss community matters.",
            "Liu Yang": "His favorite student-like figure. The boy's curiosity reminds him of his teaching days.",
        },
    ),
]


class TownAgent:
    """An agent in the town simulation with profile and mutable state."""

    def __init__(self, profile: AgentProfile, world: TownWorld) -> None:
        self.profile = profile
        self.state = AgentState()
        self._world = world

        # Place agent at their home
        home = world.get_home_for(profile.name)
        if home:
            cx, cy = home.center
            self.state.x = cx
            self.state.y = cy
            self.state.current_location_name = home.name

    @property
    def name(self) -> str:
        return self.profile.name

    @property
    def position(self) -> Tuple[int, int]:
        return (self.state.x, self.state.y)

    @property
    def location(self) -> Optional[Location]:
        return self._world.get_location_at(self.state.x, self.state.y)

    def perceive(self, all_agents: List[TownAgent]) -> Dict[str, Any]:
        """Build a perception of the current environment."""
        current_loc = self.location
        loc_name = current_loc.name if current_loc else "somewhere on the road"
        loc_desc = current_loc.description if current_loc else "An open area between buildings."

        # Find nearby agents at the same location
        nearby = []
        for other in all_agents:
            if other.name == self.name:
                continue
            other_loc = other.location
            if current_loc and other_loc and current_loc.name == other_loc.name:
                nearby.append(
                    {
                        "name": other.name,
                        "action": other.state.current_action,
                        "relationship": self.profile.relationships.get(other.name, "stranger"),
                    }
                )

        return {
            "location": loc_name,
            "location_description": loc_desc,
            "time": self._world.time.display,
            "time_24h": self._world.time.time_24h,
            "nearby_agents": nearby,
            "current_action": self.state.current_action,
        }

    def move_to_location(self, target_name: str) -> bool:
        """Move toward a named location. Returns True if arrived."""
        loc = self._world.get_location_by_name(target_name)
        if loc is None:
            return False

        tx, ty = loc.center
        new_x, new_y = self._world.move_toward(
            self.state.x, self.state.y, tx, ty, speed=3
        )
        self.state.x = new_x
        self.state.y = new_y

        # Check if we arrived
        current = self._world.get_location_at(new_x, new_y)
        if current and current.name == loc.name:
            self.state.current_location_name = loc.name
            self.state.destination = None
            return True

        self.state.destination = target_name
        self.state.current_location_name = current.name if current else ""
        return False

    def set_action(self, action: str) -> None:
        self.state.current_action = action

    def summary(self) -> str:
        loc = self.location
        loc_name = loc.name if loc else "on the road"
        return f"{self.name} is {self.state.current_action} at {loc_name}"

    def apply_evolution(self, mutation: Any) -> bool:
        """Apply an evolve_agent mutation. Returns True if successful."""
        p = mutation.payload
        field_name = p.get("field", "")
        value = p.get("value")
        if not field_name or value is None:
            return False

        if field_name == "personality":
            self.profile.personality_history.append(self.profile.personality)
            self.profile.personality = str(value)
            self.profile.evolved_traits["personality_evolved"] = True
            return True
        elif field_name == "occupation":
            self.profile.evolved_traits["previous_occupation"] = self.profile.occupation
            self.profile.occupation = str(value)
            self.profile.evolved_traits["occupation_evolved"] = True
            return True
        elif field_name == "routine":
            if isinstance(value, list):
                self.profile.routine_history.append(list(self.profile.daily_routine))
                self.profile.daily_routine = value
                self.profile.evolved_traits["routine_evolved"] = True
                return True
        return False

    def apply_relationship_evolution(self, other_name: str, new_relationship: str) -> None:
        """Update or add a relationship with another agent."""
        self.profile.relationships[other_name] = new_relationship
