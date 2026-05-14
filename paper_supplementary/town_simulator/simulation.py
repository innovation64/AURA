"""Main simulation engine for AURA Town."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


from .agents import AGENT_PROFILES, AgentProfile, TownAgent
from .config import DEFAULT_CONFIG, TownConfig
from .private_state_evolution import (
    evolve_private_state,
    update_beliefs_about_others,
)
from .fallback import (
    fallback_action,
    fallback_chat_response,
    fallback_conversation,
    fallback_daily_plan,
    fallback_evolution_check,
    fallback_reflection,
)
from .llm_engine import LLMEngine
from .map_generator import MapGenerator
from .memory import TownMemory
from .probe import TownProbeRunner
from .prompts import (
    ACTION_DECISION_SYSTEM,
    ACTION_DECISION_USER,
    CONVERSATION_SYSTEM,
    CONVERSATION_USER,
    DAILY_PLAN_SYSTEM,
    DAILY_PLAN_USER,
    EDGE_EXPANSION_SYSTEM,
    EDGE_EXPANSION_USER,
    EVOLUTION_TRIGGER_SYSTEM,
    EVOLUTION_TRIGGER_USER,
    REFLECTION_SYSTEM,
    REFLECTION_USER,
    format_activity_for_evolution,
    format_nearby,
    format_plan,
    format_relationships,
    format_routine,
    format_utilization_for_evolution,
)
from .procedural_evolution import ProceduralEvolver
from .regions import RegionInfo, RegionManager
from .spatial_explorer import SpatialExplorer
from .status_movement import StatusMovementResolver
from .world import TownWorld

logger = logging.getLogger(__name__)


@dataclass
class SimEvent:
    """A single event in the simulation log."""

    time: str
    agent: str
    event_type: str  # action, conversation, reflection, movement, plan
    description: str
    details: Dict[str, Any] = field(default_factory=dict)


class TownSimulation:
    """Orchestrates the AURA Town multi-agent simulation."""

    def __init__(self, config: Optional[TownConfig] = None) -> None:
        self.config = config or DEFAULT_CONFIG
        self.world = TownWorld(
            self.config.grid_width, self.config.grid_height,
            world_seed=self.config.world_seed,
        )
        self.world.load_map()

        self.llm = LLMEngine(self.config, seed=getattr(self.config, "llm_seed", None))
        self.probe_runner = TownProbeRunner(self.llm)
        self.agents: List[TownAgent] = []
        self.memories: Dict[str, TownMemory] = {}
        self.events: List[SimEvent] = []
        self._initialized = False
        self._conversation_pairs_this_tick: set = set()
        self._tick_index = 0
        # Ablation flags (default: all enabled)
        self._memory_enabled = True
        self._reflection_enabled = True
        self._react_action_mode = False  # True = use ReAct-style action decision
        # Evolution
        self._evolver = None
        self._activity_signals: List[Any] = []
        self._evolution_events: List[SimEvent] = []
        self._init_evolver()
        # Procedural evolution (runs every tick, no LLM needed)
        self._procedural_evolver = ProceduralEvolver(
            weather_interval=self.config.weather_transition_interval,
            season_length=self.config.season_length,
            micro_event_prob=self.config.micro_event_probability,
        ) if self.config.procedural_evolve_enabled else None

        # Phase 1-3 new systems
        self._spatial_explorer = SpatialExplorer(self.world)
        self._status_resolver = StatusMovementResolver(self.world)
        self._region_manager = RegionManager(self.world)
        self._map_generator = MapGenerator(llm_engine=self.llm)

    def initialize(self) -> None:
        """Set up agents and their memories."""
        self.agents = []
        self.memories = {}
        self.events = []
        self._tick_index = 0
        self.world.time.day = 1
        self.world.time.hour = 6
        self.world.time.minute = 0

        location_names = self.world.get_all_location_names()

        for profile in AGENT_PROFILES:
            agent = TownAgent(profile, self.world)
            self.agents.append(agent)
            self.memories[agent.name] = TownMemory(
                config=self.config,
                llm_engine=self.llm,
                max_items=self.config.max_memories,
            )

        self._initialized = True
        # Register existing location names with building generator
        self.world.init_building_generator()
        # Seed initial explored chunks from agent positions
        from .chunks import CHUNK_SIZE
        for agent in self.agents:
            cx = agent.state.x // CHUNK_SIZE
            cy = agent.state.y // CHUNK_SIZE
            agent.state.explored_chunks.add((cx, cy))
            # Also add neighboring chunks for initial visibility
            for ddx in range(-1, 2):
                for ddy in range(-1, 2):
                    agent.state.explored_chunks.add((cx + ddx, cy + ddy))
        self._log_event("System", "system", "Simulation initialized with 5 agents.")

    def step(self) -> List[SimEvent]:
        """Advance the simulation by one tick. Returns new events."""
        if not self._initialized:
            self.initialize()

        new_events: List[SimEvent] = []
        self._conversation_pairs_this_tick = set()

        # Check if day is over
        if not self.world.is_daytime():
            # Put everyone to sleep and start a new day
            for agent in self.agents:
                agent.set_action("sleeping")
            self.world.new_day()
            evt = self._log_event(
                "System", "system",
                f"Day {self.world.time.day} begins.",
            )
            new_events.append(evt)
            # Generate new daily plans
            plan_events = self._generate_all_daily_plans()
            new_events.extend(plan_events)
            # Advance one tick into the new day
            self.world.advance_time(self.config.tick_minutes)
            self._tick_index += 1
            return new_events

        # First tick of the sim — generate initial plans
        if self.world.time.hour == 6 and self.world.time.minute == 0:
            for agent in self.agents:
                if not agent.state.daily_plan:
                    plan_events = self._generate_daily_plan(agent)
                    new_events.extend(plan_events)

        # Each agent: perceive → reason → act
        for agent in self.agents:
            try:
                tick_events = self._agent_tick(agent)
                new_events.extend(tick_events)
            except Exception as e:
                logger.error("Agent %s tick failed: %s", agent.name, e)

        # Check for social interactions (agents at the same location)
        conv_events = self._handle_social_interactions()
        new_events.extend(conv_events)

        # Private-state evolution: deterministic, per-tick. Drives
        # availability / emotional_state / unspoken_goal off the agent's
        # current action + workplace load + recent events. Beliefs-about-
        # others refresh only on co-location, so second-order ToM probes
        # still surface stale beliefs.
        self._evolve_private_states()

        # Handle reflections (skip when ablated)
        if self._reflection_enabled:
            for agent in self.agents:
                mem = self.memories[agent.name]
                if mem.should_reflect():
                    ref_events = self._reflect(agent)
                    new_events.extend(ref_events)

        # Evolution: collect signals, track utilization, check edges, run evolver
        self._collect_activity_signals()
        self.world.tick_utilization()
        for agent in self.agents:
            loc = agent.location
            if loc:
                self.world.record_visit(loc.name)

        edge_events = self._check_edge_exploration()
        new_events.extend(edge_events)

        evo_events = self._run_evolution()
        new_events.extend(evo_events)
        self._evolution_events.extend(evo_events)

        # Advance world time
        self.world.advance_time(self.config.tick_minutes)
        self._tick_index += 1

        return new_events

    def _agent_tick(self, agent: TownAgent) -> List[SimEvent]:
        """Run one tick for a single agent: perceive → decide → act."""
        events: List[SimEvent] = []

        # Bubble management: clear expired bubbles
        if agent.state.speech_bubble is not None:
            agent.state.speech_bubble_tick += 1
            if agent.state.speech_bubble_tick >= 3:
                agent.state.speech_bubble = None
                agent.state.speech_bubble_tick = 0
                agent.state.interaction_partner = None
        if agent.state.thought_bubble is not None:
            agent.state.thought_bubble_tick += 1
            if agent.state.thought_bubble_tick >= 2:
                agent.state.thought_bubble = None
                agent.state.thought_bubble_tick = 0

        # Track explored chunks for this agent
        from .chunks import CHUNK_SIZE
        cx = agent.state.x // CHUNK_SIZE
        cy = agent.state.y // CHUNK_SIZE
        agent.state.explored_chunks.add((cx, cy))

        # Player-controlled agents use pending_action or idle
        if agent.state.player_controlled:
            if agent.state.pending_action is not None:
                action_data = agent.state.pending_action
                agent.state.pending_action = None
            else:
                # No pending action — keep current state, skip autonomous decision
                if agent.state.conversation_cooldown > 0:
                    agent.state.conversation_cooldown -= 1
                return events
            # Process the manual action_data below (skip perception/LLM)
        else:
            # Normal autonomous flow
            perception = agent.perceive(self.agents)
            sim_time = self.world.time.total_minutes + (self.world.time.day - 1) * 24 * 60

            # Record observation
            obs_text = (
                f"{agent.name} is at {perception['location']}. "
                f"Time: {perception['time']}. "
                f"Currently: {perception['current_action']}."
            )
            if perception["nearby_agents"]:
                names = [a["name"] for a in perception["nearby_agents"]]
                obs_text += f" Nearby: {', '.join(names)}."
            if self._memory_enabled:
                self.memories[agent.name].add_observation(obs_text, sim_time)

            if self._react_action_mode:
                # ReAct mode: interleave reasoning with tool calls (reactive)
                probe_context = "No probe data."  # No proactive probing in ReAct
                action_data = self._react_decide_action(agent, perception)
            else:
                # AURA mode: proactive probing before reasoning
                probe_context = self._run_probe(agent, perception)
                action_data = self._decide_action(agent, perception, probe_context)

        if action_data:
            new_action = action_data.get("action", agent.state.current_action)
            target_location = action_data.get("location", "")
            thought = action_data.get("thought", "")

            # Phase 3A: Status-driven movement override
            # If the action changed, check if a better location exists
            if new_action != agent.state.current_action and not target_location:
                resolved_loc = self._status_resolver.resolve_location(
                    new_action, agent.state.x, agent.state.y,
                    agent.name, agent.location,
                )
                if resolved_loc:
                    target_location = resolved_loc.name

            # Phase 1A: Exploration via SpatialExplorer
            action_lower = new_action.lower() if new_action else ""
            if "explore" in action_lower or (agent.state.curiosity > 0.6 and not target_location):
                if agent.state.exploration_goal is None:
                    goal = self._spatial_explorer.select_exploration_target(
                        agent.state.x, agent.state.y,
                        agent.state.explored_chunks,
                        goal_hint=thought if "explore" in action_lower else None,
                        preferred_biomes=agent.state.preferred_biomes or None,
                    )
                    if goal:
                        agent.state.exploration_goal = goal
                        evt = self._log_event(
                            agent.name, "exploration",
                            f"{agent.name} set exploration goal: {goal.reason}",
                            details={"goal_type": goal.goal_type, "target_chunk": list(goal.target_chunk)},
                        )
                        events.append(evt)

            # Execute exploration movement if we have a goal
            if agent.state.exploration_goal is not None:
                from .spatial_explorer import SpatialExplorer as SE
                new_x, new_y = SE.step_toward_goal(
                    agent.state.x, agent.state.y,
                    agent.state.exploration_goal,
                    speed=3,
                )
                agent.state.x = new_x
                agent.state.y = new_y
                # Update explored bounds
                self.world._explored_min_x = min(self.world._explored_min_x, new_x)
                self.world._explored_min_y = min(self.world._explored_min_y, new_y)
                self.world._explored_max_x = max(self.world._explored_max_x, new_x + 1)
                self.world._explored_max_y = max(self.world._explored_max_y, new_y + 1)
                # Track explored chunks and immediately generate buildings
                from .chunks import CHUNK_SIZE as CS
                ecx, ecy = new_x // CS, new_y // CS
                agent.state.explored_chunks.add((ecx, ecy))
                agent.state.exploration_history.append((ecx, ecy))
                # Immediately generate buildings for this chunk + neighbors
                for ddx in range(-1, 2):
                    for ddy in range(-1, 2):
                        new_locs = self.world.ensure_chunk_locations(ecx + ddx, ecy + ddy)
                        for loc in new_locs:
                            evt = self._log_event(
                                agent.name, "exploration",
                                f"{agent.name} discovered {loc.emoji} {loc.name} ({loc.type})",
                                details={"location": loc.name, "type": loc.type},
                            )
                            events.append(evt)
                # Check if goal reached
                if SE.is_goal_reached(new_x, new_y, agent.state.exploration_goal):
                    evt = self._log_event(
                        agent.name, "exploration",
                        f"{agent.name} reached exploration goal: {agent.state.exploration_goal.reason}",
                    )
                    events.append(evt)
                    agent.state.exploration_goal = None
                    agent.state.curiosity = max(0, agent.state.curiosity - 0.3)
                # Update location tracking
                loc = self.world.get_location_at(new_x, new_y)
                agent.state.current_location_name = loc.name if loc else ""
            elif target_location:
                # Normal movement toward named location
                current_loc = agent.location
                current_name = current_loc.name if current_loc else ""
                if target_location.lower() != current_name.lower():
                    arrived = agent.move_to_location(target_location)
                    if arrived:
                        evt = self._log_event(
                            agent.name, "movement",
                            f"{agent.name} arrived at {target_location}.",
                        )
                        events.append(evt)
                    else:
                        evt = self._log_event(
                            agent.name, "movement",
                            f"{agent.name} is walking toward {target_location}.",
                        )
                        events.append(evt)

            # Update action
            old_action = agent.state.current_action
            agent.set_action(new_action)

            # Set thought bubble from the "thought" field
            if thought:
                agent.state.thought_bubble = thought[:60]
                agent.state.thought_bubble_tick = 0

            # Derive mood from action keywords
            action_lower = new_action.lower()
            if any(w in action_lower for w in ["happy", "enjoy", "laugh", "fun", "celebrate"]):
                agent.state.mood = "happy"
            elif any(w in action_lower for w in ["think", "read", "study", "write", "research", "reflect"]):
                agent.state.mood = "thinking"
            elif any(w in action_lower for w in ["sleep", "rest", "nap", "tired", "relax"]):
                agent.state.mood = "tired"
            elif any(w in action_lower for w in ["excit", "discover", "explore", "amazing", "wow"]):
                agent.state.mood = "excited"
            else:
                agent.state.mood = "neutral"

            if new_action != old_action:
                emoji = action_data.get("emoji", "")
                evt = self._log_event(
                    agent.name, "action",
                    f"{agent.name} is now {new_action}. {emoji}",
                    details={"thought": thought},
                )
                events.append(evt)

        # Decrement conversation cooldown
        if agent.state.conversation_cooldown > 0:
            agent.state.conversation_cooldown -= 1

        return events

    def _decide_action(
        self,
        agent: TownAgent,
        perception: Dict[str, Any],
        probe_context: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to decide the agent's next action."""
        mem = self.memories[agent.name]
        location_names = ", ".join(self.world.get_all_location_names())

        # Respect ablation settings: only include memories if enabled
        if self._memory_enabled:
            memories_text = mem.format_recent_for_prompt(5)
        else:
            memories_text = "(no memory available)"

        system_msg = ACTION_DECISION_SYSTEM.format(
            name=agent.profile.name,
            age=agent.profile.age,
            occupation=agent.profile.occupation,
            personality=agent.profile.personality,
            locations=location_names,
        )
        user_msg = ACTION_DECISION_USER.format(
            time=perception["time"],
            location=perception["location"],
            current_action=perception["current_action"],
            plan=format_plan(agent.state.daily_plan),
            nearby=format_nearby(perception["nearby_agents"]),
            memories=memories_text,
            name=agent.profile.name,
            probe=probe_context or "No probe data.",
        )

        try:
            result = self.llm.chat_json(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
            if "raw" not in result:
                return result
        except Exception as e:
            logger.info("Action decision LLM failed for %s, using fallback: %s", agent.name, e)

        # Fallback: deterministic action from daily routine
        return fallback_action(
            agent,
            self.world.time.hour,
            self.world.time.minute,
        )

    def _react_decide_action(
        self,
        agent: "TownAgent",
        perception: dict,
    ) -> dict:
        """ReAct-style action decision: tools called DURING reasoning, not before.

        This is the key baseline comparison: AURA probes proactively (before reasoning),
        while ReAct calls tools reactively (during reasoning).
        """
        import json as _json

        location_names = ", ".join(self.world.get_all_location_names())
        mem = self.memories[agent.name]
        memories_text = mem.format_recent_for_prompt(5) if self._memory_enabled else "(no memory)"

        # Build tool descriptions for the ReAct agent
        tool_info = [
            ("get_time", "Get the current simulation time"),
            ("get_nearby_agents", "Get agents at the same location with their actions"),
            ("get_recent_events", "Get recent events in the town"),
            ("get_recent_memories", "Get the agent's recent memories"),
        ]
        tool_desc = "\n".join(f"- {n}: {d}" for n, d in tool_info)

        system_msg = (
            f"You are simulating {agent.profile.name}, a {agent.profile.age}-year-old "
            f"{agent.profile.occupation}.\nPersonality: {agent.profile.personality}\n\n"
            f"You must decide what to do next. You can use tools to gather information "
            f"DURING your reasoning process.\n\n"
            f"Available tools:\n{tool_desc}\n\n"
            f"Format:\n"
            f"Thought: [reasoning]\n"
            f"Action: [tool_name] (if you need info)\n"
            f"Observation: [result]\n"
            f"... (max 3 tool calls)\n"
            f"Thought: [final reasoning]\n"
            f'Decision: {{"action": "...", "location": "...", "thought": "...", "emoji": "..."}}\n\n'
            f"Available locations: {location_names}"
        )

        user_msg = (
            f"Current time: {perception.get('time', 'unknown')}\n"
            f"Current location: {perception.get('location', 'unknown')}\n"
            f"Current activity: {perception.get('current_action', 'unknown')}\n"
            f"Today's plan:\n{format_plan(agent.state.daily_plan)}\n\n"
            f"What does {agent.profile.name} do next?"
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        # ReAct loop: up to 3 tool calls during reasoning
        for step in range(4):
            try:
                resp_text = self.llm.chat(messages, max_tokens=512)
            except Exception as e:
                logger.info("ReAct LLM failed for %s: %s", agent.name, e)
                break

            # Check for Decision
            if "Decision:" in resp_text:
                try:
                    dec_str = resp_text.split("Decision:")[-1].strip()
                    si = dec_str.index("{")
                    ei = dec_str.rindex("}") + 1
                    return _json.loads(dec_str[si:ei])
                except (ValueError, _json.JSONDecodeError):
                    pass

            # Check for Action (tool call)
            if "Action:" in resp_text and "Decision:" not in resp_text:
                action_lines = [l for l in resp_text.split("\n") if l.strip().startswith("Action:")]
                if action_lines:
                    tool_req = action_lines[0].split("Action:")[1].strip().lower()
                    observation = self._react_execute_tool(tool_req, agent, perception)
                    messages.append({"role": "assistant", "content": resp_text})
                    messages.append({"role": "user", "content": f"Observation: {observation}"})
                    continue

            # Try to parse JSON from response
            try:
                si = resp_text.index("{")
                ei = resp_text.rindex("}") + 1
                return _json.loads(resp_text[si:ei])
            except (ValueError, _json.JSONDecodeError):
                pass
            break

        # Fallback
        return fallback_action(agent, self.world.time.hour, self.world.time.minute)

    def _react_execute_tool(self, tool_req: str, agent: "TownAgent", perception: dict) -> str:
        """Execute a tool for ReAct action decision."""
        import json as _json

        if "time" in tool_req:
            return _json.dumps({"time": perception.get("time"), "hour": self.world.time.hour})
        elif "nearby" in tool_req:
            nearby = perception.get("nearby_agents", [])
            return _json.dumps([{"name": a["name"], "action": a["action"]} for a in nearby])
        elif "event" in tool_req:
            events = self.events[-8:]
            return _json.dumps([{"time": e.time, "description": e.description} for e in events])
        elif "memor" in tool_req:
            mem = self.memories.get(agent.name)
            if mem and self._memory_enabled:
                return mem.format_recent_for_prompt(5)
            return "No memories available."
        elif "plan" in tool_req:
            return _json.dumps(agent.state.daily_plan[:5])
        return _json.dumps({"error": f"unknown tool: {tool_req}"})

    def _run_probe(self, agent: TownAgent, perception: Dict[str, Any]) -> str:
        if not self.config.probe_enabled:
            return "No probe data."

        if self._tick_index - agent.state.last_probe_tick < self.config.probe_cooldown_ticks:
            return agent.state.last_probe_summary or "No probe data."

        memory = self.memories[agent.name]
        try:
            result = self.probe_runner.run(
                agent=agent,
                agents=self.agents,
                events=self.events,
                perception=perception,
                memory=memory,
                max_steps=self.config.probe_max_steps,
            )
        except Exception as e:
            logger.info("Probe failed for %s (LLM unavailable): %s", agent.name, e)
            return "No probe data."

        agent.state.last_probe_tick = self._tick_index
        agent.state.last_probe_steps = [
            {
                "tool": step.tool,
                "arguments": step.arguments,
                "ok": step.ok,
                "output": step.output,
                "error": step.error,
            }
            for step in result.steps
        ]

        if result.steps:
            self._log_event(
                agent.name,
                "probe",
                f"{agent.name} probed the environment. {result.summary}",
                details={"steps": agent.state.last_probe_steps},
            )

        # Filter out probe steps that duplicate already-known perception data
        redundant_tools = {"world.time", "world.location", "world.nearby_agents", "memory.recent"}
        result.steps = [s for s in result.steps if s.tool not in redundant_tools]
        formatted_prompt = result.to_prompt()
        # Cache the rich formatted output so cooldown ticks return useful context
        agent.state.last_probe_summary = formatted_prompt
        return formatted_prompt

    def _generate_all_daily_plans(self) -> List[SimEvent]:
        """Generate daily plans for all agents."""
        events = []
        for agent in self.agents:
            events.extend(self._generate_daily_plan(agent))
        return events

    def _generate_daily_plan(self, agent: TownAgent) -> List[SimEvent]:
        """Use LLM to generate a daily plan for an agent."""
        events: List[SimEvent] = []
        mem = self.memories[agent.name]
        location_names = ", ".join(self.world.get_all_location_names())

        system_msg = DAILY_PLAN_SYSTEM.format(
            name=agent.profile.name,
            age=agent.profile.age,
            occupation=agent.profile.occupation,
            personality=agent.profile.personality,
            routine=format_routine(agent.profile.daily_routine),
            relationships=format_relationships(agent.profile.relationships),
            locations=location_names,
        )
        user_msg = DAILY_PLAN_USER.format(
            day=self.world.time.day,
            time=self.world.time.display,
            memories=mem.format_recent_for_prompt(5),
            name=agent.profile.name,
        )

        try:
            result = self.llm.chat_json(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
            if isinstance(result, list):
                plan_items = result
            elif isinstance(result, dict) and "raw" not in result:
                # Might be wrapped in a key
                plan_items = result.get("plan", result.get("activities", [result]))
            else:
                plan_items = []

            plan_strings = []
            for item in plan_items:
                if isinstance(item, dict):
                    t = item.get("time", "")
                    a = item.get("activity", item.get("action", ""))
                    plan_strings.append(f"{t} - {a}")
                elif isinstance(item, str):
                    plan_strings.append(item)

            agent.state.daily_plan = plan_strings
            agent.state.plan_index = 0

            evt = self._log_event(
                agent.name, "plan",
                f"{agent.name} made a plan for the day.",
                details={"plan": plan_strings},
            )
            events.append(evt)

            # Record in memory
            sim_time = self.world.time.total_minutes + (self.world.time.day - 1) * 24 * 60
            mem.add_observation(
                f"{agent.name}'s plan for the day: {'; '.join(plan_strings[:3])}...",
                sim_time,
                kind="plan",
                importance=4,
            )

        except Exception as e:
            logger.info("Plan generation LLM failed for %s, using fallback: %s", agent.name, e)
            agent.state.daily_plan = fallback_daily_plan(agent.profile)
            agent.state.plan_index = 0

            evt = self._log_event(
                agent.name, "plan",
                f"{agent.name} made a plan for the day (routine-based).",
                details={"plan": agent.state.daily_plan},
            )
            events.append(evt)

        return events

    def _handle_social_interactions(self) -> List[SimEvent]:
        """Check for agents at the same public location and trigger conversations."""
        events: List[SimEvent] = []

        # Group agents by location
        location_agents: Dict[str, List[TownAgent]] = {}
        for agent in self.agents:
            loc = agent.location
            if loc and loc.type != "home":
                location_agents.setdefault(loc.name, []).append(agent)

        for loc_name, agents_here in location_agents.items():
            if len(agents_here) < 2:
                continue

            # Try conversations between pairs
            for i in range(len(agents_here)):
                for j in range(i + 1, len(agents_here)):
                    a1 = agents_here[i]
                    a2 = agents_here[j]

                    # Avoid repeated conversations this tick
                    pair_key = tuple(sorted([a1.name, a2.name]))
                    if pair_key in self._conversation_pairs_this_tick:
                        continue

                    # Check cooldowns
                    if a1.state.conversation_cooldown > 0 or a2.state.conversation_cooldown > 0:
                        continue

                    self._conversation_pairs_this_tick.add(pair_key)
                    conv_events = self._generate_conversation(a1, a2, loc_name)
                    events.extend(conv_events)
                    # Set cooldown (don't chat every single tick)
                    a1.state.conversation_cooldown = 2
                    a2.state.conversation_cooldown = 2

        return events

    def _generate_conversation(
        self, agent1: TownAgent, agent2: TownAgent, location: str
    ) -> List[SimEvent]:
        """Generate a conversation between two agents using LLM."""
        events: List[SimEvent] = []
        sim_time = self.world.time.total_minutes + (self.world.time.day - 1) * 24 * 60

        rel_1to2 = agent1.profile.relationships.get(agent2.name, "acquaintance")
        rel_2to1 = agent2.profile.relationships.get(agent1.name, "acquaintance")

        mem1 = self.memories[agent1.name]
        mem2 = self.memories[agent2.name]
        context_lines = []
        for m in mem1.get_recent(3):
            context_lines.append(f"  {agent1.name}: {m.content}")
        for m in mem2.get_recent(3):
            context_lines.append(f"  {agent2.name}: {m.content}")

        system_msg = CONVERSATION_SYSTEM.format(
            agent1_name=agent1.name,
            agent1_occupation=agent1.profile.occupation,
            agent1_personality=agent1.profile.personality,
            agent2_name=agent2.name,
            agent2_occupation=agent2.profile.occupation,
            agent2_personality=agent2.profile.personality,
            relationship_1to2=rel_1to2,
            relationship_2to1=rel_2to1,
        )
        user_msg = CONVERSATION_USER.format(
            location=location,
            time=self.world.time.display,
            agent1_name=agent1.name,
            agent1_action=agent1.state.current_action,
            agent2_name=agent2.name,
            agent2_action=agent2.state.current_action,
            context="\n".join(context_lines) if context_lines else "No recent context.",
        )

        try:
            result = self.llm.chat_json(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=600,
            )

            if isinstance(result, list):
                exchanges = result
            elif isinstance(result, dict):
                exchanges = result.get("conversation", result.get("exchanges", []))
            else:
                exchanges = []

            if not exchanges:
                return events

            # Build conversation text
            conv_lines = []
            for ex in exchanges:
                if isinstance(ex, dict):
                    speaker = ex.get("speaker", "???")
                    text = ex.get("text", ex.get("message", ""))
                    conv_lines.append(f'{speaker}: "{text}"')

            conv_text = "\n".join(conv_lines)

            # Set speech bubbles on both agents with their last dialogue line
            for ex in reversed(exchanges):
                if isinstance(ex, dict):
                    speaker = ex.get("speaker", "")
                    text = ex.get("text", ex.get("message", ""))
                    if speaker == agent1.name and text:
                        agent1.state.speech_bubble = text[:60]
                        agent1.state.speech_bubble_tick = 0
                    elif speaker == agent2.name and text:
                        agent2.state.speech_bubble = text[:60]
                        agent2.state.speech_bubble_tick = 0
            agent1.state.interaction_partner = agent2.name
            agent2.state.interaction_partner = agent1.name

            # Log the conversation event
            evt = self._log_event(
                f"{agent1.name} & {agent2.name}",
                "conversation",
                f"{agent1.name} and {agent2.name} had a conversation at {location}.",
                details={"dialogue": conv_lines, "location": location},
            )
            events.append(evt)

            # Store in both agents' memories
            summary = f"Had a conversation with {agent2.name} at {location}: {conv_lines[0] if conv_lines else ''}"
            mem1.add_conversation(summary, sim_time, agent2.name)

            summary2 = f"Had a conversation with {agent1.name} at {location}: {conv_lines[0] if conv_lines else ''}"
            mem2.add_conversation(summary2, sim_time, agent1.name)

        except Exception as e:
            logger.info("Conversation LLM failed, using fallback: %s", e)
            exchanges = fallback_conversation(
                agent1, agent2, location, self.world.time.hour,
            )
            conv_lines = [f'{ex["speaker"]}: "{ex["text"]}"' for ex in exchanges]
            # Set speech bubbles for fallback conversations
            for ex in reversed(exchanges):
                if ex.get("speaker") == agent1.name:
                    agent1.state.speech_bubble = ex["text"][:60]
                    agent1.state.speech_bubble_tick = 0
                elif ex.get("speaker") == agent2.name:
                    agent2.state.speech_bubble = ex["text"][:60]
                    agent2.state.speech_bubble_tick = 0
            agent1.state.interaction_partner = agent2.name
            agent2.state.interaction_partner = agent1.name
            evt = self._log_event(
                f"{agent1.name} & {agent2.name}",
                "conversation",
                f"{agent1.name} and {agent2.name} had a conversation at {location}.",
                details={"dialogue": conv_lines, "location": location},
            )
            events.append(evt)

            summary = f"Had a conversation with {agent2.name} at {location}: {conv_lines[0] if conv_lines else ''}"
            mem1.add_conversation(summary, sim_time, agent2.name)
            summary2 = f"Had a conversation with {agent1.name} at {location}: {conv_lines[0] if conv_lines else ''}"
            mem2.add_conversation(summary2, sim_time, agent1.name)

        return events

    def _evolve_private_states(self) -> None:
        """Deterministic per-tick update of every agent's private state.

        Runs after physical-action ticks and social interactions, so the
        rules see the post-step `current_action` and updated co-location.
        Mutates `agent.state.availability`, `.emotional_state`,
        `.unspoken_goal`, and `.beliefs_about_others`. Side-effect free
        wrt LLM calls; pure function of sim state + recent events.
        """
        # Snapshot recent event descriptions once (cheap)
        recent_descs = [e.description for e in self.events[-8:]]

        # Phase A: own private state per agent
        per_agent_summary: Dict[str, Dict[str, Any]] = {}
        for agent in self.agents:
            loc = agent.location
            loc_name = loc.name if loc else ""
            nearby = [
                o for o in self.agents
                if o.name != agent.name
                and o.location is not None
                and loc is not None
                and o.location.name == loc.name
            ]
            update = evolve_private_state(
                agent_name=agent.name,
                current_action=agent.state.current_action,
                location_name=loc_name,
                nearby_count=len(nearby),
                hour_24=self.world.time.hour,
                minute=self.world.time.minute,
                recent_event_descriptions=recent_descs,
                prev_availability=agent.state.availability,
                prev_emotional_state=agent.state.emotional_state,
            )
            agent.state.availability = update.availability
            agent.state.emotional_state = update.emotional_state
            agent.state.unspoken_goal = update.unspoken_goal
            per_agent_summary[agent.name] = {
                "name": agent.name,
                "location": loc_name,
                "availability": update.availability,
                "emotional_state": update.emotional_state,
                "rule_fired": update.rule_fired,
            }

        # Phase B: refresh second-order beliefs only on co-location.
        # Each agent's beliefs about other co-located agents update to
        # the latest observed availability / emotional_state. Beliefs
        # about non-co-located agents are NOT refreshed -- they go stale,
        # which is the substrate the second-order ToM probe queries.
        for agent in self.agents:
            loc = agent.location
            if loc is None:
                continue
            co_located = [
                per_agent_summary[o.name]
                for o in self.agents
                if o.name != agent.name
                and o.location is not None
                and o.location.name == loc.name
            ]
            if not co_located:
                continue
            agent.state.beliefs_about_others = update_beliefs_about_others(
                self_agent_name=agent.name,
                self_beliefs=agent.state.beliefs_about_others,
                co_located_others=co_located,
            )

    def _reflect(self, agent: TownAgent) -> List[SimEvent]:
        """Generate reflections for an agent."""
        events: List[SimEvent] = []
        mem = self.memories[agent.name]
        sim_time = self.world.time.total_minutes + (self.world.time.day - 1) * 24 * 60

        system_msg = REFLECTION_SYSTEM.format(
            name=agent.name,
            personality=agent.profile.personality,
        )
        user_msg = REFLECTION_USER.format(
            name=agent.name,
            memories=mem.format_recent_for_prompt(15),
        )

        try:
            result = self.llm.chat_json(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )

            if isinstance(result, list):
                insights = result
            elif isinstance(result, dict):
                insights = result.get("insights", result.get("reflections", []))
            else:
                insights = []

            for insight in insights:
                if isinstance(insight, str):
                    mem.add_reflection(insight, sim_time)
                    evt = self._log_event(
                        agent.name, "reflection",
                        f"{agent.name} reflected: {insight}",
                    )
                    events.append(evt)

        except Exception as e:
            logger.info("Reflection LLM failed for %s, using fallback: %s", agent.name, e)
            insights = fallback_reflection(agent.name, agent.profile.personality)
            for insight in insights:
                mem.add_reflection(insight, sim_time)
                evt = self._log_event(
                    agent.name, "reflection",
                    f"{agent.name} reflected: {insight}",
                )
                events.append(evt)

        return events

    # ── Evolution system ──────────────────────────────────────────

    def _init_evolver(self) -> None:
        """Create the evolver: LLMEvolver if LLM available, else NoOpEvolver."""
        if not self.config.evolve_enabled:
            return
        try:
            from aura.defaults.llm_evolver import LLMEvolver
            from aura.evolve import NoOpEvolver
            if self.config.llm_available:
                self._evolver = LLMEvolver(
                    llm=self.llm,
                    evolve_interval=self.config.evolve_interval,
                    max_mutations=self.config.evolve_max_mutations,
                )
            else:
                self._evolver = NoOpEvolver()
        except ImportError:
            logger.debug("AURA evolve module not available")

    def _collect_activity_signals(self) -> None:
        """Collect activity signals from all agents for this tick."""
        try:
            from aura.evolve_types import ActivitySignal
        except ImportError:
            return

        for agent in self.agents:
            loc = agent.location
            loc_name = loc.name if loc else "road"
            nearby = []
            for other in self.agents:
                if other.name == agent.name:
                    continue
                other_loc = other.location
                if loc and other_loc and loc.name == other_loc.name:
                    nearby.append(other.name)

            sig = ActivitySignal(
                agent_name=agent.name,
                location=loc_name,
                action=agent.state.current_action,
                nearby_agents=tuple(nearby),
            )
            self._activity_signals.append(sig)

    def _check_edge_exploration(self) -> List[SimEvent]:
        """Detect agents at grid edges and trigger chunk generation + procedural buildings."""
        events: List[SimEvent] = []

        for agent in self.agents:
            # Ensure chunks around agent are generated (procedural buildings)
            cx, cy = self.world.chunk_manager.world_to_chunk(
                agent.state.x, agent.state.y
            )
            # Generate for current chunk and its neighbors
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    chunk_key = (cx + dx, cy + dy)
                    agent.state.explored_chunks.add(chunk_key)
                    new_locs = self.world.ensure_chunk_locations(cx + dx, cy + dy)
                    for loc in new_locs:
                        evt = self._log_event(
                            "World", "evolution",
                            f"New location discovered: {loc.name} ({loc.emoji})",
                            details={"location": loc.name, "biome": self.world.chunk_manager.get_biome_at(loc.x, loc.y)},
                        )
                        events.append(evt)

            # Update curiosity: increases when visiting same locations repeatedly
            loc = agent.location
            if loc:
                agent.state.explored_locations.add(loc.name)
                total_known = len(agent.state.explored_locations)
                # Curiosity builds when agent keeps visiting known places
                if total_known > 0:
                    repeat_ratio = self.world._visit_counts.get(loc.name, 0) / max(1, total_known)
                    agent.state.curiosity = min(1.0, agent.state.curiosity + repeat_ratio * 0.02)

            # If curiosity is high and no exploration target, use FrontierDetector
            if agent.state.curiosity > 0.5 and agent.state.exploration_target is None:
                from .chunks import CHUNK_SIZE
                from .spatial_explorer import FrontierDetector
                frontiers = FrontierDetector.get_frontiers_near(
                    agent.state.explored_chunks, agent.state.x, agent.state.y, max_results=1
                )
                if frontiers:
                    fx, fy = frontiers[0]
                    agent.state.exploration_target = (
                        fx * CHUNK_SIZE + CHUNK_SIZE // 2,
                        fy * CHUNK_SIZE + CHUNK_SIZE // 2,
                    )
                else:
                    agent.state.exploration_target = (
                        agent.state.x + 16,
                        agent.state.y,
                    )
                agent.state.curiosity = 0.0  # reset after triggering exploration

            # Legacy: also trigger LLM-based expansion if evolve is enabled
            if self.config.evolve_enabled:
                direction = self.world.is_at_edge(agent.state.x, agent.state.y)
                if direction:
                    expand_events = self._expand_toward(agent, direction)
                    events.extend(expand_events)

        return events

    def _expand_toward(self, agent: TownAgent, direction: str) -> List[SimEvent]:
        """Generate new locations for grid expansion using LLM."""
        events: List[SimEvent] = []
        expand_amount = 15

        # Build coordinate rules based on direction
        coord_rules = {
            "east": f"x should be between {self.world.width} and {self.world.width + expand_amount - 1}",
            "west": f"x should be between 0 and {expand_amount - 1} (existing locations shift right by {expand_amount})",
            "south": f"y should be between {self.world.height} and {self.world.height + expand_amount - 1}",
            "north": f"y should be between 0 and {expand_amount - 1} (existing locations shift down by {expand_amount})",
        }

        new_area = {
            "east": f"x: {self.world.width}-{self.world.width + expand_amount}, y: 0-{self.world.height}",
            "west": f"x: 0-{expand_amount}, y: 0-{self.world.height}",
            "south": f"x: 0-{self.world.width}, y: {self.world.height}-{self.world.height + expand_amount}",
            "north": f"x: 0-{self.world.width}, y: 0-{expand_amount}",
        }

        existing_names = ", ".join(self.world.get_all_location_names())

        # Try LLM generation
        new_locations = []
        try:
            system_msg = EDGE_EXPANSION_SYSTEM.format(
                direction=direction,
                existing_locations=existing_names,
                coordinate_rules=coord_rules.get(direction, ""),
            )
            user_msg = EDGE_EXPANSION_USER.format(
                agent_name=agent.name,
                occupation=agent.profile.occupation,
                direction=direction,
                agent_x=agent.state.x,
                agent_y=agent.state.y,
                expand_amount=expand_amount,
                new_area=new_area.get(direction, ""),
            )

            result = self.llm.chat_json([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ])

            if isinstance(result, dict):
                new_locations = result.get("locations", [])
            elif isinstance(result, list):
                new_locations = result

        except Exception as e:
            logger.info("Edge expansion LLM failed: %s", e)

        # Apply the expansion via mutation
        try:
            from aura.evolve_types import MutationType, WorldMutation
            mutation = WorldMutation(
                type=MutationType.EXPAND_GRID,
                target=direction,
                payload={
                    "direction": direction,
                    "amount": expand_amount,
                    "locations": new_locations,
                },
                reason=f"{agent.name} explored toward {direction}",
            )
            if self.world.apply_mutation(mutation):
                evt = self._log_event(
                    agent.name, "evolution",
                    f"The town expanded {direction}! Grid is now {self.world.width}x{self.world.height}.",
                    details={"direction": direction, "new_locations": [l.get("name", "?") for l in new_locations if isinstance(l, dict)]},
                )
                events.append(evt)
        except ImportError:
            logger.debug("Cannot import evolve_types for edge expansion")

        return events

    def _run_evolution(self) -> List[SimEvent]:
        """Run the evolver and apply mutations."""
        events: List[SimEvent] = []

        # Always run procedural evolver first (lightweight, every tick)
        if self._procedural_evolver is not None:
            proc_events = self._procedural_evolver.tick(
                self._tick_index,
                self.world.properties,
                self.world._visit_counts,
                self.world.locations,
            )
            for pe in proc_events:
                evt = self._log_event(
                    "World", pe.get("type", "evolution"),
                    pe["description"],
                )
                events.append(evt)

        if not self.config.evolve_enabled or self._evolver is None:
            # Try fallback evolution
            fallback_mutations = fallback_evolution_check(
                self.world.get_world_properties(),
                self._tick_index,
                self.world.get_utilization(),
            )
            for m_dict in fallback_mutations:
                events.extend(self._apply_mutation_dict(m_dict))
            return events

        try:
            from aura.evolve_types import ActivitySignal, MutationType

            # Check if evolution should run
            if not self._evolver.should_evolve(self.world, self._activity_signals, self._tick_index):
                return events

            result = self._evolver.evolve(self.world, self._activity_signals, self._tick_index)

            for mutation in result.mutations:
                if mutation.type in (MutationType.EVOLVE_AGENT,):
                    # Apply to agent
                    for agent in self.agents:
                        if agent.name == mutation.target:
                            if agent.apply_evolution(mutation):
                                evt = self._log_event(
                                    mutation.target, "evolution",
                                    f"{mutation.target} evolved: {mutation.reason}",
                                    details={"mutation_type": mutation.type.value, "payload": mutation.payload},
                                )
                                events.append(evt)
                            break

                elif mutation.type == MutationType.EVOLVE_RELATIONSHIP:
                    # Apply relationship evolution
                    p = mutation.payload
                    other = p.get("other_agent", "")
                    rel = p.get("relationship", "")
                    for agent in self.agents:
                        if agent.name == mutation.target:
                            agent.apply_relationship_evolution(other, rel)
                            evt = self._log_event(
                                mutation.target, "evolution",
                                f"{mutation.target}'s relationship with {other} evolved: {rel}",
                                details={"mutation_type": mutation.type.value},
                            )
                            events.append(evt)
                            break

                else:
                    # Apply to world
                    if self.world.apply_mutation(mutation):
                        evt = self._log_event(
                            "World", "evolution",
                            f"World evolved: {mutation.reason}",
                            details={"mutation_type": mutation.type.value, "target": mutation.target},
                        )
                        events.append(evt)

            if result.mutations:
                result.applied = True

        except ImportError:
            logger.debug("AURA evolve module not available for evolution")
        except Exception as e:
            logger.warning("Evolution failed: %s", e)

        # Clear accumulated signals after evolution attempt
        self._activity_signals.clear()

        return events

    def _apply_mutation_dict(self, m_dict: Dict[str, Any]) -> List[SimEvent]:
        """Apply a raw mutation dict (from fallback) to the world."""
        events: List[SimEvent] = []
        try:
            from aura.evolve_types import MutationType, WorldMutation
            type_map = {t.value: t for t in MutationType}
            mt = type_map.get(m_dict.get("type", ""))
            if mt is None:
                return events
            mutation = WorldMutation(
                type=mt,
                target=m_dict.get("target", ""),
                payload=m_dict.get("payload", {}),
                reason=m_dict.get("reason", ""),
            )
            if self.world.apply_mutation(mutation):
                evt = self._log_event(
                    "World", "evolution",
                    f"World evolved: {mutation.reason}",
                    details={"mutation_type": mutation.type.value, "target": mutation.target},
                )
                events.append(evt)
        except ImportError:
            pass
        return events

    def _log_event(
        self,
        agent: str,
        event_type: str,
        description: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> SimEvent:
        """Create and store a simulation event."""
        evt = SimEvent(
            time=self.world.time.display,
            agent=agent,
            event_type=event_type,
            description=description,
            details=details or {},
        )
        self.events.append(evt)
        return evt

    # ── State export for UI ─────────────────────────────────────────

    def get_state(self) -> Dict[str, Any]:
        """Export current simulation state for the UI."""
        agents_data = []
        all_explored_chunks = set()
        for agent in self.agents:
            mem = self.memories.get(agent.name)
            all_explored_chunks.update(agent.state.explored_chunks)

            # Serialize exploration goal
            exploration_goal_data = None
            if agent.state.exploration_goal is not None:
                eg = agent.state.exploration_goal
                exploration_goal_data = {
                    "goal_type": eg.goal_type,
                    "target_chunk": list(eg.target_chunk),
                    "target_world": list(eg.target_world) if eg.target_world else None,
                    "path_length": len(eg.path),
                    "priority": eg.priority,
                    "reason": eg.reason,
                    "path_preview": [list(p) for p in eg.path[:20]],  # first 20 waypoints
                }

            agents_data.append({
                "name": agent.name,
                "emoji": agent.profile.emoji,
                "x": agent.state.x,
                "y": agent.state.y,
                "action": agent.state.current_action,
                "location": agent.state.current_location_name or "on the road",
                "occupation": agent.profile.occupation,
                "personality": agent.profile.personality,
                "plan": agent.state.daily_plan,
                "probe_summary": agent.state.last_probe_summary,
                "probe_steps": agent.state.last_probe_steps,
                "memories": [
                    {"content": m.content, "kind": m.kind, "importance": m.importance}
                    for m in (mem.get_recent(10) if mem else [])
                ],
                "relationships": agent.profile.relationships,
                "player_controlled": agent.state.player_controlled,
                "speech_bubble": agent.state.speech_bubble,
                "thought_bubble": agent.state.thought_bubble,
                "mood": agent.state.mood,
                "interaction_partner": agent.state.interaction_partner,
                "destination": agent.state.destination,
                "exploration_goal": exploration_goal_data,
                "curiosity": agent.state.curiosity,
            })

        # Build chunk biome data for current explored area
        chunk_biomes = self.world.chunk_manager.get_biome_map(
            self.world._explored_min_x,
            self.world._explored_min_y,
            self.world._explored_max_x - self.world._explored_min_x,
            self.world._explored_max_y - self.world._explored_min_y,
        )

        # Compute frontier chunks for visualization
        from .spatial_explorer import FrontierDetector
        frontier_chunks = FrontierDetector.get_frontiers(all_explored_chunks)

        # Auto-detect regions from explored chunks
        self._region_manager.auto_detect_regions(all_explored_chunks)

        # Get world map data
        world_map_data = self._region_manager.get_world_map_data(self.agents)

        return {
            "time": self.world.time.display,
            "day": self.world.time.day,
            "hour": self.world.time.hour,
            "minute": self.world.time.minute,
            "grid_width": self.world.width,
            "grid_height": self.world.height,
            "probe_enabled": self.config.probe_enabled,
            "probe_max_steps": self.config.probe_max_steps,
            "evolve_enabled": self.config.evolve_enabled,
            "agents": agents_data,
            "locations": [
                {
                    "name": loc.name,
                    "type": loc.type,
                    "emoji": loc.emoji,
                    "x": loc.x,
                    "y": loc.y,
                    "width": loc.width,
                    "height": loc.height,
                    "description": loc.description,
                    "interior_objects": loc.interior_objects,
                    "items": loc.items,
                    "atmosphere": loc.atmosphere,
                    "region_id": loc.region_id,
                }
                for loc in self.world.locations
            ],
            "explored_chunks": [f"{cx},{cy}" for cx, cy in all_explored_chunks],
            "frontier_chunks": [f"{cx},{cy}" for cx, cy in frontier_chunks[:50]],
            "events": [
                {
                    "time": e.time,
                    "agent": e.agent,
                    "type": e.event_type,
                    "description": e.description,
                    "details": e.details,
                }
                for e in self.events[-50:]  # Last 50 events
            ],
            "world_properties": self.world.get_world_properties(),
            "evolution_log": self.world.evolution_log[-20:],
            "chunk_biomes": chunk_biomes,
            "world_map": {
                "regions": world_map_data.regions,
                "agent_positions": world_map_data.agent_positions,
                "connections": world_map_data.connections,
                "world_bounds": world_map_data.world_bounds,
            },
        }

    def update_probe_settings(self, enabled: bool, max_steps: int) -> None:
        self.config.probe_enabled = bool(enabled)
        self.config.probe_max_steps = max(0, int(max_steps))

    def update_ablation_settings(
        self, memory_enabled: bool = True, reflection_enabled: bool = True,
    ) -> None:
        """Toggle memory and reflection subsystems for ablation experiments."""
        self._memory_enabled = bool(memory_enabled)
        self._reflection_enabled = bool(reflection_enabled)

    def update_action_mode(self, react_mode: bool = False) -> None:
        """Switch between AURA proactive and ReAct reactive action decision."""
        self._react_action_mode = bool(react_mode)

    # ── Chat with environment enrichment ──────────────────────────────

    def chat(
        self,
        user_name: str,
        message: str,
        read_only: bool = False,
    ) -> Dict[str, Any]:
        """Process a user chat message through the environment agent.

        Flow: User Query → Env Agent (gather context + probe) → LLM → Response
        Returns the full pipeline data so the UI can visualise each stage.

        Args:
            read_only: when True, the chat call does NOT write the chat event
                to the global event log nor add an observation to the
                speaker's memory. This is the mode the RQ2 paired-snapshot
                runner uses so that condition-A's chat does not pollute
                the simulation state seen by condition-B's chat at the same
                query position. Default False to preserve existing demo /
                interactive behaviour.
        """
        # Find the agent the user is "playing as"
        agent = None
        for a in self.agents:
            if a.name == user_name:
                agent = a
                break
        if agent is None:
            return {"error": f"Agent '{user_name}' not found"}

        # ── Stage 1: gather environmental context ──
        perception = agent.perceive(self.agents)
        mem = self.memories.get(agent.name)

        env_context: Dict[str, Any] = {
            "location": perception["location"],
            "location_description": perception.get("location_description", ""),
            "time": perception["time"],
            "nearby_agents": perception["nearby_agents"],
            "current_action": perception["current_action"],
            "recent_memories": [
                m.content for m in (mem.get_recent(5) if mem else [])
            ],
            "recent_events": [
                {
                    "time": e.time,
                    "description": e.description,
                    "type": e.event_type,
                }
                for e in self.events[-8:]
            ],
        }

        # ── Stage 2: run probe tools for extra context ──
        # Honour the configured probe budget directly. The earlier
        # `min(1, ...)` cap was a latency-time legacy that silently
        # disabled multi-step probing in RQ2 chat paths even when the
        # experiment configured B≥2. Now the runner controls the budget.
        probe_data: Optional[Dict[str, Any]] = None
        if self.config.probe_enabled and mem is not None:
            try:
                probe_result = self.probe_runner.run(
                    agent=agent,
                    agents=self.agents,
                    events=self.events,
                    perception=perception,
                    memory=mem,
                    max_steps=max(0, self.config.probe_max_steps),
                )
                probe_data = {
                    "summary": probe_result.summary,
                    "steps": [
                        {
                            "tool": s.tool,
                            "arguments": s.arguments,
                            "ok": s.ok,
                            "output": s.output,
                            "error": s.error,
                        }
                        for s in probe_result.steps
                    ],
                }
            except Exception as exc:
                logger.warning("Chat probe failed for %s: %s", agent.name, exc)

        # ── Stage 3: build enriched prompt and call LLM ──
        context_text = self._build_env_context_prompt(env_context, probe_data)

        system_msg = (
            f"You are an AI assistant in a collaborative town environment.\n"
            f"The user ({user_name}, {agent.profile.occupation}, age {agent.profile.age}) "
            f"is asking a question. The Environment Agent has gathered the following "
            f"real-time context to help you respond more accurately:\n\n"
            f"{context_text}\n\n"
            f"Ground your answer in the environmental context above. "
            f"Use specific names, locations, times, and events from the context to give an informative answer. "
            f"Do not invent details (directions, distances, sensory descriptions) that are not in the context. "
            f"If the context does not contain enough information to answer fully, state what you do know and acknowledge the gap. "
            f"Be helpful, specific, and concise. Reply in the same language as the user's message."
        )

        try:
            ai_response = self.llm.chat(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": message},
                ],
                max_tokens=800,
            )
        except Exception as exc:
            logger.info("Chat LLM failed, using fallback: %s", exc)
            ai_response = fallback_chat_response(agent, message, env_context)

        # Log the chat as an event + store in memory ONLY in interactive mode.
        # The RQ2 paired-snapshot runner uses read_only=True so cross-condition
        # state stays identical at every query position.
        if not read_only:
            sim_time = self.world.time.total_minutes + (self.world.time.day - 1) * 24 * 60
            self._log_event(
                user_name,
                "chat",
                f"{user_name} asked: {message[:80]}",
                details={"message": message, "response": ai_response[:200]},
            )
            if mem is not None:
                mem.add_observation(
                    f"Asked AI: '{message[:60]}' and received an answer.",
                    sim_time,
                    kind="conversation",
                    importance=5,
                )

        return {
            "user": user_name,
            "message": message,
            "env_context": env_context,
            "probe": probe_data,
            "ai_response": ai_response,
            "read_only": read_only,
        }

    def _build_env_context_prompt(
        self,
        env_context: Dict[str, Any],
        probe_data: Optional[Dict[str, Any]],
    ) -> str:
        """Format gathered environmental context into a readable prompt section."""
        lines: List[str] = []
        lines.append(f"Location: {env_context['location']}")
        if env_context.get("location_description"):
            lines.append(f"  {env_context['location_description']}")
        lines.append(f"Time: {env_context['time']}")
        lines.append(f"Current Activity: {env_context['current_action']}")

        nearby = env_context.get("nearby_agents") or []
        if nearby:
            lines.append("Nearby People:")
            for a in nearby:
                rel = a.get("relationship", "unknown")
                lines.append(f"  - {a['name']}: {a['action']} (relationship: {rel})")

        mems = env_context.get("recent_memories") or []
        if mems:
            lines.append("Recent Memories:")
            for m in mems[:5]:
                lines.append(f"  - {m}")

        evts = env_context.get("recent_events") or []
        if evts:
            lines.append("Recent Events:")
            for e in evts[:5]:
                lines.append(f"  - [{e['time']}] {e['description']}")

        if probe_data and probe_data.get("steps"):
            lines.append("Environment Probe Results:")
            for s in probe_data["steps"]:
                status = "ok" if s["ok"] else "error"
                lines.append(f"  - {s['tool']} -> {status}")
                if s["ok"] and s.get("output"):
                    lines.append(f"    output: {str(s['output'])[:200]}")

        return "\n".join(lines)

    # ── Directional movement & exploration ──────────────────────────

    def move_agent_direction(self, name: str, direction: str, steps: int = 2) -> Optional[Dict[str, Any]]:
        """Move a controlled agent N/S/E/W by grid cells. Returns updated state or None."""
        agent = None
        for a in self.agents:
            if a.name == name:
                agent = a
                break
        if agent is None:
            return None

        dx, dy = 0, 0
        if direction == "north":
            dy = -steps
        elif direction == "south":
            dy = steps
        elif direction == "east":
            dx = steps
        elif direction == "west":
            dx = -steps
        else:
            return None

        agent.state.x += dx
        agent.state.y += dy

        # Update explored bounds
        self.world._explored_min_x = min(self.world._explored_min_x, agent.state.x)
        self.world._explored_min_y = min(self.world._explored_min_y, agent.state.y)
        self.world._explored_max_x = max(self.world._explored_max_x, agent.state.x + 1)
        self.world._explored_max_y = max(self.world._explored_max_y, agent.state.y + 1)

        # Update location tracking
        loc = self.world.get_location_at(agent.state.x, agent.state.y)
        agent.state.current_location_name = loc.name if loc else ""

        # Track explored chunks
        from .chunks import CHUNK_SIZE
        cx = agent.state.x // CHUNK_SIZE
        cy = agent.state.y // CHUNK_SIZE
        agent.state.explored_chunks.add((cx, cy))

        # Trigger chunk generation around agent
        for ddx in range(-1, 2):
            for ddy in range(-1, 2):
                self.world.ensure_chunk_locations(cx + ddx, cy + ddy)

        return self.get_state()

    def explore_direction(self, name: str, direction: str) -> Optional[Dict[str, Any]]:
        """Set exploration_target 20 cells in the given direction (supports 8 compass directions)."""
        agent = None
        for a in self.agents:
            if a.name == name:
                agent = a
                break
        if agent is None:
            return None

        offsets = {
            "north": (0, -20),
            "south": (0, 20),
            "east": (20, 0),
            "west": (-20, 0),
            "northeast": (14, -14),
            "northwest": (-14, -14),
            "southeast": (14, 14),
            "southwest": (-14, 14),
        }
        off = offsets.get(direction)
        if off is None:
            return None

        agent.state.exploration_target = (agent.state.x + off[0], agent.state.y + off[1])
        agent.state.destination = f"Exploring {direction}"
        return self.get_state()

    # ── Player control ──────────────────────────────────────────────

    def set_agent_control(self, name: str, controlled: bool) -> Optional[Dict[str, Any]]:
        """Toggle player control mode for an agent. Returns updated agent state dict or None."""
        for agent in self.agents:
            if agent.name == name:
                agent.state.player_controlled = bool(controlled)
                if not controlled:
                    agent.state.pending_action = None
                return self.get_state()
        return None

    def set_agent_action(self, name: str, action_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Queue a manual action for a player-controlled agent. Returns updated state or None."""
        for agent in self.agents:
            if agent.name == name:
                if not agent.state.player_controlled:
                    return None
                agent.state.pending_action = action_data
                return self.get_state()
        return None

    def initiate_conversation(self, agent_name: str, target_name: str) -> Optional[Dict[str, Any]]:
        """Trigger a conversation between a controlled agent and a target at the same location.

        Returns dict with conversation lines and updated state, or None on failure.
        """
        agent = None
        target = None
        for a in self.agents:
            if a.name == agent_name:
                agent = a
            if a.name == target_name:
                target = a

        if agent is None or target is None:
            return None

        # Check same location
        agent_loc = agent.location
        target_loc = target.location
        if not agent_loc or not target_loc or agent_loc.name != target_loc.name:
            return None

        conv_events = self._generate_conversation(agent, target, agent_loc.name)
        # Set cooldown
        agent.state.conversation_cooldown = 2
        target.state.conversation_cooldown = 2

        conversation_lines = []
        for evt in conv_events:
            conversation_lines.extend(evt.details.get("dialogue", []))

        return {
            "conversation": conversation_lines,
            "state": self.get_state(),
        }

    def get_agent_detail(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a specific agent."""
        for agent in self.agents:
            if agent.name == agent_name:
                mem = self.memories.get(agent.name)
                return {
                    "name": agent.name,
                    "emoji": agent.profile.emoji,
                    "age": agent.profile.age,
                    "occupation": agent.profile.occupation,
                    "personality": agent.profile.personality,
                    "location": agent.state.current_location_name or "on the road",
                    "action": agent.state.current_action,
                    "plan": agent.state.daily_plan,
                    "probe_summary": agent.state.last_probe_summary,
                    "probe_steps": agent.state.last_probe_steps,
                    "recent_memories": [
                        f"[{m.kind}] {m.content}"
                        for m in (mem.get_recent(15) if mem else [])
                    ],
                    "reflections": [
                        m.content for m in (mem.get_reflections() if mem else [])
                    ],
                    "relationships": agent.profile.relationships,
                    "memory_count": mem.count if mem else 0,
                }
        return None
