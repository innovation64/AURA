"""Active environment probing inspired by tool-calling loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from aura.tools import Tool, ToolCall, ToolRegistry, ToolResult

from .llm_engine import LLMEngine
from .agents import TownAgent
from .memory import TownMemory

if TYPE_CHECKING:
    from .simulation import SimEvent


@dataclass
class ProbeStep:
    tool: str
    arguments: Dict[str, Any]
    ok: bool
    output: Any = None
    error: Optional[str] = None


@dataclass
class ProbeResult:
    summary: str
    steps: List[ProbeStep] = field(default_factory=list)

    def to_prompt(self) -> str:
        if not self.steps:
            return "No probe data."
        lines = ["Environment probe findings:"]
        for step in self.steps:
            if not step.ok:
                continue
            if step.output is None:
                continue
            # Format probe output as structured context, not raw tool calls
            if step.tool == "world.location":
                if isinstance(step.output, dict):
                    lines.append(
                        f"- Current location detail: {step.output.get('name', '?')} "
                        f"({step.output.get('type', '?')}) — {step.output.get('description', '')}"
                    )
                else:
                    lines.append(f"- Location: {step.output}")
            elif step.tool == "world.time":
                if isinstance(step.output, dict):
                    lines.append(f"- Time: {step.output.get('time', step.output)}")
                else:
                    lines.append(f"- Time: {step.output}")
            elif step.tool == "world.nearby_agents":
                if isinstance(step.output, list) and step.output:
                    agents_str = ", ".join(
                        f"{a.get('name', '?')} ({a.get('action', '?')})"
                        for a in step.output[:5]
                    )
                    lines.append(f"- Nearby: {agents_str}")
                elif isinstance(step.output, list):
                    lines.append("- Nearby: no one nearby")
                else:
                    lines.append(f"- Nearby: {step.output}")
            elif step.tool == "memory.recent":
                if isinstance(step.output, list) and step.output:
                    lines.append(f"- Recent memories: {'; '.join(str(m) for m in step.output[:3])}")
                else:
                    lines.append("- No recent memories")
            elif step.tool == "world.events_recent":
                if isinstance(step.output, list) and step.output:
                    lines.append(f"- Recent events: {'; '.join(str(e) for e in step.output[:3])}")
            elif step.tool == "world.location_activities":
                if isinstance(step.output, dict):
                    loc_name = step.output.get("location", "?")
                    agents_there = step.output.get("agents_there", [])
                    if agents_there:
                        desc = ", ".join(f"{a.get('name','?')} ({a.get('action','?')})" for a in agents_there[:5])
                        lines.append(f"- At {loc_name}: {desc}")
                    else:
                        lines.append(f"- At {loc_name}: no one there")
                else:
                    lines.append(f"- Location activities: {step.output}")
            elif step.tool == "social.opportunities":
                if isinstance(step.output, list) and step.output:
                    opps = "; ".join(
                        f"{o.get('name','?')} at {o.get('location','?')} ({o.get('relationship','?')}, {'same location' if o.get('same_location') else 'different location'})"
                        for o in step.output[:3]
                    )
                    lines.append(f"- Social opportunities: {opps}")
                else:
                    lines.append("- No social opportunities found")
            else:
                lines.append(f"- {step.tool}: {step.output}")
        if len(lines) == 1:
            return "No probe data."
        return "\n".join(lines)


@dataclass
class ProbeDecision:
    action: str
    tool: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class LLMProbePlanner:
    """Use LLM to decide which tool to call next."""

    def __init__(self, llm: LLMEngine) -> None:
        self.llm = llm

    def decide(
        self,
        tools: List[Tool],
        context: Dict[str, Any],
        last_result: Optional[ToolResult],
    ) -> ProbeDecision:
        tool_lines = []
        for tool in tools:
            tool_lines.append(f"- {tool.name}: {tool.description}")
        tool_text = "\n".join(tool_lines)

        system = (
            "You are an environment probe controller for a town simulation agent. "
            "Your job is to gather NOVEL context the agent does NOT already have. "
            "The agent ALREADY knows: current location, time, nearby agents, and recent memories. "
            "DO NOT call tools that duplicate known info (world.time, world.location, world.nearby_agents, memory.recent). "
            "Instead, prioritize tools that provide NEW information to help execute the daily plan: "
            "1) social.opportunities — find the best interaction partner based on relationships, "
            "2) world.location_activities — scout a DESTINATION from the plan before moving there, "
            "3) world.agents_summary — overview of all agents (useful for coordination), "
            "4) world.events_recent — recent events that may require plan adjustment. "
            "Return JSON ONLY with keys: action (call_tool|stop), tool, arguments, reason."
        )
        agent_name = context.get("agent", "unknown")
        location = context.get("location", "unknown")
        current_action = context.get("action", "unknown")
        user = (
            f"Agent: {agent_name}\n"
            f"Location: {location}\n"
            f"Current action: {current_action}\n"
            f"Daily plan (next items): {context.get('daily_plan', [])}\n"
            f"Time: {context.get('time', 'unknown')}\n"
            f"Nearby: {context.get('nearby', [])}\n\n"
            "Available tools:\n"
            f"{tool_text}\n\n"
            f"Last tool result: {last_result}\n\n"
            "What tool should be called to help the agent decide an appropriate next action? "
            "Respond with a single JSON object."
        )

        result = self.llm.chat_json(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=200,
        )

        if isinstance(result, dict) and result.get("action") in {"call_tool", "stop"}:
            return ProbeDecision(
                action=str(result.get("action")),
                tool=result.get("tool"),
                arguments=result.get("arguments") or {},
                reason=str(result.get("reason", "")),
            )

        return ProbeDecision(action="stop", reason="Unrecognized probe decision output.")


class TownProbeRunner:
    """Runs a bounded tool-calling loop to gather extra context."""

    def __init__(self, llm: LLMEngine) -> None:
        self.llm = llm

    def _build_tools(
        self,
        agent: TownAgent,
        agents: List[TownAgent],
        memory: TownMemory,
        events: List["SimEvent"],
    ) -> List[Tool]:
        def _time() -> Dict[str, Any]:
            return {
                "time": agent._world.time.display,
                "day": agent._world.time.day,
                "time_24h": agent._world.time.time_24h,
            }

        def _location() -> Dict[str, Any]:
            loc = agent.location
            if not loc:
                return {"error": "agent location unknown"}
            return {
                "name": loc.name,
                "type": loc.type,
                "emoji": loc.emoji,
                "description": loc.description,
                "x": loc.x,
                "y": loc.y,
                "width": loc.width,
                "height": loc.height,
            }

        def _nearby(limit: int = 5) -> List[Dict[str, Any]]:
            nearby = []
            loc = agent.location
            if not loc:
                return nearby
            for other in agents:
                if other.name == agent.name:
                    continue
                other_loc = other.location
                if other_loc and other_loc.name == loc.name:
                    nearby.append(
                        {
                            "name": other.name,
                            "action": other.state.current_action,
                            "location": other.state.current_location_name,
                        }
                    )
            return nearby[: max(limit, 0)]

        def _agents_summary(limit: int = 10) -> List[Dict[str, Any]]:
            summary = [
                {
                    "name": other.name,
                    "action": other.state.current_action,
                    "location": other.state.current_location_name,
                }
                for other in agents
            ]
            return summary[: max(limit, 0)]

        def _recent_memories(limit: int = 5) -> List[str]:
            return [m.content for m in memory.get_recent(limit)]

        def _recent_events(limit: int = 5) -> List[str]:
            recent = events[-max(limit, 0) :] if limit else []
            return [e.description for e in recent]

        def _location_activities(location: str = "") -> Dict[str, Any]:
            """Check what agents and activities are at a DIFFERENT location."""
            target = location.strip()
            if not target:
                return {"error": "specify a location name"}
            result_agents = []
            for other in agents:
                if other.name == agent.name:
                    continue
                if other.state.current_location_name.lower() == target.lower():
                    result_agents.append({
                        "name": other.name,
                        "action": other.state.current_action,
                    })
            return {
                "location": target,
                "agents_there": result_agents,
                "count": len(result_agents),
            }

        def _agent_private_state(agent_name: str = "") -> Dict[str, Any]:
            """Read the private (non-scene-visible) state of a named agent.

            Surfaces the four private fields the AURATown public/private split
            keeps out of the passive scene snapshot: availability, emotional
            state, unspoken goal, and the 5-minute-old beliefs-about-others
            dictionary refreshed on co-location. Used by IntentFrame-driven
            probes when the literal-vs-implicit gap is high enough to need
            mood/availability context the surface query did not request.
            """
            target = (agent_name or "").strip()
            if not target:
                return {"error": "specify an agent name"}
            other = next((a for a in agents if a.name.lower() == target.lower()), None)
            if other is None:
                return {"error": f"unknown agent {agent_name!r}"}
            st = other.state
            return {
                "agent": other.name,
                "availability": st.availability,
                "emotional_state": st.emotional_state,
                "unspoken_goal": st.unspoken_goal,
                "current_action": st.current_action,
                "current_location": st.current_location_name,
            }

        def _agent_belief_about(believer: str = "", target: str = "") -> Dict[str, Any]:
            """Read what `believer` currently believes about `target`'s state.

            Beliefs lag actual state — they only refresh on co-location (see
            `private_state_evolution.update_beliefs_about_others`). Probing this
            surfaces second-order theory-of-mind cases such as
            ``does Lin Wei think Zhang Hao is at home?`` where the believer's
            stale memory may diverge from the target's current ground truth.
            """
            b = (believer or "").strip()
            t = (target or "").strip()
            if not b or not t:
                return {"error": "specify believer and target agent names"}
            ag = next((a for a in agents if a.name.lower() == b.lower()), None)
            if ag is None:
                return {"error": f"unknown believer {believer!r}"}
            belief = ag.state.beliefs_about_others.get(t)
            if not belief:
                # try case-insensitive lookup
                for k, v in ag.state.beliefs_about_others.items():
                    if k.lower() == t.lower():
                        belief = v
                        break
            if not belief:
                return {
                    "believer": b,
                    "target": t,
                    "belief": None,
                    "note": (
                        f"{b} has no recorded belief about {t} — they have not "
                        "been co-located recently."
                    ),
                }
            return {"believer": b, "target": t, "belief": belief}

        def _social_opportunities() -> List[Dict[str, Any]]:
            """Find the best social interaction opportunities based on proximity and relationships."""
            opportunities = []
            loc = agent.location
            if not loc:
                return opportunities
            for other in agents:
                if other.name == agent.name:
                    continue
                other_loc = other.location
                same_location = other_loc and other_loc.name == loc.name
                # Check relationship
                rel = agent.profile.relationships.get(other.name, "acquaintance")
                action = other.state.current_action.lower()
                # Score opportunity
                score = 0
                if same_location:
                    score += 3
                if "friend" in rel.lower() or "close" in rel.lower():
                    score += 2
                if any(w in action for w in ["chat", "idle", "walk", "sit", "relax", "break"]):
                    score += 1  # available for interaction
                if score >= 2:
                    opportunities.append({
                        "name": other.name,
                        "location": other.state.current_location_name,
                        "action": other.state.current_action,
                        "relationship": rel,
                        "same_location": same_location,
                        "opportunity_score": score,
                    })
            opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)
            return opportunities[:5]

        return [
            Tool(name="world.time", description="Get the current simulation time.", handler=_time),
            Tool(
                name="world.location",
                description="Get the agent's current location details.",
                handler=_location,
            ),
            Tool(
                name="world.nearby_agents",
                description="List agents at the same location.",
                handler=_nearby,
            ),
            Tool(
                name="world.agents_summary",
                description="List all agents with their location and action.",
                handler=_agents_summary,
            ),
            Tool(
                name="memory.recent",
                description="Get the agent's recent memories.",
                handler=_recent_memories,
            ),
            Tool(
                name="world.events_recent",
                description="Get recent global events.",
                handler=_recent_events,
            ),
            Tool(
                name="world.location_activities",
                description="Check what agents and activities are at a specific location (use to scout destinations before moving).",
                handler=_location_activities,
            ),
            Tool(
                name="social.opportunities",
                description="Find the best social interaction opportunities based on proximity and relationships.",
                handler=_social_opportunities,
            ),
            Tool(
                name="agent.private_state",
                description=(
                    "Read the private state of a named agent (availability, "
                    "emotional_state, unspoken_goal, current_action, "
                    "current_location). Use when the surface query may hide a "
                    "mood/availability/goal need the scene snapshot doesn't expose."
                ),
                handler=_agent_private_state,
            ),
            Tool(
                name="agent.belief_about",
                description=(
                    "Read what `believer` currently believes about `target`'s "
                    "state. Beliefs lag actual state and only refresh on "
                    "co-location, so this surfaces second-order ToM cases such "
                    "as 'does X think Y is busy?' where X's stale memory may "
                    "diverge from Y's current ground truth."
                ),
                handler=_agent_belief_about,
            ),
        ]

    def run(
        self,
        agent: TownAgent,
        agents: List[TownAgent],
        events: List["SimEvent"],
        perception: Dict[str, Any],
        memory: TownMemory,
        max_steps: int = 2,
    ) -> ProbeResult:
        tools = self._build_tools(agent, agents, memory, events)
        registry = ToolRegistry(tools)
        planner = LLMProbePlanner(self.llm)
        steps: List[ProbeStep] = []
        last_result: Optional[ToolResult] = None

        context = {
            "agent": agent.name,
            "location": perception.get("location"),
            "action": perception.get("current_action"),
            "nearby": [a.get("name") for a in perception.get("nearby_agents", [])],
            "time": perception.get("time"),
            "daily_plan": [p.get("activity", str(p)) if isinstance(p, dict) else str(p) for p in (agent.state.daily_plan or [])[:3]],
        }

        for _ in range(max(0, max_steps)):
            decision = planner.decide(tools, context, last_result)
            if decision.action != "call_tool" or not decision.tool:
                break

            call = ToolCall(name=decision.tool, arguments=decision.arguments)
            result = registry.execute(call)
            steps.append(
                ProbeStep(
                    tool=call.name,
                    arguments=call.arguments,
                    ok=result.ok,
                    output=result.output,
                    error=result.error,
                )
            )
            last_result = result

            # Update context summary for next iteration
            context["last_tool"] = {
                "name": call.name,
                "ok": result.ok,
                "output": result.output,
                "error": result.error,
            }

        summary = "No probe actions."
        if steps:
            summary = f"Used {len(steps)} probe tool(s)."

        return ProbeResult(summary=summary, steps=steps)
