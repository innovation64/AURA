"""
Baseline implementations for comparison experiments.

Provides:
1. Vanilla LLM — direct GPT call without any environment context
2. Static Context — LLM with one-shot environment snapshot (RAG-style)
3. ReAct Agent — interleaved reasoning + tool calling (reactive probing)
4. Reflexion — ReAct + self-reflection (Shinn et al., 2023)
5. Plan-and-Solve — explicit planning before execution (Wang et al., 2023)
6. Generative Agents — perceive-retrieve-plan-reflect (Park et al., 2023)

IMPORTANT: For fair comparison, all tool-using baselines share the same
tool set and budget as AURA (default 5 steps).
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from evaluation.config import EvalConfig

# Make the AURA package importable so RQ2 baselines can use the live
# LLMIntentInferrer (the same one exercised in RQ-Intent), rather than
# the server-side fixed-budget TownProbeRunner.
_AURA_SRC = Path(__file__).resolve().parent.parent / "AURA" / "src"
if str(_AURA_SRC) not in sys.path:
    sys.path.insert(0, str(_AURA_SRC))


def _get_client(config: EvalConfig) -> "OpenAI":
    if OpenAI is None:
        raise ImportError("pip install openai")
    return OpenAI(api_key=config.api_key, base_url=config.base_url)


# =============================================================================
# Shared tool set — same tools available to both ReAct and AURA
# =============================================================================

SHARED_TOOLS = [
    {"name": "get_time", "description": "Get the current simulation time"},
    {"name": "get_location", "description": "Get details about the agent's current location"},
    {"name": "get_nearby_agents", "description": "Get list of agents at the same location with their actions"},
    {"name": "get_all_agents", "description": "Get summary of all agents' locations and actions"},
    {"name": "get_recent_memories", "description": "Get the agent's most recent memories and reflections"},
    {"name": "get_recent_events", "description": "Get recent events in the town (conversations, movements, etc.)"},
    {"name": "get_agent_plan", "description": "Get the agent's daily plan and schedule"},
    {"name": "get_location_info", "description": "Get details about a specific location (who's there, what's happening)"},
]

# Disclosure cost per tool. Used by RQ2 Pareto analysis to penalise
# Fixed_Probe-style unconditional probing (which always touches the
# high-disclosure tools) on queries that don't need that information.
# Scale: 0 = public-observable, 1 = moderate (plans/locations of others),
# 2 = high (private memories, recent reflections, conversation logs).
# These labels are conservative — every tool here is in fact accessible
# to the agent in this town; the score reflects the *kind* of information
# revealed when the tool's result is surfaced to the user-facing answer.
TOOL_DISCLOSURE: Dict[str, int] = {
    "get_time": 0,
    "get_location": 0,
    "get_nearby_agents": 0,
    "get_all_agents": 1,
    "get_recent_memories": 2,
    "get_recent_events": 1,
    "get_agent_plan": 1,
    "get_location_info": 0,
}


def _execute_tool(tool_name: str, env_state: Dict, agent_name: str, args: Optional[Dict] = None) -> str:
    """Execute a tool — shared implementation for all baselines."""
    agent = next((a for a in env_state.get("agents", []) if a["name"] == agent_name), {})

    if tool_name == "get_time":
        return json.dumps({"time": env_state.get("time", "unknown"), "hour": env_state.get("hour", 0)})

    elif tool_name == "get_location":
        loc = agent.get("location", "unknown")
        agents_at_loc = [
            {"name": a["name"], "action": a["action"]}
            for a in env_state.get("agents", [])
            if a.get("location") == loc
        ]
        return json.dumps({"location": loc, "action": agent.get("action", "unknown"), "agents_here": agents_at_loc})

    elif tool_name == "get_nearby_agents":
        nearby = [
            {"name": a["name"], "action": a["action"], "location": a["location"]}
            for a in env_state.get("agents", [])
            if a.get("location") == agent.get("location") and a["name"] != agent_name
        ]
        return json.dumps(nearby)

    elif tool_name == "get_all_agents":
        summary = [
            {"name": a["name"], "location": a["location"], "action": a["action"]}
            for a in env_state.get("agents", [])
        ]
        return json.dumps(summary)

    elif tool_name == "get_recent_memories":
        memories = agent.get("memories", [])[:10]
        return json.dumps({"memories": [m.get("content", str(m)) for m in memories] if memories else ["No memories available"]})

    elif tool_name == "get_recent_events":
        events = env_state.get("events", [])[-10:]
        return json.dumps([
            {"time": e.get("time"), "type": e.get("type", ""), "description": e.get("description", "")}
            for e in events
        ])

    elif tool_name == "get_agent_plan":
        plan = agent.get("daily_plan", agent.get("plan", []))
        return json.dumps({"plan": plan if plan else ["No plan available"]})

    elif tool_name == "get_location_info":
        target_loc = (args or {}).get("location", agent.get("location", ""))
        agents_there = [
            {"name": a["name"], "action": a["action"]}
            for a in env_state.get("agents", [])
            if a.get("location") == target_loc
        ]
        return json.dumps({"location": target_loc, "agents": agents_there, "count": len(agents_there)})

    return json.dumps({"error": f"unknown tool: {tool_name}"})


def _resolve_location_from_query(query: str, locations: List[Dict[str, Any]], fallback: str = "") -> str:
    """Resolve an explicitly mentioned public location from a query.

    This is deliberately conservative: use an exact location-name match first,
    then a unique non-generic token match such as "library" -> "Town Library".
    Ambiguous type mentions such as "park" when multiple parks exist fall back
    to the caller's default instead of guessing.
    """
    q = query.lower()
    stop = {
        "the", "town", "place", "places", "building", "buildings", "location",
        "nearby", "closest", "right", "here", "there", "from", "where",
    }

    for loc in sorted(locations, key=lambda x: len(str(x.get("name", ""))), reverse=True):
        name = str(loc.get("name", "")).strip()
        if name and name.lower() in q:
            return name

    matches = []
    for loc in locations:
        name = str(loc.get("name", "")).strip()
        if not name:
            continue
        tokens = [
            tok.strip(".,?!:;()[]{}'\"").lower()
            for tok in name.replace("-", " ").split()
        ]
        useful = [tok for tok in tokens if len(tok) > 3 and tok not in stop]
        score = sum(1 for tok in useful if tok in q)
        if score:
            matches.append((score, len(name), name))

    if not matches:
        return fallback
    matches.sort(key=lambda item: (-item[0], item[1], item[2]))
    best_score = matches[0][0]
    best = [m for m in matches if m[0] == best_score]
    if len(best) == 1:
        return best[0][2]
    return fallback


# =============================================================================
# Baseline 1: Vanilla LLM (no environment context)
# =============================================================================

def vanilla_llm_chat(config: EvalConfig, agent_name: str, query: str) -> Dict[str, Any]:
    """
    Direct LLM call without any environment context.
    This is the weakest baseline — the model has zero grounding.
    """
    client = _get_client(config)

    messages = [
        {
            "role": "system",
            "content": (
                f"You are {agent_name}, a resident of a small town. "
                "Answer the user's question based on what you would reasonably know."
            ),
        },
        {"role": "user", "content": query},
    ]

    t0 = time.time()
    resp = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=512,
    )
    latency = time.time() - t0

    return {
        "response": resp.choices[0].message.content.strip(),
        "latency": latency,
        "method": "vanilla_llm",
        "tool_calls": 0,
        "env_context": {},
    }


# =============================================================================
# Baseline 2: Static Context (RAG-style, no probe)
# =============================================================================

def static_context_chat(
    config: EvalConfig,
    agent_name: str,
    query: str,
    env_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    LLM call with static environment context (like simple RAG).
    Context is provided but NOT proactively gathered — just the current snapshot.
    """
    client = _get_client(config)

    agent = next((a for a in env_state.get("agents", []) if a["name"] == agent_name), {})
    nearby = [a for a in env_state.get("agents", [])
              if a.get("location") == agent.get("location") and a["name"] != agent_name]

    context = {
        "location": agent.get("location", "unknown"),
        "time": env_state.get("time", "unknown"),
        "current_action": agent.get("action", "unknown"),
        "nearby_agents": [{"name": a["name"], "action": a["action"]} for a in nearby],
    }

    messages = [
        {
            "role": "system",
            "content": (
                f"You are an AI assistant helping {agent_name} in a small town.\n"
                f"Current context:\n"
                f"- Location: {context['location']}\n"
                f"- Time: {context['time']}\n"
                f"- Current activity: {context['current_action']}\n"
                f"- Nearby: {json.dumps(context['nearby_agents'])}\n\n"
                "Use this context to answer the question accurately. "
                "If you are unsure about something, say so rather than guessing."
            ),
        },
        {"role": "user", "content": query},
    ]

    t0 = time.time()
    resp = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=512,
    )
    latency = time.time() - t0

    return {
        "response": resp.choices[0].message.content.strip(),
        "latency": latency,
        "method": "static_context",
        "tool_calls": 0,
        "env_context": context,
    }


# =============================================================================
# Baseline 3: ReAct-style (reactive tool calling during reasoning)
# Uses SAME tool set and budget as AURA for fair comparison.
# =============================================================================

def fixed_probe_chat(
    config: EvalConfig,
    agent_name: str,
    query: str,
    env_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Cross-regime control: unconditionally fire every SHARED_TOOLS probe.

    No gap inference, no tool selection — every query receives the same
    saturated probe dump regardless of intent. Mirrors the spirit of the
    RQ-Intent fixed-private baseline (always probe), adapted to the RQ2
    town's tool registry which has no private-state tool. Tests whether
    AURA's gap-routed "skip when not needed" advantage matters on the
    factual lookup regime.
    """
    client = _get_client(config)

    probe_outputs: Dict[str, str] = {}
    t0 = time.time()
    for tool_spec in SHARED_TOOLS:
        tool_name = tool_spec["name"]
        # get_location_info needs an args.location; for fixed_probe we
        # default to the agent's own location to keep the call deterministic.
        args = None
        if tool_name == "get_location_info":
            agent = next((a for a in env_state.get("agents", []) if a["name"] == agent_name), {})
            args = {"location": agent.get("location", "")}
        try:
            probe_outputs[tool_name] = _execute_tool(tool_name, env_state, agent_name, args)
        except Exception as e:
            probe_outputs[tool_name] = json.dumps({"error": str(e)})

    probe_block = "\n".join(
        f"- {name}: {out[:600]}" for name, out in probe_outputs.items()
    )

    messages = [
        {
            "role": "system",
            "content": (
                f"You are an AI assistant helping {agent_name} answer a question about their town.\n\n"
                f"Probe results (every available tool was invoked unconditionally):\n{probe_block}\n\n"
                "Use the probe results to answer the user's question. "
                "If a probe is irrelevant to the query, ignore it. "
                "If you are unsure, say so rather than guessing."
            ),
        },
        {"role": "user", "content": query},
    ]

    resp = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=512,
    )
    latency = time.time() - t0

    return {
        "response": resp.choices[0].message.content.strip(),
        "latency": latency,
        "method": "fixed_probe",
        "tool_calls": len(SHARED_TOOLS),
        "env_context": probe_outputs,
    }


def aura_gap_routed_chat(
    config: EvalConfig,
    agent_name: str,
    query: str,
    env_state: Dict[str, Any],
    configured_budget: int = 5,
) -> Dict[str, Any]:
    """AURA's gap-routed mechanism actually exercised on RQ2.

    The server-side AURA_Full RQ2 condition uses TownProbeRunner with a
    fixed `probe_max_steps`, which never exercises the
    LLM-IntentInferrer the paper claims as the contribution. This baseline
    plugs the live LLMIntentInferrer onto the RQ2 town and tools so the
    gap → budget → probe-selection → synthesise loop is what gets
    measured.

    Returns the standard baseline payload plus `gap`, `recommended_probes`,
    and `actual_probes` so downstream cost / privacy analyses can attribute
    over- or under-probing to the inferrer.
    """
    # Late import: avoids paying the AURA-package import cost on every
    # baseline that doesn't need it, and keeps a clean failure path if
    # AURA/src is not on sys.path for some reason.
    from aura.intent import LLMIntentInferrer
    from aura.types import SceneState

    client = _get_client(config)
    inferrer = LLMIntentInferrer(client=client, model=config.model)

    agents = env_state.get("agents", [])
    locations = env_state.get("locations", [])
    agent = next((a for a in agents if a["name"] == agent_name), {})
    other_names = [a["name"] for a in agents if a["name"] != agent_name]
    agent_location = agent.get("location", "")
    nearby_rows = [
        f"{a.get('name', '?')} ({a.get('action', 'unknown')})"
        for a in agents
        if a.get("name") != agent_name and a.get("location") == agent_location
    ]

    agent_rows = []
    for a in agents:
        name = a.get("name", "?")
        loc = a.get("location", "unknown")
        action = a.get("action", "unknown")
        agent_rows.append(f"{name} at {loc} ({action})")

    location_rows = []
    for loc in locations[:24]:
        name = loc.get("name", "?")
        loc_type = loc.get("type", "place")
        x = loc.get("x")
        y = loc.get("y")
        coord = f", x={x}, y={y}" if x is not None and y is not None else ""
        location_rows.append(f"{name} ({loc_type}{coord})")
    if len(locations) > 24:
        location_rows.append(f"... {len(locations) - 24} more locations")

    scene_summary = (
        f"Time: {env_state.get('time', '?')}. "
        f"{agent_name} is at {agent.get('location', '?')}, doing: {agent.get('action', '?')}. "
        f"Other agents present in town: {', '.join(other_names) or '(none)'}. "
        "Nearby people at the same location: "
        f"{'; '.join(nearby_rows) if nearby_rows else 'none'}. "
        "Known public town locations: "
        f"{'; '.join(location_rows) if location_rows else '(none listed)'}. "
        "Current public town snapshot: "
        f"{'; '.join(agent_rows) if agent_rows else '(no agent snapshot)'}."
    )
    scene_entities = list(dict.fromkeys(
        [a.get("name", "") for a in agents]
        + [loc.get("name", "") for loc in locations]
    ))
    scene = SceneState(
        summary=scene_summary,
        entities=[e for e in scene_entities if e],
        context={
            "current_agent": agent,
            "agents": agents,
            "locations": locations,
        },
    )
    tool_names = [t["name"] for t in SHARED_TOOLS]

    t0 = time.time()
    frame = inferrer.infer(query, scene, [], available_tools=tool_names)

    # gap → budget mapping mirrors aura.core.intent_gap_to_budget exactly.
    g = frame.gap or 0.0
    if g < 0.20:
        dyn_budget = 0
    elif g < 0.40:
        dyn_budget = 1
    elif g < 0.60:
        dyn_budget = 2
    elif g < 0.80:
        dyn_budget = 3
    else:
        dyn_budget = 5
    dyn_budget = min(configured_budget, dyn_budget)

    # Probe selection: prefer the inferrer's recommended_probes, filtered
    # to actually-available tools; truncate to dyn_budget.
    candidates = [p for p in (frame.recommended_probes or []) if p in tool_names]
    probes_to_run = candidates[:dyn_budget]

    probe_outputs: Dict[str, str] = {}
    for tn in probes_to_run:
        args = None
        if tn == "get_location_info":
            args = {
                "location": _resolve_location_from_query(
                    query, locations, fallback=agent.get("location", ""),
                )
            }
        try:
            probe_outputs[tn] = _execute_tool(tn, env_state, agent_name, args)
        except Exception as e:
            probe_outputs[tn] = json.dumps({"error": str(e)})

    if probe_outputs:
        probe_block = "\n".join(f"- {n}: {o[:600]}" for n, o in probe_outputs.items())
        sys_prompt = (
            f"You are an AI assistant helping {agent_name}.\n"
            f"Scene: {scene_summary}\n\n"
            f"Inferred intent — gap={g:.2f}; implicit_need={frame.implicit_need}\n"
            f"Probe results (selected by gap-routing):\n{probe_block}\n\n"
            "Answer the user's question."
        )
    else:
        sys_prompt = (
            f"You are an AI assistant helping {agent_name}.\n"
            f"Scene: {scene_summary}\n\n"
            f"Inferred intent — gap={g:.2f} (low; literal answer is likely sufficient).\n"
            "Answer the user's question using only the scene description."
        )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": query},
    ]
    resp = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=512,
    )
    latency = time.time() - t0

    return {
        "response": resp.choices[0].message.content.strip(),
        "latency": latency,
        "method": "aura_gap_routed",
        "tool_calls": len(probe_outputs),
        "env_context": probe_outputs,
        "gap": g,
        "recommended_probes": candidates,
        "actual_probes": list(probe_outputs.keys()),
        "implicit_need": frame.implicit_need,
        "should_alert": frame.should_alert,
    }


def react_chat(
    config: EvalConfig,
    agent_name: str,
    query: str,
    env_state: Dict[str, Any],
    max_react_steps: int = 5,  # FIXED: same budget as AURA explore_max_steps
) -> Dict[str, Any]:
    """
    ReAct-style agent: interleaves Thought -> Action -> Observation in a single LLM stream.
    Tools are called DURING reasoning (reactive), not before (proactive).
    Uses the same tool set and budget as AURA for fair comparison.
    """
    client = _get_client(config)

    tool_names = [t["name"] for t in SHARED_TOOLS]
    tool_descriptions = "\n".join(f"- {t['name']}: {t['description']}" for t in SHARED_TOOLS)

    system_prompt = f"""You are an AI assistant helping {agent_name} answer a question about their town.

You have access to these tools:
{tool_descriptions}

Use the ReAct format:
Thought: [your reasoning about what information you need]
Action: [tool_name]
Observation: [tool result will be inserted here]
... (repeat if needed, max {max_react_steps} tool calls)
Thought: [final reasoning]
Answer: [your final answer to the user]

IMPORTANT:
- Use tools to gather information BEFORE answering.
- If you are unsure, use a tool to verify rather than guessing.
- If you have enough information, skip to the Answer directly."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    all_context = {}
    tool_call_count = 0
    t0 = time.time()

    for step in range(max_react_steps + 1):
        resp = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=512,
        )
        content = resp.choices[0].message.content.strip()

        # Check if there's an Action to execute
        if "Action:" in content and "Answer:" not in content:
            # Extract tool name
            action_line = [line for line in content.split("\n") if line.strip().startswith("Action:")]
            if action_line:
                tool_name = action_line[0].split("Action:")[1].strip().lower()
                # Find matching tool
                matched = None
                for tn in tool_names:
                    if tn in tool_name:
                        matched = tn
                        break

                if matched:
                    observation = _execute_tool(matched, env_state, agent_name)
                    all_context[matched] = observation
                    tool_call_count += 1
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": f"Observation: {observation}"})
                    continue

        # No more actions or Answer found
        break

    latency = time.time() - t0

    # Extract final answer
    answer = content
    if "Answer:" in content:
        answer = content.split("Answer:")[-1].strip()

    return {
        "response": answer,
        "latency": latency,
        "method": "react",
        "tool_calls": tool_call_count,
        "react_steps": min(step + 1, max_react_steps),
        "env_context": all_context,
    }


# =============================================================================
# Run all baselines for a given query set
# =============================================================================

def run_all_baselines(
    config: EvalConfig,
    agent_name: str,
    queries: List[Dict],
    env_state: Dict[str, Any],
    agent_profile: Optional[dict] = None,
) -> Dict[str, List[Dict]]:
    """Run all baseline methods on a set of queries."""
    results = {
        "vanilla_llm": [],
        "static_context": [],
        "react": [],
        "generative_agents": [],
        "contextagent": [],
    }
    memory_stream = _MemoryStream()

    for qi, q in enumerate(queries):
        query_text = q["query"]
        if qi % 10 == 0:
            print(f"  Baseline query {qi}/{len(queries)}...")

        # Vanilla LLM
        try:
            r = vanilla_llm_chat(config, agent_name, query_text)
            r["query_id"] = q["id"]
            r["query"] = query_text
            r["category"] = q["category"]
            results["vanilla_llm"].append(r)
        except Exception as e:
            results["vanilla_llm"].append({"query_id": q["id"], "error": str(e)})

        # Static Context
        try:
            r = static_context_chat(config, agent_name, query_text, env_state)
            r["query_id"] = q["id"]
            r["query"] = query_text
            r["category"] = q["category"]
            results["static_context"].append(r)
        except Exception as e:
            results["static_context"].append({"query_id": q["id"], "error": str(e)})

        # ReAct (same budget as AURA)
        try:
            r = react_chat(config, agent_name, query_text, env_state, max_react_steps=5)
            r["query_id"] = q["id"]
            r["query"] = query_text
            r["category"] = q["category"]
            results["react"].append(r)
        except Exception as e:
            results["react"].append({"query_id": q["id"], "error": str(e)})

        # Generative Agents (Park et al., 2023)
        try:
            r = generative_agents_chat(config, agent_name, query_text, env_state, memory_stream)
            r["query_id"] = q["id"]
            r["query"] = query_text
            r["category"] = q["category"]
            results["generative_agents"].append(r)
        except Exception as e:
            results["generative_agents"].append({"query_id": q["id"], "error": str(e)})

        # ContextAgent-style (Yang et al., NeurIPS 2025)
        try:
            r = contextagent_chat(config, agent_name, query_text, env_state, agent_profile)
            r["query_id"] = q["id"]
            r["query"] = query_text
            r["category"] = q["category"]
            results["contextagent"].append(r)
        except Exception as e:
            results["contextagent"].append({"query_id": q["id"], "error": str(e)})

        time.sleep(0.5)  # Rate limit

    return results

# =============================================================================
# ReAct-style ACTION DECISION (for RQ1 — not chat)
# Uses interleaved Thought -> Action -> Observation during action reasoning.
# =============================================================================

def react_action_decision(
    config: "EvalConfig",
    agent_name: str,
    agent_profile: dict,
    perception: dict,
    env_state: dict,
    daily_plan: list,
    location_names: list,
    max_react_steps: int = 3,
) -> dict:
    """
    ReAct-style action decision: interleaves reasoning with tool calls.
    Unlike AURA where probing happens BEFORE reasoning, here tools are called
    DURING the reasoning process (true ReAct paradigm).
    """
    import time as _time

    client = _get_client(config)

    tool_descriptions = "\n".join(
        f"- {t['name']}: {t['description']}" for t in SHARED_TOOLS
    )

    system_prompt = (
        f"You are simulating {agent_name}, a {agent_profile.get('age', 30)}-year-old "
        f"{agent_profile.get('occupation', 'resident')}.\n"
        f"Personality: {agent_profile.get('personality', 'friendly')}\n\n"
        f"You must decide what {agent_name} does next. You can use tools to gather information.\n\n"
        f"Available tools:\n{tool_descriptions}\n\n"
        f"Use the ReAct format:\n"
        f"Thought: [your reasoning about what to do next]\n"
        f"Action: [tool_name] (optional - only if you need more info)\n"
        f"Observation: [tool result will be inserted here]\n"
        f"... (repeat if needed, max {max_react_steps} tool calls)\n"
        f"Thought: [final reasoning]\n"
        f'Decision: {{"action": "...", "location": "...", "thought": "...", "emoji": "..."}}\n\n'
        f"Available locations: {', '.join(location_names)}\n\n"
        f'IMPORTANT: Your Decision must be valid JSON on a single line after "Decision: ".'
    )

    user_msg = (
        f"Current time: {perception.get('time', 'unknown')}\n"
        f"Current location: {perception.get('location', 'unknown')}\n"
        f"Current activity: {perception.get('current_action', 'unknown')}\n"
        f"Today's plan: {json.dumps(daily_plan[:5])}\n\n"
        f"What does {agent_name} do next? Use tools if you need more information."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    tool_call_count = 0
    t0 = _time.time()

    for step in range(max_react_steps + 1):
        resp = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=512,
        )
        content_text = resp.choices[0].message.content.strip()

        # Check if there is a Decision
        if "Decision:" in content_text:
            try:
                decision_str = content_text.split("Decision:")[-1].strip()
                start_idx = decision_str.index("{")
                end_idx = decision_str.rindex("}") + 1
                decision = json.loads(decision_str[start_idx:end_idx])
                decision["method"] = "react"
                decision["tool_calls"] = tool_call_count
                decision["latency"] = _time.time() - t0
                return decision
            except (ValueError, json.JSONDecodeError):
                pass

        # Check for Action to execute
        if "Action:" in content_text and "Decision:" not in content_text:
            action_line = [
                l for l in content_text.split("\n")
                if l.strip().startswith("Action:")
            ]
            if action_line:
                tool_name = action_line[0].split("Action:")[1].strip().lower()
                tool_names = [t["name"] for t in SHARED_TOOLS]
                matched = None
                for tn in tool_names:
                    if tn in tool_name:
                        matched = tn
                        break
                if matched:
                    observation = _execute_tool(matched, env_state, agent_name)
                    tool_call_count += 1
                    messages.append({"role": "assistant", "content": content_text})
                    messages.append(
                        {"role": "user", "content": f"Observation: {observation}"}
                    )
                    continue

        break

    # Fallback: try to parse any JSON from the last response
    latency = _time.time() - t0
    try:
        start_idx = content_text.index("{")
        end_idx = content_text.rindex("}") + 1
        result = json.loads(content_text[start_idx:end_idx])
        result["method"] = "react"
        result["tool_calls"] = tool_call_count
        result["latency"] = latency
        return result
    except (ValueError, json.JSONDecodeError):
        return {
            "action": "idle",
            "location": perception.get("location", ""),
            "thought": "Could not decide",
            "emoji": "",
            "method": "react",
            "tool_calls": tool_call_count,
            "latency": latency,
        }


# =============================================================================
# Baseline 4: Reflexion — ReAct + self-reflection on failures
# (Shinn et al., 2023: "Reflexion: Language Agents with Verbal Reinforcement")
# =============================================================================

def reflexion_chat(
    config: EvalConfig,
    agent_name: str,
    query: str,
    env_state: Dict[str, Any],
    max_react_steps: int = 5,
    max_reflexion_rounds: int = 2,
) -> Dict[str, Any]:
    """
    Reflexion agent: runs ReAct, self-evaluates, reflects on errors, retries.
    Each round: Act -> Evaluate -> Reflect -> Retry (if needed).
    """
    client = _get_client(config)
    t0 = time.time()
    total_tool_calls = 0
    reflections: List[str] = []

    for reflexion_round in range(max_reflexion_rounds + 1):
        # Build reflection-augmented prompt
        reflection_block = ""
        if reflections:
            reflection_block = (
                "\n\n## Previous Reflections (learn from past mistakes)\n"
                + "\n".join(f"- Round {i+1}: {r}" for i, r in enumerate(reflections))
                + "\n\nUse these reflections to improve your answer.\n"
            )

        tool_names = [t["name"] for t in SHARED_TOOLS]
        tool_descriptions = "\n".join(
            f"- {t['name']}: {t['description']}" for t in SHARED_TOOLS
        )

        system_prompt = f"""You are an AI assistant helping {agent_name} answer a question about their town.

You have access to these tools:
{tool_descriptions}

Use the ReAct format:
Thought: [your reasoning]
Action: [tool_name]
Observation: [tool result]
... (max {max_react_steps} tool calls)
Thought: [final reasoning]
Answer: [your final answer]

IMPORTANT: Use tools to verify facts. If uncertain, probe before answering.
{reflection_block}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        # Run ReAct loop
        round_tool_calls = 0
        content = ""
        for step in range(max_react_steps + 1):
            resp = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=512,
            )
            content = resp.choices[0].message.content.strip()

            if "Action:" in content and "Answer:" not in content:
                action_line = [
                    line for line in content.split("\n")
                    if line.strip().startswith("Action:")
                ]
                if action_line:
                    tool_name = action_line[0].split("Action:")[1].strip().lower()
                    matched = None
                    for tn in tool_names:
                        if tn in tool_name:
                            matched = tn
                            break
                    if matched:
                        observation = _execute_tool(matched, env_state, agent_name)
                        round_tool_calls += 1
                        messages.append({"role": "assistant", "content": content})
                        messages.append(
                            {"role": "user", "content": f"Observation: {observation}"}
                        )
                        continue
            break

        total_tool_calls += round_tool_calls

        # Extract answer
        answer = content
        if "Answer:" in content:
            answer = content.split("Answer:")[-1].strip()

        # Self-evaluate (only if not last round)
        if reflexion_round < max_reflexion_rounds:
            eval_prompt = f"""Evaluate your answer to: "{query}"
Your answer: "{answer}"

Check:
1. Did you verify facts using tools, or did you guess?
2. Are there claims that might be wrong?
3. Did you use enough tools to gather information?

If the answer is good, reply: PASS
If the answer could be improved, reply:
REFLECT: [what went wrong and how to improve]"""

            eval_messages = [
                {"role": "system", "content": "You are a critical self-evaluator."},
                {"role": "user", "content": eval_prompt},
            ]
            eval_resp = client.chat.completions.create(
                model=config.model,
                messages=eval_messages,
                temperature=0.1,
                max_tokens=256,
            )
            eval_text = eval_resp.choices[0].message.content.strip()

            if "PASS" in eval_text and "REFLECT" not in eval_text:
                break  # Answer is satisfactory
            elif "REFLECT:" in eval_text:
                reflection = eval_text.split("REFLECT:")[-1].strip()
                reflections.append(reflection)
                continue  # Try again with reflection
            else:
                break  # Ambiguous evaluation, keep current answer

    latency = time.time() - t0
    return {
        "response": answer,
        "latency": latency,
        "method": "reflexion",
        "tool_calls": total_tool_calls,
        "reflexion_rounds": len(reflections),
        "reflections": reflections,
        "env_context": {},
    }


def reflexion_action_decision(
    config: "EvalConfig",
    agent_name: str,
    agent_profile: dict,
    perception: dict,
    env_state: dict,
    daily_plan: list,
    location_names: list,
    max_react_steps: int = 3,
    max_reflexion_rounds: int = 1,
) -> dict:
    """Reflexion-style action decision with self-critique."""
    client = _get_client(config)
    t0 = time.time()
    total_tool_calls = 0
    reflections: List[str] = []

    tool_descriptions = "\n".join(
        f"- {t['name']}: {t['description']}" for t in SHARED_TOOLS
    )

    for reflexion_round in range(max_reflexion_rounds + 1):
        reflection_block = ""
        if reflections:
            reflection_block = (
                "\n\nReflections from previous attempts:\n"
                + "\n".join(f"- {r}" for r in reflections)
                + "\nImprove your decision based on these reflections.\n"
            )

        system_prompt = (
            f"You are simulating {agent_name}, a {agent_profile.get('age', 30)}-year-old "
            f"{agent_profile.get('occupation', 'resident')}.\n"
            f"Personality: {agent_profile.get('personality', 'friendly')}\n\n"
            f"Available tools:\n{tool_descriptions}\n\n"
            f"Available locations: {', '.join(location_names)}\n\n"
            f"Use ReAct format, then output Decision as JSON.\n"
            f'Decision: {{"action": "...", "location": "...", "thought": "...", "emoji": "..."}}'
            f"{reflection_block}"
        )

        user_msg = (
            f"Time: {perception.get('time', 'unknown')}, "
            f"Location: {perception.get('location', 'unknown')}, "
            f"Activity: {perception.get('current_action', 'unknown')}\n"
            f"Plan: {json.dumps(daily_plan[:5])}\n"
            f"What does {agent_name} do next?"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        round_tool_calls = 0
        content_text = ""
        for step in range(max_react_steps + 1):
            resp = client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=512,
            )
            content_text = resp.choices[0].message.content.strip()

            if "Decision:" in content_text:
                break

            if "Action:" in content_text:
                action_line = [
                    ln for ln in content_text.split("\n")
                    if ln.strip().startswith("Action:")
                ]
                if action_line:
                    tool_name = action_line[0].split("Action:")[1].strip().lower()
                    tool_names = [t["name"] for t in SHARED_TOOLS]
                    matched = next((tn for tn in tool_names if tn in tool_name), None)
                    if matched:
                        obs = _execute_tool(matched, env_state, agent_name)
                        round_tool_calls += 1
                        messages.append({"role": "assistant", "content": content_text})
                        messages.append({"role": "user", "content": f"Observation: {obs}"})
                        continue
            break

        total_tool_calls += round_tool_calls

        # Try parse decision
        if "Decision:" in content_text:
            try:
                dec_str = content_text.split("Decision:")[-1].strip()
                si = dec_str.index("{")
                ei = dec_str.rindex("}") + 1
                decision = json.loads(dec_str[si:ei])
                decision["method"] = "reflexion"
                decision["tool_calls"] = total_tool_calls
                decision["latency"] = time.time() - t0
                decision["reflexion_rounds"] = len(reflections)
                return decision
            except (ValueError, json.JSONDecodeError):
                pass

        # Self-evaluate for reflection
        if reflexion_round < max_reflexion_rounds:
            reflections.append(
                f"Failed to produce valid decision in round {reflexion_round+1}. "
                "Focus on outputting a valid JSON Decision."
            )
            continue

    latency = time.time() - t0
    return {
        "action": "idle",
        "location": perception.get("location", ""),
        "thought": "Reflexion could not decide",
        "emoji": "",
        "method": "reflexion",
        "tool_calls": total_tool_calls,
        "latency": latency,
        "reflexion_rounds": len(reflections),
    }


# =============================================================================
# Baseline 5: Plan-and-Solve — explicit planning before execution
# (Wang et al., 2023: "Plan-and-Solve Prompting")
# =============================================================================

def plan_and_solve_chat(
    config: EvalConfig,
    agent_name: str,
    query: str,
    env_state: Dict[str, Any],
    max_steps: int = 5,
) -> Dict[str, Any]:
    """
    Plan-and-Solve: first generates a plan of what information to gather,
    then executes the plan step-by-step using tools.
    """
    client = _get_client(config)
    t0 = time.time()

    tool_descriptions = "\n".join(
        f"- {t['name']}: {t['description']}" for t in SHARED_TOOLS
    )

    # Phase 1: Planning
    plan_prompt = f"""You need to answer this question about {agent_name}'s town: "{query}"

Available tools:
{tool_descriptions}

Create a step-by-step plan to gather the necessary information. List 1-{max_steps} steps.
Format each step as: STEP N: [tool_name] — [why this information is needed]

If you can answer without tools, write: STEP 1: DIRECT — [reason]"""

    plan_messages = [
        {"role": "system", "content": "You are a planning agent. Create efficient plans."},
        {"role": "user", "content": plan_prompt},
    ]

    plan_resp = client.chat.completions.create(
        model=config.model,
        messages=plan_messages,
        temperature=0.3,
        max_tokens=512,
    )
    plan_text = plan_resp.choices[0].message.content.strip()

    # Phase 2: Execute plan
    tool_names = [t["name"] for t in SHARED_TOOLS]
    gathered_context = {}
    tool_call_count = 0

    for line in plan_text.split("\n"):
        if not line.strip().startswith("STEP"):
            continue
        if "DIRECT" in line:
            break

        # Extract tool name from plan step
        matched = None
        line_lower = line.lower()
        for tn in tool_names:
            if tn in line_lower:
                matched = tn
                break
        if matched and tool_call_count < max_steps:
            result = _execute_tool(matched, env_state, agent_name)
            gathered_context[matched] = result
            tool_call_count += 1

    # Phase 3: Synthesize answer using gathered context
    context_str = "\n".join(
        f"[{tool}]: {data}" for tool, data in gathered_context.items()
    )

    synth_prompt = f"""You are helping {agent_name} in a small town.

Question: {query}

Information gathered:
{context_str if context_str else "(no tools used)"}

Answer the question accurately using the gathered information.
If information is missing, say what you don't know rather than guessing."""

    synth_messages = [
        {"role": "system", "content": synth_prompt},
        {"role": "user", "content": query},
    ]

    synth_resp = client.chat.completions.create(
        model=config.model,
        messages=synth_messages,
        temperature=config.temperature,
        max_tokens=512,
    )
    answer = synth_resp.choices[0].message.content.strip()
    latency = time.time() - t0

    return {
        "response": answer,
        "latency": latency,
        "method": "plan_and_solve",
        "tool_calls": tool_call_count,
        "plan": plan_text,
        "env_context": gathered_context,
    }


def plan_and_solve_action_decision(
    config: "EvalConfig",
    agent_name: str,
    agent_profile: dict,
    perception: dict,
    env_state: dict,
    daily_plan: list,
    location_names: list,
    max_steps: int = 3,
) -> dict:
    """Plan-and-Solve action decision: plan information needs, gather, decide."""
    client = _get_client(config)
    t0 = time.time()

    tool_descriptions = "\n".join(
        f"- {t['name']}: {t['description']}" for t in SHARED_TOOLS
    )

    # Phase 1: Plan
    plan_prompt = (
        f"You are deciding what {agent_name} ({agent_profile.get('occupation', 'resident')}) "
        f"should do next.\nTime: {perception.get('time')}, Location: {perception.get('location')}\n"
        f"Plan: {json.dumps(daily_plan[:5])}\n\n"
        f"Available tools:\n{tool_descriptions}\n\n"
        f"What information do you need? List 1-{max_steps} STEP lines, each naming a tool."
    )
    plan_resp = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": "You are a planning agent."},
            {"role": "user", "content": plan_prompt},
        ],
        temperature=0.3,
        max_tokens=256,
    )
    plan_text = plan_resp.choices[0].message.content.strip()

    # Phase 2: Execute
    tool_names = [t["name"] for t in SHARED_TOOLS]
    gathered = {}
    tool_call_count = 0
    for line in plan_text.split("\n"):
        if tool_call_count >= max_steps:
            break
        line_lower = line.lower()
        matched = next((tn for tn in tool_names if tn in line_lower), None)
        if matched:
            gathered[matched] = _execute_tool(matched, env_state, agent_name)
            tool_call_count += 1

    # Phase 3: Decide
    context_str = "\n".join(f"[{t}]: {d}" for t, d in gathered.items())
    decide_prompt = (
        f"You are {agent_name}, {agent_profile.get('occupation', 'resident')}.\n"
        f"Personality: {agent_profile.get('personality', 'friendly')}\n"
        f"Time: {perception.get('time')}, Location: {perception.get('location')}\n"
        f"Information:\n{context_str}\n\n"
        f"Available locations: {', '.join(location_names)}\n"
        f'Output ONLY a JSON: {{"action": "...", "location": "...", "thought": "...", "emoji": "..."}}'
    )
    dec_resp = client.chat.completions.create(
        model=config.model,
        messages=[{"role": "user", "content": decide_prompt}],
        temperature=config.temperature,
        max_tokens=256,
    )
    dec_text = dec_resp.choices[0].message.content.strip()
    latency = time.time() - t0

    try:
        si = dec_text.index("{")
        ei = dec_text.rindex("}") + 1
        decision = json.loads(dec_text[si:ei])
        decision["method"] = "plan_and_solve"
        decision["tool_calls"] = tool_call_count
        decision["latency"] = latency
        return decision
    except (ValueError, json.JSONDecodeError):
        return {
            "action": "idle",
            "location": perception.get("location", ""),
            "thought": "Plan-and-Solve could not decide",
            "emoji": "",
            "method": "plan_and_solve",
            "tool_calls": tool_call_count,
            "latency": latency,
        }


# =============================================================================
# Baseline 6: Generative Agents (Park et al., 2023)
#
# Reimplements the perceive → retrieve → plan → reflect loop from
# "Generative Agents: Interactive Simulacra of Human Behavior" (NeurIPS 2023).
#
# KEY DIFFERENCE from AURA: environment perception is a PASSIVE observation
# string — no proactive probing, no bounded tool calls. The agent sees only
# what is provided in its fixed observation window.
#
# Memory stream uses the same 3-factor retrieval as the original paper:
#   score(m, q, t) = α·recency(m,t) + β·importance(m) + γ·relevance(m,q)
# =============================================================================

class _MemoryStream:
    """Simplified memory stream following Park et al. (2023).

    Each memory has: content, timestamp, importance (1-10), access_count.
    Retrieval uses recency × importance × relevance (keyword overlap).
    Reflection is triggered every `reflection_threshold` new observations.
    """

    def __init__(
        self,
        max_items: int = 200,
        reflection_threshold: int = 10,
        alpha: float = 1.0,   # recency weight
        beta: float = 1.0,    # importance weight
        gamma: float = 1.0,   # relevance weight
        decay_rate: float = 0.995,
    ):
        self.memories: List[Dict[str, Any]] = []
        self.max_items = max_items
        self.reflection_threshold = reflection_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.decay_rate = decay_rate
        self._obs_since_reflection = 0

    def add(self, content: str, importance: float = 5.0, mem_type: str = "observation") -> None:
        self.memories.append({
            "content": content,
            "importance": importance / 10.0,
            "type": mem_type,
            "timestamp": len(self.memories),
            "access_count": 0,
        })
        if mem_type == "observation":
            self._obs_since_reflection += 1
        if len(self.memories) > self.max_items:
            self.memories = self.memories[-self.max_items:]

    def should_reflect(self) -> bool:
        return self._obs_since_reflection >= self.reflection_threshold

    def reset_reflection_counter(self) -> None:
        self._obs_since_reflection = 0

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        if not self.memories:
            return []
        now = len(self.memories)
        query_tokens = set(query.lower().split())
        scored: List[tuple] = []
        for m in self.memories:
            # Recency: exponential decay
            age = now - m["timestamp"]
            recency = self.decay_rate ** age
            # Importance: normalized 0-1
            importance = m["importance"]
            # Relevance: keyword overlap
            mem_tokens = set(m["content"].lower().split())
            overlap = len(query_tokens & mem_tokens) / max(len(query_tokens | mem_tokens), 1)
            score = self.alpha * recency + self.beta * importance + self.gamma * overlap
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m["content"] for _, m in scored[:k]]


def generative_agents_chat(
    config: EvalConfig,
    agent_name: str,
    query: str,
    env_state: Dict[str, Any],
    memory_stream: Optional[_MemoryStream] = None,
) -> Dict[str, Any]:
    """
    Generative Agents baseline (Park et al., 2023) for chat queries.

    Key design: the agent receives a PASSIVE observation string (no tool calls).
    It retrieves relevant memories and generates a response.
    NO proactive probing — pure perceive-retrieve-respond.
    """
    client = _get_client(config)
    if memory_stream is None:
        memory_stream = _MemoryStream()

    agent = next((a for a in env_state.get("agents", []) if a["name"] == agent_name), {})

    # Passive observation string (fixed format, like the original paper)
    observation = (
        f"{agent_name} is at {agent.get('location', 'unknown')}. "
        f"It is {env_state.get('time', 'unknown')}. "
        f"{agent_name} is currently {agent.get('action', 'idle')}."
    )
    nearby = [a for a in env_state.get("agents", [])
              if a.get("location") == agent.get("location") and a["name"] != agent_name]
    if nearby:
        others = ", ".join(f"{a['name']} ({a['action']})" for a in nearby)
        observation += f" Nearby: {others}."

    # Add observation to memory
    memory_stream.add(observation, importance=3.0, mem_type="observation")

    # Retrieve relevant memories
    retrieved = memory_stream.retrieve(query, k=5)

    # Build prompt (following Park et al. style)
    memory_block = "\n".join(f"- {m}" for m in retrieved) if retrieved else "(no memories)"

    messages = [
        {
            "role": "system",
            "content": (
                f"You are {agent_name}, a resident of a small town.\n\n"
                f"Current observation:\n{observation}\n\n"
                f"Relevant memories:\n{memory_block}\n\n"
                "Answer the user's question based on your observation and memories. "
                "If you don't know something, say so."
            ),
        },
        {"role": "user", "content": query},
    ]

    t0 = time.time()
    resp = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=512,
    )
    latency = time.time() - t0

    return {
        "response": resp.choices[0].message.content.strip(),
        "latency": latency,
        "method": "generative_agents",
        "tool_calls": 0,
        "observation": observation,
        "memories_retrieved": len(retrieved),
        "env_context": {"observation": observation},
    }


def generative_agents_action_decision(
    config: "EvalConfig",
    agent_name: str,
    agent_profile: dict,
    perception: dict,
    env_state: dict,
    daily_plan: list,
    location_names: list,
    memory_stream: Optional[_MemoryStream] = None,
) -> dict:
    """
    Generative Agents baseline (Park et al., 2023) for action decisions.

    Follows perceive → retrieve → plan → act. NO tool calls.
    The agent sees only its fixed observation string + retrieved memories.
    Reflection is triggered every 10 observations (as in original paper).
    """
    client = _get_client(config)
    if memory_stream is None:
        memory_stream = _MemoryStream()

    # Perceive: passive observation string
    observation = (
        f"{agent_name} is at {perception.get('location', 'unknown')}. "
        f"It is {perception.get('time', 'unknown')}. "
        f"{agent_name} is {perception.get('current_action', 'idle')}."
    )
    nearby_agents = [
        a for a in env_state.get("agents", [])
        if a.get("location") == perception.get("location") and a["name"] != agent_name
    ]
    if nearby_agents:
        others = ", ".join(f"{a['name']} ({a['action']})" for a in nearby_agents)
        observation += f" Nearby: {others}."

    memory_stream.add(observation, importance=3.0, mem_type="observation")

    # Reflect if threshold reached
    reflection_text = ""
    if memory_stream.should_reflect():
        recent = memory_stream.retrieve("recent events and observations", k=10)
        reflect_prompt = (
            f"You are {agent_name}. Based on your recent experiences, "
            "what are 1-3 high-level insights or reflections?\n\n"
            "Recent experiences:\n" + "\n".join(f"- {m}" for m in recent) +
            "\n\nReflections (be concise):"
        )
        try:
            reflect_resp = client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": reflect_prompt}],
                temperature=0.5,
                max_tokens=256,
            )
            reflection_text = reflect_resp.choices[0].message.content.strip()
            memory_stream.add(reflection_text, importance=8.0, mem_type="reflection")
            memory_stream.reset_reflection_counter()
        except Exception:
            pass

    # Retrieve: get relevant memories for decision-making
    context_query = f"{agent_name} deciding what to do at {perception.get('time', '')}"
    retrieved = memory_stream.retrieve(context_query, k=5)
    memory_block = "\n".join(f"- {m}" for m in retrieved) if retrieved else "(no memories)"

    # Plan + Act: decide next action
    system_prompt = (
        f"You are simulating {agent_name}, a {agent_profile.get('age', 30)}-year-old "
        f"{agent_profile.get('occupation', 'resident')}.\n"
        f"Personality: {agent_profile.get('personality', 'friendly')}\n\n"
        f"Current observation:\n{observation}\n\n"
        f"Relevant memories:\n{memory_block}\n\n"
        f"Today's plan: {json.dumps(daily_plan[:5])}\n\n"
        f"Available locations: {', '.join(location_names)}\n\n"
        f"Based on your observation, memories, and plan, decide what to do next.\n"
        f'Output ONLY a JSON: {{"action": "...", "location": "...", "thought": "...", "emoji": "..."}}'
    )

    t0 = time.time()
    resp = client.chat.completions.create(
        model=config.model,
        messages=[{"role": "user", "content": system_prompt}],
        temperature=config.temperature,
        max_tokens=256,
    )
    content_text = resp.choices[0].message.content.strip()
    latency = time.time() - t0

    try:
        si = content_text.index("{")
        ei = content_text.rindex("}") + 1
        decision = json.loads(content_text[si:ei])
        decision["method"] = "generative_agents"
        decision["tool_calls"] = 0
        decision["latency"] = latency
        decision["memories_retrieved"] = len(retrieved)
        decision["reflected"] = bool(reflection_text)
        return decision
    except (ValueError, json.JSONDecodeError):
        return {
            "action": "idle",
            "location": perception.get("location", ""),
            "thought": "Could not decide",
            "emoji": "",
            "method": "generative_agents",
            "tool_calls": 0,
            "latency": latency,
            "memories_retrieved": len(retrieved),
            "reflected": bool(reflection_text),
        }


# =============================================================================
# Baseline 7: ContextAgent-style (Yang et al., NeurIPS 2025)
#
# Adapted from "ContextAgent: Context-Aware Proactive LLM Agents with
# Open-World Sensory Perceptions". The key idea: extract multi-dimensional
# context (sensory + persona) from the environment, use LLM to predict
# whether proactive assistance is needed, then respond.
#
# KEY DIFFERENCE from AURA:
# - Single-pass inference (no bounded probing loop)
# - No feedback adaptation (no attention tracker / collaborative paradigm)
# - Context is extracted once, not iteratively refined
# =============================================================================

def _extract_multidim_context(
    env_state: Dict[str, Any],
    agent_name: str,
    agent_profile: Optional[dict] = None,
) -> Dict[str, Any]:
    """Extract multi-dimensional context following ContextAgent's approach.

    Dimensions:
    1. Spatial: location, nearby entities
    2. Temporal: time, schedule alignment
    3. Social: nearby agents, relationship context
    4. Activity: current action, recent events
    5. Persona: agent profile, preferences (if available)
    """
    agent = next((a for a in env_state.get("agents", []) if a["name"] == agent_name), {})
    nearby = [
        a for a in env_state.get("agents", [])
        if a.get("location") == agent.get("location") and a["name"] != agent_name
    ]
    recent_events = env_state.get("events", [])[-5:]

    context = {
        "spatial": {
            "location": agent.get("location", "unknown"),
            "nearby_count": len(nearby),
        },
        "temporal": {
            "time": env_state.get("time", "unknown"),
            "hour": env_state.get("hour", 0),
        },
        "social": {
            "nearby_agents": [
                {"name": a["name"], "action": a["action"]}
                for a in nearby
            ],
        },
        "activity": {
            "current_action": agent.get("action", "idle"),
            "recent_events": [
                e.get("description", str(e))
                for e in recent_events if isinstance(e, dict)
            ],
        },
        "persona": {},
    }
    if agent_profile:
        context["persona"] = {
            "occupation": agent_profile.get("occupation", ""),
            "personality": agent_profile.get("personality", ""),
        }
    return context


def contextagent_chat(
    config: EvalConfig,
    agent_name: str,
    query: str,
    env_state: Dict[str, Any],
    agent_profile: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    ContextAgent-style baseline (Yang et al., NeurIPS 2025) for chat.

    Single-pass: extract multi-dimensional context → LLM inference.
    No probing loop, no feedback adaptation.
    """
    client = _get_client(config)
    context = _extract_multidim_context(env_state, agent_name, agent_profile)

    context_block = (
        f"Spatial: {agent_name} is at {context['spatial']['location']}, "
        f"{context['spatial']['nearby_count']} others nearby.\n"
        f"Temporal: {context['temporal']['time']}\n"
        f"Social: {json.dumps(context['social']['nearby_agents'])}\n"
        f"Activity: {context['activity']['current_action']}\n"
        f"Recent events: {'; '.join(context['activity']['recent_events'][:3])}\n"
    )
    if context["persona"]:
        context_block += (
            f"Persona: {context['persona'].get('occupation', '')}, "
            f"{context['persona'].get('personality', '')}\n"
        )

    messages = [
        {
            "role": "system",
            "content": (
                f"You are an AI assistant for {agent_name} in a small town.\n\n"
                f"Multi-dimensional context:\n{context_block}\n"
                "Use ALL dimensions of context to provide an accurate, "
                "well-grounded response. If information is missing, acknowledge it."
            ),
        },
        {"role": "user", "content": query},
    ]

    t0 = time.time()
    resp = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=512,
    )
    latency = time.time() - t0

    return {
        "response": resp.choices[0].message.content.strip(),
        "latency": latency,
        "method": "contextagent",
        "tool_calls": 0,
        "context_dimensions": list(context.keys()),
        "env_context": context,
    }


def contextagent_action_decision(
    config: "EvalConfig",
    agent_name: str,
    agent_profile: dict,
    perception: dict,
    env_state: dict,
    daily_plan: list,
    location_names: list,
) -> dict:
    """
    ContextAgent-style action decision (NeurIPS 2025 adaptation).

    Multi-dimensional context extraction → proactive need prediction → action.
    No tool calls, no iterative probing.
    """
    client = _get_client(config)
    context = _extract_multidim_context(env_state, agent_name, agent_profile)

    # Step 1: Proactive need prediction (ContextAgent's key innovation)
    # In the original paper this is a classifier; we use LLM for fair comparison
    need_prompt = (
        f"Given this context about {agent_name}:\n"
        f"- Location: {context['spatial']['location']}\n"
        f"- Time: {context['temporal']['time']}\n"
        f"- Nearby: {json.dumps(context['social']['nearby_agents'])}\n"
        f"- Current: {context['activity']['current_action']}\n"
        f"- Recent: {'; '.join(context['activity']['recent_events'][:3])}\n"
        f"- Plan: {json.dumps(daily_plan[:5])}\n\n"
        f"Does {agent_name} need to change their current activity? "
        f"Consider social opportunities, schedule alignment, and environmental cues.\n"
        f"Reply YES or NO, then explain briefly."
    )

    # Step 2: Make decision with full context
    system_prompt = (
        f"You are simulating {agent_name}, a {agent_profile.get('age', 30)}-year-old "
        f"{agent_profile.get('occupation', 'resident')}.\n"
        f"Personality: {agent_profile.get('personality', 'friendly')}\n\n"
        f"Multi-dimensional context:\n"
        f"- Spatial: at {context['spatial']['location']}, {context['spatial']['nearby_count']} nearby\n"
        f"- Temporal: {context['temporal']['time']}\n"
        f"- Social: {json.dumps(context['social']['nearby_agents'])}\n"
        f"- Activity: {context['activity']['current_action']}\n"
        f"- Events: {'; '.join(context['activity']['recent_events'][:3])}\n"
        f"- Plan: {json.dumps(daily_plan[:5])}\n\n"
        f"Available locations: {', '.join(location_names)}\n\n"
        f"Based on all context dimensions, decide what to do next.\n"
        f'Output ONLY JSON: {{"action": "...", "location": "...", "thought": "...", "emoji": "..."}}'
    )

    t0 = time.time()
    resp = client.chat.completions.create(
        model=config.model,
        messages=[{"role": "user", "content": system_prompt}],
        temperature=config.temperature,
        max_tokens=256,
    )
    content_text = resp.choices[0].message.content.strip()
    latency = time.time() - t0

    try:
        si = content_text.index("{")
        ei = content_text.rindex("}") + 1
        decision = json.loads(content_text[si:ei])
        decision["method"] = "contextagent"
        decision["tool_calls"] = 0
        decision["latency"] = latency
        decision["context_dimensions"] = 5
        return decision
    except (ValueError, json.JSONDecodeError):
        return {
            "action": "idle",
            "location": perception.get("location", ""),
            "thought": "ContextAgent could not decide",
            "emoji": "",
            "method": "contextagent",
            "tool_calls": 0,
            "latency": latency,
            "context_dimensions": 5,
        }
