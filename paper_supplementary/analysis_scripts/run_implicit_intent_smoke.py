"""Day-8 smoke: 5 implicit-intent queries x 3 conditions x 1 seed (real LLM).

Validates the ToM retrofit end-to-end by driving three answer strategies
against the same hand-built scene and a minimal probe registry. We do not
use AURAAgent.run() here because the default-backend stubs short-circuit
Reasoner/Interactor to deterministic placeholders; this smoke directly
exercises the LLM + AURA's IntentInferrer + a scripted probe loop, which
is what the Day 9 ablation will do for real.

Conditions:
  - literal   : single LLM call with scene only, no tools.
  - no_intent : LLM call may issue ReAct-style tool calls up to fixed B=2.
  - tom       : IntentInferrer maps query -> gap; a high gap triggers
                private-state probes before the answering LLM call.

Output: evaluation/results/intent_smoke.json. Budget: ~$0.30 total on
gpt-4o-mini (5 * 3 runs x roughly 2-4 calls each).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

_AURA_SRC = Path(__file__).resolve().parent.parent / "AURA" / "src"
if str(_AURA_SRC) not in sys.path:
    sys.path.insert(0, str(_AURA_SRC))

from openai import OpenAI

from aura.intent import LLMIntentInferrer
from aura.types import MemoryItem, SceneState


# ---------------------------------------------------------------------------
# Hand-built scene (public info only)
# ---------------------------------------------------------------------------

SCENE_SUMMARY = (
    "Sunrise Cafe, 10:15 AM. Present: Lin Wei (cafe owner, behind the counter), "
    "Zhang Hao (sipping coffee, reading), Chen Mei (at a corner table, writing). "
    "Wang Jun is reportedly at the library today."
)

PUBLIC_STATE: Dict[str, Dict[str, Any]] = {
    "Lin Wei":   {"location": "Sunrise Cafe", "action": "serving customers"},
    "Zhang Hao": {"location": "Sunrise Cafe", "action": "reading a novel"},
    "Chen Mei":  {"location": "Sunrise Cafe", "action": "writing in a notebook"},
    "Wang Jun":  {"location": "Town Library", "action": "studying"},
}

PRIVATE_STATE: Dict[str, Dict[str, Any]] = {
    "Lin Wei":   {"availability": "busy",            "emotional_state": "cheerful", "unspoken_goal": "hosting a small poetry reading tonight"},
    "Zhang Hao": {"availability": "available",       "emotional_state": "relaxed",  "unspoken_goal": "wants to invite Chen Mei to a book discussion"},
    "Chen Mei":  {"availability": "do_not_disturb", "emotional_state": "focused", "unspoken_goal": "drafting a difficult letter"},
    "Wang Jun":  {"availability": "busy",            "emotional_state": "stressed", "unspoken_goal": "preparing for a grant deadline"},
}

# Second-order ToM: each agent's beliefs about the others' private state.
# These beliefs are deliberately not always correct -- Lin Wei can THINK
# Zhang Hao is busy when he's actually relaxed, which is what makes
# second-order ToM non-trivial. Ground-truth here is the BELIEF, not
# the target's actual state.
BELIEFS_ABOUT_OTHERS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Lin Wei": {
        "Zhang Hao": {"availability": "available",       "emotional_state": "relaxed"},
        "Chen Mei":  {"availability": "do_not_disturb", "emotional_state": "focused"},
        "Wang Jun":  {"availability": "unknown",         "emotional_state": "unknown"},
    },
    "Zhang Hao": {
        "Lin Wei":  {"availability": "busy",      "emotional_state": "cheerful"},
        # Zhang Hao MISREADS Chen Mei as approachable, which is wrong vs ground truth:
        "Chen Mei": {"availability": "available", "emotional_state": "thoughtful"},
        "Wang Jun": {"availability": "busy",      "emotional_state": "unknown"},
    },
    "Chen Mei": {
        # Chen Mei OVERESTIMATES Lin Wei's busyness — ground truth busy, belief do_not_disturb:
        "Lin Wei":   {"availability": "do_not_disturb", "emotional_state": "stressed"},
        "Zhang Hao": {"availability": "available",      "emotional_state": "relaxed"},
        "Wang Jun":  {"availability": "unknown",        "emotional_state": "unknown"},
    },
    "Wang Jun": {
        # Wang Jun has been at the library all day and isn't up to date:
        "Lin Wei":   {"availability": "available", "emotional_state": "neutral"},
        "Zhang Hao": {"availability": "unknown",   "emotional_state": "unknown"},
        "Chen Mei":  {"availability": "unknown",   "emotional_state": "unknown"},
    },
}


# ---------------------------------------------------------------------------
# Probe tools (python-side; each returns a JSON string)
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS = [
    "get_all_agents",
    "get_nearby_agents",
    "get_agent_plan",
    "get_agent_private_state",
    "get_agent_belief_about",
]


def tool_exec(name: str, args: Dict[str, Any]) -> str:
    if name == "get_all_agents":
        return json.dumps([{"name": n, **PUBLIC_STATE[n]} for n in PUBLIC_STATE])
    if name == "get_nearby_agents":
        loc = args.get("location", "Sunrise Cafe")
        return json.dumps([{"name": n, **v} for n, v in PUBLIC_STATE.items() if v["location"] == loc])
    if name == "get_agent_plan":
        plans = {
            "Lin Wei":   ["serve morning customers", "prep for 19:00 poetry reading"],
            "Zhang Hao": ["finish novel chapter 7", "attend poetry reading at 19:00"],
            "Chen Mei":  ["finish letter", "deliver letter before 17:00"],
            "Wang Jun":  ["library until 18:00", "dinner break 12:30"],
        }
        n = args.get("agent_name", "")
        return json.dumps({"agent_name": n, "plan": plans.get(n, [])})
    if name == "get_agent_private_state":
        n = args.get("agent_name", "")
        data = PRIVATE_STATE.get(n)
        return json.dumps(data or {"error": f"unknown agent {n!r}"})
    if name == "get_agent_belief_about":
        believer = args.get("believer", "")
        target = args.get("target", "")
        data = BELIEFS_ABOUT_OTHERS.get(believer, {}).get(target)
        return json.dumps(
            data or {"error": f"{believer!r} has no belief recorded about {target!r}"}
        )
    return json.dumps({"error": f"unknown tool {name!r}"})


TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_all_agents",
            "description": "List every agent with their public location and action.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_nearby_agents",
            "description": "List agents at a given location.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_agent_plan",
            "description": "Daily plan for a named agent.",
            "parameters": {
                "type": "object",
                "properties": {"agent_name": {"type": "string"}},
                "required": ["agent_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_agent_private_state",
            "description": "Probe an agent's PRIVATE state (availability, emotional_state, unspoken_goal). This information is NOT in the public scene.",
            "parameters": {
                "type": "object",
                "properties": {"agent_name": {"type": "string"}},
                "required": ["agent_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_agent_belief_about",
            "description": "Second-order ToM probe: ask what `believer` BELIEVES about `target`'s private state. The belief may differ from the target's actual state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "believer": {"type": "string"},
                    "target": {"type": "string"},
                },
                "required": ["believer", "target"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Per-condition answer strategies
# ---------------------------------------------------------------------------

# Module-level backbone model. Override via set_backbone_model() before
# invoking the answer functions. Kept as a module-level handle (not an arg)
# so existing callers do not break.
BACKBONE_MODEL = "gpt-4o-mini"


def set_backbone_model(name: str) -> None:
    global BACKBONE_MODEL
    BACKBONE_MODEL = name


def set_scene(
    summary: str,
    public_state: Dict[str, Dict[str, Any]],
    private_state: Dict[str, Dict[str, Any]],
    beliefs_about_others: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    """Swap the module-level scene state used by tool_exec / answer fns.

    Lets multi-scene runners (e.g. run_implicit_intent_v2.py) cycle through
    different (SCENE_SUMMARY, PUBLIC_STATE, PRIVATE_STATE, BELIEFS_ABOUT_OTHERS)
    blocks per query without forking the answer-strategy logic. Existing
    single-scene runners that never call this keep the original Sunrise Cafe
    defaults at import time.
    """
    global SCENE_SUMMARY, PUBLIC_STATE, PRIVATE_STATE, BELIEFS_ABOUT_OTHERS
    SCENE_SUMMARY = summary
    PUBLIC_STATE = public_state
    PRIVATE_STATE = private_state
    BELIEFS_ABOUT_OTHERS = beliefs_about_others


def _literal_answer(client: OpenAI, query: str) -> Tuple[str, int, float]:
    t0 = time.time()
    resp = client.chat.completions.create(
        model=BACKBONE_MODEL,
        messages=[
            {"role": "system", "content": f"You are a helpful assistant. The current environment is: {SCENE_SUMMARY}. Answer the user's question using ONLY this scene description. Do not make up information not present."},
            {"role": "user", "content": query},
        ],
        temperature=0.1, max_tokens=300,
    )
    return resp.choices[0].message.content or "", 0, time.time() - t0


def _react_answer(client: OpenAI, query: str, budget: int) -> Tuple[str, int, float]:
    """No-intent condition: let the LLM decide tool use up to `budget` steps.

    Invariant: every tool_call in an assistant message MUST receive a
    corresponding tool-role response before the next API call; otherwise
    OpenAI rejects with 400 'tool_call_ids did not have response messages'.
    We therefore process each batch atomically (all tool_calls in the
    batch either fully execute or fully record a synthetic 'budget
    exhausted' response), and only THEN check whether the budget is hit.
    """
    t0 = time.time()
    messages = [
        {"role": "system", "content": (
            f"Scene: {SCENE_SUMMARY}\n\n"
            f"You can call tools to gather additional information. Max {budget} tool calls. "
            "Answer the user's question."
        )},
        {"role": "user", "content": query},
    ]
    calls = 0
    for _ in range(budget + 1):
        resp = client.chat.completions.create(
            model=BACKBONE_MODEL,
            messages=messages,
            tools=TOOL_SCHEMA, tool_choice="auto",
            temperature=0.1, max_tokens=400,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            messages.append(msg)
            # Atomic batch: respond to EVERY tool_call_id even if over budget.
            for tc in msg.tool_calls:
                if calls < budget:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}
                    out = tool_exec(tc.function.name, args)
                    calls += 1
                else:
                    out = json.dumps({"error": "budget exhausted; no further probes"})
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": out[:2000],
                })
            if calls >= budget:
                # Force the model to answer now
                messages.append({"role": "user", "content": "Based on what you have, answer now."})
                final = client.chat.completions.create(
                    model=BACKBONE_MODEL, messages=messages,
                    temperature=0.1, max_tokens=400,
                )
                return final.choices[0].message.content or "", calls, time.time() - t0
            continue
        # Model produced an answer
        return msg.content or "", calls, time.time() - t0
    return "", calls, time.time() - t0


def _tom_answer(client: OpenAI, query: str, configured_budget: int) -> Tuple[str, int, float, Optional[float]]:
    """ToM condition: IntentInferrer -> dynamic budget -> directed probing -> answer."""
    t0 = time.time()
    inferrer = LLMIntentInferrer(client=client, model=BACKBONE_MODEL)
    scene = SceneState(summary=SCENE_SUMMARY, entities=list(PUBLIC_STATE.keys()))
    mems = [MemoryItem(content="earlier: user mentioned wanting to socialize today")]
    frame = inferrer.infer(query, scene, mems, available_tools=AVAILABLE_TOOLS)

    # Map gap to dynamic budget: low gap skips tools entirely; high gap
    # opens the full configured budget. Uses the same intent_gap_to_budget
    # mapping as core.py (inlined here to avoid the full AURAAgent wiring
    # for the Day 9 ablation).
    g = frame.gap or 0.0
    if g < 0.20:   dyn_budget = 0
    elif g < 0.40: dyn_budget = 1
    elif g < 0.60: dyn_budget = 2
    elif g < 0.80: dyn_budget = 3
    else:           dyn_budget = 5
    dyn_budget = min(configured_budget, dyn_budget)

    # If intent says "literal suffices", fall back to the literal branch.
    if dyn_budget == 0:
        ans, _, _ = _literal_answer(client, query)
        # Still prepend heads-up if alert flag is set
        if frame.should_alert and frame.implicit_need:
            ans = f"[heads-up] You may also be wondering: {frame.implicit_need[0]}\n\n{ans}"
        return ans, 0, time.time() - t0, g

    # Otherwise run a directed probe loop that prefers the recommended_probes.
    preferred = set(frame.recommended_probes or [])
    system_hint = (
        f"Scene: {SCENE_SUMMARY}\n\n"
        f"The user's query has implicit need: {frame.implicit_need}. "
        f"You may call up to {dyn_budget} tools. Prefer these first if relevant: "
        f"{sorted(preferred) if preferred else 'any available'}. "
        "Then answer both the literal and the implicit need."
    )
    messages = [
        {"role": "system", "content": system_hint},
        {"role": "user", "content": query},
    ]
    calls = 0
    for _ in range(dyn_budget + 1):
        resp = client.chat.completions.create(
            model=BACKBONE_MODEL, messages=messages,
            tools=TOOL_SCHEMA, tool_choice="auto",
            temperature=0.1, max_tokens=400,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            messages.append(msg)
            # Atomic batch (same invariant as _react_answer): every
            # tool_call_id must receive a tool-role response or OpenAI
            # rejects the follow-up call with 400.
            for tc in msg.tool_calls:
                if calls < dyn_budget:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}
                    out = tool_exec(tc.function.name, args)
                    calls += 1
                else:
                    out = json.dumps({"error": "budget exhausted; no further probes"})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out[:2000]})
            if calls >= dyn_budget:
                messages.append({"role": "user", "content": "Based on what you have, answer now, covering BOTH the literal and the implicit need."})
                final = client.chat.completions.create(
                    model=BACKBONE_MODEL, messages=messages,
                    temperature=0.1, max_tokens=400,
                )
                ans = final.choices[0].message.content or ""
                if frame.should_alert and frame.implicit_need:
                    ans = f"[heads-up] You may also be wondering: {frame.implicit_need[0]}\n\n{ans}"
                return ans, calls, time.time() - t0, g
            continue
        ans = msg.content or ""
        if frame.should_alert and frame.implicit_need:
            ans = f"[heads-up] You may also be wondering: {frame.implicit_need[0]}\n\n{ans}"
        return ans, calls, time.time() - t0, g
    return "", calls, time.time() - t0, g


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

SMOKE_QUERY_IDS = [101, 103, 108, 111, 116]
CONFIGURED_BUDGET = 3


def _load_queries(ids: List[int]) -> List[Dict[str, Any]]:
    src = Path(__file__).resolve().parent.parent / "evaluation" / "data" / "implicit_intent_queries.json"
    with open(src) as f:
        all_q = json.load(f)["queries"]
    by_id = {q["id"]: q for q in all_q}
    return [by_id[i] for i in ids if i in by_id]


def main() -> int:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERR: OPENAI_API_KEY not set", file=sys.stderr)
        return 2
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    queries = _load_queries(SMOKE_QUERY_IDS)
    conditions = ["literal", "no_intent", "tom"]
    print(f"Running {len(queries)} queries x {len(conditions)} conditions = {len(queries) * len(conditions)} runs\n")

    results: Dict[str, Any] = {
        "meta": {"scene": SCENE_SUMMARY, "query_ids": SMOKE_QUERY_IDS, "configured_budget": CONFIGURED_BUDGET},
        "per_query": [],
    }

    for q in queries:
        row: Dict[str, Any] = {
            "id": q["id"], "subcategory": q.get("subcategory"),
            "query": q["query"], "implicit_need": q.get("implicit_need"),
            "by_condition": {},
        }
        print(f"----- id={q['id']} ({q.get('subcategory')}) -----")
        print(f"Q: {q['query']}")
        print(f"implicit_need: {q.get('implicit_need')}")

        # literal
        ans, calls, dur = _literal_answer(client, q["query"])
        row["by_condition"]["literal"] = {"answer": ans, "probes": calls, "latency": round(dur, 2), "gap": None}
        print(f"  [literal   ] {dur:.2f}s probes={calls}")
        print(f"                -> {ans[:180]}")

        # no_intent
        ans, calls, dur = _react_answer(client, q["query"], CONFIGURED_BUDGET)
        row["by_condition"]["no_intent"] = {"answer": ans, "probes": calls, "latency": round(dur, 2), "gap": None}
        print(f"  [no_intent ] {dur:.2f}s probes={calls}")
        print(f"                -> {ans[:180]}")

        # tom
        ans, calls, dur, gap = _tom_answer(client, q["query"], CONFIGURED_BUDGET)
        row["by_condition"]["tom"] = {"answer": ans, "probes": calls, "latency": round(dur, 2), "gap": gap}
        print(f"  [tom       ] {dur:.2f}s probes={calls} gap={gap}")
        print(f"                -> {ans[:180]}")
        print()

        results["per_query"].append(row)

    out = Path(__file__).resolve().parent.parent / "evaluation" / "results" / "intent_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[write] {out}")

    # Aggregate
    print("\n=== Aggregate (probes per condition) ===")
    for cond in conditions:
        total = sum(r["by_condition"][cond]["probes"] for r in results["per_query"])
        avg_lat = sum(r["by_condition"][cond]["latency"] for r in results["per_query"]) / len(results["per_query"])
        print(f"  {cond:<10}: probes {total} total, avg {total / len(results['per_query']):.2f}   latency avg {avg_lat:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
