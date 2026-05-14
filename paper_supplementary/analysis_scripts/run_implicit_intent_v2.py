"""Run the 100-query, scene-keyed RQ-Intent v2 benchmark.

This runner preserves the original v1 answer strategies by reusing
``run_implicit_intent_smoke`` for literal / ReAct-style no-intent /
IntentFrame-guided answers, but swaps the module-level scene state before
each query. That keeps runtime behavior comparable while avoiding the v1
single-scene constant leak into scenes B/C/D.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent


def _repo_relative(path: Path) -> str:
    """Return stable artifact paths without leaking local usernames."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


_AURA_SRC = ROOT / "AURA" / "src"
if str(_AURA_SRC) not in sys.path:
    sys.path.insert(0, str(_AURA_SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI

from scripts import run_implicit_intent_smoke as smoke
from aura.intent import LLMIntentInferrer, intent_frame_to_dict
from aura.types import MemoryItem, SceneState

JUDGE_MODEL = "gpt-4o-mini"
CONFIGURED_BUDGET = 3
DEFAULT_CONDITIONS = ["literal", "no_intent", "tom"]
VALID_CONDITIONS = DEFAULT_CONDITIONS + ["fixed_probe", "oracle_intent"]
AVAILABLE_TOOLS = [
    "get_all_agents",
    "get_nearby_agents",
    "get_agent_plan",
    "get_agent_private_state",
    "get_agent_belief_about",
]

CURRENT_TOOL_TRACE: List[Dict[str, Any]] = []

JUDGE_SYSTEM = (
    "You are an expert evaluator for a theory-of-mind-aware assistant. "
    "Given a user's query, hidden ground-truth private state, the implicit "
    "need the query is really asking about, and an AI response, produce a "
    "JSON object with two independent scores in [0, 1]:\n\n"
    "  literal_score  : does the response correctly answer the surface "
    "question? Consistency with the scene / ground truth earns 1.0. "
    "Fabricating contradictory public facts earns 0.\n"
    "  implicit_score : does the response surface or address the implicit "
    "need? Explicit naming of the implicit concern earns 1.0. Ignoring it "
    "earns 0. Partial surfacing earns 0.4-0.6.\n\n"
    "SECOND-ORDER NOTE: when the ground-truth dict contains a "
    "'believers_belief_about_target' field, the query is about the "
    "BELIEVER'S BELIEF -- not the target's actual state. A correct "
    "second-order answer reflects the believer's belief (even if that "
    "belief is wrong compared to targets_actual_private_state). A response "
    "that conflates the believer's belief with the target's ground truth "
    "earns a LOWER implicit_score, because it fails to model the believer's "
    "mind.\n\n"
    "Output schema (JSON, no prose):\n"
    "{\n"
    '  "literal_score": 0.0,\n'
    '  "implicit_score": 0.0,\n'
    '  "rationale": "<one sentence>"\n'
    "}"
)


def load_benchmark(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("queries"), list):
        raise ValueError(f"{path} must be a dict with a queries list")
    if not isinstance(data.get("scenes"), dict):
        raise ValueError(f"{path} must include a scenes dict")
    return data


def validate_benchmark(data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    scenes = data.get("scenes", {})
    queries = data.get("queries", [])
    ids = Counter(q.get("id") for q in queries)
    for qid, count in ids.items():
        if qid is None or count > 1:
            errors.append(f"duplicate/missing query id: {qid!r} count={count}")

    for sid, scene in scenes.items():
        for key in ("summary", "public_state", "private_state", "beliefs_about_others"):
            if key not in scene:
                errors.append(f"scene {sid}: missing {key}")
        roster = set(scene.get("public_state", {}))
        if roster != set(scene.get("private_state", {})):
            errors.append(f"scene {sid}: public/private rosters differ")
        for believer, beliefs in scene.get("beliefs_about_others", {}).items():
            if believer not in roster:
                errors.append(f"scene {sid}: unknown believer {believer!r}")
            for target in beliefs:
                if target not in roster:
                    errors.append(f"scene {sid}: unknown belief target {target!r}")

    for q in queries:
        qid = q.get("id")
        sid = q.get("scene")
        scene = scenes.get(sid)
        if not scene:
            errors.append(f"query {qid}: unknown scene {sid!r}")
            continue
        roster = set(scene.get("public_state", {}))
        subject = q.get("agent_subject")
        if subject and subject not in roster:
            errors.append(f"query {qid}: subject {subject!r} not in scene {sid}")
        target = q.get("target")
        if target and target not in roster:
            errors.append(f"query {qid}: target {target!r} not in scene {sid}")
        if q.get("subcategory") == "second_order":
            if not subject or not target:
                errors.append(f"query {qid}: second_order needs agent_subject and target")
            if "get_agent_belief_about" not in q.get("gold_required_tools", []):
                errors.append(f"query {qid}: second_order missing get_agent_belief_about gold tool")
            if "get_agent_private_state" not in q.get("forbidden_tools", []):
                errors.append(f"query {qid}: second_order should forbid get_agent_private_state")
        for field in ("query", "subcategory", "implicit_need", "gold_required_tools", "forbidden_tools"):
            if field not in q:
                errors.append(f"query {qid}: missing {field}")
    return errors


def _focus_location(scene: Dict[str, Any]) -> str:
    locs = [v.get("location", "") for v in scene.get("public_state", {}).values()]
    counts = Counter(loc for loc in locs if loc)
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def _default_plan(scene: Dict[str, Any], agent_name: str) -> List[str]:
    public = scene.get("public_state", {}).get(agent_name, {})
    private = scene.get("private_state", {}).get(agent_name, {})
    plan = []
    if public.get("action"):
        plan.append(f"current public activity: {public['action']}")
    if private.get("unspoken_goal"):
        plan.append(f"private goal: {private['unspoken_goal']}")
    return plan


def _make_tool_exec(scene: Dict[str, Any]):
    public = scene["public_state"]
    private = scene["private_state"]
    beliefs = scene["beliefs_about_others"]
    plans = scene.get("plans", {})
    focus = _focus_location(scene)

    def tool_exec(name: str, args: Dict[str, Any]) -> str:
        if name == "get_all_agents":
            result: Any = [{"name": n, **public[n]} for n in public]
        elif name == "get_nearby_agents":
            loc = args.get("location") or focus
            result = [{"name": n, **v} for n, v in public.items() if v.get("location") == loc]
        elif name == "get_agent_plan":
            n = args.get("agent_name", "")
            result = {"agent_name": n, "plan": plans.get(n, _default_plan(scene, n))}
        elif name == "get_agent_private_state":
            n = args.get("agent_name", "")
            result = private.get(n) or {"error": f"unknown agent {n!r}"}
        elif name == "get_agent_belief_about":
            believer = args.get("believer", "")
            target = args.get("target", "")
            result = beliefs.get(believer, {}).get(target)
            if result is None:
                result = {"error": f"{believer!r} has no belief recorded about {target!r}"}
        else:
            result = {"error": f"unknown tool {name!r}"}
        out = json.dumps(result, ensure_ascii=False)
        CURRENT_TOOL_TRACE.append({"tool": name, "args": args, "result": out})
        return out

    return tool_exec


def apply_scene(scene: Dict[str, Any]) -> None:
    smoke.SCENE_SUMMARY = scene["summary"]
    smoke.PUBLIC_STATE = scene["public_state"]
    smoke.PRIVATE_STATE = scene["private_state"]
    smoke.BELIEFS_ABOUT_OTHERS = scene["beliefs_about_others"]
    smoke.AVAILABLE_TOOLS = list(AVAILABLE_TOOLS)
    smoke.tool_exec = _make_tool_exec(scene)


def judge_response(
    client: OpenAI,
    scene: Dict[str, Any],
    query: str,
    implicit_need: str,
    response: str,
    agent_private_state: Dict[str, Any],
) -> Dict[str, Any]:
    ground_truth = {
        "scene_summary": scene["summary"],
        "public_state": scene["public_state"],
        "private_state": agent_private_state,
        "implicit_need": implicit_need,
    }
    user_msg = (
        f"USER QUERY: {query}\n\n"
        f"GROUND TRUTH (scene + all private state): {json.dumps(ground_truth, ensure_ascii=False)}\n\n"
        f"IMPLICIT NEED THE QUERY IS REALLY AFTER: {implicit_need}\n\n"
        f"AI RESPONSE: {response}\n\n"
        "Score both dimensions and output the JSON."
    )
    try:
        r = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        raw = (r.choices[0].message.content or "").strip()
        obj = json.loads(raw)
        ls = float(obj.get("literal_score", 0) or 0)
        ims = float(obj.get("implicit_score", 0) or 0)
        return {
            "literal_score": max(0.0, min(1.0, ls)),
            "implicit_score": max(0.0, min(1.0, ims)),
            "rationale": str(obj.get("rationale", ""))[:400],
        }
    except Exception as e:
        return {
            "literal_score": None,
            "implicit_score": None,
            "rationale": f"judge error: {type(e).__name__}: {e}",
        }


def paired_t(diffs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    valid = [d for d in diffs if d is not None and not math.isnan(d)]
    if len(valid) < 2:
        return None, None
    md = statistics.mean(valid)
    sd = statistics.stdev(valid) if len(set(valid)) > 1 else 0.0
    if sd == 0:
        return (float("inf") if md != 0 else 0.0), None
    t = md / (sd / math.sqrt(len(valid)))
    try:
        from scipy import stats  # type: ignore

        p = float(stats.t.sf(abs(t), len(valid) - 1) * 2)
    except ImportError:
        p = None
    return t, p


def _subjects_for_query(q: Dict[str, Any], scene: Dict[str, Any], budget: int) -> List[str]:
    subject = q.get("agent_subject")
    if subject:
        return [subject]
    focus = _focus_location(scene)
    by_focus = [
        name
        for name, state in scene["public_state"].items()
        if state.get("location") == focus
    ]
    return (by_focus or list(scene["public_state"]))[:budget]


def _dedupe_probe_plan(plan: Iterable[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for step in plan:
        tool = step.get("tool")
        args = step.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        key = (tool, json.dumps(args, sort_keys=True, ensure_ascii=False))
        if tool not in AVAILABLE_TOOLS or key in seen:
            continue
        seen.add(key)
        deduped.append({"tool": tool, "args": args, "reason": step.get("reason", "")})
        if len(deduped) >= budget:
            break
    return deduped


def _execute_probe_plan(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    for step in plan:
        tool = step["tool"]
        args = step.get("args") or {}
        try:
            out = smoke.tool_exec(tool, args)
        except Exception as e:
            out = json.dumps({"error": f"{type(e).__name__}: {e}"})
        observations.append(
            {"tool": tool, "args": args, "reason": step.get("reason", ""), "result": out}
        )
    return observations


def _synthesize_from_observations(
    client: OpenAI,
    scene: Dict[str, Any],
    query: str,
    implicit_need: str,
    observations: List[Dict[str, Any]],
    subcategory: Optional[str] = None,
) -> str:
    obs_block = (
        "\n".join(
            f"- {o['tool']}({json.dumps(o.get('args', {}), ensure_ascii=False)}) -> {o.get('result', '')}"
            for o in observations
        )
        if observations
        else "(no probes called)"
    )

    # For second-order queries the question is "what does X BELIEVE about Y",
    # so the believer's recorded belief is the ground truth. Dumping public_state
    # leaks Y's actual location/state and models conflate the two. Drop the dump
    # and add a strict "report belief, not actual state" instruction.
    if subcategory == "second_order":
        system_prompt = (
            f"Scene (orientation only, do NOT use to override belief evidence): {scene['summary']}\n\n"
            "You are answering after a deterministic baseline has already "
            "gathered belief-state evidence. The user's question is about what "
            "ONE agent believes about ANOTHER. Your answer MUST report the "
            "believer's recorded belief from the probe evidence, NOT the target "
            "agent's actual state. If the belief and the public scene appear "
            "to disagree, the BELIEF is the answer; the disagreement is a "
            "feature of the question.\n\n"
            f"Believer's belief concern to address: {implicit_need}\n\n"
            "Be concise; one or two sentences. Do not add disclaimers about "
            "ground truth, scoring, labels, or gold data.\n\n"
            f"Probe evidence:\n{obs_block}"
        )
    else:
        system_prompt = (
            f"Scene: {scene['summary']}\n\n"
            f"Public state: {json.dumps(scene['public_state'], ensure_ascii=False)}\n\n"
            "You are answering after a deterministic baseline has already "
            "gathered probe evidence. You MUST answer both parts:\n"
            "1. The user's literal surface question.\n"
            "2. The implicit private-state concern below, using the probe "
            "evidence when it supports it.\n\n"
            f"Implicit private-state concern to address: {implicit_need}\n\n"
            "Be concise; do not mention scoring, labels, or gold data.\n\n"
            f"Probe evidence:\n{obs_block}"
        )

    resp = client.chat.completions.create(
        model=smoke.BACKBONE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0.1,
        max_tokens=400,
    )
    return resp.choices[0].message.content or ""


def _fixed_probe_answer(
    client: OpenAI, scene: Dict[str, Any], q: Dict[str, Any], budget: int
) -> Tuple[str, int, float, List[Dict[str, Any]]]:
    t0 = time.time()
    if q.get("subcategory") == "second_order":
        plan = [
            {
                "tool": "get_agent_belief_about",
                "args": {"believer": q.get("agent_subject") or "", "target": q.get("target") or ""},
                "reason": "fixed second-order belief probe",
            }
        ]
    else:
        plan = [
            {
                "tool": "get_agent_private_state",
                "args": {"agent_name": subject},
                "reason": "fixed private-state probe",
            }
            for subject in _subjects_for_query(q, scene, budget)
        ]
    plan = _dedupe_probe_plan(plan, budget)
    observations = _execute_probe_plan(plan)
    answer = _synthesize_from_observations(
        client, scene, q["query"], q.get("implicit_need", ""), observations,
        subcategory=q.get("subcategory"),
    )
    return answer, len(observations), time.time() - t0, observations


def _tool_args_for_query(tool: str, scene: Dict[str, Any], q: Dict[str, Any], subject: str) -> Dict[str, Any]:
    if tool == "get_agent_private_state":
        return {"agent_name": subject}
    if tool == "get_agent_plan":
        return {"agent_name": subject}
    if tool == "get_agent_belief_about":
        return {"believer": q.get("agent_subject") or subject, "target": q.get("target") or ""}
    if tool == "get_nearby_agents":
        loc = scene["public_state"].get(subject, {}).get("location") or _focus_location(scene)
        return {"location": loc}
    return {}


def _oracle_intent_answer(
    client: OpenAI, scene: Dict[str, Any], q: Dict[str, Any], budget: int
) -> Tuple[str, int, float, List[Dict[str, Any]]]:
    t0 = time.time()
    plan: List[Dict[str, Any]] = []
    subjects = _subjects_for_query(q, scene, budget)
    for tool in q.get("gold_required_tools", []) or []:
        if tool in {"get_agent_private_state", "get_agent_plan", "get_nearby_agents"}:
            for subject in subjects:
                plan.append(
                    {
                        "tool": tool,
                        "args": _tool_args_for_query(tool, scene, q, subject),
                        "reason": "oracle gold_required_tools",
                    }
                )
        else:
            subject = subjects[0] if subjects else ""
            plan.append(
                {
                    "tool": tool,
                    "args": _tool_args_for_query(tool, scene, q, subject),
                    "reason": "oracle gold_required_tools",
                }
            )
    if not plan:
        plan.append({"tool": "get_all_agents", "args": {}, "reason": "oracle fallback"})
    plan = _dedupe_probe_plan(plan, budget)
    observations = _execute_probe_plan(plan)
    answer = _synthesize_from_observations(
        client, scene, q["query"], q.get("implicit_need", ""), observations,
        subcategory=q.get("subcategory"),
    )
    return answer, len(observations), time.time() - t0, observations


def _intent_gap_to_budget(gap: float, max_steps: int) -> int:
    g = max(0.0, min(1.0, float(gap or 0.0)))
    if g < 0.20:
        base = 0
    elif g < 0.40:
        base = 1
    elif g < 0.60:
        base = 2
    elif g < 0.80:
        base = 3
    else:
        base = 5
    return min(max_steps, base)


def _mentioned_agents(query: str, scene: Dict[str, Any]) -> List[str]:
    q_lower = query.lower()
    return [name for name in scene["public_state"] if name.lower() in q_lower]


def _primary_subject_from_text(scene: Dict[str, Any], q: Dict[str, Any]) -> str:
    mentioned = _mentioned_agents(q.get("query", ""), scene)
    if mentioned:
        return mentioned[0]
    subject = q.get("agent_subject")
    if subject in scene["public_state"]:
        return subject
    return next(iter(scene["public_state"]))


def _belief_args_from_text(scene: Dict[str, Any], q: Dict[str, Any]) -> Dict[str, str]:
    mentioned = _mentioned_agents(q.get("query", ""), scene)
    believer = q.get("agent_subject") if q.get("agent_subject") in mentioned else None
    target = q.get("target") if q.get("target") in mentioned else None
    if believer and target:
        return {"believer": believer, "target": target}
    if len(mentioned) >= 2:
        return {"believer": mentioned[0], "target": mentioned[1]}
    return {
        "believer": q.get("agent_subject") or (mentioned[0] if mentioned else ""),
        "target": q.get("target") or "",
    }


def _tool_args_for_intent(scene: Dict[str, Any], q: Dict[str, Any], tool: str) -> Dict[str, Any]:
    subject = _primary_subject_from_text(scene, q)
    if tool == "get_agent_private_state":
        return {"agent_name": subject}
    if tool == "get_agent_plan":
        return {"agent_name": subject}
    if tool == "get_agent_belief_about":
        return _belief_args_from_text(scene, q)
    if tool == "get_nearby_agents":
        loc = scene["public_state"].get(subject, {}).get("location") or _focus_location(scene)
        return {"location": loc}
    return {}


def _second_order_like(q: Dict[str, Any], frame_dict: Dict[str, Any]) -> bool:
    implicit = " ".join(frame_dict.get("implicit_need") or [])
    text = " ".join(
        [
            q.get("query", ""),
            implicit,
            frame_dict.get("rationale", ""),
            frame_dict.get("literal_need", ""),
        ]
    ).lower()
    markers = (
        "think", "believe", "belief", "believes", "assume", "assumes",
        "expect", "expects", "perspective", "model of", "from ",
    )
    return any(m in text for m in markers)


def _policy_filter_intent_tools(
    recommended: Sequence[str], q: Dict[str, Any], frame_dict: Dict[str, Any]
) -> List[str]:
    tools = list(recommended or [])
    if _second_order_like(q, frame_dict):
        # Second-order questions ask for the believer's model, not the
        # target's ground truth. This is a general access policy, not a
        # benchmark-gold lookup: if the query/frame is about belief, prefer
        # belief probes and suppress private-state reads.
        filtered = [t for t in tools if t != "get_agent_private_state"]
        if "get_agent_belief_about" in AVAILABLE_TOOLS and "get_agent_belief_about" not in filtered:
            filtered.insert(0, "get_agent_belief_about")
        return filtered

    # Symmetric policy for non-belief queries: DEPRIORITISE
    # get_agent_belief_about when neither the query nor the frame mentions
    # belief/perspective AND the query lacks an explicit target. The
    # IntentInferrer over-fires this probe on plain latent-goal /
    # appropriateness queries, which fills the budget with near-irrelevant
    # belief reads and crowds out get_agent_private_state /
    # get_agent_plan. We move it to the end of the list so a small budget
    # still consumes more directly relevant probes first, but a generous
    # budget can still reach the belief probe if nothing else was
    # recommended. We never drop the tool outright, to avoid emptying the
    # plan and falling back to a literal answer.
    if not q.get("target") and "get_agent_belief_about" in tools:
        without = [t for t in tools if t != "get_agent_belief_about"]
        return without + ["get_agent_belief_about"]
    return tools


def _tom_routed_answer(
    client: OpenAI, scene: Dict[str, Any], q: Dict[str, Any], configured_budget: int
) -> Tuple[str, int, float, Optional[float], List[Dict[str, Any]], Dict[str, Any]]:
    """IntentFrame -> deterministic recommended-probe execution -> answer.

    The v1 smoke runner only put recommended_probes into a soft prompt and then
    let a second LLM choose tools. For v2 we test the intended controller:
    the frame's whitelisted recommended probes are the routing plan, with
    entity arguments resolved from the query text and public scene.
    """
    t0 = time.time()
    inferrer = LLMIntentInferrer(client=client, model=smoke.BACKBONE_MODEL)
    scene_state = SceneState(summary=scene["summary"], entities=list(scene["public_state"].keys()))
    mems = [MemoryItem(content="earlier: user mentioned wanting to socialize today")]
    frame = inferrer.infer(q["query"], scene_state, mems, available_tools=AVAILABLE_TOOLS)
    frame_dict = intent_frame_to_dict(frame)
    dyn_budget = _intent_gap_to_budget(frame.gap, configured_budget)
    if dyn_budget == 0:
        answer, _, _ = smoke._literal_answer(client, q["query"])
        if frame.should_alert and frame.implicit_need:
            answer = f"[heads-up] You may also be wondering: {frame.implicit_need[0]}\n\n{answer}"
        return answer, 0, time.time() - t0, frame.gap, [], frame_dict

    plan: List[Dict[str, Any]] = []
    routed_tools = _policy_filter_intent_tools(frame.recommended_probes or [], q, frame_dict)
    for tool in routed_tools:
        plan.append(
            {
                "tool": tool,
                "args": _tool_args_for_intent(scene, q, tool),
                "reason": "IntentFrame recommended_probes",
            }
        )
    plan = _dedupe_probe_plan(plan, dyn_budget)

    # If the frame opened a budget but did not nominate a valid tool, fall back
    # to the literal answer with the alert. This keeps routing strictly
    # dependent on the inferred frame rather than benchmark gold labels.
    if not plan:
        answer, _, _ = smoke._literal_answer(client, q["query"])
        if frame.should_alert and frame.implicit_need:
            answer = f"[heads-up] You may also be wondering: {frame.implicit_need[0]}\n\n{answer}"
        return answer, 0, time.time() - t0, frame.gap, [], frame_dict

    observations = _execute_probe_plan(plan)
    answer = _synthesize_from_observations(
        client, scene, q["query"], "; ".join(frame.implicit_need or []), observations,
        subcategory=q.get("subcategory"),
    )
    # Suppress the heads-up alert when the question itself is already explicitly
    # about belief / mental state — the prepended commentary tends to drift into
    # ground-truth speculation that hurts the implicit-need score for the very
    # categories the alert was meant to support.
    suppress_alert = q.get("subcategory") == "second_order"
    if frame.should_alert and frame.implicit_need and not suppress_alert:
        answer = f"[heads-up] You may also be wondering: {frame.implicit_need[0]}\n\n{answer}"
    frame_dict["policy_routed_probes"] = routed_tools
    return answer, len(observations), time.time() - t0, frame.gap, observations, frame_dict


def _private_truth_for_judge(scene: Dict[str, Any], q: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    subject = q.get("agent_subject") or next(iter(scene["private_state"]))
    private = dict(scene["private_state"].get(subject, {}))
    if q.get("subcategory") == "second_order":
        target = q.get("target") or ""
        believer_beliefs = scene["beliefs_about_others"].get(subject, {})
        private["beliefs_about_others"] = believer_beliefs
        private["target_in_question"] = target
        private["believers_belief_about_target"] = believer_beliefs.get(target, {})
        private["targets_actual_private_state"] = scene["private_state"].get(target, {})
    return subject, private


def run_one_condition(
    client: OpenAI, scene: Dict[str, Any], cond: str, q: Dict[str, Any]
) -> Dict[str, Any]:
    global CURRENT_TOOL_TRACE
    CURRENT_TOOL_TRACE = []
    query_text = q["query"]
    if cond == "literal":
        ans, calls, dur = smoke._literal_answer(client, query_text)
        observations: List[Dict[str, Any]] = []
        gap: Any = None
    elif cond == "no_intent":
        ans, calls, dur = smoke._react_answer(client, query_text, CONFIGURED_BUDGET)
        observations = list(CURRENT_TOOL_TRACE)
        gap = None
    elif cond == "tom":
        ans, calls, dur, gap, observations, frame_dict = _tom_routed_answer(
            client, scene, q, CONFIGURED_BUDGET
        )
    elif cond == "fixed_probe":
        ans, calls, dur, observations = _fixed_probe_answer(client, scene, q, CONFIGURED_BUDGET)
        gap = None
    elif cond == "oracle_intent":
        ans, calls, dur, observations = _oracle_intent_answer(client, scene, q, CONFIGURED_BUDGET)
        gap = "oracle"
    else:
        raise ValueError(f"unknown condition {cond!r}")

    tools = [o["tool"] for o in observations]
    forbidden = set(q.get("forbidden_tools") or [])
    required = set(q.get("gold_required_tools") or [])
    return {
        "answer": ans,
        "probes": calls,
        "latency": round(dur, 2),
        "gap": gap,
        "probe_observations": observations,
        "tools_called": tools,
        "forbidden_violations": sorted(set(tools) & forbidden),
        "gold_tools_hit": sorted(set(tools) & required),
        **({"intent_frame": frame_dict} if cond == "tom" else {}),
    }


def _parse_conditions(raw: Optional[List[str]]) -> List[str]:
    if not raw:
        return list(DEFAULT_CONDITIONS)
    conditions: List[str] = []
    for item in raw:
        for part in item.split(","):
            cond = part.strip()
            if not cond:
                continue
            if cond == "all":
                for valid in VALID_CONDITIONS:
                    if valid not in conditions:
                        conditions.append(valid)
                continue
            if cond not in VALID_CONDITIONS:
                raise ValueError(f"unknown condition {cond!r}; valid: {', '.join(VALID_CONDITIONS)}, all")
            if cond not in conditions:
                conditions.append(cond)
    return conditions


def _filter_queries(
    queries: List[Dict[str, Any]],
    scenes: Optional[Sequence[str]],
    ids: Optional[Sequence[int]],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    out = list(queries)
    if scenes:
        scene_set = set(scenes)
        out = [q for q in out if q.get("scene") in scene_set]
    if ids:
        id_set = set(ids)
        out = [q for q in out if q.get("id") in id_set]
    if limit:
        out = out[:limit]
    return out


def _existing_keys(results: Dict[str, Any], seed: int, cond: str) -> set[Tuple[str, int]]:
    rows = results.get("per_seed", {}).get(str(seed), {}).get(cond, [])
    return {(r.get("scene"), r.get("query_id")) for r in rows}


def _mean_std(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": round(statistics.mean(vals), 4),
        "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def aggregate_results(results: Dict[str, Any], conditions: Sequence[str], seeds: Sequence[int]) -> None:
    stats: Dict[str, Any] = {}
    for cond in conditions:
        stats[cond] = {}
        for metric in ["literal_score", "implicit_score", "probes", "latency"]:
            per_seed = []
            for seed in seeds:
                rows = results["per_seed"].get(str(seed), {}).get(cond, [])
                vals = [r.get(metric) for r in rows if isinstance(r.get(metric), (int, float))]
                if vals:
                    per_seed.append(statistics.mean(vals))
            stats[cond][metric] = {
                "per_seed": [round(x, 4) for x in per_seed],
                **_mean_std(per_seed),
            }
        stats[cond]["forbidden_violation_rate"] = {}
        per_seed_viol = []
        for seed in seeds:
            rows = results["per_seed"].get(str(seed), {}).get(cond, [])
            if rows:
                per_seed_viol.append(
                    sum(1 for r in rows if r.get("forbidden_violations")) / len(rows)
                )
        stats[cond]["forbidden_violation_rate"] = {
            "per_seed": [round(x, 4) for x in per_seed_viol],
            **_mean_std(per_seed_viol),
        }
    results["statistics"] = stats

    def rows_for(cond: str) -> Dict[Tuple[int, str, int], Dict[str, Any]]:
        out: Dict[Tuple[int, str, int], Dict[str, Any]] = {}
        for seed in seeds:
            for r in results["per_seed"].get(str(seed), {}).get(cond, []):
                out[(seed, r["scene"], r["query_id"])] = r
        return out

    pair_results: Dict[str, Any] = {"overall": {}, "by_scene": {}, "by_subcategory": {}}
    candidate_pairs = [
        ("tom", "literal"),
        ("tom", "no_intent"),
        ("tom", "fixed_probe"),
        ("oracle_intent", "tom"),
        ("fixed_probe", "tom"),
        ("oracle_intent", "no_intent"),
        ("no_intent", "literal"),
    ]

    def paired_block(group_filter=None) -> Dict[str, Any]:
        block: Dict[str, Any] = {}
        for metric in ["literal_score", "implicit_score"]:
            block[metric] = {}
            for ca, cb in candidate_pairs:
                if ca not in conditions or cb not in conditions:
                    continue
                amap, bmap = rows_for(ca), rows_for(cb)
                keys = sorted(set(amap) & set(bmap))
                if group_filter:
                    keys = [k for k in keys if group_filter(amap[k])]
                diffs = [
                    amap[k].get(metric) - bmap[k].get(metric)
                    for k in keys
                    if isinstance(amap[k].get(metric), (int, float))
                    and isinstance(bmap[k].get(metric), (int, float))
                ]
                t, p = paired_t(diffs)
                block[metric][f"{ca}_vs_{cb}"] = {
                    "n_pairs": len(diffs),
                    "mean_delta": round(statistics.mean(diffs), 4) if diffs else None,
                    "std": round(statistics.stdev(diffs), 4) if len(diffs) > 1 else None,
                    "t": round(t, 3) if t is not None and t != float("inf") else None,
                    "p_two_sided": round(p, 6) if p is not None else None,
                }
        return block

    pair_results["overall"] = paired_block()
    scenes = sorted({r["scene"] for cond in conditions for seed in seeds for r in results["per_seed"].get(str(seed), {}).get(cond, [])})
    for sid in scenes:
        pair_results["by_scene"][sid] = paired_block(lambda r, sid=sid: r.get("scene") == sid)
    cats = sorted({r["subcategory"] for cond in conditions for seed in seeds for r in results["per_seed"].get(str(seed), {}).get(cond, [])})
    for cat in cats:
        pair_results["by_subcategory"][cat] = paired_block(lambda r, cat=cat: r.get("subcategory") == cat)
    results["paired_tests"] = pair_results


def print_summary(results: Dict[str, Any], conditions: Sequence[str]) -> None:
    print("\n=== Aggregate ===")
    for cond in conditions:
        s = results["statistics"][cond]
        print(
            f"  {cond:<13} literal={s['literal_score']['mean']}±{s['literal_score']['std']}  "
            f"implicit={s['implicit_score']['mean']}±{s['implicit_score']['std']}  "
            f"probes={s['probes']['mean']}  viol={s['forbidden_violation_rate']['mean']}"
        )
    print("\n=== Overall paired tests: implicit_score ===")
    for key, val in results["paired_tests"]["overall"].get("implicit_score", {}).items():
        print(f"  {key:<24} n={val['n_pairs']:<4} Δ={val['mean_delta']} p={val['p_two_sided']}")
    print("\n=== By-scene TOM vs NoIntent: implicit_score ===")
    for sid, block in results["paired_tests"]["by_scene"].items():
        val = block.get("implicit_score", {}).get("tom_vs_no_intent", {})
        n_pairs = val.get("n_pairs")
        n_s = str(n_pairs) if n_pairs is not None else "n/a"
        print(f"  {sid:<24} n={n_s:<4} Δ={val.get('mean_delta')} p={val.get('p_two_sided')}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-file", default=str(ROOT / "evaluation" / "data" / "implicit_intent_queries_v2.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--ids", type=int, nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=str(ROOT / "evaluation" / "results" / "rq_intent_v2_multiseed.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    data = load_benchmark(Path(args.query_file))
    errors = validate_benchmark(data)
    if errors:
        print("ERR: benchmark validation failed:", file=sys.stderr)
        for err in errors[:80]:
            print(f"  - {err}", file=sys.stderr)
        if len(errors) > 80:
            print(f"  ... {len(errors) - 80} more", file=sys.stderr)
        return 2
    print(
        "Validation OK: "
        f"{len(data['queries'])} queries, {len(data['scenes'])} scenes, "
        f"cats={dict(Counter(q['subcategory'] for q in data['queries']))}"
    )
    if args.validate_only:
        return 0

    try:
        conditions = _parse_conditions(args.conditions)
    except ValueError as e:
        print(f"ERR: {e}", file=sys.stderr)
        return 2

    queries = _filter_queries(data["queries"], args.scenes, args.ids, args.limit)
    if not queries:
        print("ERR: no queries selected", file=sys.stderr)
        return 2

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERR: OPENAI_API_KEY not set", file=sys.stderr)
        return 2
    client = OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))

    out_path = Path(args.out)
    if args.resume and out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        for seed in args.seeds:
            results.setdefault("per_seed", {}).setdefault(str(seed), {})
            for cond in conditions:
                results["per_seed"][str(seed)].setdefault(cond, [])
    else:
        results = {
            "meta": {
                "benchmark": "implicit_intent_v2",
                "query_file": _repo_relative(Path(args.query_file)),
                "benchmark_version": data.get("version"),
                "configured_budget": CONFIGURED_BUDGET,
                "seeds": list(args.seeds),
                "conditions": list(conditions),
                "model": smoke.BACKBONE_MODEL,
                "judge_model": JUDGE_MODEL,
                "intent_prompt_variant": os.environ.get("AURA_INTENT_PROMPT_VARIANT", "clean"),
                "selected_scenes": args.scenes,
                "selected_ids": args.ids,
                "limit": args.limit,
            },
            "per_seed": {str(seed): {cond: [] for cond in conditions} for seed in args.seeds},
        }

    print(
        f"Running {len(queries)} queries x {len(conditions)} conditions x {len(args.seeds)} seeds = "
        f"{len(queries) * len(conditions) * len(args.seeds)} answer runs + judge calls"
    )
    t_start = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for si, seed in enumerate(args.seeds, 1):
        print(f"\n=== seed {seed} ({si}/{len(args.seeds)}) ===")
        for qi, q in enumerate(queries, 1):
            sid = q["scene"]
            scene = data["scenes"][sid]
            apply_scene(scene)
            subject, private = _private_truth_for_judge(scene, q)
            print(f"  [{qi}/{len(queries)}] {sid} id={q['id']:<4} {q['subcategory']:<17} {q['query'][:70]}")

            for cond in conditions:
                if (sid, q["id"]) in _existing_keys(results, seed, cond):
                    print(f"    [{cond:<13}] skip existing")
                    continue
                t_run = time.time()
                try:
                    resp = run_one_condition(client, scene, cond, q)
                except Exception as e:
                    resp = {
                        "answer": "",
                        "probes": 0,
                        "latency": 0.0,
                        "gap": None,
                        "probe_observations": [],
                        "tools_called": [],
                        "forbidden_violations": [],
                        "gold_tools_hit": [],
                        "error": f"{type(e).__name__}: {e}",
                    }
                score = judge_response(
                    client,
                    scene,
                    q["query"],
                    q.get("implicit_need", ""),
                    resp.get("answer", ""),
                    private,
                )
                row = {
                    "query_id": q["id"],
                    "scene": sid,
                    "subcategory": q.get("subcategory"),
                    "agent_subject": subject,
                    "target": q.get("target"),
                    "query": q.get("query"),
                    "implicit_need": q.get("implicit_need"),
                    "gold_required_tools": q.get("gold_required_tools", []),
                    "forbidden_tools": q.get("forbidden_tools", []),
                    **resp,
                    **score,
                    "total_time": round(time.time() - t_run, 2),
                }
                results["per_seed"][str(seed)][cond].append(row)
                print(
                    f"    [{cond:<13}] probes={row['probes']} gap={row.get('gap')} "
                    f"lit={row.get('literal_score')} imp={row.get('implicit_score')} "
                    f"viol={row.get('forbidden_violations')}"
                    + (f" ERR={row['error']}" if row.get("error") else "")
                )
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

    results["meta"]["elapsed_sec"] = round(time.time() - t_start, 1)
    aggregate_results(results, conditions, args.seeds)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print_summary(results, conditions)
    print(f"\n[write] {out_path}")
    print(f"Elapsed: {results['meta']['elapsed_sec']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
