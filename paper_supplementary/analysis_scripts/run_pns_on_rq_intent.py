"""Plan-and-Solve baseline on the RQ-Intent 25 implicit-intent queries.

Adds the missing baseline that the audit identified as critical: if
Plan-and-Solve outperforms AURA on RQ2 factual grounding, does it ALSO
outperform AURA on RQ-Intent (where the literal-vs-implicit gap is
non-zero)? If yes, AURA's mechanism story is in trouble. If no, AURA's
contribution is correctly scoped to gap-non-zero queries.

Reuses verbatim from `scripts.run_implicit_intent_smoke`:
  - SCENE_SUMMARY, PUBLIC_STATE, PRIVATE_STATE, BELIEFS_ABOUT_OTHERS
  - AVAILABLE_TOOLS, TOOL_SCHEMA, tool_exec
  - judge_response from run_implicit_intent_full

Adds: `_plan_and_solve_answer` — a plan-then-execute condition with the
same tool budget as the other conditions.

Output: evaluation/results/rq_intent_pns_multiseed.json
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
_AURA_SRC = ROOT / "AURA" / "src"
if str(_AURA_SRC) not in sys.path:
    sys.path.insert(0, str(_AURA_SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI

from scripts.run_implicit_intent_smoke import (
    SCENE_SUMMARY, PUBLIC_STATE, PRIVATE_STATE, BELIEFS_ABOUT_OTHERS,
    AVAILABLE_TOOLS, tool_exec, TOOL_SCHEMA,
)
from scripts.run_implicit_intent_full import judge_response

QUERIES_PATH = ROOT / "evaluation" / "data" / "implicit_intent_queries.json"
OUTPUT_PATH = ROOT / "evaluation" / "results" / "rq_intent_pns_multiseed.json"
MODEL = os.environ.get("AURA_BACKBONE_MODEL", "gpt-4o-mini")
SEEDS = [42, 123, 456]
CONFIGURED_BUDGET = 3


def make_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


def _system_prompt_scene() -> str:
    """Same scene/context prefix used by the literal/no_intent/intent runs."""
    return (
        "You are an assistant for a 5-agent town simulation.\n\n"
        f"SCENE: {SCENE_SUMMARY}\n\n"
        f"PUBLIC STATE (visible to anyone):\n{json.dumps(PUBLIC_STATE, ensure_ascii=False, indent=2)}\n\n"
        "Some agent state (availability, mood, goals, beliefs) is private and "
        "only retrievable via probe tools. Use them when the surface query "
        "may hide a private-state need."
    )


def _plan_and_solve_answer(
    client: OpenAI, query: str, budget: int, seed: Optional[int] = None,
) -> Tuple[str, int, float]:
    """Plan-then-execute baseline.

    Phase 1: ask the model to enumerate up to `budget` tool calls that
    would gather the evidence needed.
    Phase 2: execute the plan in order via the OpenAI tool API.
    Phase 3: synthesize a final answer over the gathered observations.

    Mirrors `_react_answer`'s budget semantics so total tool calls per
    query are comparable. Returns (response_text, n_tool_calls,
    wall_clock_seconds).
    """
    t0 = time.time()
    scene_msg = _system_prompt_scene()

    # ── Phase 1: planning
    plan_system = (
        scene_msg
        + "\n\nYou are a planning assistant. Given the user's query, list "
        f"up to {budget} probe tool calls (with arguments) you would run "
        "to gather the evidence needed. Available tools: "
        + ", ".join(AVAILABLE_TOOLS)
        + ".\n\nOutput JSON with this schema (no prose):\n"
        '{"plan": [{"tool": "<name>", "args": {...}, "reason": "..."}, ...]}\n\n'
        "If the query can be answered without tools, return "
        '{"plan": []} so we skip Phase 2.'
    )
    kwargs = dict(
        model=MODEL,
        messages=[
            {"role": "system", "content": plan_system},
            {"role": "user", "content": query},
        ],
        temperature=0.1,
        max_tokens=400,
        response_format={"type": "json_object"},
    )
    if seed is not None:
        kwargs["seed"] = seed
    try:
        resp = client.chat.completions.create(**kwargs)
    except TypeError:
        kwargs.pop("seed", None)
        resp = client.chat.completions.create(**kwargs)

    plan_raw = resp.choices[0].message.content or "{}"
    try:
        plan = json.loads(plan_raw).get("plan", [])
        if not isinstance(plan, list):
            plan = []
    except json.JSONDecodeError:
        plan = []

    # ── Phase 2: execute the plan
    observations: List[Dict[str, Any]] = []
    for step in plan[:budget]:
        if not isinstance(step, dict):
            continue
        tool = step.get("tool")
        args = step.get("args") or {}
        if tool not in AVAILABLE_TOOLS:
            observations.append({"tool": tool, "args": args, "result": f"unknown tool {tool!r}"})
            continue
        try:
            out = tool_exec(tool, args if isinstance(args, dict) else {})
        except Exception as e:
            out = json.dumps({"error": str(e)})
        observations.append({"tool": tool, "args": args, "result": out})

    # ── Phase 3: synthesize final answer
    obs_block = (
        "\n".join(
            f"- {o['tool']}({json.dumps(o.get('args', {}), ensure_ascii=False)}) -> {o['result']}"
            for o in observations
        )
        if observations
        else "(no probes called)"
    )
    answer_system = (
        scene_msg
        + "\n\nYou now have evidence from probe tools. Use it to answer the "
        "user's query. Be concise (2-3 sentences). If the implicit need behind "
        "the surface query is visible in the evidence, surface it explicitly.\n\n"
        f"PROBE EVIDENCE:\n{obs_block}"
    )
    kwargs2 = dict(
        model=MODEL,
        messages=[
            {"role": "system", "content": answer_system},
            {"role": "user", "content": query},
        ],
        temperature=0.4,
        max_tokens=320,
    )
    if seed is not None:
        kwargs2["seed"] = seed
    try:
        ans_resp = client.chat.completions.create(**kwargs2)
    except TypeError:
        kwargs2.pop("seed", None)
        ans_resp = client.chat.completions.create(**kwargs2)
    answer = (ans_resp.choices[0].message.content or "").strip()
    return answer, len(observations), time.time() - t0


def main() -> int:
    with open(QUERIES_PATH) as f:
        data = json.load(f)
    queries = data.get("queries") if isinstance(data, dict) else data
    print(f"[setup] {len(queries)} queries × {len(SEEDS)} seeds × 1 condition (Plan_and_Solve)")
    print(f"[setup] backbone={MODEL}, judge=gpt-4o-mini, budget={CONFIGURED_BUDGET}")

    client = make_client()
    output: Dict[str, Any] = {
        "schema_version": "1.0",
        "backbone": MODEL,
        "judge_model": "gpt-4o-mini",
        "configured_budget": CONFIGURED_BUDGET,
        "seeds": SEEDS,
        "n_queries": len(queries),
        "condition": "plan_and_solve",
        "per_seed": {},
    }
    t_global = time.time()

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        details: List[Dict[str, Any]] = []
        t_seed = time.time()
        for qi, q in enumerate(queries):
            agent = q.get("agent_subject") or q.get("agent")
            agent_private = PRIVATE_STATE.get(agent, {})
            try:
                ans, n_probes, dur = _plan_and_solve_answer(
                    client, q["query"], CONFIGURED_BUDGET, seed=seed,
                )
                judge = judge_response(
                    client, q["query"], q.get("implicit_need", ""),
                    ans, agent_private,
                )
                details.append({
                    "query_id": q.get("id", qi),
                    "subcategory": q.get("subcategory"),
                    "query": q["query"],
                    "agent_subject": agent,
                    "implicit_need": q.get("implicit_need"),
                    "response": ans[:600],
                    "probes_called": n_probes,
                    "latency": round(dur, 3),
                    "literal_score": judge.get("literal_score"),
                    "implicit_score": judge.get("implicit_score"),
                    "judge_rationale": judge.get("rationale", "")[:200],
                })
                if (qi + 1) % 5 == 0:
                    elapsed = time.time() - t_seed
                    avg_lit = statistics.mean(
                        d["literal_score"] for d in details if d.get("literal_score") is not None
                    )
                    avg_imp = statistics.mean(
                        d["implicit_score"] for d in details if d.get("implicit_score") is not None
                    )
                    print(f"  q{qi+1}/{len(queries)}  elapsed={elapsed:.0f}s  "
                          f"lit={avg_lit:.3f} imp={avg_imp:.3f}")
            except Exception as e:
                details.append({
                    "query_id": q.get("id", qi),
                    "query": q["query"],
                    "error": str(e),
                })

        # per-seed aggregate
        valid = [d for d in details if d.get("literal_score") is not None]
        agg = {
            "n_valid": len(valid),
            "n_total": len(details),
            "literal_score": (
                statistics.mean(d["literal_score"] for d in valid) if valid else 0.0
            ),
            "implicit_score": (
                statistics.mean(d["implicit_score"] for d in valid) if valid else 0.0
            ),
            "mean_latency": (
                statistics.mean(d["latency"] for d in valid) if valid else 0.0
            ),
            "mean_probes": (
                statistics.mean(d["probes_called"] for d in valid) if valid else 0.0
            ),
        }
        output["per_seed"][str(seed)] = {
            "summary": agg,
            "details": details,
        }
        print(f"  seed {seed}: lit={agg['literal_score']:.3f} "
              f"imp={agg['implicit_score']:.3f} "
              f"probes={agg['mean_probes']:.2f} "
              f"lat={agg['mean_latency']:.2f}s "
              f"valid={agg['n_valid']}/{agg['n_total']}")
        # checkpoint
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    # multi-seed aggregate
    seeds_lit = [output["per_seed"][str(s)]["summary"]["literal_score"] for s in SEEDS]
    seeds_imp = [output["per_seed"][str(s)]["summary"]["implicit_score"] for s in SEEDS]
    seeds_lat = [output["per_seed"][str(s)]["summary"]["mean_latency"] for s in SEEDS]

    output["multi_seed_summary"] = {
        "literal_score_mean": round(statistics.mean(seeds_lit), 4),
        "literal_score_std": round(statistics.stdev(seeds_lit), 4) if len(seeds_lit) > 1 else 0.0,
        "implicit_score_mean": round(statistics.mean(seeds_imp), 4),
        "implicit_score_std": round(statistics.stdev(seeds_imp), 4) if len(seeds_imp) > 1 else 0.0,
        "implicit_per_seed": seeds_imp,
        "mean_latency": round(statistics.mean(seeds_lat), 2),
        "n_seeds": len(seeds_lit),
    }
    output["wall_clock_s"] = round(time.time() - t_global, 1)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n=== MULTI-SEED AGGREGATE ===")
    s = output["multi_seed_summary"]
    print(f"  Plan_and_Solve  literal={s['literal_score_mean']:.3f} ± {s['literal_score_std']:.4f}")
    print(f"  Plan_and_Solve  implicit={s['implicit_score_mean']:.3f} ± {s['implicit_score_std']:.4f}")
    print(f"  per-seed implicit: {seeds_imp}")
    print(f"  mean latency = {s['mean_latency']:.2f}s")
    print(f"\n[done] wall={output['wall_clock_s']:.1f}s   saved → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
