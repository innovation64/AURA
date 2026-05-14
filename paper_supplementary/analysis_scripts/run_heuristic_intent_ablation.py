"""Heuristic vs LLM IntentInferrer ablation.

Runs the SAME 25 implicit-intent queries x 3 seeds as the headline
RQ-ToM experiment (run_implicit_intent_full.py) but with
HeuristicIntentInferrer instead of LLMIntentInferrer. All other
plumbing (gap-to-budget map, probe loop, judge) is identical so the
contrast isolates the inferrer backend.

Output: evaluation/results/rq_tom_heuristic_ablation.json
Budget estimate: 25 queries x 3 seeds x ~2 LLM calls/run + 75 judge
calls ~ $0.40-0.70 on gpt-4o-mini. The heuristic itself is free.
"""

from __future__ import annotations

import argparse
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

_AURA_SRC = Path(__file__).resolve().parent.parent / "AURA" / "src"
if str(_AURA_SRC) not in sys.path:
    sys.path.insert(0, str(_AURA_SRC))

from openai import OpenAI

from scripts.run_implicit_intent_smoke import (
    SCENE_SUMMARY, PUBLIC_STATE, PRIVATE_STATE, BELIEFS_ABOUT_OTHERS,
    AVAILABLE_TOOLS, tool_exec, TOOL_SCHEMA,
    _literal_answer,
)
from scripts.run_implicit_intent_full import judge_response, JUDGE_MODEL, paired_t

from aura.intent import HeuristicIntentInferrer
from aura.types import MemoryItem, SceneState

CONFIGURED_BUDGET = 3


def _tom_heuristic_answer(client: OpenAI, query: str, configured_budget: int
                          ) -> Tuple[str, int, float, Optional[float]]:
    """ToM condition with HEURISTIC backend. Same plumbing as the LLM path
    in run_implicit_intent_smoke._tom_answer; only the inferrer differs."""
    t0 = time.time()
    inferrer = HeuristicIntentInferrer()
    scene = SceneState(summary=SCENE_SUMMARY, entities=list(PUBLIC_STATE.keys()))
    mems = [MemoryItem(content="earlier: user mentioned wanting to socialize today")]
    frame = inferrer.infer(query, scene, mems, available_tools=AVAILABLE_TOOLS)

    g = frame.gap or 0.0
    if g < 0.20:   dyn_budget = 0
    elif g < 0.40: dyn_budget = 1
    elif g < 0.60: dyn_budget = 2
    elif g < 0.80: dyn_budget = 3
    else:           dyn_budget = 5
    dyn_budget = min(configured_budget, dyn_budget)

    if dyn_budget == 0:
        ans, _, _ = _literal_answer(client, query)
        if frame.should_alert and frame.implicit_need:
            ans = f"[heads-up] You may also be wondering: {frame.implicit_need[0]}\n\n{ans}"
        return ans, 0, time.time() - t0, g

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
            model="gpt-4o-mini", messages=messages,
            tools=TOOL_SCHEMA, tool_choice="auto",
            temperature=0.1, max_tokens=400,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                if calls < dyn_budget:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}
                    out = tool_exec(tc.function.name, args)
                    calls += 1
                else:
                    out = json.dumps({"error": "budget exhausted"})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out[:2000]})
            if calls >= dyn_budget:
                messages.append({"role": "user", "content":
                                 "Based on what you have, answer now, covering BOTH the literal and the implicit need."})
                final = client.chat.completions.create(
                    model="gpt-4o-mini", messages=messages,
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


def _load_queries() -> List[Dict[str, Any]]:
    src = Path(__file__).resolve().parent.parent / "evaluation" / "data" / "implicit_intent_queries.json"
    with open(src) as f:
        return json.load(f)["queries"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                        / "evaluation" / "results"
                                        / "rq_tom_heuristic_ablation.json"))
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERR: OPENAI_API_KEY not set", file=sys.stderr)
        return 2
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    queries = _load_queries()
    if args.limit:
        queries = queries[:args.limit]

    print(f"Heuristic-backend ablation: {len(queries)} queries x {len(args.seeds)} seeds = "
          f"{len(queries) * len(args.seeds)} answer + {len(queries) * len(args.seeds)} judge calls")

    results: Dict[str, Any] = {
        "meta": {
            "scene": SCENE_SUMMARY,
            "configured_budget": CONFIGURED_BUDGET,
            "seeds": list(args.seeds),
            "model": "gpt-4o-mini",
            "judge_model": JUDGE_MODEL,
            "inferrer_backend": "HeuristicIntentInferrer",
        },
        "per_seed": {str(s): [] for s in args.seeds},
    }

    t_start = time.time()
    for si, seed in enumerate(args.seeds):
        print(f"\n=== seed {seed} ({si + 1}/{len(args.seeds)}) ===")
        for qi, q in enumerate(queries, 1):
            subject = q.get("agent_subject") or list(PRIVATE_STATE.keys())[0]
            private = dict(PRIVATE_STATE.get(subject, {}))
            if q.get("subcategory") == "second_order":
                target = q.get("target") or ""
                believer_beliefs = BELIEFS_ABOUT_OTHERS.get(subject, {})
                private["beliefs_about_others"] = believer_beliefs
                private["target_in_question"] = target
                private["believers_belief_about_target"] = believer_beliefs.get(target, {})
                private["targets_actual_private_state"] = PRIVATE_STATE.get(target, {})
            print(f"  [{qi}/{len(queries)}] id={q['id']:<4} {q.get('subcategory'):<17} {q['query'][:60]}")

            t_run = time.time()
            try:
                ans, calls, dur, gap = _tom_heuristic_answer(client, q["query"], CONFIGURED_BUDGET)
                resp = {"answer": ans, "probes": calls, "latency": round(dur, 2), "gap": gap}
            except Exception as e:
                resp = {"answer": "", "probes": 0, "latency": 0.0, "gap": None,
                        "error": f"{type(e).__name__}: {e}"}

            try:
                score = judge_response(client, q["query"], q.get("implicit_need", ""),
                                       resp.get("answer", ""), private)
            except Exception as e:
                score = {"literal_score": None, "implicit_score": None,
                         "rationale": f"judge exc: {e}"}

            row = {
                "query_id": q["id"],
                "subcategory": q.get("subcategory"),
                "agent_subject": subject,
                **resp,
                **score,
                "total_time": round(time.time() - t_run, 2),
            }
            results["per_seed"][str(seed)].append(row)
            print(f"    [tom_heur] probes={resp['probes']} gap={resp.get('gap')}  "
                  f"lit={score.get('literal_score')} imp={score.get('implicit_score')}")

    # Aggregation
    print("\n=== Aggregation ===")
    lit_per_seed: List[float] = []
    imp_per_seed: List[float] = []
    probes_per_seed: List[float] = []
    lat_per_seed: List[float] = []
    for seed in args.seeds:
        rows = results["per_seed"][str(seed)]
        lit_vals = [r["literal_score"] for r in rows if r.get("literal_score") is not None]
        imp_vals = [r["implicit_score"] for r in rows if r.get("implicit_score") is not None]
        probe_vals = [r["probes"] for r in rows]
        lat_vals = [r["latency"] for r in rows]
        if lit_vals:
            lit_per_seed.append(sum(lit_vals) / len(lit_vals))
        if imp_vals:
            imp_per_seed.append(sum(imp_vals) / len(imp_vals))
        if probe_vals:
            probes_per_seed.append(sum(probe_vals) / len(probe_vals))
        if lat_vals:
            lat_per_seed.append(sum(lat_vals) / len(lat_vals))

    def ms(vs: List[float]) -> str:
        if not vs: return "n/a"
        if len(vs) == 1: return f"{vs[0]:.4f}"
        return f"{statistics.mean(vs):.4f} ± {statistics.stdev(vs):.4f}"

    print(f"  tom_heuristic  literal={ms(lit_per_seed)}  implicit={ms(imp_per_seed)}  "
          f"probes={ms(probes_per_seed)}  latency={ms(lat_per_seed)}")

    # Per-subcategory
    by_sub: Dict[str, List[float]] = {}
    for seed in args.seeds:
        for r in results["per_seed"][str(seed)]:
            sub = r.get("subcategory", "?")
            if r.get("implicit_score") is not None:
                by_sub.setdefault(sub, []).append(r["implicit_score"])
    print("  per-subcategory implicit_score:")
    for sub, vs in sorted(by_sub.items()):
        print(f"    {sub:<17} n={len(vs):3d}  mean={statistics.mean(vs):.3f}")

    results["aggregate"] = {
        "literal_score_mean": statistics.mean(lit_per_seed) if lit_per_seed else None,
        "literal_score_std": statistics.stdev(lit_per_seed) if len(lit_per_seed) > 1 else None,
        "implicit_score_mean": statistics.mean(imp_per_seed) if imp_per_seed else None,
        "implicit_score_std": statistics.stdev(imp_per_seed) if len(imp_per_seed) > 1 else None,
        "probes_mean": statistics.mean(probes_per_seed) if probes_per_seed else None,
        "latency_mean": statistics.mean(lat_per_seed) if lat_per_seed else None,
        "per_subcategory_implicit_mean": {sub: statistics.mean(vs) for sub, vs in by_sub.items()},
        "wall_time_seconds": round(time.time() - t_start, 1),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
