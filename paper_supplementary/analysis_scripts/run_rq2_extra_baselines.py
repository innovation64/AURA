"""RQ2 supplementary: run Reflexion + Plan-and-Solve baselines that the
original multi-seed runner skipped.

Both baselines are already implemented in `evaluation/baselines.py`:
  - reflexion_chat: ReAct + self-reflection, up to 2 retry rounds (≤15 LLM calls)
  - plan_and_solve_chat: explicit plan-then-execute, ≤5 tool calls

For each seed in {42, 123, 456}, we:
  1. /api/reset (with seed payload — propagates to TownConfig.world_seed
     and llm_seed via our seed-fix patch)
  2. Warm 10 sim steps deterministically
  3. For each of the 50 chat queries: capture gt_state, ask each baseline,
     score with judge_factual_accuracy + judge_context_utilization
  4. Advance 1 sim tick between queries so each query sees a fresh state

Output: evaluation/results/rq2_extra_baselines_multiseed.json
Schema mirrors rq2_factual_accuracy_multiseed.json so downstream scripts
(rescore_rq2_strict.py etc.) can ingest it as additional conditions.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from statistics import mean

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.config import EvalConfig  # noqa: E402
from evaluation.baselines import reflexion_chat, plan_and_solve_chat  # noqa: E402
from evaluation.llm_judge import judge_factual_accuracy, judge_context_utilization  # noqa: E402
from evaluation.run_experiments import AURAClient  # noqa: E402

SEEDS = [42, 123, 456]
QUERIES_PATH = ROOT / "evaluation" / "data" / "chat_queries.json"
OUTPUT_PATH = ROOT / "evaluation" / "results" / "rq2_extra_baselines_multiseed.json"
SERVER = os.environ.get("AURA_SERVER", "http://127.0.0.1:7861")
N_QUERIES = int(os.environ.get("N_QUERIES", "50"))
WARMUP_STEPS = 10

BASELINES = [
    ("Reflexion", reflexion_chat),
    ("Plan_and_Solve", plan_and_solve_chat),
]


def aggregate_condition(details: list[dict]) -> dict:
    fa_values = [
        r["factual_accuracy"]["accuracy"]
        for r in details
        if isinstance(r.get("factual_accuracy"), dict) and "accuracy" in r["factual_accuracy"]
    ]
    cu_values = [
        r["context_utilization"]["utilization"]
        for r in details
        if isinstance(r.get("context_utilization"), dict) and "utilization" in r["context_utilization"]
    ]
    lat_values = [r["latency"] for r in details if "latency" in r]
    n_errors = sum(1 for r in details if "error" in r)
    return {
        "avg_factual_accuracy": round(mean(fa_values), 4) if fa_values else 0.0,
        "avg_context_utilization": round(mean(cu_values), 4) if cu_values else 0.0,
        "avg_latency": round(mean(lat_values), 3) if lat_values else 0.0,
        "num_valid": len(fa_values),
        "num_errors": n_errors,
    }


def main() -> int:
    config = EvalConfig(aura_server=SERVER)
    client = AURAClient(SERVER)

    with open(QUERIES_PATH) as f:
        queries = json.load(f)["queries"][:N_QUERIES]
    print(f"[setup] {len(queries)} queries × {len(SEEDS)} seeds × {len(BASELINES)} baselines")

    output = {
        "schema_version": "1.0",
        "seeds": SEEDS,
        "n_queries": len(queries),
        "baselines": [name for name, _ in BASELINES],
        "per_seed": {},
    }

    t_global = time.time()

    for seed in SEEDS:
        print(f"\n{'='*60}\n[seed {seed}] reset (seed payload propagated to server)\n{'='*60}")
        reset_resp = client.reset(seed=seed)
        if not reset_resp.get("ok"):
            print(f"  ERROR: reset failed: {reset_resp}")
            continue
        # Warm up
        for _ in range(WARMUP_STEPS):
            client.step()
            time.sleep(0.2)

        agents = client.state()["state"]["agents"]
        agent_names = [a["name"] for a in agents]

        # Phase A: collect snapshots for THIS seed
        print(f"[seed {seed}] Phase A: collecting {len(queries)} snapshots...")
        snapshots: list[dict] = []
        for qi in range(len(queries)):
            gt_state = client.state()["state"]
            snapshots.append({
                "qi": qi,
                "agent": agent_names[qi % len(agent_names)],
                "gt_state": gt_state,
                "tick": gt_state.get("tick", WARMUP_STEPS + qi),
            })
            client.step()
            time.sleep(0.1)
        print(f"[seed {seed}] {len(snapshots)} snapshots captured")

        seed_results: dict = {}

        # Phase B: run each baseline against the snapshots
        for bl_name, bl_fn in BASELINES:
            print(f"\n[seed {seed}] running {bl_name}...")
            details = []
            t0 = time.time()
            for qi, q in enumerate(queries):
                if qi % 10 == 0:
                    elapsed = time.time() - t0
                    print(f"  q{qi}/{len(queries)}  elapsed={elapsed:.0f}s")
                snap = snapshots[qi]
                agent_name = snap["agent"]
                gt_state = snap["gt_state"]
                try:
                    bl_result = bl_fn(config, agent_name, q["query"], gt_state)
                    response = bl_result.get("response", "")
                    env_context = bl_result.get("env_context", {})
                    latency = bl_result.get("latency", 0.0)

                    fa = judge_factual_accuracy(config, q["query"], gt_state, response, agent_name)
                    cu = judge_context_utilization(config, env_context, response)

                    details.append({
                        "query_id": q["id"],
                        "category": q["category"],
                        "query": q["query"],
                        "agent": agent_name,
                        "response": response[:600],
                        "latency": round(latency, 3),
                        "factual_accuracy": fa,
                        "context_utilization": cu,
                        "tool_calls": bl_result.get("tool_calls", 0),
                        "reflexion_rounds": bl_result.get("reflexion_rounds", None),
                        "snapshot_tick": snap["tick"],
                    })
                except Exception as e:
                    details.append({
                        "query_id": q["id"],
                        "category": q["category"],
                        "query": q["query"],
                        "agent": agent_name,
                        "error": str(e),
                    })

            summary = aggregate_condition(details)
            seed_results[bl_name] = {
                "summary": summary,
                "details": details,
                "avg_factual_accuracy": summary["avg_factual_accuracy"],
                "avg_context_utilization": summary["avg_context_utilization"],
                "avg_latency": summary["avg_latency"],
                "num_valid": summary["num_valid"],
            }
            print(f"  {bl_name} seed {seed}: FA={summary['avg_factual_accuracy']} "
                  f"CU={summary['avg_context_utilization']} "
                  f"lat={summary['avg_latency']}s "
                  f"errors={summary['num_errors']}")

        output["per_seed"][str(seed)] = seed_results

        # Save partial after every seed in case of interruption
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[seed {seed}] partial saved → {OUTPUT_PATH}")

    # Aggregate across seeds
    print(f"\n{'='*60}\nMULTI-SEED AGGREGATE\n{'='*60}")
    multi_seed_summary: dict = {}
    for bl_name, _ in BASELINES:
        per_seed_fa = []
        per_seed_lat = []
        for seed in SEEDS:
            sr = output["per_seed"].get(str(seed), {}).get(bl_name)
            if sr:
                per_seed_fa.append(sr["avg_factual_accuracy"])
                per_seed_lat.append(sr["avg_latency"])
        if per_seed_fa:
            from statistics import stdev
            multi_seed_summary[bl_name] = {
                "fa_mean": round(mean(per_seed_fa), 4),
                "fa_std": round(stdev(per_seed_fa), 4) if len(per_seed_fa) > 1 else 0.0,
                "fa_per_seed": per_seed_fa,
                "lat_mean": round(mean(per_seed_lat), 2),
                "n_seeds": len(per_seed_fa),
            }
            print(f"  {bl_name}: FA = {multi_seed_summary[bl_name]['fa_mean']:.4f} "
                  f"± {multi_seed_summary[bl_name]['fa_std']:.4f} "
                  f"(n_seeds={len(per_seed_fa)}, lat={multi_seed_summary[bl_name]['lat_mean']:.1f}s)")

    output["multi_seed_summary"] = multi_seed_summary
    output["wall_clock_s"] = round(time.time() - t_global, 1)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[done] wall={output['wall_clock_s']:.1f}s   saved → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
