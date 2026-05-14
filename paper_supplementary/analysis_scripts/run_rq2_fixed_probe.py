"""RQ2 cross-regime control: fixed_probe baseline.

Every query unconditionally receives all 8 SHARED_TOOLS outputs. No gap
inference, no tool selection. Tests whether AURA's gap-routed "skip when
not needed" advantage matters on the zero-gap factual regime, by
contrasting the RQ-Intent fixed_probe finding (0.851 implicit, slightly
above AURA Intent 0.803, n.s.) with whether unconditional probing
similarly approaches AURA Full's 0.640 FA on RQ2 -- or tanks because
zero-gap queries don't need the extra context.

Output: evaluation/results/rq2_fixed_probe_multiseed.json
Schema mirrors rq2_extra_baselines_multiseed.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, stdev

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.config import EvalConfig  # noqa: E402
from evaluation.baselines import fixed_probe_chat  # noqa: E402
from evaluation.llm_judge import judge_factual_accuracy, judge_context_utilization  # noqa: E402
from evaluation.run_experiments import AURAClient  # noqa: E402

SEEDS = [42, 123, 456]
QUERIES_PATH = ROOT / "evaluation" / "data" / "chat_queries.json"
OUTPUT_PATH = ROOT / "evaluation" / "results" / "rq2_fixed_probe_multiseed.json"
SERVER = os.environ.get("AURA_SERVER", "http://127.0.0.1:7861")
N_QUERIES = int(os.environ.get("N_QUERIES", "50"))
WARMUP_STEPS = 10

BASELINE_NAME = "Fixed_Probe"
BASELINE_FN = fixed_probe_chat


def aggregate_condition(details):
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


def main():
    config = EvalConfig(aura_server=SERVER)
    client = AURAClient(SERVER)

    with open(QUERIES_PATH) as f:
        queries = json.load(f)["queries"][:N_QUERIES]
    print(f"[setup] {len(queries)} queries × {len(SEEDS)} seeds × 1 baseline ({BASELINE_NAME})")

    output = {
        "schema_version": "1.0",
        "seeds": SEEDS,
        "n_queries": len(queries),
        "baseline": BASELINE_NAME,
        "per_seed": {},
    }

    t_global = time.time()

    for seed in SEEDS:
        print(f"\n{'='*60}\n[seed {seed}] reset\n{'='*60}")
        reset_resp = client.reset(seed=seed)
        if not reset_resp.get("ok"):
            print(f"  ERROR: reset failed: {reset_resp}")
            continue
        for _ in range(WARMUP_STEPS):
            client.step()
            time.sleep(0.2)

        agents = client.state()["state"]["agents"]
        agent_names = [a["name"] for a in agents]

        # Phase A: snapshots
        print(f"[seed {seed}] Phase A: collecting {len(queries)} snapshots...")
        snapshots = []
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

        # Phase B: fixed_probe
        print(f"\n[seed {seed}] running {BASELINE_NAME}...")
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
                bl_result = BASELINE_FN(config, agent_name, q["query"], gt_state)
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
                    "snapshot_tick": snap["tick"],
                })
            except Exception as e:
                details.append({
                    "query_id": q["id"],
                    "category": q["category"],
                    "query": q["query"],
                    "agent": agent_name,
                    "error": f"{type(e).__name__}: {e}",
                })

        summary = aggregate_condition(details)
        output["per_seed"][str(seed)] = {
            BASELINE_NAME: {
                "summary": summary,
                "details": details,
                "avg_factual_accuracy": summary["avg_factual_accuracy"],
                "avg_context_utilization": summary["avg_context_utilization"],
                "avg_latency": summary["avg_latency"],
                "num_valid": summary["num_valid"],
            }
        }
        print(f"  {BASELINE_NAME} seed {seed}: FA={summary['avg_factual_accuracy']} "
              f"CU={summary['avg_context_utilization']} "
              f"lat={summary['avg_latency']}s "
              f"errors={summary['num_errors']}")

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[seed {seed}] partial saved → {OUTPUT_PATH}")

    # Aggregate across seeds
    print(f"\n{'='*60}\nMULTI-SEED AGGREGATE\n{'='*60}")
    per_seed_fa = []
    per_seed_lat = []
    for seed in SEEDS:
        sr = output["per_seed"].get(str(seed), {}).get(BASELINE_NAME)
        if sr:
            per_seed_fa.append(sr["avg_factual_accuracy"])
            per_seed_lat.append(sr["avg_latency"])
    if per_seed_fa:
        output["multi_seed_summary"] = {
            BASELINE_NAME: {
                "fa_mean": round(mean(per_seed_fa), 4),
                "fa_std": round(stdev(per_seed_fa), 4) if len(per_seed_fa) > 1 else 0.0,
                "fa_per_seed": per_seed_fa,
                "lat_mean": round(mean(per_seed_lat), 2),
                "n_seeds": len(per_seed_fa),
            }
        }
        print(f"  {BASELINE_NAME}: FA = {output['multi_seed_summary'][BASELINE_NAME]['fa_mean']:.4f} "
              f"± {output['multi_seed_summary'][BASELINE_NAME]['fa_std']:.4f} "
              f"(n_seeds={len(per_seed_fa)}, lat={output['multi_seed_summary'][BASELINE_NAME]['lat_mean']:.2f}s)")

    output["wall_clock_s"] = round(time.time() - t_global, 1)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[done] wall={output['wall_clock_s']:.1f}s   saved → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
