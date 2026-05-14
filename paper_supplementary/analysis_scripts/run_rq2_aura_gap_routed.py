"""RQ2 mechanism baseline: AURA gap-routed (LLMIntentInferrer actually firing).

The headline AURA_Full numbers in evaluation/results/rq2_factual_accuracy_multiseed.json
were produced by the town-server TownProbeRunner with a fixed
`probe_max_steps`, which never exercises the LLM IntentInferrer the
paper attributes the gain to. This script runs `aura_gap_routed_chat`
on the same 50 chat queries × 3 seeds, so the gap → budget →
probe-selection loop is what FA / latency / probe-count are
measured against.

Output: evaluation/results/rq2_aura_gap_routed_multiseed.json
Schema mirrors run_rq2_fixed_probe.py with extra `gap`, `recommended_probes`,
`actual_probes` per query.
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
from evaluation.baselines import aura_gap_routed_chat  # noqa: E402
from evaluation.llm_judge import judge_factual_accuracy, judge_context_utilization  # noqa: E402
from evaluation.run_experiments import AURAClient  # noqa: E402

SEEDS = [
    int(s.strip())
    for s in os.environ.get("SEEDS", "42,123,456").split(",")
    if s.strip()
]
QUERIES_PATH = ROOT / "evaluation" / "data" / "chat_queries.json"
OUTPUT_PATH = ROOT / "evaluation" / "results" / "rq2_aura_gap_routed_multiseed.json"
SERVER = os.environ.get("AURA_SERVER", "http://127.0.0.1:7861")
N_QUERIES = int(os.environ.get("N_QUERIES", "50"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "10"))
RESUME = os.environ.get("RESUME", "0").lower() in {"1", "true", "yes"}

BASELINE_NAME = "AURA_GapRouted"
BASELINE_FN = aura_gap_routed_chat


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
    probe_values = [r.get("tool_calls", 0) for r in details]
    gap_values = [r["gap"] for r in details if r.get("gap") is not None]
    n_errors = sum(1 for r in details if "error" in r)
    return {
        "avg_factual_accuracy": round(mean(fa_values), 4) if fa_values else 0.0,
        "avg_context_utilization": round(mean(cu_values), 4) if cu_values else 0.0,
        "avg_latency": round(mean(lat_values), 3) if lat_values else 0.0,
        "avg_probes": round(mean(probe_values), 3) if probe_values else 0.0,
        "avg_gap": round(mean(gap_values), 3) if gap_values else 0.0,
        "num_valid": len(fa_values),
        "num_errors": n_errors,
    }


def main():
    config = EvalConfig(aura_server=SERVER)
    client = AURAClient(SERVER)

    with open(QUERIES_PATH) as f:
        queries = json.load(f)["queries"][:N_QUERIES]
    print(f"[setup] {len(queries)} queries × {len(SEEDS)} seeds × 1 baseline ({BASELINE_NAME})")

    if RESUME and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            output = json.load(f)
        output.setdefault("per_seed", {})
        output["seeds"] = SEEDS
        output["n_queries"] = len(queries)
        output["baseline"] = BASELINE_NAME
        print(f"[resume] loaded partial output from {OUTPUT_PATH}")
    else:
        output = {
            "schema_version": "1.0",
            "seeds": SEEDS,
            "n_queries": len(queries),
            "baseline": BASELINE_NAME,
            "per_seed": {},
        }

    t_global = time.time()

    for seed in SEEDS:
        existing = output.get("per_seed", {}).get(str(seed), {}).get(BASELINE_NAME)
        if existing and existing.get("num_valid") == len(queries):
            print(f"\n[seed {seed}] already complete ({len(queries)} queries); skipping")
            continue

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
                    "gap": bl_result.get("gap"),
                    "recommended_probes": bl_result.get("recommended_probes", []),
                    "actual_probes": bl_result.get("actual_probes", []),
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
                "avg_probes": summary["avg_probes"],
                "avg_gap": summary["avg_gap"],
                "num_valid": summary["num_valid"],
            }
        }
        print(f"  {BASELINE_NAME} seed {seed}: FA={summary['avg_factual_accuracy']} "
              f"CU={summary['avg_context_utilization']} "
              f"probes={summary['avg_probes']} "
              f"gap={summary['avg_gap']} "
              f"lat={summary['avg_latency']}s "
              f"errors={summary['num_errors']}")

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[seed {seed}] partial saved → {OUTPUT_PATH}")

    print(f"\n{'='*60}\nMULTI-SEED AGGREGATE\n{'='*60}")
    per_seed_fa, per_seed_lat, per_seed_probes, per_seed_gap = [], [], [], []
    for seed in SEEDS:
        sr = output["per_seed"].get(str(seed), {}).get(BASELINE_NAME)
        if sr:
            per_seed_fa.append(sr["avg_factual_accuracy"])
            per_seed_lat.append(sr["avg_latency"])
            per_seed_probes.append(sr["avg_probes"])
            per_seed_gap.append(sr["avg_gap"])
    if per_seed_fa:
        output["multi_seed_summary"] = {
            BASELINE_NAME: {
                "fa_mean": round(mean(per_seed_fa), 4),
                "fa_std": round(stdev(per_seed_fa), 4) if len(per_seed_fa) > 1 else 0.0,
                "fa_per_seed": per_seed_fa,
                "lat_mean": round(mean(per_seed_lat), 2),
                "probes_mean": round(mean(per_seed_probes), 3),
                "gap_mean": round(mean(per_seed_gap), 3),
                "n_seeds": len(per_seed_fa),
            }
        }
        s = output["multi_seed_summary"][BASELINE_NAME]
        print(f"  {BASELINE_NAME}: FA = {s['fa_mean']:.4f} ± {s['fa_std']:.4f} "
              f"(n_seeds={len(per_seed_fa)}, probes={s['probes_mean']:.2f}, "
              f"gap={s['gap_mean']:.3f}, lat={s['lat_mean']:.2f}s)")

    output["wall_clock_s"] = round(time.time() - t_global, 1)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[done] wall={output['wall_clock_s']:.1f}s   saved → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
