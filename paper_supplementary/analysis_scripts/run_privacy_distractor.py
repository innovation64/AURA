"""Privacy-sensitive distractor slice runner.

Runs each baseline against `evaluation/data/privacy_distractor_queries.json`,
where every query carries a `forbidden_tools` list specifying which tools
would over-disclose for that particular factual question. Records FA
(via the same judge as RQ2) and a per-query `privacy_violation` flag
(1 if any forbidden tool was actually fired, 0 otherwise).

Output: evaluation/results/rq2_privacy_distractor_multiseed.json
Schema mirrors run_rq2_extra_baselines.py with extra
`tools_actually_called` and `privacy_violation` per query record.

Why moderate + RQ2 town: reuses the same SHARED_TOOLS and judge so
results are directly comparable to existing RQ2 numbers; forbidden_tools
flag the disclosure-heavy public tools (memories / events / plans) that
unconditional probing must touch but a gap-routed controller should
skip.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.config import EvalConfig  # noqa: E402
from evaluation.baselines import (  # noqa: E402
    vanilla_llm_chat,
    static_context_chat,
    react_chat,
    plan_and_solve_chat,
    fixed_probe_chat,
    aura_gap_routed_chat,
)
from evaluation.llm_judge import judge_factual_accuracy  # noqa: E402
from evaluation.run_experiments import AURAClient  # noqa: E402

SEEDS = [
    int(s.strip())
    for s in os.environ.get("SEEDS", "42,123,456").split(",")
    if s.strip()
]
QUERIES_PATH = ROOT / "evaluation" / "data" / "privacy_distractor_queries.json"
OUTPUT_PATH = ROOT / "evaluation" / "results" / "rq2_privacy_distractor_multiseed.json"
SERVER = os.environ.get("AURA_SERVER", "http://127.0.0.1:7861")
N_QUERIES = int(os.environ.get("N_QUERIES", "0"))  # 0 = all
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "10"))
RESUME = os.environ.get("RESUME", "0").lower() in {"1", "true", "yes"}

# Baselines to run. Skipping Reflexion (env_context = {} so we can't
# attribute tool calls cleanly) and the server-side AURA_Full (uses
# TownProbeRunner, not SHARED_TOOLS). Filter via env BASELINES if needed.
ALL_BASELINES: List[tuple] = [
    ("Vanilla_LLM",    lambda cfg, name, q, st: vanilla_llm_chat(cfg, name, q)),
    ("Static_Context", static_context_chat),
    ("ReAct",          react_chat),
    ("Plan_and_Solve", plan_and_solve_chat),
    ("Fixed_Probe",    fixed_probe_chat),
    ("AURA_GapRouted", aura_gap_routed_chat),
]
_filter = os.environ.get("BASELINES", "").strip()
BASELINES = (
    [(n, fn) for n, fn in ALL_BASELINES if n in _filter.split(",")] if _filter else ALL_BASELINES
)


def tools_called(bl_result: Dict[str, Any]) -> List[str]:
    """Best-effort extraction of the tool-name set actually fired.

    AURA_GapRouted records `actual_probes` directly. For everything else,
    `env_context` keys ARE the tool names (per baselines.py convention).
    """
    if "actual_probes" in bl_result and isinstance(bl_result["actual_probes"], list):
        return list(bl_result["actual_probes"])
    ctx = bl_result.get("env_context") or {}
    if isinstance(ctx, dict):
        return [k for k in ctx.keys() if isinstance(k, str)]
    return []


def aggregate(details: List[Dict[str, Any]]) -> Dict[str, float]:
    fa_values = [
        r["factual_accuracy"]["accuracy"]
        for r in details
        if isinstance(r.get("factual_accuracy"), dict) and "accuracy" in r["factual_accuracy"]
    ]
    viol_values = [r["privacy_violation"] for r in details if "privacy_violation" in r]
    probe_values = [r.get("tool_calls", 0) for r in details]
    lat_values = [r["latency"] for r in details if "latency" in r]
    n_errors = sum(1 for r in details if "error" in r)
    return {
        "avg_factual_accuracy": round(mean(fa_values), 4) if fa_values else 0.0,
        "privacy_violation_rate": round(mean(viol_values), 4) if viol_values else 0.0,
        "avg_probes": round(mean(probe_values), 3) if probe_values else 0.0,
        "avg_latency": round(mean(lat_values), 3) if lat_values else 0.0,
        "num_valid": len(fa_values),
        "num_errors": n_errors,
    }


def main() -> int:
    config = EvalConfig(aura_server=SERVER)
    client = AURAClient(SERVER)

    with open(QUERIES_PATH) as f:
        spec = json.load(f)
    queries = spec["queries"]
    if N_QUERIES:
        queries = queries[:N_QUERIES]
    print(f"[setup] {len(queries)} privacy queries × {len(SEEDS)} seeds × "
          f"{len(BASELINES)} baselines = {len(queries) * len(SEEDS) * len(BASELINES)} runs")
    print(f"[setup] baselines: {[n for n, _ in BASELINES]}")

    if RESUME and OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            output = json.load(f)
        output.setdefault("per_seed", {})
        print(f"[resume] loaded partial output from {OUTPUT_PATH}")
    else:
        output = {
            "schema_version": "1.0",
            "seeds": SEEDS,
            "n_queries": len(queries),
            "baselines": [name for name, _ in BASELINES],
            "moderate_default_forbidden": spec.get("moderate_default_forbidden", []),
            "per_seed": {},
        }

    t_global = time.time()

    for seed in SEEDS:
        seed_block = output["per_seed"].setdefault(str(seed), {})
        # Skip seed entirely if every baseline already complete
        all_done = all(
            seed_block.get(name, {}).get("num_valid") == len(queries)
            for name, _ in BASELINES
        )
        if all_done:
            print(f"\n[seed {seed}] all baselines complete; skipping")
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
        snapshots: List[Dict[str, Any]] = []
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

        for bl_name, bl_fn in BASELINES:
            existing = seed_block.get(bl_name, {})
            if existing.get("num_valid") == len(queries):
                print(f"  [seed {seed}] {bl_name}: already complete, skipping")
                continue

            print(f"\n[seed {seed}] running {bl_name}...")
            details: List[Dict[str, Any]] = []
            t0 = time.time()
            for qi, q in enumerate(queries):
                if qi % 10 == 0:
                    elapsed = time.time() - t0
                    print(f"  q{qi}/{len(queries)}  elapsed={elapsed:.0f}s")
                snap = snapshots[qi]
                agent_name = snap["agent"]
                gt_state = snap["gt_state"]
                forbidden = set(q.get("forbidden_tools", []))
                try:
                    bl_result = bl_fn(config, agent_name, q["query"], gt_state)
                    response = bl_result.get("response", "")
                    latency = bl_result.get("latency", 0.0)
                    fa = judge_factual_accuracy(config, q["query"], gt_state, response, agent_name)
                    fired = tools_called(bl_result)
                    violations = [t for t in fired if t in forbidden]
                    details.append({
                        "query_id": q["id"],
                        "category": q["category"],
                        "query": q["query"],
                        "agent": agent_name,
                        "response": response[:600],
                        "latency": round(latency, 3),
                        "factual_accuracy": fa,
                        "tool_calls": bl_result.get("tool_calls", 0),
                        "tools_actually_called": fired,
                        "forbidden_tools": list(forbidden),
                        "violations": violations,
                        "privacy_violation": 1 if violations else 0,
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

            summary = aggregate(details)
            seed_block[bl_name] = {
                "summary": summary,
                "details": details,
                **{k: summary[k] for k in (
                    "avg_factual_accuracy", "privacy_violation_rate",
                    "avg_probes", "avg_latency", "num_valid",
                )},
            }
            print(f"  {bl_name} seed {seed}: FA={summary['avg_factual_accuracy']} "
                  f"viol_rate={summary['privacy_violation_rate']} "
                  f"probes={summary['avg_probes']} "
                  f"lat={summary['avg_latency']}s "
                  f"errors={summary['num_errors']}")

            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_PATH, "w") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"[seed {seed}] partial saved → {OUTPUT_PATH}")

    print(f"\n{'='*60}\nMULTI-SEED AGGREGATE\n{'='*60}")
    multi_seed_summary: Dict[str, Dict[str, float]] = {}
    for bl_name, _ in BASELINES:
        per_seed_fa: List[float] = []
        per_seed_viol: List[float] = []
        per_seed_probes: List[float] = []
        per_seed_lat: List[float] = []
        for seed in SEEDS:
            sr = output["per_seed"].get(str(seed), {}).get(bl_name)
            if sr:
                per_seed_fa.append(sr["avg_factual_accuracy"])
                per_seed_viol.append(sr["privacy_violation_rate"])
                per_seed_probes.append(sr["avg_probes"])
                per_seed_lat.append(sr["avg_latency"])
        if per_seed_fa:
            multi_seed_summary[bl_name] = {
                "fa_mean": round(mean(per_seed_fa), 4),
                "fa_std": round(stdev(per_seed_fa), 4) if len(per_seed_fa) > 1 else 0.0,
                "viol_rate_mean": round(mean(per_seed_viol), 4),
                "viol_rate_std": round(stdev(per_seed_viol), 4) if len(per_seed_viol) > 1 else 0.0,
                "probes_mean": round(mean(per_seed_probes), 3),
                "lat_mean": round(mean(per_seed_lat), 2),
                "n_seeds": len(per_seed_fa),
            }
            s = multi_seed_summary[bl_name]
            print(f"  {bl_name:<16} FA={s['fa_mean']:.4f}±{s['fa_std']:.4f}  "
                  f"viol_rate={s['viol_rate_mean']:.4f}±{s['viol_rate_std']:.4f}  "
                  f"probes={s['probes_mean']:.2f}  lat={s['lat_mean']:.2f}s")

    output["multi_seed_summary"] = multi_seed_summary
    output["wall_clock_s"] = round(time.time() - t_global, 1)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[done] wall={output['wall_clock_s']:.1f}s   saved → {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
