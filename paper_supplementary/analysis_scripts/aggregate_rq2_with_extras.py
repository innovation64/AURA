"""Merge RQ2 main multi-seed (5 conditions) + extra baselines (Reflexion +
Plan-and-Solve) into a single unified table with both lenient FA and strict
precision rescore for every condition.

Inputs:
  evaluation/results/rq2_factual_accuracy_multiseed.json   (5 conditions)
  evaluation/results/rq2_extra_baselines_multiseed.json    (Reflexion + PnS)

Output:
  evaluation/results/rq2_unified_multiseed.json
  printout: 7-condition table (lenient FA + strict precision + paired contrasts)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "evaluation" / "results" / "rq2_factual_accuracy_multiseed.json"
EXTRA = ROOT / "evaluation" / "results" / "rq2_extra_baselines_multiseed.json"
OUT = ROOT / "evaluation" / "results" / "rq2_unified_multiseed.json"


def strict_precision(fa: dict) -> float | None:
    c = fa.get("correct_claims", 0)
    x = fa.get("contradicted_claims", 0)
    if c + x == 0:
        return None
    return c / (c + x)


def is_hallucination(fa: dict) -> bool:
    return fa.get("contradicted_claims", 0) > 0


def is_perfect(fa: dict) -> bool:
    return fa.get("contradicted_claims", 0) == 0 and fa.get("correct_claims", 0) >= 1


def paired_t(deltas: list[float]) -> tuple[float, float]:
    n = len(deltas)
    if n < 2:
        return (0.0, 1.0)
    m = mean(deltas)
    s = stdev(deltas) if n > 1 else 0.0
    if s == 0:
        return (float("inf") if m != 0 else 0.0, 0.0 if m != 0 else 1.0)
    t = m / (s / math.sqrt(n))
    try:
        from scipy.stats import t as t_dist
        p = 2 * (1 - t_dist.cdf(abs(t), df=n - 1))
    except ImportError:
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return (t, p)


def collect(per_seed_obj: dict, target_conditions: list[str] | None = None) -> dict:
    """Pull (cond, qid, seed) → metrics from a per_seed JSON dict."""
    result: dict[str, dict[int, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    for seed_str, conds in per_seed_obj.items():
        try:
            seed = int(seed_str)
        except ValueError:
            continue
        for cond, cd in conds.items():
            if target_conditions is not None and cond not in target_conditions:
                continue
            for r in cd.get("details", []):
                fa = r.get("factual_accuracy")
                if not isinstance(fa, dict) or "precision" not in fa:
                    continue
                qid = r.get("query_id")
                result[cond][qid][seed] = {
                    "lenient_acc": fa.get("accuracy"),
                    "strict_p": strict_precision(fa),
                    "halluc": 1 if is_hallucination(fa) else 0,
                    "perfect": 1 if is_perfect(fa) else 0,
                    "category": r.get("category"),
                    "n_correct": fa.get("correct_claims", 0),
                    "n_contradicted": fa.get("contradicted_claims", 0),
                    "latency": r.get("latency"),
                }
    return result


def main() -> int:
    with open(MAIN) as f:
        main_d = json.load(f)
    with open(EXTRA) as f:
        extra_d = json.load(f)

    main_data = collect(main_d.get("per_seed", {}))
    extra_data = collect(
        extra_d.get("per_seed", {}),
        target_conditions=["Reflexion", "Plan_and_Solve"],
    )

    # Merge — extra conditions are independent, just add them
    all_data: dict[str, dict[int, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    for cond, qmap in main_data.items():
        for qid, smap in qmap.items():
            for seed, m in smap.items():
                all_data[cond][qid][seed] = m
    for cond, qmap in extra_data.items():
        for qid, smap in qmap.items():
            for seed, m in smap.items():
                all_data[cond][qid][seed] = m

    # Per-condition aggregates (cell-level)
    cond_agg: dict[str, dict] = {}
    cond_query_avg: dict[str, dict[int, dict]] = defaultdict(dict)
    for cond, qmap in all_data.items():
        sp_cells, sp_q, len_cells, h_cells, p_cells, lat_cells = [], [], [], [], [], []
        for qid, smap in qmap.items():
            sp_q_seeds = []
            len_q_seeds = []
            for seed, m in smap.items():
                if m["strict_p"] is not None:
                    sp_cells.append(m["strict_p"])
                    sp_q_seeds.append(m["strict_p"])
                if m["lenient_acc"] is not None:
                    len_cells.append(m["lenient_acc"])
                    len_q_seeds.append(m["lenient_acc"])
                h_cells.append(m["halluc"])
                p_cells.append(m["perfect"])
                if m["latency"] is not None:
                    lat_cells.append(m["latency"])
            cond_query_avg[cond][qid] = {
                "strict_p_mean": mean(sp_q_seeds) if sp_q_seeds else None,
                "lenient_mean": mean(len_q_seeds) if len_q_seeds else None,
            }
        cond_agg[cond] = {
            "strict_precision_mean": mean(sp_cells) if sp_cells else 0.0,
            "strict_precision_std": stdev(sp_cells) if len(sp_cells) > 1 else 0.0,
            "lenient_FA_mean": mean(len_cells) if len_cells else 0.0,
            "hallucination_rate": mean(h_cells) if h_cells else 0.0,
            "perfect_rate": mean(p_cells) if p_cells else 0.0,
            "n_cells": len(h_cells),
            "mean_latency_s": mean(lat_cells) if lat_cells else 0.0,
        }

    # Paired contrasts: AURA_Full vs each baseline (query-level mean across seeds)
    contrasts = []
    if "AURA_Full" in cond_query_avg:
        for ref in ["Vanilla_LLM", "Static_Context", "ReAct", "Reflexion",
                    "Plan_and_Solve", "AURA_NoProbe"]:
            if ref not in cond_query_avg:
                continue
            common = sorted(set(cond_query_avg["AURA_Full"]) & set(cond_query_avg[ref]))
            deltas_sp, deltas_len = [], []
            for qid in common:
                f = cond_query_avg["AURA_Full"][qid]
                r = cond_query_avg[ref][qid]
                if f["strict_p_mean"] is not None and r["strict_p_mean"] is not None:
                    deltas_sp.append(f["strict_p_mean"] - r["strict_p_mean"])
                if f["lenient_mean"] is not None and r["lenient_mean"] is not None:
                    deltas_len.append(f["lenient_mean"] - r["lenient_mean"])
            t_sp, p_sp = paired_t(deltas_sp)
            t_len, p_len = paired_t(deltas_len)
            contrasts.append({
                "vs": ref,
                "n_pairs_strict": len(deltas_sp),
                "delta_strict_p": mean(deltas_sp) if deltas_sp else 0.0,
                "p_strict": p_sp,
                "delta_lenient_FA": mean(deltas_len) if deltas_len else 0.0,
                "p_lenient": p_len,
            })

    output = {
        "schema_version": "1.0",
        "inputs": {"main": str(MAIN), "extra": str(EXTRA)},
        "per_condition": cond_agg,
        "paired_AURA_Full_vs": contrasts,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print
    print("=" * 86)
    print("RQ2 UNIFIED 7-condition TABLE  (lenient FA + strict precision)")
    print("=" * 86)
    order = ["Vanilla_LLM", "Static_Context", "ReAct", "Reflexion",
             "Plan_and_Solve", "AURA_NoProbe", "AURA_Full"]
    print(f"\n{'Condition':<18s} {'StrictP':>10s} {'LenientFA':>10s} "
          f"{'Halluc%':>9s} {'Perfect%':>9s} {'Lat(s)':>9s} {'N':>5s}")
    print("-" * 86)
    for cond in order:
        if cond not in cond_agg:
            continue
        a = cond_agg[cond]
        print(f"{cond:<18s} "
              f"{a['strict_precision_mean']:>10.3f} "
              f"{a['lenient_FA_mean']:>10.3f} "
              f"{a['hallucination_rate']*100:>8.1f}% "
              f"{a['perfect_rate']*100:>8.1f}% "
              f"{a['mean_latency_s']:>9.2f} "
              f"{a['n_cells']:>5d}")

    print(f"\n--- Paired contrasts (AURA_Full vs ...) ---")
    print(f"{'Ref':<18s} {'ΔStrictP':>10s} {'p':>10s} {'ΔLenient':>10s} {'p':>10s} {'n_pairs':>8s}")
    print("-" * 76)
    for c in contrasts:
        print(f"{c['vs']:<18s} "
              f"{c['delta_strict_p']:>+10.3f} "
              f"{c['p_strict']:>10.4f} "
              f"{c['delta_lenient_FA']:>+10.3f} "
              f"{c['p_lenient']:>10.4f} "
              f"{c['n_pairs_strict']:>8d}")

    print(f"\nSaved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
