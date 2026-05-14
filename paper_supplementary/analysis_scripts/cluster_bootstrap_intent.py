"""Cluster-bootstrap CIs for RQ-Intent paired contrasts.

The 75 paired comparisons in rq2_implicit_intent_v2_with_second_order.json
come from 25 queries x 3 seeds. Treating them as 75 independent pairs
overstates power because rows within the same query are correlated (same
implicit_need, same target, similar surface form). This script reports:

  1. seed-level paired t-test                           (N=3 — primary)
  2. query-level paired t-test on per-query mean        (N=25 — primary)
  3. cluster bootstrap CI clustered on query_id         (N_eff=25)
  4. naive query-level pooled t-test (N=75)             (for back-compat)

For each contrast x metric, prints mean delta, query-level paired t/p,
and 95% cluster-bootstrap CI. Writes a JSON sidecar.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _paired_t(diffs: List[float]) -> Tuple[Optional[float], Optional[float], int]:
    valid = [d for d in diffs if d is not None and not math.isnan(d)]
    n = len(valid)
    if n < 2:
        return None, None, n
    md = statistics.mean(valid)
    sd = statistics.stdev(valid) if len(set(valid)) > 1 else 0.0
    if sd == 0:
        return (float("inf") if md != 0 else 0.0), None, n
    t = md / (sd / math.sqrt(n))
    try:
        from scipy import stats  # type: ignore
        p = float(stats.t.sf(abs(t), n - 1) * 2)
    except ImportError:
        p = None
    return t, p, n


def _cluster_bootstrap_ci(
    per_query_diffs: Dict[int, List[float]],
    n_iter: int = 5000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> Tuple[float, float, float]:
    """Resample query_ids with replacement. Within each chosen query,
    keep all its observations (so seeds within a query move together).
    Returns (mean, low, high) at the (1-alpha) level.
    """
    rng = random.Random(rng_seed)
    qids = list(per_query_diffs.keys())
    if not qids:
        return float("nan"), float("nan"), float("nan")

    overall = [d for diffs in per_query_diffs.values() for d in diffs]
    point_mean = statistics.mean(overall) if overall else float("nan")

    boots = []
    for _ in range(n_iter):
        sample = rng.choices(qids, k=len(qids))
        flat = []
        for qid in sample:
            flat.extend(per_query_diffs[qid])
        if flat:
            boots.append(sum(flat) / len(flat))
    boots.sort()
    if not boots:
        return point_mean, float("nan"), float("nan")
    lo = boots[int((alpha / 2) * len(boots))]
    hi = boots[int((1 - alpha / 2) * len(boots))]
    return point_mean, lo, hi


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="RQ-Intent multiseed result JSON")
    p.add_argument("--out", default=None,
                   help="Output JSON path (default: alongside input with _cluster_bootstrap suffix)")
    p.add_argument("--n-iter", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    seeds = sorted(int(s) for s in data["per_seed"].keys())
    conditions = list(data["per_seed"][str(seeds[0])].keys())
    contrasts = [("tom", "literal"), ("tom", "no_intent"), ("no_intent", "literal")]
    metrics = ("literal_score", "implicit_score")

    out: Dict[str, Any] = {
        "meta": {
            "input": str(Path(args.input).name),
            "seeds": seeds,
            "conditions": conditions,
            "n_iter": args.n_iter,
        },
        "stats": {},
    }

    print(f"Input: {args.input}")
    print(f"Seeds: {seeds}, conditions: {conditions}\n")

    for metric in metrics:
        print(f"=== {metric} ===")
        out["stats"][metric] = {}
        for ca, cb in contrasts:
            # per_query_diffs: {qid: [diff_seed0, diff_seed1, ...]}
            per_query: Dict[int, List[float]] = defaultdict(list)
            seed_means: List[float] = []  # seed-level paired diff means

            for s in seeds:
                a_rows = {r["query_id"]: r for r in data["per_seed"][str(s)][ca]}
                b_rows = {r["query_id"]: r for r in data["per_seed"][str(s)][cb]}
                seed_diffs = []
                for qid, ra in a_rows.items():
                    rb = b_rows.get(qid)
                    if not rb:
                        continue
                    va, vb = ra.get(metric), rb.get(metric)
                    if va is None or vb is None:
                        continue
                    d = va - vb
                    per_query[qid].append(d)
                    seed_diffs.append(d)
                if seed_diffs:
                    seed_means.append(statistics.mean(seed_diffs))

            # Query-level paired t: collapse seeds within each query
            query_means = [statistics.mean(v) for v in per_query.values() if v]
            t_q, p_q, n_q = _paired_t(query_means)

            # Seed-level paired t: collapse queries within each seed
            t_s, p_s, n_s = _paired_t(seed_means)

            # Naive pooled (for back-compat)
            naive = [d for v in per_query.values() for d in v]
            t_n, p_n, n_n = _paired_t(naive)

            # Cluster bootstrap on query_id
            mean_b, lo_b, hi_b = _cluster_bootstrap_ci(
                dict(per_query), n_iter=args.n_iter, rng_seed=args.seed,
            )

            row = {
                "n_pairs_naive": n_n,
                "naive_pooled": {
                    "mean_delta": round(statistics.mean(naive), 4) if naive else None,
                    "t": round(t_n, 3) if t_n is not None and t_n != float("inf") else None,
                    "p_two_sided": round(p_n, 5) if p_n is not None else None,
                },
                "query_level": {
                    "n": n_q,
                    "mean_delta": round(mean_b, 4),
                    "t": round(t_q, 3) if t_q is not None and t_q != float("inf") else None,
                    "p_two_sided": round(p_q, 5) if p_q is not None else None,
                },
                "seed_level": {
                    "n": n_s,
                    "mean_delta": round(statistics.mean(seed_means), 4) if seed_means else None,
                    "t": round(t_s, 3) if t_s is not None and t_s != float("inf") else None,
                    "p_two_sided": round(p_s, 5) if p_s is not None else None,
                },
                "cluster_bootstrap_query": {
                    "mean": round(mean_b, 4),
                    "ci_lo": round(lo_b, 4),
                    "ci_hi": round(hi_b, 4),
                    "alpha": 0.05,
                    "n_iter": args.n_iter,
                },
            }
            out["stats"][metric][f"{ca}_vs_{cb}"] = row

            print(
                f"  {ca}_vs_{cb:<10}  Δ={row['naive_pooled']['mean_delta']:+.4f}  "
                f"q-level n={n_q} t={row['query_level']['t']} p={row['query_level']['p_two_sided']}  "
                f"seed-level n={n_s} t={row['seed_level']['t']} p={row['seed_level']['p_two_sided']}  "
                f"cluster-boot 95% CI=[{lo_b:+.4f}, {hi_b:+.4f}]"
            )
        print()

    out_path = Path(args.out) if args.out else (
        Path(args.input).parent / (Path(args.input).stem + "_cluster_bootstrap.json")
    )
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[write] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
