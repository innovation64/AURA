"""Aggregate RQ2 multi-seed factual accuracy from per-query details.

The upstream run produced per-seed per-condition `details` arrays but failed
to populate the top-level `per_seed[*][cond].avg_factual_accuracy` and
`statistics[cond]` aggregates. This script recomputes both from the raw
`details[].factual_accuracy.accuracy` values and writes them back in-place.

Usage:
    python -m scripts.aggregate_rq2_multiseed [path_to_json]

The input file defaults to evaluation/results/rq2_factual_accuracy_multiseed.json.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "evaluation" / "results" / "rq2_factual_accuracy_multiseed.json"


def paired_t(x: list[float], y: list[float]) -> tuple[float | None, float | None]:
    diffs = [a - b for a, b in zip(x, y)]
    n = len(diffs)
    if n < 2:
        return None, None
    md = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if len(set(diffs)) > 1 else 0.0
    if sd == 0:
        return (float("inf") if md != 0 else 0.0), None
    t = md / (sd / math.sqrt(n))
    try:
        from scipy import stats  # type: ignore

        pv = float(stats.t.sf(abs(t), n - 1) * 2)
    except ImportError:
        pv = None
    return t, pv


def aggregate(path: Path) -> dict:
    with open(path) as f:
        d = json.load(f)

    seeds = d["seeds"]
    conds = list(d["per_seed"][str(seeds[0])].keys())

    # Fill per_seed[*][cond] top-level fields from each cond's inner summary,
    # or fall back to recomputing from details.
    per_seed_fa_means = {c: [] for c in conds}
    per_seed_fa_accs_for_ttest = {c: [] for c in conds}
    per_seed_latency_means = {c: [] for c in conds}

    for s in seeds:
        sk = str(s)
        for c in conds:
            entry = d["per_seed"][sk][c]
            details = entry.get("details", []) or []
            accs = []
            lats = []
            for r in details:
                fa = r.get("factual_accuracy")
                if isinstance(fa, dict) and "accuracy" in fa and fa["accuracy"] is not None:
                    accs.append(fa["accuracy"])
                if r.get("latency") is not None:
                    lats.append(r["latency"])

            if accs:
                mean_fa = sum(accs) / len(accs)
                entry["avg_factual_accuracy"] = round(mean_fa, 4)
                entry["num_valid"] = len(accs)
                per_seed_fa_means[c].append(mean_fa)
                per_seed_fa_accs_for_ttest[c].append(accs)
            else:
                entry["avg_factual_accuracy"] = None
                entry["num_valid"] = 0

            if lats:
                mean_lat = sum(lats) / len(lats)
                entry["avg_latency"] = round(mean_lat, 4)
                per_seed_latency_means[c].append(mean_lat)

            # Context utilization passthrough from inner summary
            inner = entry.get("summary") or {}
            if inner.get("avg_context_utilization") is not None:
                entry["avg_context_utilization"] = inner["avg_context_utilization"]

    # Rebuild top-level statistics
    stats_out: dict = {}
    for c in conds:
        vals = per_seed_fa_means[c]
        stats_out[c] = {
            "mean": round(statistics.mean(vals), 4) if vals else 0.0,
            "std": round(statistics.stdev(vals), 4) if len(vals) >= 2 else 0.0,
            "values": [round(v, 4) for v in vals],
            "n_seeds": len(vals),
            "latency_mean": (
                round(statistics.mean(per_seed_latency_means[c]), 4)
                if per_seed_latency_means[c] else None
            ),
        }

    d["statistics"] = stats_out

    # Paired tests vs AURA_Full
    if "AURA_Full" in conds:
        ref = per_seed_fa_means["AURA_Full"]
        pairs = {}
        for c in conds:
            if c == "AURA_Full":
                continue
            t, pv = paired_t(ref, per_seed_fa_means[c])
            ref_mean = statistics.mean(ref) if ref else 0.0
            other_mean = statistics.mean(per_seed_fa_means[c]) if per_seed_fa_means[c] else 0.0
            rel = ref_mean / other_mean if other_mean > 0 else None
            pairs[c] = {
                "delta": round(ref_mean - other_mean, 4),
                "ratio": round(rel, 2) if rel is not None else None,
                "t": round(t, 3) if t is not None and t != float("inf") else None,
                "p_two_sided": round(pv, 5) if pv is not None else None,
            }
        d["paired_tests_vs_AURA_Full"] = pairs

    return d


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if not path.exists():
        print(f"ERR: not found: {path}", file=sys.stderr)
        return 1

    backup = path.with_suffix(path.suffix + ".pre-aggregate.bak")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
        print(f"[backup] {backup.name}")

    d = aggregate(path)
    with open(path, "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)

    print(f"[write] {path.name}")
    print("\n=== RQ2 Multi-Seed Summary ===")
    for c, s in d["statistics"].items():
        print(f"  {c:<18} mean={s['mean']:.4f}  std={s['std']:.4f}  values={s['values']}")

    if "paired_tests_vs_AURA_Full" in d:
        print("\n=== Paired t-test (seed-level, n=3) vs AURA_Full ===")
        for c, stats_c in d["paired_tests_vs_AURA_Full"].items():
            print(f"  vs {c:<16}  Δ={stats_c['delta']:+.4f}  ratio={stats_c['ratio']}×  "
                  f"t={stats_c['t']}  p={stats_c['p_two_sided']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
