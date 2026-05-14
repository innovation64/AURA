"""Per-category paired analysis of AURA_Full vs AURA_NoProbe on RQ2 FA.

Overall AURA_Full vs AURA_NoProbe is null at the seed level (p=0.299, n=3),
but the null hides category heterogeneity. This script splits RQ2 factual
accuracy by the query's `category` field (spatial/social/temporal/memory/
planning) and reports:

  - Per-category seed-level paired t-test (n=3 seeds)
  - Per-category query-level paired t-test (n = queries_per_cat * seeds)
  - Overall query-level paired t-test (n = all queries * seeds, pooled)

Usage:
    python -m scripts.rq2_category_analysis

Reads: evaluation/results/rq2_factual_accuracy_multiseed.json (must have
the per-seed details[] arrays populated; if only summary fields exist,
run scripts/aggregate_rq2_multiseed.py first to verify details are intact).
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "evaluation" / "results" / "rq2_factual_accuracy_multiseed.json"
CATS = ["spatial", "social", "temporal", "memory", "planning"]


def paired_t(diffs: list[float]) -> tuple[float | None, float | None]:
    n = len(diffs)
    if n < 2:
        return None, None
    md = statistics.mean(diffs)
    if len(set(diffs)) <= 1:
        return (float("inf") if md != 0 else 0.0), None
    sd = statistics.stdev(diffs)
    if sd == 0:
        return (float("inf") if md != 0 else 0.0), None
    t = md / (sd / math.sqrt(n))
    try:
        from scipy import stats  # type: ignore

        pv = float(stats.t.sf(abs(t), n - 1) * 2)
    except ImportError:
        pv = None
    return t, pv


def main(path: Path = DEFAULT_PATH) -> int:
    with open(path) as f:
        d = json.load(f)
    seeds = d["seeds"]
    conds = list(d["per_seed"][str(seeds[0])].keys())
    target = "AURA_Full"
    contrast = "AURA_NoProbe"
    assert target in conds and contrast in conds

    # Seed-level per-category means
    per = defaultdict(lambda: defaultdict(list))  # per[cond][cat] = [seed means]
    for s in seeds:
        sk = str(s)
        for c in conds:
            details = d["per_seed"][sk][c].get("details", []) or []
            by_cat: dict[str, list[float]] = defaultdict(list)
            for r in details:
                fa = r.get("factual_accuracy")
                if isinstance(fa, dict) and fa.get("accuracy") is not None:
                    by_cat[r.get("category", "unknown")].append(fa["accuracy"])
            for cat, accs in by_cat.items():
                per[c][cat].append(sum(accs) / len(accs))

    # Display per-category summary across all conditions
    print("Per-Category Mean FA ± Std (across 3 seeds)")
    print("-" * 88)
    header = f'{"Category":<10} | ' + " | ".join(f"{c:<16}" for c in conds)
    print(header)
    print("-" * 88)
    for cat in CATS:
        row = f"{cat:<10} | "
        for c in conds:
            vals = per[c].get(cat, [])
            if vals:
                m = statistics.mean(vals)
                sd = statistics.stdev(vals) if len(vals) >= 2 else 0.0
                row += f"{m:.3f}±{sd:.3f}{'':4} | "
            else:
                row += f'{"n/a":<16} | '
        print(row)

    # Seed-level paired t: AURA_Full vs AURA_NoProbe per category
    print(f"\nSeed-level paired t ({target} vs {contrast}, n=3 seeds)")
    print("-" * 88)
    print(f'{"Category":<10} | {target:>10} | {contrast:>10} | {"Δ":>8} | {"t":>6} | {"p":>8}')
    print("-" * 88)
    for cat in CATS:
        x = per[target].get(cat, [])
        y = per[contrast].get(cat, [])
        if len(x) == len(y) >= 2:
            diffs = [a - b for a, b in zip(x, y)]
            t, pv = paired_t(diffs)
            mx, my = statistics.mean(x), statistics.mean(y)
            pstr = f"{pv:.4f}" if pv is not None else "n/a"
            tstr = f"{t:.2f}" if t is not None and t != float("inf") else "inf"
            star = " *" if pv is not None and pv < 0.05 else ""
            print(f"{cat:<10} | {mx:>10.3f} | {my:>10.3f} | {mx - my:>+8.3f} | {tstr:>6} | {pstr:>8}{star}")

    # Query-level paired t (pool across 3 seeds per query)
    print(f"\nQuery-level paired t ({target} vs {contrast}, per-query diffs pooled across 3 seeds)")
    print("-" * 88)
    print(f'{"Category":<10} | {"n":>4} | {"mean_Δ":>8} | {"std":>7} | {"t":>6} | {"p":>8}')
    print("-" * 88)
    for cat_filter in CATS + [None]:
        diffs = []
        for s in seeds:
            sk = str(s)
            tgt = {r["query_id"]: r for r in d["per_seed"][sk][target].get("details", [])
                   if isinstance(r.get("factual_accuracy"), dict)}
            ctr = {r["query_id"]: r for r in d["per_seed"][sk][contrast].get("details", [])
                   if isinstance(r.get("factual_accuracy"), dict)}
            for qid, fr in tgt.items():
                nr = ctr.get(qid)
                if not nr:
                    continue
                if cat_filter and fr.get("category") != cat_filter:
                    continue
                fa_t = fr["factual_accuracy"].get("accuracy")
                fa_c = nr["factual_accuracy"].get("accuracy")
                if fa_t is None or fa_c is None:
                    continue
                diffs.append(fa_t - fa_c)
        if len(diffs) < 2:
            continue
        md = statistics.mean(diffs)
        sd = statistics.stdev(diffs)
        t, pv = paired_t(diffs)
        label = cat_filter if cat_filter else "OVERALL"
        pstr = f"{pv:.4f}" if pv is not None else "n/a"
        tstr = f"{t:.2f}" if t is not None and t != float("inf") else "inf"
        star = " *" if pv is not None and pv < 0.05 else ""
        print(f"{label:<10} | {len(diffs):>4} | {md:>+8.4f} | {sd:>7.4f} | {tstr:>6} | {pstr:>8}{star}")

    # Also write machine-readable output for paper tables
    out = {}
    out["per_category_seed_level"] = {}
    out["per_category_query_level"] = {}
    for cat in CATS:
        x = per[target].get(cat, [])
        y = per[contrast].get(cat, [])
        if len(x) == len(y) >= 2:
            diffs = [a - b for a, b in zip(x, y)]
            t, pv = paired_t(diffs)
            out["per_category_seed_level"][cat] = {
                f"{target}_mean": round(statistics.mean(x), 4),
                f"{contrast}_mean": round(statistics.mean(y), 4),
                "delta": round(statistics.mean(diffs), 4),
                "t": round(t, 3) if t not in (None, float("inf")) else None,
                "p_two_sided": round(pv, 5) if pv is not None else None,
            }
    for cat_filter in CATS + [None]:
        diffs = []
        for s in seeds:
            sk = str(s)
            tgt = {r["query_id"]: r for r in d["per_seed"][sk][target].get("details", [])
                   if isinstance(r.get("factual_accuracy"), dict)}
            ctr = {r["query_id"]: r for r in d["per_seed"][sk][contrast].get("details", [])
                   if isinstance(r.get("factual_accuracy"), dict)}
            for qid, fr in tgt.items():
                nr = ctr.get(qid)
                if not nr:
                    continue
                if cat_filter and fr.get("category") != cat_filter:
                    continue
                fa_t = fr["factual_accuracy"].get("accuracy")
                fa_c = nr["factual_accuracy"].get("accuracy")
                if fa_t is None or fa_c is None:
                    continue
                diffs.append(fa_t - fa_c)
        if len(diffs) < 2:
            continue
        md, sd = statistics.mean(diffs), statistics.stdev(diffs)
        t, pv = paired_t(diffs)
        label = cat_filter if cat_filter else "overall"
        out["per_category_query_level"][label] = {
            "n": len(diffs),
            "mean_delta": round(md, 4),
            "std": round(sd, 4),
            "t": round(t, 3) if t not in (None, float("inf")) else None,
            "p_two_sided": round(pv, 5) if pv is not None else None,
        }

    out_path = path.parent / "rq2_category_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[write] machine-readable output -> {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
