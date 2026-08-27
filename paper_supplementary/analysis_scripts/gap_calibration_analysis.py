"""Gap-score calibration analysis for the ACL ARR rebuttal.

Question: does a higher IntentFrame.gap score (the scalar AURA's ToM module
uses to decide whether/how much to probe) actually predict more marginal
benefit from probing?

No new LLM calls: reuses existing per-query judge scores from
`evaluation/results/rq_intent_v2_multiseed.json` (the full 100-query,
4-scene, 3-seed multiseed run: scenes A_cafe_morning, B_library_afternoon,
C_garden_evening, D_postevent_night; conditions literal, no_intent, tom,
fixed_probe, oracle_intent; model gpt-4o-mini).

This supersedes `scripts/analyze_adaptive_budget.py`, which only reads
`evaluation/results/rq2_implicit_intent_v2_with_second_order.json` — a
SINGLE-SCENE subset (scene "A_cafe_morning" / Sunrise Cafe only, 25 queries x
3 seeds = 75 rows per condition, all 5 subcategories represented but only
5 queries each). We deliberately use the broader 4-scene file here for full
benchmark coverage; the narrower file is kept as a fallback and its scope is
reported honestly if the broader file is unavailable.

Benefit measure (per query, per seed):
    benefit = implicit_score(tom) - implicit_score(no_intent)
i.e. how much the ToM-probing condition outscored the no-probing baseline
on the SAME query, joined on (seed, query_id).

Gap value: taken from the "tom" row (gap is only populated in that
condition in this dataset — it is the IntentFrame.gap the agent computed
before deciding whether to probe).

Binning: the observed gap values are highly discretized (only 6 distinct
values in {0.4, 0.5, 0.6, 0.65, 0.7, 0.8} in the full-coverage file), so we
use fixed-width 0.1 bins over the observed [0.4, 0.8] range rather than
quantile bins (quantile bins would arbitrarily split ties at 0.5/0.6/0.7
between adjacent bins). This gives 4 bins, each with a healthy n.

Outputs:
  - printed table to stdout
  - evaluation/results/gap_calibration_analysis.json
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "evaluation" / "results"

PRIMARY_SRC = RESULTS_DIR / "rq_intent_v2_multiseed.json"
FALLBACK_SRC = RESULTS_DIR / "rq2_implicit_intent_v2_with_second_order.json"
OUT_PATH = RESULTS_DIR / "gap_calibration_analysis.json"

N_BINS = 4
BIN_WIDTH = 0.1
BIN_LO = 0.4  # observed floor; adjusted at runtime if data differs


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def spearman(xs, ys):
    """Spearman rank correlation with average ranks for ties (no scipy dep)."""
    n = len(xs)
    if n < 2:
        return None

    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg_rank
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    return pearson(rx, ry)


def load_pairs(path: Path, scene_field_present: bool):
    """Return list of dicts: seed, query_id, subcategory, scene, gap, benefit."""
    with open(path) as f:
        d = json.load(f)

    pairs = []
    scenes_seen = set()
    subcats_seen = set()
    for seed, conds in d["per_seed"].items():
        tom_by_id = {r["query_id"]: r for r in conds["tom"]}
        ni_by_id = {r["query_id"]: r for r in conds["no_intent"]}
        for qid, tom_row in tom_by_id.items():
            ni_row = ni_by_id.get(qid)
            if ni_row is None:
                continue
            gap = tom_row.get("gap")
            tom_score = tom_row.get("implicit_score")
            ni_score = ni_row.get("implicit_score")
            if gap is None or tom_score is None or ni_score is None:
                continue
            scene = tom_row.get("scene")
            subcat = tom_row.get("subcategory")
            scenes_seen.add(scene)
            subcats_seen.add(subcat)
            pairs.append(
                {
                    "seed": seed,
                    "query_id": qid,
                    "scene": scene,
                    "subcategory": subcat,
                    "gap": float(gap),
                    "benefit": float(tom_score) - float(ni_score),
                }
            )
    meta = d.get("meta", {})
    return pairs, scenes_seen, subcats_seen, meta


def normal_ci95(vals):
    n = len(vals)
    if n < 2:
        return (None, None)
    m = statistics.mean(vals)
    sd = statistics.stdev(vals)
    se = sd / math.sqrt(n)
    return (round(m - 1.96 * se, 4), round(m + 1.96 * se, 4))


def main() -> int:
    if PRIMARY_SRC.exists():
        src = PRIMARY_SRC
        pairs, scenes_seen, subcats_seen, meta = load_pairs(src, scene_field_present=True)
        source_note = (
            f"Full-coverage multiseed file: {src.name}. "
            f"Scenes covered: {sorted(s for s in scenes_seen if s)} "
            f"({len(scenes_seen)} of 4). Subcategories: {sorted(subcats_seen)}. "
            f"Model: {meta.get('model')}, seeds: {meta.get('seeds')}."
        )
    else:
        src = FALLBACK_SRC
        pairs, scenes_seen, subcats_seen, meta = load_pairs(src, scene_field_present=False)
        source_note = (
            f"FALLBACK (primary file missing): {src.name}. "
            f"This file covers only ONE scene ({meta.get('scene', 'unknown')}), "
            f"25 queries x 3 seeds = 75 rows per condition. "
            f"Subcategories: {sorted(subcats_seen)}. Treat conclusions as "
            f"single-scene only, not full-benchmark."
        )

    print(f"[source] {source_note}\n")

    n_total = len(pairs)
    gaps = [p["gap"] for p in pairs]
    benefits = [p["benefit"] for p in pairs]

    gmin, gmax = min(gaps), max(gaps)
    print(f"n joined (seed, query_id) pairs: {n_total}")
    print(f"gap range observed: [{gmin}, {gmax}]")
    unique_gaps = sorted(set(round(g, 4) for g in gaps))
    print(f"unique gap values ({len(unique_gaps)}): {unique_gaps}\n")

    pearson_r = pearson(gaps, benefits)
    spearman_r = spearman(gaps, benefits)

    # --- Binning ---
    # Fixed-width 0.1 bins spanning the observed range, since gap is
    # highly discretized (few unique values) rather than continuous.
    # This avoids quantile-bin edge effects arbitrarily splitting ties.
    bin_width = BIN_WIDTH
    lo = math.floor(gmin / bin_width) * bin_width
    hi = math.ceil(gmax / bin_width) * bin_width
    n_bins = max(1, round((hi - lo) / bin_width))
    edges = [round(lo + i * bin_width, 4) for i in range(n_bins + 1)]

    bin_rows = [[] for _ in range(n_bins)]
    for p in pairs:
        g = p["gap"]
        idx = min(int((g - lo) / bin_width), n_bins - 1)
        bin_rows[idx].append(p)

    print(f"{'bin range':<14} | {'n':>4} | {'mean benefit':>13} | {'std':>7} | {'95% CI':>20}")
    bin_table = []
    bin_means = []
    for i in range(n_bins):
        lo_e, hi_e = edges[i], edges[i + 1]
        rows = bin_rows[i]
        n = len(rows)
        if n == 0:
            print(f"[{lo_e:.2f},{hi_e:.2f}) | {n:>4} | {'(empty)':>13} | {'':>7} | {'':>20}")
            bin_table.append(
                {"range": [lo_e, hi_e], "n": 0, "mean_benefit": None, "std": None, "ci95": [None, None]}
            )
            continue
        bvals = [r["benefit"] for r in rows]
        mean_b = statistics.mean(bvals)
        std_b = statistics.stdev(bvals) if n > 1 else 0.0
        ci = normal_ci95(bvals)
        bin_means.append(mean_b)
        label = f"[{lo_e:.2f},{hi_e:.2f})" if i < n_bins - 1 else f"[{lo_e:.2f},{hi_e:.2f}]"
        print(f"{label:<14} | {n:>4} | {mean_b:>13.4f} | {std_b:>7.4f} | {str(ci):>20}")
        bin_table.append(
            {
                "range": [lo_e, hi_e],
                "n": n,
                "mean_benefit": round(mean_b, 4),
                "std": round(std_b, 4),
                "ci95": list(ci),
            }
        )

    # --- Monotonicity check ---
    non_decreasing = all(
        bin_means[i] <= bin_means[i + 1] + 1e-9 for i in range(len(bin_means) - 1)
    )
    n_inversions = sum(
        1 for i in range(len(bin_means) - 1) if bin_means[i] > bin_means[i + 1] + 1e-9
    )

    print("\n=== Summary stats ===")
    print(f"Pearson r(gap, benefit)  = {round(pearson_r, 4) if pearson_r is not None else None}")
    print(f"Spearman rho(gap, benefit) = {round(spearman_r, 4) if spearman_r is not None else None}")
    print(f"Bin means non-decreasing across bins: {non_decreasing} (inversions: {n_inversions})")
    print(f"Bin means: {[round(m, 4) for m in bin_means]}")

    all_close = False
    if len(bin_means) > 1:
        spread = max(bin_means) - min(bin_means)
        all_close = spread < 0.02
    if all_close:
        print(
            "\n[WARNING] Bin means are nearly identical (spread < 0.02) — "
            "gap shows essentially no discriminative power over benefit at "
            "this granularity."
        )

    small_n_bins = [b for b in bin_table if b["n"] is not None and 0 < b["n"] < 10]
    if small_n_bins:
        print(
            f"[WARNING] {len(small_n_bins)} bin(s) have n < 10 — treat their "
            f"CIs as unreliable: {[b['range'] for b in small_n_bins]}"
        )

    out = {
        "source_file": str(src.relative_to(ROOT)),
        "source_note": source_note,
        "n_pairs": n_total,
        "gap_range_observed": [gmin, gmax],
        "unique_gap_values": unique_gaps,
        "pearson_gap_benefit": round(pearson_r, 4) if pearson_r is not None else None,
        "spearman_gap_benefit": round(spearman_r, 4) if spearman_r is not None else None,
        "bin_width": bin_width,
        "bins": bin_table,
        "bin_means_non_decreasing": non_decreasing,
        "n_inversions": n_inversions,
        "degenerate_flat_bins": all_close,
        "small_n_bins": [b["range"] for b in small_n_bins],
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[write] {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
