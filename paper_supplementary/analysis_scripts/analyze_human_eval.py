"""Analyse the 6 collected RQ5 human-eval annotations.

Loads:
  evaluation/results/human_eval_forms.json    (scenarios with _label_a / _label_b)
  evaluation/results/annotations/*.json       (per-rater ratings)

Computes:
  per-dimension mean per system (AURA vs Vanilla baseline)
  paired Wilcoxon signed-rank test per dimension (paired by rater+scenario)
  Cohen's d effect size for paired data
  per-category breakdown (spatial / social / temporal / memory / planning)
  inter-rater agreement on "AURA better" counts (Krippendorff-style)
  rater-level mean and AURA-win rate

Output:
  evaluation/results/rq5_analysis.json   (machine-readable)
  Plus a Markdown table printed to stdout.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT       = Path(__file__).resolve().parent.parent
RESULTS    = ROOT / "evaluation" / "results"
SCEN_PATH  = RESULTS / "human_eval_forms.json"
ANNO_DIR   = RESULTS / "annotations"
OUT_PATH   = RESULTS / "rq5_analysis.json"

DIMS = [
    "response_helpfulness",
    "environmental_awareness",
    "agent_believability",
    "factual_accuracy",
]


def paired_wilcoxon(diffs: List[float]) -> Tuple[float, float, int]:
    """Paired Wilcoxon signed-rank test. Returns (W, p, n_nonzero).
    Falls back to a normal approximation if scipy is unavailable.
    """
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n < 6:
        return float("nan"), float("nan"), n
    try:
        from scipy import stats  # type: ignore
        res = stats.wilcoxon(nz, alternative="two-sided", zero_method="wilcox", correction=True)
        return float(res.statistic), float(res.pvalue), n
    except ImportError:
        # Normal approximation
        ranks = sorted(range(n), key=lambda i: abs(nz[i]))
        rank_of = [0] * n
        for r, i in enumerate(ranks, 1):
            rank_of[i] = r
        wp = sum(rank_of[i] for i in range(n) if nz[i] > 0)
        wn = sum(rank_of[i] for i in range(n) if nz[i] < 0)
        w  = min(wp, wn)
        mu = n * (n + 1) / 4
        sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (w - mu) / sd if sd > 0 else 0
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
        return w, p, n


def cohens_d_paired(diffs: List[float]) -> float:
    if len(diffs) < 2:
        return float("nan")
    m = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    return m / sd if sd > 0 else float("nan")


def mean_std(vs: List[float]) -> Tuple[float, float]:
    if not vs:
        return float("nan"), float("nan")
    if len(vs) == 1:
        return vs[0], 0.0
    return statistics.mean(vs), statistics.stdev(vs)


def main() -> int:
    scenarios = json.loads(SCEN_PATH.read_text())
    sid_to_scen = {s["id"]: s for s in scenarios}

    annotators: Dict[str, dict] = {}
    for fp in sorted(ANNO_DIR.glob("*.json")):
        with open(fp) as f:
            d = json.load(f)
        annotators[d["annotator_id"]] = d

    print(f"Loaded {len(scenarios)} scenarios and {len(annotators)} raters: "
          f"{list(annotators.keys())}\n")

    # Build per-dimension paired arrays: (aura_score, baseline_score) per
    # (rater, scenario) pair.
    paired: Dict[str, List[Tuple[float, float, str, int, str]]] = {d: [] for d in DIMS}
    # Flat lists for raw means
    raw: Dict[str, Dict[str, List[float]]] = {
        sys_: {d: [] for d in DIMS} for sys_ in ("aura", "baseline")
    }
    # Per-category accumulator: cat -> dim -> [diffs]
    by_cat: Dict[str, Dict[str, List[float]]] = {}

    for aid, ann in annotators.items():
        ratings = ann["ratings"]
        for sid, scen in sid_to_scen.items():
            cat = scen.get("category", "")
            for dim in DIMS:
                key_a = f"s{sid}_a_{dim}"
                key_b = f"s{sid}_b_{dim}"
                va = ratings.get(key_a)
                vb = ratings.get(key_b)
                if va is None or vb is None:
                    continue
                if scen["_label_a"] == "aura":
                    aura, base = float(va), float(vb)
                else:
                    aura, base = float(vb), float(va)
                paired[dim].append((aura, base, aid, sid, cat))
                raw["aura"][dim].append(aura)
                raw["baseline"][dim].append(base)
                by_cat.setdefault(cat, {d: [] for d in DIMS})
                by_cat[cat][dim].append(aura - base)

    # Stats per dimension
    print(f"{'Dimension':<26} {'AURA':>14} {'Vanilla':>14} {'Δ':>8} {'W':>8} {'p':>9} {'d':>7} {'N':>4}")
    print("-" * 100)
    summary: Dict[str, dict] = {}
    for dim in DIMS:
        a = raw["aura"][dim]
        b = raw["baseline"][dim]
        diffs = [x[0] - x[1] for x in paired[dim]]
        ma, sa = mean_std(a)
        mb, sb = mean_std(b)
        d_ms = mean_std(diffs)
        W, p, n_nz = paired_wilcoxon(diffs)
        d_size = cohens_d_paired(diffs)
        n = len(diffs)

        print(f"{dim:<26} {ma:>5.2f} ± {sa:>4.2f}  {mb:>5.2f} ± {sb:>4.2f}  "
              f"{d_ms[0]:>+7.3f}  {W:>8.1f} {p:>9.4g} {d_size:>+7.3f} {n:>4d}")

        summary[dim] = {
            "aura_mean":   round(ma, 3), "aura_std":   round(sa, 3),
            "vanilla_mean":round(mb, 3), "vanilla_std":round(sb, 3),
            "delta_mean":  round(d_ms[0], 3), "delta_std": round(d_ms[1], 3),
            "wilcoxon_W":  None if math.isnan(W) else round(W, 2),
            "wilcoxon_p":  None if math.isnan(p) else float(f"{p:.6g}"),
            "cohens_d":    None if math.isnan(d_size) else round(d_size, 3),
            "n_pairs":     n, "n_nonzero": n_nz,
        }

    # Per-category breakdown
    print()
    print("Per-category Δ (AURA − Vanilla), pooled across 6 raters:")
    print(f"{'Category':<12} " + " ".join(f"{d.split('_')[0]:>8}" for d in DIMS) + f"{'avg':>8}")
    print("-" * 60)
    cat_summary: Dict[str, Dict[str, float]] = {}
    for cat in sorted(by_cat):
        cat_summary[cat] = {}
        row = [f"{cat:<12}"]
        avgs = []
        for dim in DIMS:
            ds = by_cat[cat][dim]
            if ds:
                m = statistics.mean(ds)
                avgs.append(m)
                cat_summary[cat][dim] = round(m, 3)
                row.append(f"{m:>+8.2f}")
            else:
                row.append(f"{'-':>8}")
        avg = statistics.mean(avgs) if avgs else float("nan")
        cat_summary[cat]["avg"] = round(avg, 3)
        row.append(f"{avg:>+8.2f}")
        print("".join(row))

    # Per-rater AURA-win rate
    print()
    print("Per-rater AURA-better rate (across 50 scenarios × 4 dims = 200 comparisons):")
    print(f"{'Rater':<18} {'AURA wins':>10} {'ties':>6} {'AURA loses':>11} {'mean Δ':>8}")
    print("-" * 60)
    rater_summary: Dict[str, dict] = {}
    for aid in sorted(annotators):
        ann = annotators[aid]
        wins = ties = losses = 0
        all_diffs: List[float] = []
        for sid, scen in sid_to_scen.items():
            for dim in DIMS:
                va = ann["ratings"].get(f"s{sid}_a_{dim}")
                vb = ann["ratings"].get(f"s{sid}_b_{dim}")
                if va is None or vb is None:
                    continue
                aura = va if scen["_label_a"] == "aura" else vb
                base = vb if scen["_label_a"] == "aura" else va
                d = aura - base
                all_diffs.append(d)
                if d > 0: wins += 1
                elif d < 0: losses += 1
                else: ties += 1
        m = statistics.mean(all_diffs) if all_diffs else 0
        total = wins + ties + losses
        rater_summary[aid] = {
            "aura_wins": wins, "ties": ties, "aura_losses": losses,
            "win_rate": round(wins / total, 3) if total else 0,
            "mean_delta": round(m, 3),
        }
        print(f"{aid:<18} {wins:>4d}/{total} ({wins/total*100:>4.1f}%) "
              f"{ties:>5d}  {losses:>4d} ({losses/total*100:>4.1f}%) {m:>+7.3f}")

    # Save
    out = {
        "n_raters": len(annotators),
        "raters": list(annotators),
        "n_scenarios": len(scenarios),
        "n_paired_per_dim": len(paired[DIMS[0]]),
        "by_dimension": summary,
        "by_category": cat_summary,
        "by_rater": rater_summary,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
