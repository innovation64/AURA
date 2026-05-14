"""RQ5 conservative re-analysis: aggregate to per-rater means before testing.

The published table treats N=8 raters * 50 scenarios * 4 dims = 400 paired
observations as independent for the Wilcoxon test. They are not: ratings
within a rater (and within a scenario) are correlated. This script produces
two stricter alternatives reviewers will accept:

  (1) Per-rater aggregation: average each rater's deltas across 50 scenarios,
      run Wilcoxon over the resulting N=8 paired means.
  (2) Cluster bootstrap on rater_id (5000 resamples) to give a non-parametric
      95% CI on the mean delta.

Inputs:
  - evaluation/results/human_eval_forms.json    (50 scenarios with A/B labels)
  - evaluation/results/annotations/<rater>.json (8 raters, 50 scenarios * 4 dims * A/B)

Output:
  - evaluation/results/rq5_rater_aggregated.json
"""

import json
import math
import random
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
FORMS = ROOT / "evaluation/results/human_eval_forms.json"
ANNOT = ROOT / "evaluation/results/annotations"
OUT = ROOT / "evaluation/results/rq5_rater_aggregated.json"

DIMENSIONS = [
    "response_helpfulness",
    "environmental_awareness",
    "agent_believability",
    "factual_accuracy",
]


def wilcoxon_signed_rank(deltas: list[float]) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank for n>=6 (asymptotic). For n=8 the normal
    approx is fine. Returns (W+, p-value)."""
    n = len(deltas)
    abs_vals = [(abs(x), 1 if x > 0 else (-1 if x < 0 else 0)) for x in deltas if x != 0]
    if not abs_vals:
        return (0.0, 1.0)
    abs_vals.sort()
    # Average ranks for ties
    ranks = [0.0] * len(abs_vals)
    i = 0
    while i < len(abs_vals):
        j = i
        while j + 1 < len(abs_vals) and abs_vals[j + 1][0] == abs_vals[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # ranks 1..n
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    w_plus = sum(r for r, (_, sign) in zip(ranks, abs_vals) if sign > 0)
    w_minus = sum(r for r, (_, sign) in zip(ranks, abs_vals) if sign < 0)
    n_eff = len(abs_vals)
    mu = n_eff * (n_eff + 1) / 4
    sigma = math.sqrt(n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24)
    if sigma == 0:
        return (w_plus, 1.0)
    z = (w_plus - mu) / sigma
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return (w_plus, p)


def cluster_bootstrap_ci(rater_deltas: dict[str, list[float]], n_resamples: int = 5000, seed: int = 42) -> tuple[float, float, float]:
    """Resample raters with replacement, compute the mean of within-rater
    deltas. Returns (mean, lo95, hi95).
    """
    rng = random.Random(seed)
    rater_ids = list(rater_deltas.keys())
    means = []
    for _ in range(n_resamples):
        sample = [rng.choice(rater_ids) for _ in rater_ids]
        all_deltas = []
        for rid in sample:
            all_deltas.extend(rater_deltas[rid])
        if all_deltas:
            means.append(mean(all_deltas))
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples)]
    obs = mean([mean(v) for v in rater_deltas.values() if v])
    return (obs, lo, hi)


def main():
    with open(FORMS) as f:
        forms = json.load(f)
    label_by_scenario = {s["id"]: (s["_label_a"], s["_label_b"]) for s in forms}

    rater_files = sorted(ANNOT.glob("*.json"))
    if not rater_files:
        raise SystemExit(f"no rater annotations in {ANNOT}")

    # Build per-(rater, dim) list of (scenario_id, aura_score, vanilla_score, delta)
    # per_rater_per_dim_deltas[rater][dim] = [delta_per_scenario, ...]
    per_rater_per_dim_deltas: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for rf in rater_files:
        with open(rf) as f:
            r = json.load(f)
        rid = r.get("annotator_id", rf.stem)
        ratings = r.get("ratings", {})
        for sid in range(50):
            la, lb = label_by_scenario[sid]
            for dim in DIMENSIONS:
                key_a = f"s{sid}_a_{dim}"
                key_b = f"s{sid}_b_{dim}"
                if key_a not in ratings or key_b not in ratings:
                    continue
                ra = ratings[key_a]
                rb = ratings[key_b]
                if la == "aura":
                    aura_score, vanilla_score = ra, rb
                else:
                    aura_score, vanilla_score = rb, ra
                per_rater_per_dim_deltas[rid][dim].append(aura_score - vanilla_score)

    output = {
        "schema_version": "1.0",
        "n_raters": len(per_rater_per_dim_deltas),
        "raters": list(per_rater_per_dim_deltas.keys()),
        "per_dimension": {},
        "method": (
            "Each rater's deltas averaged across 50 scenarios, then Wilcoxon "
            "signed-rank on the resulting N=8 paired means. Cluster bootstrap "
            "on rater_id (5000 resamples) for non-parametric 95% CI on the "
            "mean delta. This is the conservative analysis; the original 400-"
            "cell paired Wilcoxon ignores rater-level dependence."
        ),
    }

    print("=" * 80)
    print("RQ5 conservative re-analysis: rater-aggregated paired Wilcoxon (N=8)")
    print("=" * 80)
    print(f"\n{'Dimension':<28s} {'Δ̄':>8s} {'sd':>8s} {'W+':>8s} {'p (Wilc, n=8)':>16s} {'cluster CI':>22s}")
    print("-" * 95)

    for dim in DIMENSIONS:
        # Per-rater mean delta
        per_rater_mean = {rid: mean(d_list) for rid, d_list in
                          ((rid, per_rater_per_dim_deltas[rid].get(dim, [])) for rid in per_rater_per_dim_deltas)
                          if d_list}
        rater_means_list = list(per_rater_mean.values())
        if len(rater_means_list) < 2:
            continue
        m = mean(rater_means_list)
        sd = stdev(rater_means_list)
        w_plus, p = wilcoxon_signed_rank(rater_means_list)

        # Cluster bootstrap on raw deltas (cluster = rater)
        rater_deltas_dict = {rid: per_rater_per_dim_deltas[rid].get(dim, [])
                             for rid in per_rater_per_dim_deltas}
        rater_deltas_dict = {k: v for k, v in rater_deltas_dict.items() if v}
        obs_mean, lo, hi = cluster_bootstrap_ci(rater_deltas_dict)

        # Cohen's d_z on per-rater means (the "right" effect size at rater level)
        d_z = m / sd if sd > 0 else float("inf")

        # Sign test (extra-conservative)
        wins = sum(1 for x in rater_means_list if x > 0)
        ties = sum(1 for x in rater_means_list if x == 0)
        losses = sum(1 for x in rater_means_list if x < 0)

        output["per_dimension"][dim] = {
            "rater_mean_deltas": per_rater_mean,
            "n_raters": len(rater_means_list),
            "delta_mean_of_means": round(m, 4),
            "delta_std_of_means": round(sd, 4),
            "cohens_dz_rater_level": round(d_z, 4),
            "wilcoxon_W_plus_rater": w_plus,
            "wilcoxon_p_rater": round(p, 6),
            "sign_test_wins": wins,
            "sign_test_ties": ties,
            "sign_test_losses": losses,
            "cluster_bootstrap_mean": round(obs_mean, 4),
            "cluster_bootstrap_lo95": round(lo, 4),
            "cluster_bootstrap_hi95": round(hi, 4),
        }

        print(f"{dim:<28s} "
              f"{m:>+8.3f} "
              f"{sd:>8.3f} "
              f"{w_plus:>8.0f} "
              f"{p:>16.4f} "
              f"  [{lo:+.3f}, {hi:+.3f}]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {OUT}")
    print("\nNote: under rater-level aggregation, each row is a single n=8 ")
    print("Wilcoxon — a much stricter test than the original n=400 paired test.")


if __name__ == "__main__":
    main()
