"""Re-score RQ2 multi-seed factual accuracy with strict, less-soft metrics.

Reads the existing per-query LLM-judge claim classifications
(correct / contradicted / unverifiable) and computes:

  - strict_precision   = correct / (correct + contradicted)   per query
  - hallucination_rate = fraction of queries with >=1 contradicted claim
  - perfect_rate       = fraction of queries with 0 contradicted AND >=1 correct
  - claim_recall       = correct / total_claims

The existing `accuracy` field uses 0.7*precision + 0.3*completeness
where completeness is a soft "did the response answer well" judgment.
Strict precision drops that 30% soft component.

Outputs:
  evaluation/results/rq2_strict_rescore.json    (full per-query + aggregates)
  printout of paired-t and per-condition table
"""

import json
import math
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

INPUT = Path("evaluation/results/rq2_factual_accuracy_multiseed.json")
OUTPUT = Path("evaluation/results/rq2_strict_rescore.json")


def strict_precision(fa: dict) -> float | None:
    """correct / (correct + contradicted). None if denominator is 0."""
    c = fa.get("correct_claims", 0)
    x = fa.get("contradicted_claims", 0)
    if c + x == 0:
        return None  # only unverifiable claims, exclude from precision
    return c / (c + x)


def claim_recall(fa: dict) -> float | None:
    """correct / total_claims. None if total_claims is 0."""
    t = fa.get("total_claims", 0)
    if t == 0:
        return None
    return fa.get("correct_claims", 0) / t


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
    # two-sided p via Student t survival; approximate with normal for large n
    # use scipy if available, else normal approx
    try:
        from scipy.stats import t as t_dist
        p = 2 * (1 - t_dist.cdf(abs(t), df=n - 1))
    except ImportError:
        # normal approximation
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return (t, p)


def main():
    with open(INPUT) as f:
        d = json.load(f)

    # Collect per (seed, cond, query_id) records
    # cond_query_seed_metrics[cond][query_id][seed] = dict of metrics
    by_cond_query_seed: dict[str, dict[int, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    seeds = sorted(int(s) for s in d["per_seed"].keys())

    for seed_str, conds in d["per_seed"].items():
        seed = int(seed_str)
        for cond, cond_data in conds.items():
            for r in cond_data.get("details", []):
                fa = r.get("factual_accuracy")
                if not isinstance(fa, dict) or "precision" not in fa:
                    continue
                qid = r.get("query_id")
                metrics = {
                    "strict_precision": strict_precision(fa),
                    "claim_recall": claim_recall(fa),
                    "hallucinated": is_hallucination(fa),
                    "perfect": is_perfect(fa),
                    "lenient_accuracy": fa.get("accuracy"),
                    "category": r.get("category"),
                    "n_correct": fa.get("correct_claims", 0),
                    "n_contradicted": fa.get("contradicted_claims", 0),
                    "n_unverifiable": fa.get("unverifiable_claims", 0),
                    "n_total": fa.get("total_claims", 0),
                }
                by_cond_query_seed[cond][qid][seed] = metrics

    # Per-condition aggregates: average strict_precision (excluding None) over all
    # query-seed cells; hallucination_rate and perfect_rate over all cells
    cond_agg: dict[str, dict] = {}
    cond_query_avg: dict[str, dict[int, dict]] = defaultdict(dict)

    for cond, qmap in by_cond_query_seed.items():
        sp_cells = []
        cr_cells = []
        h_cells = []
        p_cells = []
        len_cells = []
        for qid, smap in qmap.items():
            for seed, m in smap.items():
                if m["strict_precision"] is not None:
                    sp_cells.append(m["strict_precision"])
                if m["claim_recall"] is not None:
                    cr_cells.append(m["claim_recall"])
                h_cells.append(1 if m["hallucinated"] else 0)
                p_cells.append(1 if m["perfect"] else 0)
                if m["lenient_accuracy"] is not None:
                    len_cells.append(m["lenient_accuracy"])

            # Per-query average across seeds (for paired tests)
            sps_q = [m["strict_precision"] for m in smap.values() if m["strict_precision"] is not None]
            crs_q = [m["claim_recall"] for m in smap.values() if m["claim_recall"] is not None]
            hs_q = [1 if m["hallucinated"] else 0 for m in smap.values()]
            ps_q = [1 if m["perfect"] else 0 for m in smap.values()]
            cond_query_avg[cond][qid] = {
                "strict_precision_mean": mean(sps_q) if sps_q else None,
                "claim_recall_mean": mean(crs_q) if crs_q else None,
                "hallucinated_mean": mean(hs_q) if hs_q else 0.0,
                "perfect_mean": mean(ps_q) if ps_q else 0.0,
            }

        cond_agg[cond] = {
            "strict_precision_mean": mean(sp_cells) if sp_cells else 0.0,
            "strict_precision_std": stdev(sp_cells) if len(sp_cells) > 1 else 0.0,
            "strict_precision_n_cells": len(sp_cells),
            "claim_recall_mean": mean(cr_cells) if cr_cells else 0.0,
            "claim_recall_std": stdev(cr_cells) if len(cr_cells) > 1 else 0.0,
            "hallucination_rate": mean(h_cells) if h_cells else 0.0,
            "perfect_rate": mean(p_cells) if p_cells else 0.0,
            "lenient_accuracy_mean": mean(len_cells) if len_cells else 0.0,
            "n_query_seed_cells": len(h_cells),
        }

    # Paired t-tests on strict_precision: AURA_Full vs each baseline,
    # at the query level (averaged across seeds within query)
    contrasts = []
    refs = ["Vanilla_LLM", "Static_Context", "ReAct", "AURA_NoProbe"]
    if "AURA_Full" in cond_query_avg:
        for ref in refs:
            if ref not in cond_query_avg:
                continue
            common_qids = sorted(set(cond_query_avg["AURA_Full"]) & set(cond_query_avg[ref]))
            deltas_sp = []
            deltas_h = []
            deltas_p = []
            for qid in common_qids:
                f = cond_query_avg["AURA_Full"][qid]
                r = cond_query_avg[ref][qid]
                if f["strict_precision_mean"] is not None and r["strict_precision_mean"] is not None:
                    deltas_sp.append(f["strict_precision_mean"] - r["strict_precision_mean"])
                deltas_h.append(f["hallucinated_mean"] - r["hallucinated_mean"])
                deltas_p.append(f["perfect_mean"] - r["perfect_mean"])
            t_sp, p_sp = paired_t(deltas_sp)
            t_h, p_h = paired_t(deltas_h)
            t_p, p_p = paired_t(deltas_p)
            contrasts.append({
                "vs": ref,
                "strict_precision_delta": mean(deltas_sp) if deltas_sp else 0.0,
                "strict_precision_t": t_sp,
                "strict_precision_p": p_sp,
                "strict_precision_n_pairs": len(deltas_sp),
                "hallucination_delta": mean(deltas_h),
                "hallucination_t": t_h,
                "hallucination_p": p_h,
                "perfect_delta": mean(deltas_p),
                "perfect_t": t_p,
                "perfect_p": p_p,
            })

    # Per-category strict precision (helps us see whether the gap is uniform)
    by_cond_cat: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for cond, qmap in by_cond_query_seed.items():
        for qid, smap in qmap.items():
            for seed, m in smap.items():
                if m["strict_precision"] is None:
                    continue
                cat = m["category"] or "unknown"
                by_cond_cat[cond][cat].append(m["strict_precision"])

    cat_agg = {}
    for cond, cmap in by_cond_cat.items():
        cat_agg[cond] = {
            cat: {"mean": mean(v), "n": len(v)}
            for cat, v in cmap.items()
        }

    output = {
        "schema_version": "1.0",
        "input": str(INPUT),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "metric_definitions": {
            "strict_precision": "correct_claims / (correct_claims + contradicted_claims) per query; query excluded if denominator is 0 (only unverifiable claims)",
            "claim_recall": "correct_claims / total_claims per query",
            "hallucination_rate": "fraction of (query, seed) cells with >=1 contradicted claim",
            "perfect_rate": "fraction with 0 contradicted AND >=1 correct claim",
            "lenient_accuracy": "the existing 0.7*precision + 0.3*completeness FA from llm_judge.py for comparison",
        },
        "per_condition": cond_agg,
        "per_condition_per_category": cat_agg,
        "paired_contrasts_AURA_Full_vs": contrasts,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print human-readable summary
    print("=" * 78)
    print("RQ2 STRICT RESCORE  (no completeness softening)")
    print("=" * 78)
    order = ["Vanilla_LLM", "Static_Context", "ReAct", "AURA_NoProbe", "AURA_Full"]
    print(f"\n{'Condition':<18s} {'StrictP':>10s} {'Recall':>10s} {'Halluc%':>10s} {'Perfect%':>10s} {'LenientFA':>10s}")
    print("-" * 78)
    for cond in order:
        if cond not in cond_agg:
            continue
        a = cond_agg[cond]
        print(f"{cond:<18s} "
              f"{a['strict_precision_mean']:>10.3f} "
              f"{a['claim_recall_mean']:>10.3f} "
              f"{a['hallucination_rate']*100:>9.1f}% "
              f"{a['perfect_rate']*100:>9.1f}% "
              f"{a['lenient_accuracy_mean']:>10.3f}")

    print(f"\n--- Paired contrasts (AURA_Full vs ...) ---")
    print(f"{'Ref':<18s} {'ΔStrictP':>10s} {'p':>10s} {'ΔHalluc':>10s} {'p':>10s} {'ΔPerfect':>10s} {'p':>10s}")
    print("-" * 78)
    for c in contrasts:
        print(f"{c['vs']:<18s} "
              f"{c['strict_precision_delta']:>+10.3f} "
              f"{c['strict_precision_p']:>10.4f} "
              f"{c['hallucination_delta']:>+10.3f} "
              f"{c['hallucination_p']:>10.4f} "
              f"{c['perfect_delta']:>+10.3f} "
              f"{c['perfect_p']:>10.4f}")

    print(f"\n--- Per-category strict precision (where gap concentrates) ---")
    cats = sorted({c for d in cat_agg.values() for c in d})
    print(f"{'Category':<18s} " + " ".join(f"{c:>14s}" for c in order if c in cat_agg))
    for cat in cats:
        row = [cat]
        for cond in order:
            if cond in cat_agg and cat in cat_agg[cond]:
                v = cat_agg[cond][cat]
                row.append(f"{v['mean']:.3f} (n={v['n']:>2d})")
            else:
                row.append("-")
        print(f"{row[0]:<18s} " + " ".join(f"{x:>14s}" for x in row[1:]))

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
