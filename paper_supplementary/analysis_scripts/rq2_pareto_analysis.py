"""RQ2 Pareto analysis: FA vs cost / disclosure across all conditions.

Loads every RQ2 multiseed result file and computes per-condition:
  - factual accuracy (mean over seeds)
  - mean tool-call count per query
  - mean disclosure score per query (sum of TOOL_DISCLOSURE for actually
    invoked tools where the tool list is recorded; falls back to mean
    tool_calls × avg disclosure across all SHARED_TOOLS when only a
    count is available)
  - mean latency per query
  - paired-test contrasts vs AURA_GapRouted (when available) and AURA_Full

Saves: evaluation/results/rq2_pareto_analysis.json
Prints: Pareto table + per-category breakdown + paired tests.

Run after run_rq2_aura_gap_routed.py (and run_rq2_fixed_probe.py) so
the gap-routed condition is included; the script gracefully skips
files that haven't been produced yet.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.baselines import SHARED_TOOLS, TOOL_DISCLOSURE  # noqa: E402

RESULTS = ROOT / "evaluation" / "results"
EXPECTED_N_QUERIES = 50
ALLOW_PARTIAL = "--allow-partial" in sys.argv

# Condition source files. Each entry: (display_name, file, seed_block_path,
# tool_list_extractor). seed_block_path is the chain of dict keys to the
# per-query "details" list. tool_list_extractor returns the list of tool
# names actually invoked for one query (or None if only a count is known).
SHARED_AVG_DISCLOSURE = (
    sum(TOOL_DISCLOSURE.values()) / len(TOOL_DISCLOSURE)
)


def _from_actual_probes(detail: Dict[str, Any]) -> Optional[List[str]]:
    """Tool list extractor for conditions that record the actual probe set."""
    return detail.get("actual_probes")


def _all_shared_tools(detail: Dict[str, Any]) -> Optional[List[str]]:
    """Fixed_Probe always invokes every shared tool."""
    n = detail.get("tool_calls", 0)
    if n == 0:
        return []
    return [t["name"] for t in SHARED_TOOLS]


def _none_extractor(detail: Dict[str, Any]) -> Optional[List[str]]:
    return None


CONDITIONS: List[Tuple[str, str, List[str], Any]] = [
    ("AURA_GapRouted",
     "rq2_aura_gap_routed_multiseed.json",
     ["per_seed", "{seed}", "AURA_GapRouted", "details"],
     _from_actual_probes),
    ("Fixed_Probe",
     "rq2_fixed_probe_multiseed.json",
     ["per_seed", "{seed}", "Fixed_Probe", "details"],
     _all_shared_tools),
    ("AURA_Full",
     "rq2_factual_accuracy_multiseed.json",
     ["per_seed", "{seed}", "AURA_Full", "details"],
     _none_extractor),
    ("AURA_NoProbe",
     "rq2_factual_accuracy_multiseed.json",
     ["per_seed", "{seed}", "AURA_NoProbe", "details"],
     _none_extractor),
    ("ReAct",
     "rq2_factual_accuracy_multiseed.json",
     ["per_seed", "{seed}", "ReAct", "details"],
     _none_extractor),
    ("Static_Context",
     "rq2_factual_accuracy_multiseed.json",
     ["per_seed", "{seed}", "Static_Context", "details"],
     _none_extractor),
    ("Vanilla_LLM",
     "rq2_factual_accuracy_multiseed.json",
     ["per_seed", "{seed}", "Vanilla_LLM", "details"],
     _none_extractor),
    ("Plan_and_Solve",
     "rq2_extra_baselines_multiseed.json",
     ["per_seed", "{seed}", "Plan_and_Solve", "details"],
     _none_extractor),
    ("Reflexion",
     "rq2_extra_baselines_multiseed.json",
     ["per_seed", "{seed}", "Reflexion", "details"],
     _none_extractor),
]

SEEDS = ["42", "123", "456"]


def disclosure_for(detail: Dict[str, Any], extractor) -> float:
    tools = extractor(detail)
    if tools is None:
        n = detail.get("tool_calls") or 0
        return n * SHARED_AVG_DISCLOSURE
    return float(sum(TOOL_DISCLOSURE.get(t, 0) for t in tools))


def fa_of(detail: Dict[str, Any]) -> Optional[float]:
    fa = detail.get("factual_accuracy")
    if isinstance(fa, dict) and "accuracy" in fa:
        return fa["accuracy"]
    return None


def category_of(detail: Dict[str, Any]) -> str:
    return detail.get("category", "?")


def load_details(file_name: str, path_template: List[str]) -> Optional[Dict[str, List[Dict]]]:
    p = RESULTS / file_name
    if not p.exists():
        return None
    data = json.load(open(p))
    out: Dict[str, List[Dict]] = {}
    for s in SEEDS:
        node: Any = data
        for key in path_template:
            k = key.format(seed=s) if "{seed}" in key else key
            if not isinstance(node, dict) or k not in node:
                node = None
                break
            node = node[k]
        if isinstance(node, list):
            out[s] = node
    return out if out else None


def per_query_avg(details_per_seed: Dict[str, List[Dict]], extractor) -> Dict[int, Dict[str, float]]:
    """Returns {query_id: {fa, probes, disclosure, latency, category}} averaged across seeds."""
    fa_acc: Dict[int, List[float]] = defaultdict(list)
    pr_acc: Dict[int, List[int]] = defaultdict(list)
    dis_acc: Dict[int, List[float]] = defaultdict(list)
    lat_acc: Dict[int, List[float]] = defaultdict(list)
    cat: Dict[int, str] = {}
    for s, details in details_per_seed.items():
        for d in details:
            fa = fa_of(d)
            if fa is None:
                continue
            qid = d["query_id"]
            fa_acc[qid].append(fa)
            pr_acc[qid].append(d.get("tool_calls", 0))
            dis_acc[qid].append(disclosure_for(d, extractor))
            lat_acc[qid].append(d.get("latency", 0.0))
            cat[qid] = category_of(d)
    out = {}
    for qid in fa_acc:
        out[qid] = {
            "fa": statistics.mean(fa_acc[qid]),
            "probes": statistics.mean(pr_acc[qid]),
            "disclosure": statistics.mean(dis_acc[qid]),
            "latency": statistics.mean(lat_acc[qid]),
            "category": cat[qid],
        }
    return out


def aggregate(per_query: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    if not per_query:
        return {}
    return {
        "fa_mean": statistics.mean(r["fa"] for r in per_query.values()),
        "probes_mean": statistics.mean(r["probes"] for r in per_query.values()),
        "disclosure_mean": statistics.mean(r["disclosure"] for r in per_query.values()),
        "latency_mean": statistics.mean(r["latency"] for r in per_query.values()),
        "n_queries": len(per_query),
    }


def paired_t(a: Dict[int, Dict], b: Dict[int, Dict], key: str = "fa") -> Optional[Dict[str, float]]:
    try:
        from scipy import stats as scst
    except ImportError:
        return None
    ids = sorted(set(a) & set(b))
    if len(ids) < 2:
        return None
    av = [a[q][key] for q in ids]
    bv = [b[q][key] for q in ids]
    res = scst.ttest_rel(av, bv)
    return {
        "n": len(ids),
        "delta": float(statistics.mean(x - y for x, y in zip(av, bv))),
        "t": float(res.statistic),
        "p": float(res.pvalue),
    }


def per_category(per_query: Dict[int, Dict]) -> Dict[str, Dict[str, float]]:
    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    for r in per_query.values():
        by_cat[r["category"]].append(r)
    out = {}
    for cat, rows in by_cat.items():
        out[cat] = {
            "fa_mean": statistics.mean(r["fa"] for r in rows),
            "probes_mean": statistics.mean(r["probes"] for r in rows),
            "disclosure_mean": statistics.mean(r["disclosure"] for r in rows),
            "n": len(rows),
        }
    return out


def main() -> int:
    print(f"=== RQ2 Pareto analysis ({len(CONDITIONS)} conditions) ===\n")
    per_cond_query: Dict[str, Dict[int, Dict[str, float]]] = {}
    per_cond_agg: Dict[str, Dict[str, float]] = {}
    per_cond_cat: Dict[str, Dict[str, Dict[str, float]]] = {}
    skipped: Dict[str, str] = {}
    paired_vs_gap: Dict[str, Dict[str, Dict[str, float]]] = {}

    for name, file_name, path_tmpl, extractor in CONDITIONS:
        details = load_details(file_name, path_tmpl)
        if details is None:
            print(f"  [skip] {name}: source file missing ({file_name})")
            skipped[name] = f"missing source file: {file_name}"
            continue
        pq = per_query_avg(details, extractor)
        if not ALLOW_PARTIAL and len(pq) < EXPECTED_N_QUERIES:
            print(
                f"  [skip] {name}: only {len(pq)}/{EXPECTED_N_QUERIES} "
                "queries present; pass --allow-partial to inspect smoke runs"
            )
            skipped[name] = f"partial result: {len(pq)}/{EXPECTED_N_QUERIES} queries"
            continue
        per_cond_query[name] = pq
        per_cond_agg[name] = aggregate(pq)
        per_cond_cat[name] = per_category(pq)

    # Aggregate Pareto table
    print(f"{'condition':<18} {'FA':>8} {'probes':>8} {'disclo':>8} {'lat(s)':>8} {'n':>4}")
    print("-" * 64)
    for name, agg in sorted(per_cond_agg.items(), key=lambda x: -x[1].get("fa_mean", 0)):
        if not agg:
            continue
        print(f"{name:<18} {agg['fa_mean']:>8.4f} {agg['probes_mean']:>8.2f} "
              f"{agg['disclosure_mean']:>8.2f} {agg['latency_mean']:>8.2f} {agg['n_queries']:>4}")

    # Paired tests vs AURA_GapRouted
    if "AURA_GapRouted" in per_cond_query:
        print("\n=== Paired tests vs AURA_GapRouted (FA, query-level) ===")
        ref = per_cond_query["AURA_GapRouted"]
        paired_vs_gap["fa"] = {}
        for name in per_cond_query:
            if name == "AURA_GapRouted":
                continue
            res = paired_t(per_cond_query[name], ref, "fa")
            if res:
                paired_vs_gap["fa"][name] = res
                print(f"  {name:<18} n={res['n']:>3}  Δ_FA={res['delta']:+.4f}  "
                      f"t={res['t']:.3f}  p={res['p']:.4g}")

        # Probe-cost paired tests (lower is better — flip sign in interpretation)
        print("\n=== Paired tests vs AURA_GapRouted (probe count, query-level) ===")
        paired_vs_gap["probes"] = {}
        for name in per_cond_query:
            if name == "AURA_GapRouted":
                continue
            res = paired_t(per_cond_query[name], ref, "probes")
            if res:
                paired_vs_gap["probes"][name] = res
                print(f"  {name:<18} n={res['n']:>3}  Δ_probes={res['delta']:+.4f}  "
                      f"t={res['t']:.3f}  p={res['p']:.4g}")

        print("\n=== Paired tests vs AURA_GapRouted (disclosure, query-level) ===")
        paired_vs_gap["disclosure"] = {}
        for name in per_cond_query:
            if name == "AURA_GapRouted":
                continue
            res = paired_t(per_cond_query[name], ref, "disclosure")
            if res:
                paired_vs_gap["disclosure"][name] = res
                print(f"  {name:<18} n={res['n']:>3}  Δ_disc={res['delta']:+.4f}  "
                      f"t={res['t']:.3f}  p={res['p']:.4g}")

    # Per-category breakdown for the central comparison
    print("\n=== AURA_GapRouted per-category (FA, probes, disclosure) ===")
    if "AURA_GapRouted" in per_cond_cat:
        for cat, row in sorted(per_cond_cat["AURA_GapRouted"].items()):
            print(f"  {cat:<10} FA={row['fa_mean']:.4f}  probes={row['probes_mean']:.2f}  "
                  f"disc={row['disclosure_mean']:.2f}  n={row['n']}")
    print("\n=== Fixed_Probe per-category (for contrast) ===")
    if "Fixed_Probe" in per_cond_cat:
        for cat, row in sorted(per_cond_cat["Fixed_Probe"].items()):
            print(f"  {cat:<10} FA={row['fa_mean']:.4f}  probes={row['probes_mean']:.2f}  "
                  f"disc={row['disclosure_mean']:.2f}  n={row['n']}")

    out_path = RESULTS / "rq2_pareto_analysis.json"
    with open(out_path, "w") as f:
        json.dump({
            "tool_disclosure": TOOL_DISCLOSURE,
            "shared_avg_disclosure": SHARED_AVG_DISCLOSURE,
            "expected_n_queries": EXPECTED_N_QUERIES,
            "allow_partial": ALLOW_PARTIAL,
            "skipped": skipped,
            "per_condition": per_cond_agg,
            "per_category": per_cond_cat,
            "paired_vs_AURA_GapRouted": paired_vs_gap,
        }, f, indent=2)
    print(f"\n[write] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
