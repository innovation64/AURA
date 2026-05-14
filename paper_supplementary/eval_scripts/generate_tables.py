#!/usr/bin/env python3
"""
Generate LaTeX tables and statistical analysis from AURA experiment results.

Supports both single-seed results and multi-seed aggregated results.

Usage:
    python generate_tables.py                    # auto-detect available results
    python generate_tables.py --output tables/   # write .tex files to directory

Multi-seed file convention (from run_experiments.py --multi-seed):
    rq1_grounding_accuracy_multiseed.json   ->  {seeds, per_seed, statistics}
    rq2_factual_accuracy_multiseed.json     ->  {seeds, per_seed, statistics}
    rq3_ablation_study_multiseed.json       ->  {seeds, per_seed, statistics}
    rq6_probe_budget_pareto_multiseed.json  ->  {seeds, per_seed, statistics}

Single-seed files (fallback):
    rq1_grounding_accuracy.json, rq2_factual_accuracy.json, etc.
"""

import json
import math
import os
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"

# Multi-seed filenames (canonical, colon-free). Older colon-prefixed
# variants moved to evaluation/results/_archive/ — see MANIFEST.md.
MULTISEED_FILES = {
    "rq1": ["rq1_grounding_accuracy_multiseed.json"],
    "rq2": ["rq2_factual_accuracy_multiseed.json"],
    "rq3": ["rq3_ablation_study_multiseed.json"],
    "rq6": ["rq6_probe_budget_pareto_multiseed.json"],
}

# Single-seed filenames (fallback)
SINGLE_FILES = {
    "rq1": "rq1_grounding_accuracy.json",
    "rq2": "rq2_factual_accuracy.json",
    "rq3": "rq3_ablation.json",
    "rq6": "rq6_probe_budget.json",
}

# ---------------------------------------------------------------------------
# Statistics utilities (no scipy dependency)
# ---------------------------------------------------------------------------

def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using math.erf."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def paired_ttest(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Paired t-test (two-tailed). Returns (t_statistic, p_value)."""
    if len(a) != len(b) or len(a) < 2:
        return (0.0, 1.0)
    diffs = [x - y for x, y in zip(a, b)]
    m = mean(diffs)
    s = std(diffs)
    if s == 0:
        return (float('inf') if m != 0 else 0.0, 0.0 if m != 0 else 1.0)
    n = len(diffs)
    t_stat = m / (s / math.sqrt(n))
    p = 2 * (1 - normal_cdf(abs(t_stat)))
    return (t_stat, p)


def bootstrap_ci(values: List[float], n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    import random
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    boot_means = []
    for _ in range(n_boot):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(mean(sample))
    boot_means.sort()
    lo = boot_means[int(n_boot * alpha / 2)]
    hi = boot_means[int(n_boot * (1 - alpha / 2))]
    return (lo, hi)


def bootstrap_diff_pvalue(a: List[float], b: List[float],
                          n_boot: int = 10000, seed: int = 42) -> float:
    """Bootstrap p-value testing H0: mean(a) == mean(b), two-tailed.
    Useful when sample size is small (e.g., 3-5 seeds)."""
    import random
    rng = random.Random(seed)
    if len(a) == 0 or len(b) == 0:
        return 1.0
    observed_diff = mean(a) - mean(b)
    combined = a + b
    n_a = len(a)
    count_extreme = 0
    for _ in range(n_boot):
        rng.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        perm_diff = mean(perm_a) - mean(perm_b)
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1
    return count_extreme / n_boot


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def fmt_mean_std(m: float, s: float, pct: bool = False, decimals: int = 3) -> str:
    """Format as 'mean +/- std' for LaTeX."""
    if pct:
        return f"{m*100:.{decimals-1}f} $\\pm$ {s*100:.{decimals-1}f}"
    return f"{m:.{decimals}f} $\\pm$ {s:.{decimals}f}"


# ---------------------------------------------------------------------------
# Data loading with multi-seed / single-seed auto-detection
# ---------------------------------------------------------------------------

def load_json(filename: str) -> Optional[Any]:
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_rq_data(rq: str) -> Tuple[Optional[dict], bool]:
    """Load RQ data, preferring multi-seed files.

    Returns: (data, is_multiseed)
    """
    for candidate in MULTISEED_FILES[rq]:
        ms = load_json(candidate)
        if ms is not None and "per_seed" in ms:
            print(f"  [OK] Loaded multi-seed data for {rq.upper()} from {candidate} "
                  f"({len(ms.get('seeds', []))} seeds)")
            return ms, True

    single = load_json(SINGLE_FILES[rq])
    if single is not None:
        print(f"  [OK] Loaded single-seed data for {rq.upper()} (no CI available)")
        return single, False

    print(f"  [SKIP] No data found for {rq.upper()}")
    return None, False


# ---------------------------------------------------------------------------
# Multi-seed metric extraction helpers
# ---------------------------------------------------------------------------

def extract_per_seed_condition_metric(per_seed: dict, condition: str,
                                       metric_path: List[str]) -> List[float]:
    """Extract a metric from per_seed results for a given condition.

    metric_path: e.g. ["summary", "overall_grounding_accuracy"]
    """
    values = []
    for seed_key, seed_data in per_seed.items():
        obj = seed_data.get(condition, {})
        for key in metric_path:
            if isinstance(obj, dict):
                obj = obj.get(key, None)
            else:
                obj = None
                break
        if obj is not None and isinstance(obj, (int, float)):
            values.append(float(obj))
    return values


def extract_per_seed_flat_metric(per_seed: dict, condition: str,
                                  metric: str) -> List[float]:
    """Extract a flat metric from per_seed results (RQ3-style, no 'summary' wrapper)."""
    values = []
    for seed_key, seed_data in per_seed.items():
        obj = seed_data.get(condition, {})
        if isinstance(obj, dict) and metric in obj:
            values.append(float(obj[metric]))
    return values


def extract_per_seed_budget_metric(per_seed: dict, budget: str,
                                    metric: str) -> List[float]:
    """Extract metric from per_seed results for RQ6 budget data."""
    values = []
    for seed_key, seed_data in per_seed.items():
        per_budget = seed_data.get("per_budget", {})
        entry = per_budget.get(str(budget), {})
        if metric in entry:
            values.append(float(entry[metric]))
    return values


# =========================================================================
# RQ1: Grounding Accuracy Table
# =========================================================================

RQ1_CONDITIONS = [
    ("Vanilla_LLM", "Vanilla LLM"),
    ("Static_Context", "Static Context"),
    ("ReAct", "ReAct"),
    ("Reflexion", "Reflexion"),
    ("Plan_and_Solve", "Plan-and-Solve"),
    ("AURA_NoProbe", "\\sys{} (No Probe)"),
    ("AURA_Full", "\\sys{} (Full, B=2)"),
]

RQ1_DIMS = [
    ("location_consistency", "Loc"),
    ("time_appropriateness", "Time"),
    ("social_awareness", "Social"),
    ("memory_utilization", "Mem"),
    ("plan_adherence", "Plan"),
]


def generate_rq1_table(output_lines: list):
    data, is_multi = load_rq_data("rq1")
    if data is None:
        return

    out = output_lines
    out.append("")
    out.append("% " + "=" * 60)
    out.append("% Table 1: Grounding Accuracy (RQ1)")
    out.append("% " + "=" * 60)
    out.append("\\begin{table*}[t]")
    out.append("\\centering")
    if is_multi:
        n_seeds = len(data.get("seeds", []))
        out.append(f"\\caption{{Grounding Accuracy (GA) across methods. "
                   f"Mean $\\pm$ std over {n_seeds} seeds. "
                   f"$^{{*}}$/$^{{**}}$/$^{{***}}$: $p<0.05/0.01/0.001$ "
                   f"vs.~\\sys{{}} (Full) via paired $t$-test.}}")
    else:
        out.append("\\caption{Grounding Accuracy (GA) across methods. "
                   "Rule = rule-based location/time consistency.}")
    out.append("\\label{tab:grounding}")
    out.append("\\begin{tabular}{lccccccccc}")
    out.append("\\toprule")
    dim_headers = " & ".join([f"\\textbf{{{d[1]}}}" for d in RQ1_DIMS])
    out.append(f"\\textbf{{Method}} & \\textbf{{GA (Overall)}} & "
               f"{dim_headers} & \\textbf{{Rule Loc}} & "
               f"\\textbf{{Latency (s)}} \\\\")
    out.append("\\midrule")

    # Collect AURA_Full values for significance testing
    aura_full_values = {}  # metric -> [values per seed]

    if is_multi:
        per_seed = data["per_seed"]
        # First pass: get AURA_Full values
        aura_full_values["overall"] = extract_per_seed_condition_metric(
            per_seed, "AURA_Full", ["summary", "overall_grounding_accuracy"])
        for dim_key, _ in RQ1_DIMS:
            aura_full_values[dim_key] = extract_per_seed_condition_metric(
                per_seed, "AURA_Full", ["summary", "dimension_scores", dim_key])

        for cond_key, cond_label in RQ1_CONDITIONS:
            if not any(cond_key in sd for sd in per_seed.values()):
                continue

            # Overall GA
            overall_vals = extract_per_seed_condition_metric(
                per_seed, cond_key, ["summary", "overall_grounding_accuracy"])
            overall_str = fmt_mean_std(mean(overall_vals), std(overall_vals))

            # Significance vs AURA_Full
            if cond_key != "AURA_Full" and len(overall_vals) >= 2:
                _, p = paired_ttest(aura_full_values["overall"], overall_vals)
                p_boot = bootstrap_diff_pvalue(aura_full_values["overall"], overall_vals)
                p_use = min(p, p_boot)  # conservative: use smaller p
                stars = sig_stars(p_use)
                if stars:
                    overall_str += f"$^{{{stars}}}$"

            # Per-dimension
            dim_strs = []
            for dim_key, _ in RQ1_DIMS:
                vals = extract_per_seed_condition_metric(
                    per_seed, cond_key, ["summary", "dimension_scores", dim_key])
                if vals:
                    dim_strs.append(fmt_mean_std(mean(vals), std(vals)))
                else:
                    dim_strs.append("--")

            # Rule-based loc
            rule_vals = extract_per_seed_condition_metric(
                per_seed, cond_key, ["summary", "rule_based_location_accuracy"])
            rule_str = fmt_mean_std(mean(rule_vals), std(rule_vals)) if rule_vals else "--"

            # Latency
            lat_vals = extract_per_seed_condition_metric(
                per_seed, cond_key, ["summary", "avg_latency_per_step"])
            lat_str = fmt_mean_std(mean(lat_vals), std(lat_vals), decimals=1) if lat_vals else "--"

            dims_joined = " & ".join(dim_strs)
            row = f"{cond_label} & {overall_str} & {dims_joined} & {rule_str} & {lat_str} \\\\"
            if cond_key == "AURA_Full":
                row = "\\midrule\n" + row
            out.append(row)
    else:
        # Single-seed fallback
        for cond_key, cond_label in RQ1_CONDITIONS:
            if cond_key not in data:
                continue
            s = data[cond_key].get("summary", {})
            dims = s.get("dimension_scores", {})
            overall = s.get("overall_grounding_accuracy", "--")
            dim_strs = [str(dims.get(d[0], "--")) for d in RQ1_DIMS]
            rule_loc = s.get("rule_based_location_accuracy", "--")
            lat = s.get("avg_latency_per_step", "--")
            dims_joined = " & ".join(dim_strs)
            row = f"{cond_label} & {overall} & {dims_joined} & {rule_loc} & {lat} \\\\"
            if cond_key == "AURA_Full":
                row = "\\midrule\n" + row
            out.append(row)

    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table*}")


# =========================================================================
# RQ2: Factual Accuracy Table
# =========================================================================

RQ2_CONDITIONS = [
    ("Vanilla_LLM", "Vanilla LLM"),
    ("Static_Context", "Static Context"),
    ("ReAct", "ReAct"),
    ("Reflexion", "Reflexion"),
    ("Plan_and_Solve", "Plan-and-Solve"),
    ("AURA_NoProbe", "\\sys{} (No Probe)"),
    ("AURA_Full", "\\sys{} (Full)"),
]

RQ2_CATEGORIES = ["spatial", "social", "temporal", "memory", "planning"]


def generate_rq2_table(output_lines: list):
    data, is_multi = load_rq_data("rq2")
    if data is None:
        return

    out = output_lines
    out.append("")
    out.append("% " + "=" * 60)
    out.append("% Table 2: Factual Accuracy (RQ2)")
    out.append("% " + "=" * 60)
    out.append("\\begin{table*}[t]")
    out.append("\\centering")
    if is_multi:
        n_seeds = len(data.get("seeds", []))
        out.append(f"\\caption{{Factual Accuracy (FA) and per-category breakdown. "
                   f"Mean $\\pm$ std over {n_seeds} seeds.}}")
    else:
        out.append("\\caption{Factual Accuracy (FA) and Context Utilization (CU) "
                   "on environment-grounded queries.}")
    out.append("\\label{tab:factual}")

    cat_headers = " & ".join([f"\\textbf{{{c.title()}}}" for c in RQ2_CATEGORIES])
    out.append(f"\\begin{{tabular}}{{lcc{('c' * len(RQ2_CATEGORIES))}c}}")
    out.append("\\toprule")
    out.append(f"\\textbf{{Method}} & \\textbf{{FA (\\%)}} & \\textbf{{CU (\\%)}} & "
               f"{cat_headers} & \\textbf{{Latency (s)}} \\\\")
    out.append("\\midrule")

    if is_multi:
        per_seed = data["per_seed"]
        # AURA_Full reference for significance
        aura_fa = extract_per_seed_condition_metric(
            per_seed, "AURA_Full", ["summary", "avg_factual_accuracy"])

        for cond_key, cond_label in RQ2_CONDITIONS:
            if not any(cond_key in sd for sd in per_seed.values()):
                continue

            fa_vals = extract_per_seed_condition_metric(
                per_seed, cond_key, ["summary", "avg_factual_accuracy"])
            cu_vals = extract_per_seed_condition_metric(
                per_seed, cond_key, ["summary", "avg_context_utilization"])
            lat_vals = extract_per_seed_condition_metric(
                per_seed, cond_key, ["summary", "avg_latency"])

            fa_str = fmt_mean_std(mean(fa_vals), std(fa_vals), pct=True, decimals=2) if fa_vals else "--"
            cu_str = fmt_mean_std(mean(cu_vals), std(cu_vals), pct=True, decimals=2) if cu_vals else "--"
            lat_str = fmt_mean_std(mean(lat_vals), std(lat_vals), decimals=1) if lat_vals else "--"

            # Significance
            if cond_key != "AURA_Full" and fa_vals and len(fa_vals) >= 2:
                _, p = paired_ttest(aura_fa, fa_vals)
                p_boot = bootstrap_diff_pvalue(aura_fa, fa_vals)
                p_use = min(p, p_boot)
                stars = sig_stars(p_use)
                if stars:
                    fa_str += f"$^{{{stars}}}$"

            # Per-category
            cat_strs = []
            for cat in RQ2_CATEGORIES:
                cat_vals = extract_per_seed_condition_metric(
                    per_seed, cond_key, ["summary", "by_category", cat])
                if cat_vals:
                    cat_strs.append(fmt_mean_std(mean(cat_vals), std(cat_vals),
                                                  pct=True, decimals=2))
                else:
                    cat_strs.append("--")

            cats_joined = " & ".join(cat_strs)
            row = f"{cond_label} & {fa_str} & {cu_str} & {cats_joined} & {lat_str} \\\\"
            if cond_key == "AURA_Full":
                row = "\\midrule\n" + row
            out.append(row)
    else:
        for cond_key, cond_label in RQ2_CONDITIONS:
            if cond_key not in data:
                continue
            s = data[cond_key].get("summary", {})
            fa = s.get("avg_factual_accuracy", 0)
            cu = s.get("avg_context_utilization", 0)
            lat = s.get("avg_latency", 0)
            by_cat = s.get("by_category", {})
            cat_strs = [f"{by_cat.get(c, 0)*100:.1f}" for c in RQ2_CATEGORIES]
            cats_joined = " & ".join(cat_strs)
            row = (f"{cond_label} & {fa*100:.1f} & {cu*100:.1f} & "
                   f"{cats_joined} & {lat:.1f} \\\\")
            if cond_key == "AURA_Full":
                row = "\\midrule\n" + row
            out.append(row)

    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table*}")


# =========================================================================
# RQ3: Ablation Study Table
# =========================================================================

RQ3_CONDITIONS = [
    "Full (B=2)", "-Probing", "-Memory", "-Reflection",
    "-Memory&Reflect", "Vanilla (All off)",
]


def generate_rq3_table(output_lines: list):
    data, is_multi = load_rq_data("rq3")
    if data is None:
        return

    out = output_lines
    out.append("")
    out.append("% " + "=" * 60)
    out.append("% Table 3: Ablation Study (RQ3)")
    out.append("% " + "=" * 60)
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    if is_multi:
        n_seeds = len(data.get("seeds", []))
        out.append(f"\\caption{{Ablation study: contribution of each component. "
                   f"Mean $\\pm$ std over {n_seeds} seeds.}}")
    else:
        out.append("\\caption{Ablation study: contribution of each component to "
                   "grounding accuracy (GA) and factual accuracy (FA).}")
    out.append("\\label{tab:ablation}")
    out.append("\\begin{tabular}{lcccccc}")
    out.append("\\toprule")
    out.append("\\textbf{Configuration} & \\textbf{Probe} & \\textbf{Mem} & "
               "\\textbf{Refl} & \\textbf{GA} & \\textbf{FA (\\%)} & "
               "\\textbf{Latency (s)} \\\\")
    out.append("\\midrule")

    check = "\\cmark"
    cross = "\\xmark"

    if is_multi:
        per_seed = data["per_seed"]
        # Reference: Full (B=2)
        full_ga = extract_per_seed_flat_metric(per_seed, "Full (B=2)", "avg_ga")

        for name in RQ3_CONDITIONS:
            if not any(name in sd for sd in per_seed.values()):
                continue

            # Get one seed's data to read boolean flags
            sample = None
            for sd in per_seed.values():
                if name in sd:
                    sample = sd[name]
                    break

            p = check if sample.get("probe") else cross
            m = check if sample.get("memory") else cross
            r = check if sample.get("reflection") else cross

            ga_vals = extract_per_seed_flat_metric(per_seed, name, "avg_ga")
            fa_vals = extract_per_seed_flat_metric(per_seed, name, "avg_fa")
            lat_vals = extract_per_seed_flat_metric(per_seed, name, "avg_latency")

            ga_str = fmt_mean_std(mean(ga_vals), std(ga_vals), decimals=4) if ga_vals else "--"
            fa_str = fmt_mean_std(mean(fa_vals), std(fa_vals), pct=True, decimals=2) if fa_vals else "--"
            lat_str = fmt_mean_std(mean(lat_vals), std(lat_vals), decimals=1) if lat_vals else "--"

            # Delta from Full
            if name != "Full (B=2)" and ga_vals and full_ga and len(ga_vals) >= 2:
                _, p_val = paired_ttest(full_ga, ga_vals)
                stars = sig_stars(p_val)
                if stars:
                    ga_str += f"$^{{{stars}}}$"

            out.append(f"{name} & {p} & {m} & {r} & {ga_str} & {fa_str} & {lat_str} \\\\")
    else:
        for name in RQ3_CONDITIONS:
            if name not in data:
                continue
            s = data[name]
            p = check if s.get("probe") else cross
            m = check if s.get("memory") else cross
            r = check if s.get("reflection") else cross
            ga = s.get("avg_ga", 0)
            fa = s.get("avg_fa", 0)
            lat = s.get("avg_latency", 0)
            out.append(f"{name} & {p} & {m} & {r} & {ga:.4f} & {fa*100:.1f} & {lat:.1f} \\\\")

    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")


# =========================================================================
# RQ6: Probe Budget vs GA/Latency (data for plotting + table)
# =========================================================================

def generate_rq6_table_and_data(output_lines: list):
    data, is_multi = load_rq_data("rq6")
    if data is None:
        return

    out = output_lines
    out.append("")
    out.append("% " + "=" * 60)
    out.append("% Table/Figure data: Probe Budget Pareto (RQ6)")
    out.append("% " + "=" * 60)

    if is_multi:
        per_seed = data["per_seed"]
        seeds = data.get("seeds", [])
        n_seeds = len(seeds)

        # Determine budget range from first seed
        first_seed_data = list(per_seed.values())[0]
        budgets = sorted(first_seed_data.get("per_budget", {}).keys(), key=int)

        out.append(f"% Probe budget Pareto data (mean +/- std, {n_seeds} seeds)")
        out.append("\\begin{table}[t]")
        out.append("\\centering")
        out.append(f"\\caption{{Probe budget $B$ vs.~GA and latency. "
                   f"Mean $\\pm$ std over {n_seeds} seeds.}}")
        out.append("\\label{tab:budget}")
        out.append("\\begin{tabular}{cccc}")
        out.append("\\toprule")
        out.append("\\textbf{Budget $B$} & \\textbf{GA} & "
                   "\\textbf{Avg Latency (s)} & \\textbf{$N$} \\\\")
        out.append("\\midrule")

        # Also prepare pgfplots data
        pgf_lines = ["budget ga ga_err latency latency_err"]

        for b in budgets:
            ga_vals = extract_per_seed_budget_metric(per_seed, b, "avg_ga")
            lat_vals = extract_per_seed_budget_metric(per_seed, b, "avg_latency")
            n_vals = extract_per_seed_budget_metric(per_seed, b, "num_ga_judgments")

            ga_str = fmt_mean_std(mean(ga_vals), std(ga_vals)) if ga_vals else "--"
            lat_str = fmt_mean_std(mean(lat_vals), std(lat_vals), decimals=1) if lat_vals else "--"
            n_str = str(int(mean(n_vals))) if n_vals else "--"

            out.append(f"{b} & {ga_str} & {lat_str} & {n_str} \\\\")

            if ga_vals and lat_vals:
                pgf_lines.append(f"{b} {mean(ga_vals):.4f} {std(ga_vals):.4f} "
                                 f"{mean(lat_vals):.2f} {std(lat_vals):.2f}")

        out.append("\\bottomrule")
        out.append("\\end{tabular}")
        out.append("\\end{table}")

        # pgfplots filecontents
        out.append("")
        out.append("% pgfplots data for Pareto figure")
        out.append("\\begin{filecontents}{rq6_pareto.dat}")
        for line in pgf_lines:
            out.append(line)
        out.append("\\end{filecontents}")

    else:
        per_budget = data.get("per_budget", {})
        frontier = data.get("pareto_frontier", [])

        out.append("\\begin{table}[t]")
        out.append("\\centering")
        out.append("\\caption{Probe budget $B$ vs.~GA and latency (single run).}")
        out.append("\\label{tab:budget}")
        out.append("\\begin{tabular}{cccc}")
        out.append("\\toprule")
        out.append("\\textbf{Budget $B$} & \\textbf{GA} & "
                   "\\textbf{Avg Latency (s)} & \\textbf{Pareto?} \\\\")
        out.append("\\midrule")

        frontier_budgets = {p["budget"] for p in frontier}
        for b, entry in sorted(per_budget.items(), key=lambda x: int(x[0])):
            ga = entry.get("avg_ga", 0)
            lat = entry.get("avg_latency", 0)
            on_frontier = "\\cmark" if int(b) in frontier_budgets else ""
            out.append(f"{b} & {ga:.3f} & {lat:.1f} & {on_frontier} \\\\")

        out.append("\\bottomrule")
        out.append("\\end{tabular}")
        out.append("\\end{table}")

        # pgfplots
        out.append("")
        out.append("\\begin{filecontents}{rq6_pareto.dat}")
        out.append("budget ga latency")
        for b, entry in sorted(per_budget.items(), key=lambda x: int(x[0])):
            out.append(f"{b} {entry.get('avg_ga', 0)} {entry.get('avg_latency', 0)}")
        out.append("\\end{filecontents}")


# =========================================================================
# Significance Summary Table
# =========================================================================

def generate_significance_summary(output_lines: list):
    """Generate a standalone significance table: AURA_Full vs each baseline."""
    out = output_lines

    # Try multi-seed RQ1
    data_rq1, is_multi_rq1 = load_rq_data("rq1")
    data_rq2, is_multi_rq2 = load_rq_data("rq2")

    if not is_multi_rq1 and not is_multi_rq2:
        out.append("")
        out.append("% Significance tests require multi-seed data (--multi-seed).")
        out.append("% Re-run experiments with: python -m evaluation.run_experiments "
                   "--multi-seed --seeds 42 123 456 789 2024")
        return

    out.append("")
    out.append("% " + "=" * 60)
    out.append("% Significance Summary: Paired t-test + Bootstrap permutation")
    out.append("% " + "=" * 60)
    out.append("\\begin{table}[t]")
    out.append("\\centering")
    out.append("\\caption{Statistical significance: \\sys{} (Full) vs.~baselines. "
               "$p$-values from paired $t$-test / bootstrap permutation test.}")
    out.append("\\label{tab:significance}")
    out.append("\\begin{tabular}{lcccc}")
    out.append("\\toprule")
    out.append("\\textbf{Baseline} & \\textbf{$\\Delta$GA} & \\textbf{$p$(GA)} & "
               "\\textbf{$\\Delta$FA} & \\textbf{$p$(FA)} \\\\")
    out.append("\\midrule")

    baselines = [
        ("Vanilla_LLM", "Vanilla LLM"),
        ("Static_Context", "Static Context"),
        ("ReAct", "ReAct"),
        ("AURA_NoProbe", "\\sys{} (No Probe)"),
    ]

    for cond_key, cond_label in baselines:
        ga_delta_str, ga_p_str = "--", "--"
        fa_delta_str, fa_p_str = "--", "--"

        if is_multi_rq1 and data_rq1:
            per_seed = data_rq1["per_seed"]
            aura_ga = extract_per_seed_condition_metric(
                per_seed, "AURA_Full", ["summary", "overall_grounding_accuracy"])
            base_ga = extract_per_seed_condition_metric(
                per_seed, cond_key, ["summary", "overall_grounding_accuracy"])
            if aura_ga and base_ga and len(aura_ga) >= 2:
                delta = mean(aura_ga) - mean(base_ga)
                t_stat, p_t = paired_ttest(aura_ga, base_ga)
                p_boot = bootstrap_diff_pvalue(aura_ga, base_ga)
                ga_delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
                ga_p_str = f"{p_t:.4f} / {p_boot:.4f}{sig_stars(min(p_t, p_boot))}"

        if is_multi_rq2 and data_rq2:
            per_seed = data_rq2["per_seed"]
            aura_fa = extract_per_seed_condition_metric(
                per_seed, "AURA_Full", ["summary", "avg_factual_accuracy"])
            base_fa = extract_per_seed_condition_metric(
                per_seed, cond_key, ["summary", "avg_factual_accuracy"])
            if aura_fa and base_fa and len(aura_fa) >= 2:
                delta = mean(aura_fa) - mean(base_fa)
                t_stat, p_t = paired_ttest(aura_fa, base_fa)
                p_boot = bootstrap_diff_pvalue(aura_fa, base_fa)
                fa_delta_str = f"+{delta*100:.1f}\\%" if delta >= 0 else f"{delta*100:.1f}\\%"
                fa_p_str = f"{p_t:.4f} / {p_boot:.4f}{sig_stars(min(p_t, p_boot))}"

        out.append(f"{cond_label} & {ga_delta_str} & {ga_p_str} & "
                   f"{fa_delta_str} & {fa_p_str} \\\\")

    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    out.append("\\end{table}")


# =========================================================================
# Plain-text summary (for quick inspection)
# =========================================================================

def generate_plaintext_summary(output_lines: list):
    """Print a human-readable summary of key results."""
    out = output_lines
    out.append("")
    out.append("%" + "-" * 70)
    out.append("% PLAIN-TEXT SUMMARY (not LaTeX)")
    out.append("%" + "-" * 70)

    for rq in ["rq1", "rq2", "rq3", "rq6"]:
        data, is_multi = load_rq_data(rq)
        if data is None:
            continue

        out.append(f"% {rq.upper()}: {'multi-seed' if is_multi else 'single-seed'}")
        if is_multi and "statistics" in data:
            stats = data["statistics"]
            if isinstance(stats, dict):
                if "mean" in stats:
                    out.append(f"%   Overall: {stats['mean']:.4f} +/- {stats['std']:.4f}")
                else:
                    for cond, cond_stats in sorted(stats.items()):
                        if isinstance(cond_stats, dict) and "mean" in cond_stats:
                            out.append(f"%   {cond}: {cond_stats['mean']:.4f} "
                                       f"+/- {cond_stats['std']:.4f}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from AURA experiment results")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for .tex files (default: stdout)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override results directory")
    args = parser.parse_args()

    global RESULTS_DIR
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)

    print("=" * 60)
    print("AURA Statistical Analysis & LaTeX Table Generator")
    print(f"Results directory: {RESULTS_DIR}")
    print("=" * 60)

    # Check what files are available
    print("\nAvailable result files:")
    for f in sorted(RESULTS_DIR.glob("*.json")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:45s} ({size_kb:7.1f} KB)")

    # Generate all tables
    output_lines: list = []

    output_lines.append("% Auto-generated by generate_tables.py")
    output_lines.append("% Requires: \\usepackage{booktabs}, \\newcommand{\\sys}{AURA}")
    output_lines.append("% Requires: \\usepackage{pifont}")
    output_lines.append("% \\newcommand{\\cmark}{\\ding{51}}")
    output_lines.append("% \\newcommand{\\xmark}{\\ding{55}}")
    output_lines.append("")

    print("\n--- Generating tables ---")
    generate_rq1_table(output_lines)
    generate_rq2_table(output_lines)
    generate_rq3_table(output_lines)
    generate_rq6_table_and_data(output_lines)
    generate_significance_summary(output_lines)
    generate_plaintext_summary(output_lines)

    full_output = "\n".join(output_lines)

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "generated_tables.tex"
        with open(out_path, "w") as f:
            f.write(full_output)
        print(f"\nLaTeX tables written to {out_path}")
    else:
        print("\n" + "=" * 60)
        print("LATEX OUTPUT (copy below into paper)")
        print("=" * 60)
        print(full_output)

    # Also write a JSON summary of p-values for programmatic use
    pvalue_summary = _compute_pvalue_summary()
    if pvalue_summary:
        pval_path = RESULTS_DIR / "significance_pvalues.json"
        with open(pval_path, "w") as f:
            json.dump(pvalue_summary, f, indent=2)
        print(f"\nP-value summary written to {pval_path}")

    print("\nDone.")


def _compute_pvalue_summary() -> Optional[dict]:
    """Compute p-values and return as dict for programmatic use."""
    summary = {}

    for rq, metric_path_map in [
        ("rq1", {"overall_ga": ["summary", "overall_grounding_accuracy"]}),
        ("rq2", {"avg_fa": ["summary", "avg_factual_accuracy"]}),
    ]:
        data, is_multi = load_rq_data(rq)
        if not is_multi or data is None:
            continue

        per_seed = data["per_seed"]
        rq_summary = {}

        for metric_name, metric_path in metric_path_map.items():
            aura_vals = extract_per_seed_condition_metric(
                per_seed, "AURA_Full", metric_path)

            baselines = ["Vanilla_LLM", "Static_Context", "ReAct", "AURA_NoProbe"]
            for bl in baselines:
                bl_vals = extract_per_seed_condition_metric(per_seed, bl, metric_path)
                if aura_vals and bl_vals and len(aura_vals) >= 2:
                    t_stat, p_t = paired_ttest(aura_vals, bl_vals)
                    p_boot = bootstrap_diff_pvalue(aura_vals, bl_vals)
                    delta = mean(aura_vals) - mean(bl_vals)

                    key = f"{metric_name}_vs_{bl}"
                    rq_summary[key] = {
                        "aura_full_mean": round(mean(aura_vals), 4),
                        "aura_full_std": round(std(aura_vals), 4),
                        "baseline_mean": round(mean(bl_vals), 4),
                        "baseline_std": round(std(bl_vals), 4),
                        "delta": round(delta, 4),
                        "t_statistic": round(t_stat, 4),
                        "p_ttest": round(p_t, 4),
                        "p_bootstrap": round(p_boot, 4),
                        "significant_005": min(p_t, p_boot) < 0.05,
                        "n_seeds": len(aura_vals),
                    }

        if rq_summary:
            summary[rq] = rq_summary

    # RQ3: Full vs ablations
    data_rq3, is_multi_rq3 = load_rq_data("rq3")
    if is_multi_rq3 and data_rq3:
        per_seed = data_rq3["per_seed"]
        full_ga = extract_per_seed_flat_metric(per_seed, "Full (B=2)", "avg_ga")
        rq3_summary = {}
        for abl in ["-Probing", "-Memory", "-Reflection", "-Memory&Reflect", "Vanilla (All off)"]:
            abl_ga = extract_per_seed_flat_metric(per_seed, abl, "avg_ga")
            if full_ga and abl_ga and len(full_ga) >= 2:
                t_stat, p_t = paired_ttest(full_ga, abl_ga)
                p_boot = bootstrap_diff_pvalue(full_ga, abl_ga)
                delta = mean(full_ga) - mean(abl_ga)
                rq3_summary[f"ga_Full_vs_{abl}"] = {
                    "full_mean": round(mean(full_ga), 4),
                    "ablation_mean": round(mean(abl_ga), 4),
                    "delta": round(delta, 4),
                    "p_ttest": round(p_t, 4),
                    "p_bootstrap": round(p_boot, 4),
                    "significant_005": min(p_t, p_boot) < 0.05,
                }
        if rq3_summary:
            summary["rq3"] = rq3_summary

    return summary if summary else None


if __name__ == "__main__":
    main()
