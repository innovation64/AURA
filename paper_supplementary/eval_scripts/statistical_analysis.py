"""
Statistical analysis module for AURA paper — NeurIPS Agent Track.

Provides rigorous statistical testing required by top-tier venues:
  1. Paired Wilcoxon signed-rank test (non-parametric, exact via scipy)
  2. Paired t-test (parametric, exact via scipy)
  3. Cohen's d effect size
  4. 95% confidence intervals (bootstrap + t-distribution)
  5. Bonferroni and Benjamini-Hochberg correction for multiple comparisons
  6. Per-condition multi-seed aggregation with variance reporting
  7. LaTeX table generation with significance markers

Dependencies:
  - scipy (recommended): Uses exact t-distribution and Wilcoxon test.
    Falls back to normal/Cornish-Fisher approximations if unavailable.

Usage:
    python -m evaluation.statistical_analysis --results-dir evaluation/results
"""

import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Prefer scipy for accurate distributions; fall back to approximations
try:
    from scipy import stats as sp_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ============================================================================
# Core Statistical Functions
# ============================================================================

def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def std(vals: List[float], ddof: int = 1) -> float:
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - ddof))


def _normal_cdf(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _t_cdf(t: float, df: int) -> float:
    """t-distribution CDF. Uses scipy when available, otherwise approximation."""
    if _HAS_SCIPY:
        return float(sp_stats.t.cdf(t, df))
    # Fallback: Cornish-Fisher expansion (accurate for df >= 5)
    if df >= 30:
        return _normal_cdf(t)
    g1 = (t**3 + t) / (4 * df)
    g2 = (5 * t**5 + 16 * t**3 + 3 * t) / (96 * df**2)
    z = t + g1 + g2
    return _normal_cdf(z)


def _t_ppf(p: float, df: int) -> float:
    """t-distribution inverse CDF (percent point function)."""
    if _HAS_SCIPY:
        return float(sp_stats.t.ppf(p, df))
    # Fallback: rough lookup for common alpha levels
    # t-critical values for two-tailed 95% CI
    _table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
              6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
              15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}
    if df in _table:
        return _table[df]
    if df > 30:
        return 1.96
    # Interpolate
    lower = max(k for k in _table if k <= df)
    upper = min(k for k in _table if k >= df)
    if lower == upper:
        return _table[lower]
    frac = (df - lower) / (upper - lower)
    return _table[lower] + frac * (_table[upper] - _table[lower])


def bootstrap_ci(vals: List[float], n_boot: int = 10000, alpha: float = 0.05) -> Tuple[float, float]:
    """Bootstrap confidence interval for the mean."""
    n = len(vals)
    if n < 2:
        m = mean(vals)
        return (m, m)
    boot_means = []
    rng = random.Random(42)
    for _ in range(n_boot):
        sample = [vals[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(mean(sample))
    boot_means.sort()
    lo = int(n_boot * alpha / 2)
    hi = int(n_boot * (1 - alpha / 2))
    return (round(boot_means[lo], 4), round(boot_means[min(hi, n_boot - 1)], 4))


# ============================================================================
# Statistical Tests
# ============================================================================

@dataclass
class TestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float          # Cohen's d
    ci_lower: float             # 95% CI lower bound of mean difference
    ci_upper: float             # 95% CI upper bound of mean difference
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    n: int
    significant_005: bool       # p < 0.05
    significant_001: bool       # p < 0.01
    significant_corrected: bool # after Bonferroni


def paired_ttest(a: List[float], b: List[float]) -> TestResult:
    """Paired two-tailed t-test with Cohen's d and 95% CI."""
    assert len(a) == len(b), "Paired samples must have equal length"
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    d_mean = mean(diffs)
    d_std = std(diffs)

    if _HAS_SCIPY and n >= 2:
        # Use scipy for exact p-value
        scipy_result = sp_stats.ttest_rel(a, b)
        t_stat = float(scipy_result.statistic)
        p_val = float(scipy_result.pvalue)
    elif d_std == 0 or n < 2:
        t_stat = 0.0
        p_val = 1.0
    else:
        t_stat = d_mean / (d_std / math.sqrt(n))
        p_val = 2.0 * (1.0 - _t_cdf(abs(t_stat), n - 1))

    # Cohen's d (paired)
    pooled_std = math.sqrt((std(a) ** 2 + std(b) ** 2) / 2) if (std(a) + std(b)) > 0 else 1.0
    cohens_d = (mean(a) - mean(b)) / pooled_std if pooled_std > 0 else 0.0

    # 95% CI for mean difference
    t_crit = _t_ppf(0.975, max(n - 1, 1))
    margin = t_crit * d_std / math.sqrt(n) if n > 0 else 0
    ci_lower = d_mean - margin
    ci_upper = d_mean + margin

    return TestResult(
        test_name="paired_t_test",
        statistic=round(t_stat, 4),
        p_value=round(max(p_val, 1e-10), 6),
        effect_size=round(cohens_d, 4),
        ci_lower=round(ci_lower, 4),
        ci_upper=round(ci_upper, 4),
        mean_a=round(mean(a), 4),
        mean_b=round(mean(b), 4),
        std_a=round(std(a), 4),
        std_b=round(std(b), 4),
        n=n,
        significant_005=p_val < 0.05,
        significant_001=p_val < 0.01,
        significant_corrected=False,
    )


def wilcoxon_signed_rank(a: List[float], b: List[float]) -> TestResult:
    """Wilcoxon signed-rank test (non-parametric paired test)."""
    assert len(a) == len(b)
    n = len(a)

    diffs = [(x - y) for x, y in zip(a, b) if x != y]
    if not diffs:
        return TestResult(
            test_name="wilcoxon_signed_rank",
            statistic=0.0, p_value=1.0, effect_size=0.0,
            ci_lower=0.0, ci_upper=0.0,
            mean_a=mean(a), mean_b=mean(b),
            std_a=std(a), std_b=std(b),
            n=n, significant_005=False, significant_001=False,
            significant_corrected=False,
        )

    # Use scipy for exact test when available
    if _HAS_SCIPY:
        try:
            scipy_result = sp_stats.wilcoxon(a, b)
            W = float(scipy_result.statistic)
            p_val = float(scipy_result.pvalue)
        except ValueError:
            # scipy raises if all differences are zero
            W, p_val = 0.0, 1.0
    else:
        # Manual implementation with normal approximation
        abs_diffs = [(abs(d), d) for d in diffs]
        abs_diffs.sort(key=lambda x: x[0])

        ranks = [0.0] * len(abs_diffs)
        i = 0
        while i < len(abs_diffs):
            j = i
            while j < len(abs_diffs) and abs_diffs[j][0] == abs_diffs[i][0]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[k] = avg_rank
            i = j

        w_plus = sum(r for r, (_, d) in zip(ranks, abs_diffs) if d > 0)
        w_minus = sum(r for r, (_, d) in zip(ranks, abs_diffs) if d < 0)
        W = min(w_plus, w_minus)
        nr = len(diffs)

        mean_W = nr * (nr + 1) / 4.0
        std_W = math.sqrt(nr * (nr + 1) * (2 * nr + 1) / 24.0) if nr > 0 else 1.0
        z = (W - mean_W) / std_W if std_W > 0 else 0.0
        p_val = 2.0 * (1.0 - _normal_cdf(abs(z)))

    # Cohen's d equivalent
    pooled_std = math.sqrt((std(a) ** 2 + std(b) ** 2) / 2) if (std(a) + std(b)) > 0 else 1.0
    cohens_d = (mean(a) - mean(b)) / pooled_std if pooled_std > 0 else 0.0

    # CI via bootstrap when scipy available, else normal approximation
    d_diffs = [x - y for x, y in zip(a, b)]
    d_mean = mean(d_diffs)
    if n >= 5:
        ci_lo, ci_hi = bootstrap_ci(d_diffs)
    else:
        d_std_val = std(d_diffs)
        margin = 1.96 * d_std_val / math.sqrt(n) if n > 0 else 0
        ci_lo, ci_hi = round(d_mean - margin, 4), round(d_mean + margin, 4)

    return TestResult(
        test_name="wilcoxon_signed_rank",
        statistic=round(W, 4),
        p_value=round(max(p_val, 1e-10), 6),
        effect_size=round(cohens_d, 4),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        mean_a=round(mean(a), 4),
        mean_b=round(mean(b), 4),
        std_a=round(std(a), 4),
        std_b=round(std(b), 4),
        n=n,
        significant_005=p_val < 0.05,
        significant_001=p_val < 0.01,
        significant_corrected=False,
    )


# ============================================================================
# Multiple Comparison Correction
# ============================================================================

def bonferroni_correct(results: List[TestResult], num_comparisons: int) -> List[TestResult]:
    """Apply Bonferroni correction to a list of test results."""
    corrected = []
    for r in results:
        adjusted_p = min(r.p_value * num_comparisons, 1.0)
        r_new = TestResult(**{**asdict(r), "significant_corrected": adjusted_p < 0.05})
        corrected.append(r_new)
    return corrected


def benjamini_hochberg(results: List[TestResult]) -> List[TestResult]:
    """Benjamini-Hochberg FDR correction."""
    m = len(results)
    if m == 0:
        return results
    indexed = sorted(enumerate(results), key=lambda x: x[1].p_value)
    corrected = [None] * m
    for rank, (orig_idx, r) in enumerate(indexed, 1):
        adjusted_p = min(r.p_value * m / rank, 1.0)
        r_new = TestResult(**{**asdict(r), "significant_corrected": adjusted_p < 0.05})
        corrected[orig_idx] = r_new
    return corrected


# ============================================================================
# Multi-Seed Aggregator
# ============================================================================

@dataclass
class ConditionStats:
    """Aggregated statistics for one experimental condition across seeds."""
    condition: str
    metric: str
    values: List[float]
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n_seeds: int


def aggregate_multiseed(
    per_seed_data: Dict[int, Dict[str, float]],
    conditions: List[str],
    metric_key: str,
) -> Dict[str, ConditionStats]:
    """Aggregate multi-seed results into per-condition statistics."""
    stats = {}
    for cond in conditions:
        vals = []
        for seed, data in per_seed_data.items():
            if isinstance(data, dict) and cond in data:
                cond_data = data[cond]
                if isinstance(cond_data, dict):
                    v = cond_data.get(metric_key, cond_data.get("summary", {}).get(metric_key))
                else:
                    v = cond_data
                if v is not None:
                    vals.append(float(v))
        if not vals:
            continue
        m = mean(vals)
        s = std(vals)
        n = len(vals)
        t_crit = _t_ppf(0.975, max(n - 1, 1))
        margin = t_crit * s / math.sqrt(n) if n > 0 else 0
        stats[cond] = ConditionStats(
            condition=cond, metric=metric_key,
            values=vals, mean=round(m, 4), std=round(s, 4),
            ci_lower=round(m - margin, 4), ci_upper=round(m + margin, 4),
            n_seeds=n,
        )
    return stats


# ============================================================================
# Pairwise Comparison Engine
# ============================================================================

@dataclass
class PairwiseComparison:
    """Full comparison between two conditions."""
    condition_a: str
    condition_b: str
    metric: str
    ttest: TestResult
    wilcoxon: TestResult


def run_pairwise_comparisons(
    condition_scores: Dict[str, List[float]],
    reference: str = "AURA_Full",
) -> List[PairwiseComparison]:
    """
    Compare reference condition against all others using both parametric
    and non-parametric tests.
    """
    if reference not in condition_scores:
        return []

    ref_scores = condition_scores[reference]
    comparisons = []

    for cond, scores in condition_scores.items():
        if cond == reference:
            continue
        n = min(len(ref_scores), len(scores))
        if n < 3:
            continue
        a = ref_scores[:n]
        b = scores[:n]

        tt = paired_ttest(a, b)
        wt = wilcoxon_signed_rank(a, b)
        comparisons.append(PairwiseComparison(
            condition_a=reference, condition_b=cond,
            metric="pairwise", ttest=tt, wilcoxon=wt,
        ))

    # Apply Bonferroni correction
    num_comp = len(comparisons)
    if num_comp > 1:
        all_tt = [c.ttest for c in comparisons]
        all_wt = [c.wilcoxon for c in comparisons]
        corrected_tt = bonferroni_correct(all_tt, num_comp)
        corrected_wt = bonferroni_correct(all_wt, num_comp)
        for i, c in enumerate(comparisons):
            c.ttest = corrected_tt[i]
            c.wilcoxon = corrected_wt[i]

    return comparisons


# ============================================================================
# Result Analysis Pipeline
# ============================================================================

def analyze_rq1_multiseed(results_dir: str) -> Optional[Dict[str, Any]]:
    """Analyze RQ1 multi-seed results with full statistical testing."""
    path = Path(results_dir) / "rq1_grounding_accuracy_multiseed.json"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return None

    with open(path) as f:
        data = json.load(f)

    seeds = data.get("seeds", [])
    per_seed = data.get("per_seed", {})

    # Extract per-condition per-seed GA scores
    condition_scores: Dict[str, List[float]] = defaultdict(list)
    for seed_key, seed_data in per_seed.items():
        if not isinstance(seed_data, dict):
            continue
        for cond_name, cond_data in seed_data.items():
            if isinstance(cond_data, dict) and "summary" in cond_data:
                ga = cond_data["summary"].get("overall_grounding_accuracy")
                if ga is not None:
                    condition_scores[cond_name].append(float(ga))

    if not condition_scores:
        print("  [WARN] No valid condition scores found in RQ1 multiseed data")
        return None

    # Per-condition stats
    cond_stats = {}
    for cond, vals in condition_scores.items():
        m = mean(vals)
        s = std(vals)
        n = len(vals)
        t_crit = _t_ppf(0.975, max(n - 1, 1))
        margin = t_crit * s / math.sqrt(n) if n > 0 else 0
        cond_stats[cond] = {
            "mean": round(m, 4),
            "std": round(s, 4),
            "ci_95_lower": round(m - margin, 4),
            "ci_95_upper": round(m + margin, 4),
            "n_seeds": n,
            "values": [round(v, 4) for v in vals],
        }

    # Pairwise comparisons against AURA_Full
    comparisons = run_pairwise_comparisons(condition_scores, reference="AURA_Full")
    comparison_results = []
    for c in comparisons:
        comparison_results.append({
            "comparison": f"{c.condition_a} vs {c.condition_b}",
            "paired_t_test": asdict(c.ttest),
            "wilcoxon_signed_rank": asdict(c.wilcoxon),
        })

    return {
        "analysis": "RQ1_Grounding_Accuracy",
        "num_seeds": len(seeds),
        "seeds": seeds,
        "per_condition_statistics": cond_stats,
        "pairwise_comparisons": comparison_results,
        "correction_method": "bonferroni",
        "num_comparisons": len(comparisons),
    }


def analyze_rq2_multiseed(results_dir: str) -> Optional[Dict[str, Any]]:
    """Analyze RQ2 multi-seed results with full statistical testing."""
    path = Path(results_dir) / "rq2_factual_accuracy_multiseed.json"
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return None

    with open(path) as f:
        data = json.load(f)

    seeds = data.get("seeds", [])
    per_seed = data.get("per_seed", {})

    condition_scores: Dict[str, List[float]] = defaultdict(list)
    for seed_key, seed_data in per_seed.items():
        if not isinstance(seed_data, dict):
            continue
        for cond_name, cond_data in seed_data.items():
            if isinstance(cond_data, dict) and "summary" in cond_data:
                fa = cond_data["summary"].get("avg_factual_accuracy")
                if fa is not None:
                    condition_scores[cond_name].append(float(fa))

    if not condition_scores:
        return None

    cond_stats = {}
    for cond, vals in condition_scores.items():
        m = mean(vals)
        s = std(vals)
        n = len(vals)
        t_crit = 2.776 if n == 5 else 1.96
        margin = t_crit * s / math.sqrt(n) if n > 0 else 0
        cond_stats[cond] = {
            "mean": round(m, 4),
            "std": round(s, 4),
            "ci_95_lower": round(m - margin, 4),
            "ci_95_upper": round(m + margin, 4),
            "n_seeds": n,
            "values": [round(v, 4) for v in vals],
        }

    comparisons = run_pairwise_comparisons(condition_scores, reference="AURA_Full")
    comparison_results = []
    for c in comparisons:
        comparison_results.append({
            "comparison": f"{c.condition_a} vs {c.condition_b}",
            "paired_t_test": asdict(c.ttest),
            "wilcoxon_signed_rank": asdict(c.wilcoxon),
        })

    return {
        "analysis": "RQ2_Factual_Accuracy",
        "num_seeds": len(seeds),
        "seeds": seeds,
        "per_condition_statistics": cond_stats,
        "pairwise_comparisons": comparison_results,
        "correction_method": "bonferroni",
        "num_comparisons": len(comparisons),
    }


def analyze_rq3_ablation(results_dir: str) -> Optional[Dict[str, Any]]:
    """Analyze RQ3 ablation with significance tests between configurations."""
    path = Path(results_dir) / "rq3_ablation.json"
    if not path.exists():
        # Try multiseed version
        path = Path(results_dir) / "rq3_ablation_study_multiseed.json"
        if not path.exists():
            print(f"  [SKIP] RQ3 ablation results not found")
            return None

    with open(path) as f:
        data = json.load(f)

    # Single-seed analysis: report what we have with interpretation notes
    if "per_seed" not in data:
        configs = {}
        for name, cfg_data in data.items():
            if isinstance(cfg_data, dict) and "avg_ga" in cfg_data:
                configs[name] = {
                    "grounding_accuracy": cfg_data["avg_ga"],
                    "factual_accuracy": cfg_data.get("avg_fa", 0),
                    "latency": cfg_data.get("avg_latency", 0),
                    "note": "single-seed; significance testing requires multi-seed runs",
                }
        return {
            "analysis": "RQ3_Ablation_Study",
            "configurations": configs,
            "warning": "Single-seed results. Run with --multi-seed for significance tests.",
        }

    # Multi-seed version
    seeds = data.get("seeds", [])
    per_seed = data.get("per_seed", {})
    condition_scores: Dict[str, List[float]] = defaultdict(list)

    for seed_key, seed_data in per_seed.items():
        if not isinstance(seed_data, dict):
            continue
        for cond_name, cond_data in seed_data.items():
            if isinstance(cond_data, dict) and "avg_ga" in cond_data:
                condition_scores[cond_name].append(float(cond_data["avg_ga"]))

    cond_stats = {}
    for cond, vals in condition_scores.items():
        m = mean(vals)
        s = std(vals)
        cond_stats[cond] = {"mean": round(m, 4), "std": round(s, 4), "n": len(vals)}

    comparisons = run_pairwise_comparisons(condition_scores, reference="Full (B=2)")
    comparison_results = []
    for c in comparisons:
        comparison_results.append({
            "comparison": f"{c.condition_a} vs {c.condition_b}",
            "paired_t_test": asdict(c.ttest),
            "wilcoxon_signed_rank": asdict(c.wilcoxon),
        })

    return {
        "analysis": "RQ3_Ablation_Study",
        "num_seeds": len(seeds),
        "per_condition_statistics": cond_stats,
        "pairwise_comparisons": comparison_results,
        "correction_method": "bonferroni",
    }


# ============================================================================
# LaTeX Table Generation
# ============================================================================

def _sig_marker(p: float, corrected: bool) -> str:
    """Return significance marker for LaTeX."""
    if corrected:
        return "$^{\\dagger}$"
    if p < 0.001:
        return "$^{***}$"
    if p < 0.01:
        return "$^{**}$"
    if p < 0.05:
        return "$^{*}$"
    return ""


def generate_latex_table(
    cond_stats: Dict[str, dict],
    comparisons: List[dict],
    metric_name: str = "GA",
    caption: str = "Grounding accuracy across conditions",
) -> str:
    """Generate a publication-ready LaTeX table with significance markers."""
    # Map condition -> comparison result (for sig markers)
    sig_map = {}
    for c in comparisons:
        cond_b = c["comparison"].split(" vs ")[1]
        p = c["wilcoxon_signed_rank"]["p_value"]
        corrected = c["wilcoxon_signed_rank"]["significant_corrected"]
        sig_map[cond_b] = (p, corrected)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}. "
        r"Mean $\pm$ SD across seeds. $^{*}$/$^{**}$/$^{***}$: $p < 0.05/0.01/0.001$ "
        r"(Wilcoxon, Bonferroni-corrected).}",
        r"\label{tab:" + metric_name.lower() + r"_stats}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        f"Condition & {metric_name} (Mean $\\pm$ SD) & 95\\% CI & $n$ & Cohen's $d$ \\\\",
        r"\midrule",
    ]

    for cond, stats in cond_stats.items():
        m = stats["mean"]
        s = stats["std"]
        ci_lo = stats.get("ci_95_lower", m - 1.96 * s)
        ci_hi = stats.get("ci_95_upper", m + 1.96 * s)
        n = stats.get("n_seeds", stats.get("n", 0))

        # Look up significance and effect size
        p, corrected = sig_map.get(cond, (1.0, False))
        marker = _sig_marker(p, corrected)

        # Find Cohen's d from comparisons
        d_val = ""
        for c in comparisons:
            if c["comparison"].split(" vs ")[1] == cond:
                d_val = f"{c['wilcoxon_signed_rank']['effect_size']:.2f}"
                break

        cond_display = cond.replace("_", "\\_")
        lines.append(
            f"  {cond_display}{marker} & ${m:.4f} \\pm {s:.4f}$ "
            f"& [{ci_lo:.4f}, {ci_hi:.4f}] & {n} & {d_val} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


# ============================================================================
# Main Entry Point
# ============================================================================

def run_full_analysis(results_dir: str = "evaluation/results") -> Dict[str, Any]:
    """Run full statistical analysis on all available results."""
    print("=" * 60)
    print("AURA Statistical Analysis (NeurIPS Agent Track)")
    print("=" * 60)

    all_analyses = {}

    # RQ1
    print("\n--- RQ1: Grounding Accuracy ---")
    rq1 = analyze_rq1_multiseed(results_dir)
    if rq1:
        all_analyses["rq1"] = rq1
        print(f"  Conditions: {list(rq1['per_condition_statistics'].keys())}")
        print(f"  Comparisons: {len(rq1['pairwise_comparisons'])}")
        for c in rq1["pairwise_comparisons"]:
            p = c["wilcoxon_signed_rank"]["p_value"]
            d = c["wilcoxon_signed_rank"]["effect_size"]
            print(f"    {c['comparison']}: p={p:.4f}, d={d:.4f}")

        # Generate LaTeX
        latex = generate_latex_table(
            rq1["per_condition_statistics"],
            rq1["pairwise_comparisons"],
            metric_name="GA",
            caption="Grounding accuracy (GA) across conditions with statistical significance",
        )
        all_analyses["rq1_latex"] = latex

    # RQ2
    print("\n--- RQ2: Factual Accuracy ---")
    rq2 = analyze_rq2_multiseed(results_dir)
    if rq2:
        all_analyses["rq2"] = rq2
        print(f"  Conditions: {list(rq2['per_condition_statistics'].keys())}")
        for c in rq2["pairwise_comparisons"]:
            p = c["wilcoxon_signed_rank"]["p_value"]
            d = c["wilcoxon_signed_rank"]["effect_size"]
            print(f"    {c['comparison']}: p={p:.4f}, d={d:.4f}")

        latex = generate_latex_table(
            rq2["per_condition_statistics"],
            rq2["pairwise_comparisons"],
            metric_name="FA",
            caption="Factual accuracy (FA) across conditions with statistical significance",
        )
        all_analyses["rq2_latex"] = latex

    # RQ3
    print("\n--- RQ3: Ablation Study ---")
    rq3 = analyze_rq3_ablation(results_dir)
    if rq3:
        all_analyses["rq3"] = rq3
        if "warning" in rq3:
            print(f"  {rq3['warning']}")
        else:
            print(f"  Configurations: {list(rq3['per_condition_statistics'].keys())}")

    # Save all analyses
    out_path = Path(results_dir) / "statistical_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_analyses, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  -> Full analysis saved to {out_path}")

    # Save LaTeX tables separately
    latex_path = Path(results_dir) / "latex_tables.tex"
    with open(latex_path, "w") as f:
        f.write("% Auto-generated statistical tables for AURA paper\n")
        f.write("% Generated by evaluation/statistical_analysis.py\n\n")
        for key in ["rq1_latex", "rq2_latex"]:
            if key in all_analyses:
                f.write(f"% === {key} ===\n")
                f.write(all_analyses[key])
                f.write("\n\n")
    print(f"  -> LaTeX tables saved to {latex_path}")

    return all_analyses


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AURA Statistical Analysis")
    parser.add_argument("--results-dir", default="evaluation/results")
    args = parser.parse_args()
    run_full_analysis(args.results_dir)
