"""
Temporal analysis module for AURA paper — NeurIPS Agent Track.

Analyzes how agent performance varies across:
  1. Time of day (morning/afternoon/evening stratification)
  2. Simulation progression (early vs late — memory accumulation effects)
  3. Memory load (does accumulated memory help or hurt?)
  4. Per-agent learning curves

Usage:
    python -m evaluation.temporal_analysis --results-dir evaluation/results
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


# ============================================================================
# Time-of-Day Stratification
# ============================================================================

TIME_PERIODS = {
    "early_morning (6-8h)": (6, 8),
    "morning (8-12h)": (8, 12),
    "midday (12-14h)": (12, 14),
    "afternoon (14-17h)": (14, 17),
    "evening (17-20h)": (17, 20),
    "night (20-23h)": (20, 23),
}


def stratify_by_time(
    step_results: List[Dict],
    step_to_hour_fn=None,
) -> Dict[str, Dict[str, Any]]:
    """Stratify grounding accuracy by time of day."""
    if step_to_hour_fn is None:
        # Default: each step ≈ 30min, starting at 6:00 AM
        step_to_hour_fn = lambda step: min(6 + (step * 30) // 60, 22)

    period_scores: Dict[str, List[float]] = defaultdict(list)
    period_dims: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    dims = [
        "location_consistency", "time_appropriateness",
        "social_awareness", "memory_utilization", "plan_adherence",
    ]

    for r in step_results:
        judgment = r.get("judgment", {})
        if "overall" not in judgment:
            continue

        hour = step_to_hour_fn(r.get("step", 0))

        for period_name, (h_start, h_end) in TIME_PERIODS.items():
            if h_start <= hour < h_end:
                period_scores[period_name].append(judgment["overall"])
                for dim in dims:
                    if dim in judgment:
                        period_dims[period_name][dim].append(judgment[dim])
                break

    results = {}
    for period in TIME_PERIODS:
        scores = period_scores.get(period, [])
        results[period] = {
            "mean_ga": round(mean(scores), 4),
            "std_ga": round(std(scores), 4),
            "n": len(scores),
            "dimensions": {
                dim: round(mean(period_dims.get(period, {}).get(dim, [])), 4)
                for dim in dims
            },
        }

    return results


# ============================================================================
# Memory Accumulation Effect
# ============================================================================

def analyze_memory_accumulation(
    step_results: List[Dict],
    window_size: int = 10,
) -> Dict[str, Any]:
    """
    Analyze whether GA changes as agents accumulate more memories.
    Uses a sliding window to compute GA trend over simulation steps.
    """
    # Group scores by step
    step_scores: Dict[int, List[float]] = defaultdict(list)
    for r in step_results:
        judgment = r.get("judgment", {})
        if "overall" in judgment:
            step_scores[r.get("step", 0)].append(judgment["overall"])

    if not step_scores:
        return {"error": "no valid scores"}

    sorted_steps = sorted(step_scores.keys())

    # Compute per-step average
    step_avgs = []
    for step in sorted_steps:
        scores = step_scores[step]
        step_avgs.append({
            "step": step,
            "mean_ga": round(mean(scores), 4),
            "n_agents": len(scores),
        })

    # Sliding window trend
    window_avgs = []
    for i in range(0, len(step_avgs) - window_size + 1, window_size):
        window = step_avgs[i:i + window_size]
        all_ga = [w["mean_ga"] for w in window]
        window_avgs.append({
            "window_start": window[0]["step"],
            "window_end": window[-1]["step"],
            "mean_ga": round(mean(all_ga), 4),
            "std_ga": round(std(all_ga), 4),
        })

    # Linear trend estimation (simple OLS)
    if len(step_avgs) >= 5:
        xs = [sa["step"] for sa in step_avgs]
        ys = [sa["mean_ga"] for sa in step_avgs]
        n = len(xs)
        x_mean = mean(xs)
        y_mean = mean(ys)
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        denominator = sum((x - x_mean) ** 2 for x in xs)
        slope = numerator / denominator if denominator > 0 else 0
        intercept = y_mean - slope * x_mean

        # R-squared
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - y_mean) ** 2 for y in ys)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        trend = {
            "slope": round(slope, 6),
            "intercept": round(intercept, 4),
            "r_squared": round(r_squared, 4),
            "interpretation": (
                "significant degradation over time"
                if slope < -0.001 and r_squared > 0.1
                else "significant improvement over time"
                if slope > 0.001 and r_squared > 0.1
                else "no significant temporal trend"
            ),
            "per_step_change": round(slope * 100, 4),  # percentage points per step
        }
    else:
        trend = {"error": "insufficient data points for trend analysis"}

    # Quarter comparison (for paper reporting)
    q_len = max(len(sorted_steps) // 4, 1)
    quarters = {
        "Q1 (first 25%)": sorted_steps[:q_len],
        "Q2 (25-50%)": sorted_steps[q_len:2 * q_len],
        "Q3 (50-75%)": sorted_steps[2 * q_len:3 * q_len],
        "Q4 (last 25%)": sorted_steps[3 * q_len:],
    }
    quarter_stats = {}
    for q_name, q_steps in quarters.items():
        q_scores = []
        for s in q_steps:
            q_scores.extend(step_scores.get(s, []))
        quarter_stats[q_name] = {
            "mean_ga": round(mean(q_scores), 4),
            "std_ga": round(std(q_scores), 4),
            "n": len(q_scores),
        }

    return {
        "per_step": step_avgs,
        "sliding_window": window_avgs,
        "linear_trend": trend,
        "quarter_comparison": quarter_stats,
    }


# ============================================================================
# Per-Agent Learning Curves
# ============================================================================

def analyze_per_agent_curves(
    step_results: List[Dict],
) -> Dict[str, Dict[str, Any]]:
    """Analyze per-agent GA trajectories to identify individual differences."""
    agent_steps: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

    for r in step_results:
        judgment = r.get("judgment", {})
        if "overall" in judgment:
            agent = r.get("agent", "unknown")
            step = r.get("step", 0)
            agent_steps[agent][step].append(judgment["overall"])

    agent_curves = {}
    for agent, steps in agent_steps.items():
        sorted_s = sorted(steps.keys())
        avgs = [mean(steps[s]) for s in sorted_s]

        if len(avgs) >= 3:
            first_half = avgs[:len(avgs) // 2]
            second_half = avgs[len(avgs) // 2:]
            improvement = mean(second_half) - mean(first_half)
        else:
            improvement = 0

        agent_curves[agent] = {
            "mean_ga": round(mean(avgs), 4),
            "std_ga": round(std(avgs), 4),
            "n_steps": len(avgs),
            "first_half_avg": round(mean(avgs[:len(avgs) // 2]) if len(avgs) >= 2 else mean(avgs), 4),
            "second_half_avg": round(mean(avgs[len(avgs) // 2:]) if len(avgs) >= 2 else mean(avgs), 4),
            "improvement": round(improvement, 4),
            "trajectory": "improving" if improvement > 0.02 else "degrading" if improvement < -0.02 else "stable",
        }

    return agent_curves


# ============================================================================
# Cross-Condition Temporal Comparison
# ============================================================================

def compare_conditions_temporally(
    all_results: Dict[str, Dict],
) -> Dict[str, Any]:
    """
    Compare temporal patterns across conditions.
    Shows whether probing's advantage is time-dependent.
    """
    condition_temporal = {}

    for cond_name, cond_data in all_results.items():
        if not isinstance(cond_data, dict) or "details" not in cond_data:
            continue

        details = cond_data["details"]
        time_strat = stratify_by_time(details)
        mem_accum = analyze_memory_accumulation(details)

        condition_temporal[cond_name] = {
            "by_time_period": time_strat,
            "memory_accumulation_trend": mem_accum.get("linear_trend", {}),
            "quarter_comparison": mem_accum.get("quarter_comparison", {}),
        }

    # Find periods where AURA_Full advantage is largest/smallest
    if "AURA_Full" in condition_temporal and "AURA_NoProbe" in condition_temporal:
        advantage_by_period = {}
        full_periods = condition_temporal["AURA_Full"]["by_time_period"]
        noprobe_periods = condition_temporal["AURA_NoProbe"]["by_time_period"]

        for period in TIME_PERIODS:
            full_ga = full_periods.get(period, {}).get("mean_ga", 0)
            noprobe_ga = noprobe_periods.get(period, {}).get("mean_ga", 0)
            advantage_by_period[period] = {
                "aura_full_ga": full_ga,
                "aura_noprobe_ga": noprobe_ga,
                "probing_advantage": round(full_ga - noprobe_ga, 4),
            }

        # Sort by advantage
        best_period = max(advantage_by_period.items(), key=lambda x: x[1]["probing_advantage"])
        worst_period = min(advantage_by_period.items(), key=lambda x: x[1]["probing_advantage"])

        return {
            "per_condition": condition_temporal,
            "probing_advantage_by_period": advantage_by_period,
            "best_period_for_probing": {
                "period": best_period[0],
                "advantage": best_period[1]["probing_advantage"],
            },
            "worst_period_for_probing": {
                "period": worst_period[0],
                "advantage": worst_period[1]["probing_advantage"],
            },
        }

    return {"per_condition": condition_temporal}


# ============================================================================
# Main Entry Point
# ============================================================================

def run_temporal_analysis(results_dir: str = "evaluation/results") -> Dict[str, Any]:
    """Run full temporal analysis on RQ1 results."""
    print("=" * 60)
    print("AURA Temporal Analysis (NeurIPS Agent Track)")
    print("=" * 60)

    results_path = Path(results_dir)
    analysis = {}

    rq1_path = results_path / "rq1_grounding_accuracy.json"
    if not rq1_path.exists():
        print(f"  [SKIP] {rq1_path} not found")
        return {}

    with open(rq1_path) as f:
        rq1_data = json.load(f)

    # Per-condition analysis
    for cond_name, cond_data in rq1_data.items():
        if not isinstance(cond_data, dict) or "details" not in cond_data:
            continue

        print(f"\n--- {cond_name} ---")
        details = cond_data["details"]

        # Time stratification
        time_strat = stratify_by_time(details)
        print("  Time-of-day GA:")
        for period, stats in time_strat.items():
            if stats["n"] > 0:
                print(f"    {period}: {stats['mean_ga']:.4f} (n={stats['n']})")

        # Memory accumulation
        mem_accum = analyze_memory_accumulation(details)
        trend = mem_accum.get("linear_trend", {})
        if "interpretation" in trend:
            print(f"  Trend: {trend['interpretation']} "
                  f"(slope={trend.get('slope', 0):.6f}, R²={trend.get('r_squared', 0):.4f})")

        # Per-agent curves
        agent_curves = analyze_per_agent_curves(details)
        for agent, curve in agent_curves.items():
            print(f"  {agent}: GA={curve['mean_ga']:.4f}, "
                  f"trajectory={curve['trajectory']}")

        analysis[cond_name] = {
            "time_stratification": time_strat,
            "memory_accumulation": mem_accum,
            "per_agent_curves": agent_curves,
        }

    # Cross-condition comparison
    print("\n--- Cross-Condition Temporal Comparison ---")
    cross_cond = compare_conditions_temporally(rq1_data)
    analysis["cross_condition"] = cross_cond

    if "best_period_for_probing" in cross_cond:
        best = cross_cond["best_period_for_probing"]
        worst = cross_cond["worst_period_for_probing"]
        print(f"  Best period for probing: {best['period']} "
              f"(advantage={best['advantage']:+.4f})")
        print(f"  Worst period for probing: {worst['period']} "
              f"(advantage={worst['advantage']:+.4f})")

    # Save
    out_path = results_path / "temporal_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"\n  -> Temporal analysis saved to {out_path}")

    return analysis


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AURA Temporal Analysis")
    parser.add_argument("--results-dir", default="evaluation/results")
    args = parser.parse_args()
    run_temporal_analysis(args.results_dir)
