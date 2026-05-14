"""
Error analysis module for AURA paper — NeurIPS Agent Track.

Provides systematic failure mode analysis required by top-tier venues:
  1. Per-task-type failure categorization
  2. Probing benefit/harm analysis (when probing helps vs hurts)
  3. Judge disagreement analysis (rule-based vs LLM judge)
  4. Negative result documentation (Trust Game, MemoryArena regressions)
  5. Confusion matrix between conditions
  6. Qualitative error examples for paper appendix

Usage:
    python -m evaluation.error_analysis --results-dir evaluation/results
"""

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# Failure Categorization
# ============================================================================

@dataclass
class FailureCase:
    """A single failure instance with full context."""
    condition: str
    agent: str
    step: int
    action: str
    location: str
    failure_type: str         # category of failure
    failure_dimension: str    # which grounding dimension failed
    rule_score: float
    llm_score: float
    judge_disagreement: bool  # rule and LLM disagree
    context_snippet: str


FAILURE_TYPES = {
    "location_mismatch": "Action inappropriate for current location",
    "time_mismatch": "Action inappropriate for time of day",
    "social_unawareness": "Action ignores nearby agents",
    "memory_contradiction": "Action contradicts agent's own memories",
    "plan_deviation": "Action significantly deviates from daily plan",
    "hallucinated_context": "Agent references non-existent context (probing noise)",
    "over_exploration": "Agent spends excessive steps probing instead of acting",
    "judge_parse_error": "LLM judge failed to return valid scores",
}


def categorize_failures(
    step_results: List[Dict],
    condition: str,
    threshold: float = 0.5,
) -> List[FailureCase]:
    """Categorize failures from a single condition's step results."""
    failures = []

    for r in step_results:
        judgment = r.get("judgment", {})
        if "error" in judgment or "overall" not in judgment:
            if "error" in judgment:
                failures.append(FailureCase(
                    condition=condition,
                    agent=r.get("agent", ""),
                    step=r.get("step", 0),
                    action=r.get("action", ""),
                    location=r.get("location", ""),
                    failure_type="judge_parse_error",
                    failure_dimension="all",
                    rule_score=0, llm_score=0,
                    judge_disagreement=False,
                    context_snippet=str(judgment.get("error", ""))[:200],
                ))
            continue

        # Check each dimension for failures
        dim_map = {
            "location_consistency": ("location_mismatch", "rule_location"),
            "time_appropriateness": ("time_mismatch", "rule_time"),
            "social_awareness": ("social_unawareness", "rule_social"),
            "memory_utilization": ("memory_contradiction", "rule_memory"),
            "plan_adherence": ("plan_deviation", "rule_plan"),
        }

        for dim, (ftype, rule_key) in dim_map.items():
            composite_score = judgment.get(dim, 1.0)
            if composite_score < threshold:
                rule_scores = judgment.get("rule_scores", {})
                llm_scores = judgment.get("llm_scores", {})
                rule_val = rule_scores.get(rule_key, 1)
                llm_val = llm_scores.get(dim, 1)

                failures.append(FailureCase(
                    condition=condition,
                    agent=r.get("agent", ""),
                    step=r.get("step", 0),
                    action=r.get("action", ""),
                    location=r.get("location", ""),
                    failure_type=ftype,
                    failure_dimension=dim,
                    rule_score=rule_val,
                    llm_score=llm_val,
                    judge_disagreement=(rule_val > 0.5) != (llm_val > 0.5),
                    context_snippet=judgment.get("reasoning", "")[:200],
                ))

    return failures


# ============================================================================
# Probing Benefit/Harm Analysis
# ============================================================================

@dataclass
class ProbingImpact:
    """Analysis of when probing helps vs hurts."""
    task_type: str
    probing_benefit: float       # Mean GA improvement with probing
    probing_harm: float          # Mean GA decrease with probing
    net_effect: float
    benefit_count: int           # Number of cases where probing helped
    harm_count: int              # Number of cases where probing hurt
    neutral_count: int
    explanation: str


def analyze_probing_impact(
    probe_results: Dict[str, Dict],
    noprobe_results: Dict[str, Dict],
) -> List[ProbingImpact]:
    """Compare probe vs no-probe results to identify when probing helps/hurts."""
    impacts = []

    # Extract per-step per-agent scores for both conditions
    probe_scores = _extract_per_agent_scores(probe_results)
    noprobe_scores = _extract_per_agent_scores(noprobe_results)

    # Compare matched pairs
    benefit_cases = []
    harm_cases = []
    neutral_cases = []

    for key in probe_scores:
        if key in noprobe_scores:
            diff = probe_scores[key] - noprobe_scores[key]
            if diff > 0.05:
                benefit_cases.append((key, diff))
            elif diff < -0.05:
                harm_cases.append((key, diff))
            else:
                neutral_cases.append((key, diff))

    total = len(benefit_cases) + len(harm_cases) + len(neutral_cases)
    if total == 0:
        return []

    # Categorize harm cases by failure pattern
    harm_explanations = []
    for (agent, step), diff in harm_cases:
        harm_explanations.append(
            f"Agent '{agent}' at step {step}: GA dropped by {abs(diff):.3f}"
        )

    avg_benefit = (
        sum(d for _, d in benefit_cases) / max(len(benefit_cases), 1)
    )
    avg_harm = (
        sum(abs(d) for _, d in harm_cases) / max(len(harm_cases), 1)
    )

    impacts.append(ProbingImpact(
        task_type="overall",
        probing_benefit=round(avg_benefit, 4),
        probing_harm=round(avg_harm, 4),
        net_effect=round(avg_benefit - avg_harm, 4),
        benefit_count=len(benefit_cases),
        harm_count=len(harm_cases),
        neutral_count=len(neutral_cases),
        explanation=(
            f"Probing helped in {len(benefit_cases)}/{total} cases "
            f"({100*len(benefit_cases)/total:.1f}%), "
            f"hurt in {len(harm_cases)}/{total} cases "
            f"({100*len(harm_cases)/total:.1f}%). "
            f"Net effect: {'+' if avg_benefit > avg_harm else ''}"
            f"{avg_benefit - avg_harm:.4f}"
        ),
    ))

    return impacts


def _extract_per_agent_scores(results: Dict) -> Dict[Tuple[str, int], float]:
    """Extract (agent, step) -> overall_score mapping from results."""
    scores = {}
    details = results.get("details", [])
    for r in details:
        if "judgment" in r and "overall" in r.get("judgment", {}):
            key = (r.get("agent", ""), r.get("step", 0))
            scores[key] = r["judgment"]["overall"]
    return scores


# ============================================================================
# Judge Disagreement Analysis
# ============================================================================

@dataclass
class DisagreementStats:
    """Statistics about rule-based vs LLM judge disagreements."""
    total_judgments: int
    disagreements: int
    disagreement_rate: float
    rule_stricter: int         # Rule=0, LLM=1
    llm_stricter: int          # Rule=1, LLM=0
    by_dimension: Dict[str, Dict[str, int]]


def analyze_judge_disagreements(step_results: List[Dict]) -> DisagreementStats:
    """Analyze disagreements between rule-based and LLM judges."""
    total = 0
    disagreements = 0
    rule_stricter = 0
    llm_stricter = 0
    by_dim: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "agree": 0, "rule_stricter": 0, "llm_stricter": 0
    })

    dim_map = {
        "location_consistency": "rule_location",
        "time_appropriateness": "rule_time",
        "social_awareness": "rule_social",
        "memory_utilization": "rule_memory",
        "plan_adherence": "rule_plan",
    }

    for r in step_results:
        judgment = r.get("judgment", {})
        rule_scores = judgment.get("rule_scores", {})
        llm_scores = judgment.get("llm_scores", {})

        if not rule_scores or not llm_scores:
            continue

        for dim, rule_key in dim_map.items():
            rule_val = rule_scores.get(rule_key)
            llm_val = llm_scores.get(dim)
            if rule_val is None or llm_val is None:
                continue

            total += 1
            rule_pass = rule_val > 0.5
            llm_pass = llm_val > 0.5

            if rule_pass != llm_pass:
                disagreements += 1
                if rule_pass and not llm_pass:
                    llm_stricter += 1
                    by_dim[dim]["llm_stricter"] += 1
                else:
                    rule_stricter += 1
                    by_dim[dim]["rule_stricter"] += 1
            else:
                by_dim[dim]["agree"] += 1

    return DisagreementStats(
        total_judgments=total,
        disagreements=disagreements,
        disagreement_rate=round(disagreements / max(total, 1), 4),
        rule_stricter=rule_stricter,
        llm_stricter=llm_stricter,
        by_dimension=dict(by_dim),
    )


# ============================================================================
# Negative Results Analysis
# ============================================================================

@dataclass
class NegativeResult:
    """Documentation of a case where AURA underperforms."""
    benchmark: str
    task_type: str
    aura_score: float
    baseline_score: float
    baseline_name: str
    delta: float
    hypothesis: str


def analyze_negative_results(results_dir: str) -> List[NegativeResult]:
    """Scan all results for cases where AURA Full underperforms baselines."""
    negatives = []
    results_path = Path(results_dir)

    # Check InteractiveBench trust game
    trust_path = results_path / "interactivebench"
    if trust_path.exists():
        for f in trust_path.glob("*.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                # Look for condition comparisons
                if isinstance(data, dict):
                    _check_negative(data, f.stem, negatives)
            except (json.JSONDecodeError, KeyError):
                continue

    # Check MemoryArena
    ma_path = results_path / "memoryarena"
    if ma_path.exists():
        for f in ma_path.glob("*.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    _check_negative(data, f"memoryarena/{f.stem}", negatives)
            except (json.JSONDecodeError, KeyError):
                continue

    # Check RQ1 — Vanilla LLM outperforming AURA
    rq1_path = results_path / "rq1_grounding_accuracy.json"
    if rq1_path.exists():
        with open(rq1_path) as f:
            rq1 = json.load(f)
        aura_ga = _get_ga(rq1, "AURA_Full")
        for cond in ["Vanilla_LLM", "Static_Context", "ReAct", "AURA_NoProbe"]:
            bl_ga = _get_ga(rq1, cond)
            if bl_ga is not None and aura_ga is not None and bl_ga > aura_ga:
                negatives.append(NegativeResult(
                    benchmark="RQ1_Grounding",
                    task_type="action_grounding",
                    aura_score=aura_ga,
                    baseline_score=bl_ga,
                    baseline_name=cond,
                    delta=round(bl_ga - aura_ga, 4),
                    hypothesis=_hypothesize_negative("grounding", cond),
                ))

    # Check RQ3 ablation — removing components improves score
    rq3_path = results_path / "rq3_ablation.json"
    if rq3_path.exists():
        with open(rq3_path) as f:
            rq3 = json.load(f)
        full_ga = None
        for name, cfg in rq3.items():
            if isinstance(cfg, dict) and "Full" in name:
                full_ga = cfg.get("avg_ga", 0)
                break
        if full_ga is not None:
            for name, cfg in rq3.items():
                if isinstance(cfg, dict) and "avg_ga" in cfg and "Full" not in name:
                    abl_ga = cfg["avg_ga"]
                    if abl_ga > full_ga + 0.005:
                        negatives.append(NegativeResult(
                            benchmark="RQ3_Ablation",
                            task_type="ablation",
                            aura_score=full_ga,
                            baseline_score=abl_ga,
                            baseline_name=name,
                            delta=round(abl_ga - full_ga, 4),
                            hypothesis=_hypothesize_ablation(name),
                        ))

    return negatives


def _get_ga(data: dict, condition: str) -> Optional[float]:
    cond_data = data.get(condition, {})
    if isinstance(cond_data, dict) and "summary" in cond_data:
        return cond_data["summary"].get("overall_grounding_accuracy")
    return None


def _check_negative(data: dict, benchmark: str, negatives: list):
    """Check a result dict for AURA underperformance."""
    conditions = {}
    for key, val in data.items():
        if isinstance(val, dict):
            # Try to extract a score
            for metric in ["payoff_per_round", "accuracy", "success_rate", "SR"]:
                if metric in val:
                    conditions[key] = val[metric]
                    break

    aura_score = conditions.get("aura_full") or conditions.get("full")
    if aura_score is None:
        return

    for cond, score in conditions.items():
        if cond in ("aura_full", "full"):
            continue
        if score > aura_score:
            negatives.append(NegativeResult(
                benchmark=benchmark,
                task_type="interactive",
                aura_score=aura_score,
                baseline_score=score,
                baseline_name=cond,
                delta=round(score - aura_score, 4),
                hypothesis=_hypothesize_negative("interactive", cond),
            ))


def _hypothesize_negative(task_type: str, baseline: str) -> str:
    """Generate hypothesis for why AURA underperforms."""
    if "Vanilla" in baseline:
        return (
            "Probing may introduce distracting context for simple tasks where "
            "the LLM's parametric knowledge is sufficient. The additional context "
            "creates noise that dilutes the reasoning signal."
        )
    if "NoProbe" in baseline or "no_probe" in baseline:
        return (
            "Proactive probing adds latency and context volume that may overwhelm "
            "the reasoning step for this task type. The agent performs better with "
            "focused, minimal context rather than comprehensive environmental awareness."
        )
    if "memory" in baseline.lower():
        return (
            "In formal reasoning tasks, environmental memory can introduce "
            "irrelevant associations that distract from logical deduction."
        )
    return (
        f"AURA's proactive mechanism may not generalize to this task type. "
        f"The {baseline} approach's simplicity is advantageous here."
    )


def _hypothesize_ablation(config_name: str) -> str:
    """Generate hypothesis for why removing a component improves performance."""
    if "Reflection" in config_name:
        return (
            "Reflection may introduce self-doubt loops where the agent second-guesses "
            "correct initial decisions. In constrained environments with clear action "
            "mappings, direct action selection outperforms deliberative reflection. "
            "This suggests reflection should be adaptive — enabled for ambiguous "
            "situations but bypassed for routine actions."
        )
    if "Memory" in config_name:
        return (
            "Accumulated memories may create interference effects where past context "
            "biases current decisions away from the optimal action for the present state."
        )
    return f"Removing {config_name} reduces pipeline complexity, which may improve GA in simple scenarios."


# ============================================================================
# Temporal Error Analysis
# ============================================================================

def analyze_temporal_patterns(
    step_results: List[Dict],
    condition: str,
) -> Dict[str, Any]:
    """Analyze how errors distribute across time periods."""
    time_buckets = {
        "early_morning": (6, 8),
        "morning": (8, 12),
        "midday": (12, 14),
        "afternoon": (14, 17),
        "evening": (17, 20),
        "night": (20, 23),
    }

    bucket_scores: Dict[str, List[float]] = defaultdict(list)
    bucket_failures: Dict[str, int] = defaultdict(int)
    bucket_total: Dict[str, int] = defaultdict(int)

    for r in step_results:
        judgment = r.get("judgment", {})
        if "overall" not in judgment:
            continue

        # Infer hour from step (each step ≈ 30min, starting at 6:00)
        step = r.get("step", 0)
        hour = 6 + (step * 30) // 60
        hour = min(hour, 22)

        for bucket_name, (h_start, h_end) in time_buckets.items():
            if h_start <= hour < h_end:
                score = judgment["overall"]
                bucket_scores[bucket_name].append(score)
                bucket_total[bucket_name] += 1
                if score < 0.5:
                    bucket_failures[bucket_name] += 1
                break

    results = {}
    for bucket in time_buckets:
        scores = bucket_scores.get(bucket, [])
        total = bucket_total.get(bucket, 0)
        fails = bucket_failures.get(bucket, 0)
        results[bucket] = {
            "mean_ga": round(sum(scores) / max(len(scores), 1), 4),
            "failure_rate": round(fails / max(total, 1), 4),
            "n_judgments": len(scores),
            "n_failures": fails,
        }

    # Trend: does GA degrade over simulation time?
    step_scores = defaultdict(list)
    for r in step_results:
        judgment = r.get("judgment", {})
        if "overall" in judgment:
            step_scores[r.get("step", 0)].append(judgment["overall"])

    sorted_steps = sorted(step_scores.keys())
    if len(sorted_steps) >= 5:
        first_quarter = sorted_steps[:len(sorted_steps) // 4]
        last_quarter = sorted_steps[-len(sorted_steps) // 4:]
        early_avg = sum(
            sum(step_scores[s]) / len(step_scores[s]) for s in first_quarter
        ) / max(len(first_quarter), 1)
        late_avg = sum(
            sum(step_scores[s]) / len(step_scores[s]) for s in last_quarter
        ) / max(len(last_quarter), 1)
        degradation = round(early_avg - late_avg, 4)
    else:
        early_avg = late_avg = degradation = 0.0

    return {
        "condition": condition,
        "by_time_period": results,
        "temporal_degradation": {
            "early_quarter_avg": round(early_avg, 4),
            "late_quarter_avg": round(late_avg, 4),
            "degradation": degradation,
            "interpretation": (
                "GA degrades over time (memory accumulation effect)"
                if degradation > 0.02
                else "GA remains stable across simulation"
                if abs(degradation) <= 0.02
                else "GA improves over time (learning effect)"
            ),
        },
    }


# ============================================================================
# Full Error Analysis Pipeline
# ============================================================================

def run_full_error_analysis(results_dir: str = "evaluation/results") -> Dict[str, Any]:
    """Run comprehensive error analysis on all available results."""
    print("=" * 60)
    print("AURA Error Analysis (NeurIPS Agent Track)")
    print("=" * 60)

    results_path = Path(results_dir)
    analysis = {}

    # 1. Failure categorization from RQ1
    print("\n--- Failure Categorization ---")
    rq1_path = results_path / "rq1_grounding_accuracy.json"
    if rq1_path.exists():
        with open(rq1_path) as f:
            rq1_data = json.load(f)

        all_failures = {}
        failure_summary = {}
        for cond_name, cond_data in rq1_data.items():
            if not isinstance(cond_data, dict) or "details" not in cond_data:
                continue
            failures = categorize_failures(cond_data["details"], cond_name)
            all_failures[cond_name] = [asdict(f) for f in failures]

            # Summarize
            type_counts = Counter(f.failure_type for f in failures)
            dim_counts = Counter(f.failure_dimension for f in failures)
            disagree_count = sum(1 for f in failures if f.judge_disagreement)

            failure_summary[cond_name] = {
                "total_failures": len(failures),
                "by_type": dict(type_counts),
                "by_dimension": dict(dim_counts),
                "judge_disagreements": disagree_count,
            }
            print(f"  {cond_name}: {len(failures)} failures, "
                  f"{disagree_count} judge disagreements")

        analysis["failure_categorization"] = failure_summary
        analysis["failure_details"] = {
            k: v[:10] for k, v in all_failures.items()  # Top 10 per condition
        }

        # Judge disagreement analysis
        print("\n--- Judge Disagreement Analysis ---")
        for cond_name, cond_data in rq1_data.items():
            if not isinstance(cond_data, dict) or "details" not in cond_data:
                continue
            disagree = analyze_judge_disagreements(cond_data["details"])
            analysis.setdefault("judge_disagreements", {})[cond_name] = asdict(disagree)
            print(f"  {cond_name}: {disagree.disagreement_rate:.1%} disagreement rate "
                  f"(rule stricter: {disagree.rule_stricter}, LLM stricter: {disagree.llm_stricter})")

        # Temporal patterns
        print("\n--- Temporal Error Patterns ---")
        for cond_name, cond_data in rq1_data.items():
            if not isinstance(cond_data, dict) or "details" not in cond_data:
                continue
            temporal = analyze_temporal_patterns(cond_data["details"], cond_name)
            analysis.setdefault("temporal_patterns", {})[cond_name] = temporal
            deg = temporal["temporal_degradation"]
            print(f"  {cond_name}: {deg['interpretation']} "
                  f"(early={deg['early_quarter_avg']:.4f}, late={deg['late_quarter_avg']:.4f})")

        # Probing impact analysis
        print("\n--- Probing Impact Analysis ---")
        aura_full = rq1_data.get("AURA_Full", {})
        aura_noprobe = rq1_data.get("AURA_NoProbe", {})
        if aura_full and aura_noprobe:
            impacts = analyze_probing_impact(aura_full, aura_noprobe)
            analysis["probing_impact"] = [asdict(i) for i in impacts]
            for imp in impacts:
                print(f"  {imp.task_type}: {imp.explanation}")

    # 2. Negative results
    print("\n--- Negative Results ---")
    negatives = analyze_negative_results(results_dir)
    analysis["negative_results"] = [asdict(n) for n in negatives]
    if negatives:
        for n in negatives:
            print(f"  [{n.benchmark}] {n.baseline_name} > AURA by {n.delta:.4f}")
            print(f"    Hypothesis: {n.hypothesis[:100]}...")
    else:
        print("  No negative results detected (or result files missing)")

    # Save
    out_path = results_path / "error_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"\n  -> Error analysis saved to {out_path}")

    return analysis


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AURA Error Analysis")
    parser.add_argument("--results-dir", default="evaluation/results")
    args = parser.parse_args()
    run_full_error_analysis(args.results_dir)
