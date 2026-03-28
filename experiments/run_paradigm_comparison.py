#!/usr/bin/env python3
"""Experiment 1: Compare three interaction paradigms across five scenarios.

Paradigms:
    1. Reactive    — agent queries environment explicitly (baseline)
    2. Proactive   — environment pushes context to agent
    3. Collaborative — proactive + attention feedback loop (AURA full)

Metrics per scenario:
    - Time to Awareness (TTA): steps until agent first detects injected change
    - Task Completion Rate: did the agent complete the task?
    - Total Reward: cumulative reward across the episode
    - Context Hit Rate: fraction of pushes that agent used (proactive/collab only)
    - Alert Fatigue: fraction of pushes ignored (proactive/collab only)

Usage:
    python experiments/run_paradigm_comparison.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aura.paradigm.reactive import ReactiveParadigm
from aura.paradigm.proactive import ProactiveParadigm
from aura.paradigm.collaborative import CollaborativeParadigm
from aura.paradigm.base import EpisodeResult

from scenarios import all_scenarios
from agents import RandomAgent, ReactiveAgent, AdaptiveAgent, RealisticAgent

MAX_STEPS = 20
NUM_RUNS = 5  # repeat for variance


def run_paradigm(paradigm, agent, env, meta, max_steps=MAX_STEPS) -> EpisodeResult:
    """Run one episode of a paradigm."""
    env.reset()
    return paradigm.run_episode(
        agent=agent,
        env=env,
        max_steps=max_steps,
        scenario_name=meta["name"],
    )


def compute_metrics(results: List[EpisodeResult], inject_step: int) -> Dict[str, Any]:
    """Aggregate metrics across multiple runs."""
    ttAs = []
    rewards = []
    completions = 0
    context_hit_rates = []
    fatigue_scores = []

    for r in results:
        # Time to awareness: steps after injection until detection
        if r.detected_change_at_step >= 0:
            tta = r.detected_change_at_step - inject_step
            ttAs.append(max(0, tta))
        else:
            ttAs.append(MAX_STEPS)  # never detected

        rewards.append(r.total_reward)
        if r.task_completed:
            completions += 1

        if "context_hit_rate" in r.metrics:
            context_hit_rates.append(r.metrics["context_hit_rate"])
        if "alert_fatigue" in r.metrics:
            fatigue_scores.append(r.metrics["alert_fatigue"])

    n = len(results)
    avg_tta = sum(ttAs) / n if ttAs else -1
    avg_reward = sum(rewards) / n if rewards else 0
    detection_rate = sum(1 for t in ttAs if t < MAX_STEPS) / n

    metrics = {
        "avg_time_to_awareness": round(avg_tta, 2),
        "detection_rate": round(detection_rate, 3),
        "avg_total_reward": round(avg_reward, 3),
        "task_completion_rate": round(completions / n, 3),
        "avg_steps": round(sum(len(r.steps) for r in results) / n, 1),
    }
    if context_hit_rates:
        metrics["avg_context_hit_rate"] = round(sum(context_hit_rates) / len(context_hit_rates), 3)
    if fatigue_scores:
        metrics["avg_alert_fatigue"] = round(sum(fatigue_scores) / len(fatigue_scores), 3)

    return metrics


def main():
    scenarios = all_scenarios()
    paradigms = {
        "reactive": (ReactiveParadigm(), ReactiveAgent()),
        "proactive": (ProactiveParadigm(agent_type="sysadmin"), RealisticAgent(seed=42, initial_trust=0.4)),
        "collaborative": (CollaborativeParadigm(agent_type="sysadmin"), RealisticAgent(seed=42, initial_trust=0.4)),
    }

    # Also test with random baseline
    paradigms["random_baseline"] = (ReactiveParadigm(), RandomAgent())

    all_results: Dict[str, Dict[str, Any]] = {}

    print("=" * 80)
    print("AURA Paradigm Comparison Experiment")
    print("=" * 80)
    print(f"Scenarios: {len(scenarios)}, Paradigms: {len(paradigms)}, Runs per config: {NUM_RUNS}")
    print()

    for paradigm_name, (paradigm, agent) in paradigms.items():
        paradigm_results: Dict[str, Any] = {}

        for env, meta in scenarios:
            scenario_name = meta["name"]
            inject_step = meta["inject_step"]

            episode_results = []
            for run_idx in range(NUM_RUNS):
                result = run_paradigm(paradigm, agent, env, meta)
                episode_results.append(result)

            metrics = compute_metrics(episode_results, inject_step)
            paradigm_results[scenario_name] = metrics

            print(f"  [{paradigm_name:15s}] {scenario_name:25s} | "
                  f"TTA={metrics['avg_time_to_awareness']:5.1f}  "
                  f"Det={metrics['detection_rate']:.2f}  "
                  f"Reward={metrics['avg_total_reward']:.2f}  "
                  f"Complete={metrics['task_completion_rate']:.2f}")

        # Aggregate across scenarios
        all_metrics = list(paradigm_results.values())
        paradigm_results["_aggregate"] = {
            "avg_time_to_awareness": round(sum(m["avg_time_to_awareness"] for m in all_metrics) / len(all_metrics), 2),
            "detection_rate": round(sum(m["detection_rate"] for m in all_metrics) / len(all_metrics), 3),
            "avg_total_reward": round(sum(m["avg_total_reward"] for m in all_metrics) / len(all_metrics), 3),
            "task_completion_rate": round(sum(m["task_completion_rate"] for m in all_metrics) / len(all_metrics), 3),
        }

        all_results[paradigm_name] = paradigm_results
        print()

    # Print summary table
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS (averaged across all scenarios)")
    print("=" * 80)
    print(f"{'Paradigm':20s} | {'TTA↓':>6s} | {'Det%↑':>6s} | {'Reward↑':>8s} | {'Complete↑':>10s}")
    print("-" * 70)
    for name in ["random_baseline", "reactive", "proactive", "collaborative"]:
        agg = all_results[name]["_aggregate"]
        print(f"{name:20s} | {agg['avg_time_to_awareness']:6.1f} | "
              f"{agg['detection_rate']:6.3f} | {agg['avg_total_reward']:8.3f} | "
              f"{agg['task_completion_rate']:10.3f}")

    # Print proactive-specific metrics
    print("\n" + "=" * 80)
    print("PROACTIVE METRICS (proactive & collaborative only)")
    print("=" * 80)
    for name in ["proactive", "collaborative"]:
        print(f"\n{name}:")
        for scenario_name, metrics in all_results[name].items():
            if scenario_name.startswith("_"):
                continue
            chr_str = f"  CHR={metrics.get('avg_context_hit_rate', 'N/A')}"
            fat_str = f"  Fatigue={metrics.get('avg_alert_fatigue', 'N/A')}"
            print(f"  {scenario_name:25s} |{chr_str}{fat_str}")

    # Save results
    output_path = Path(__file__).parent / "results_paradigm_comparison.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    main()
