#!/usr/bin/env python3
"""Experiment 4: Feedback convergence — does the attention tracker improve over time?

Runs the collaborative paradigm over multiple episodes and tracks:
    - Context hit rate per episode (should increase)
    - Alert fatigue per episode (should decrease)
    - Source weight convergence
    - Keyword weight accumulation

This demonstrates the online learning feedback loop.

Usage:
    python experiments/run_feedback_convergence.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aura.paradigm.collaborative import CollaborativeParadigm
from scenarios import all_scenarios
from agents import AdaptiveAgent, RealisticAgent

NUM_EPISODES = 40
MAX_STEPS = 15


def main():
    scenarios = all_scenarios()
    results: Dict[str, Any] = {}

    print("=" * 80)
    print("AURA Feedback Convergence Experiment")
    print("=" * 80)
    print("Using RealisticAgent (noisy context usage) to demonstrate convergence")
    print()

    for env, meta in scenarios:
        scenario_name = meta["name"]
        inject_step = meta["inject_step"]

        # Use SAME paradigm instance across episodes — attention tracker persists
        paradigm = CollaborativeParadigm(agent_type="sysadmin", learning_rate=0.15)
        # RealisticAgent starts with low trust (0.4) and learns over episodes
        agent = RealisticAgent(seed=42, initial_trust=0.4)

        episode_metrics: List[Dict[str, Any]] = []

        print(f"\nScenario: {scenario_name}")
        print(f"{'Ep':>4s} | {'CHR':>5s} | {'Fat':>5s} | {'TTA':>3s} | "
              f"{'Src':>3s} | {'Kw':>3s} | Source Weights")
        print("-" * 80)

        for ep_idx in range(NUM_EPISODES):
            result = paradigm.run_episode(
                agent=agent,
                env=env,
                max_steps=MAX_STEPS,
                scenario_name=scenario_name,
            )

            tta = result.detected_change_at_step - inject_step if result.detected_change_at_step >= 0 else MAX_STEPS
            tta = max(0, tta)

            # Get current source weights from attention tracker
            attn_stats = paradigm.attention_tracker.get_stats()
            source_weights = attn_stats.get("source_weights", {})

            metrics = {
                "episode": ep_idx,
                "context_hit_rate": result.metrics.get("context_hit_rate", 0),
                "alert_fatigue": result.metrics.get("alert_fatigue", 0),
                "time_to_awareness": tta,
                "total_reward": result.total_reward,
                "tracked_sources": result.metrics.get("tracked_sources", 0),
                "tracked_keywords": result.metrics.get("tracked_keywords", 0),
                "attention_use_rate": result.metrics.get("attention_use_rate", 0),
                "source_weights": source_weights,
            }
            episode_metrics.append(metrics)

            # Format source weights compactly
            sw_str = " ".join(f"{k.split('.')[-1]}={v:.2f}" for k, v in sorted(source_weights.items()))
            if len(sw_str) > 40:
                sw_str = sw_str[:40] + "…"

            print(f"{ep_idx:4d} | {metrics['context_hit_rate']:5.3f} | "
                  f"{metrics['alert_fatigue']:5.3f} | {tta:3d} | "
                  f"{metrics['tracked_sources']:3d} | {metrics['tracked_keywords']:3d} | {sw_str}")

        # Compute convergence metrics
        first_half = episode_metrics[:NUM_EPISODES // 2]
        second_half = episode_metrics[NUM_EPISODES // 2:]

        avg_chr_first = sum(m["context_hit_rate"] for m in first_half) / len(first_half)
        avg_chr_second = sum(m["context_hit_rate"] for m in second_half) / len(second_half)
        avg_fatigue_first = sum(m["alert_fatigue"] for m in first_half) / len(first_half)
        avg_fatigue_second = sum(m["alert_fatigue"] for m in second_half) / len(second_half)
        avg_tta_first = sum(m["time_to_awareness"] for m in first_half) / len(first_half)
        avg_tta_second = sum(m["time_to_awareness"] for m in second_half) / len(second_half)

        # Source weight differentiation: max_weight - min_weight at end
        final_weights = episode_metrics[-1].get("source_weights", {})
        if final_weights:
            weight_values = list(final_weights.values())
            weight_spread = max(weight_values) - min(weight_values)
        else:
            weight_spread = 0.0

        convergence = {
            "chr_improvement": round(avg_chr_second - avg_chr_first, 4),
            "fatigue_reduction": round(avg_fatigue_first - avg_fatigue_second, 4),
            "tta_improvement": round(avg_tta_first - avg_tta_second, 2),
            "first_half_chr": round(avg_chr_first, 4),
            "second_half_chr": round(avg_chr_second, 4),
            "first_half_fatigue": round(avg_fatigue_first, 4),
            "second_half_fatigue": round(avg_fatigue_second, 4),
            "weight_spread": round(weight_spread, 4),
            "final_source_weights": {k: round(v, 3) for k, v in final_weights.items()},
        }

        results[scenario_name] = {
            "episodes": episode_metrics,
            "convergence": convergence,
        }

        print(f"\n  Convergence: CHR {avg_chr_first:.3f}→{avg_chr_second:.3f} "
              f"(Δ={convergence['chr_improvement']:+.4f})  "
              f"Fatigue {avg_fatigue_first:.3f}→{avg_fatigue_second:.3f} "
              f"(Δ={-convergence['fatigue_reduction']:+.4f})")
        if final_weights:
            print(f"  Source weights: {' | '.join(f'{k}={v:.3f}' for k, v in sorted(final_weights.items()))}")
            print(f"  Weight spread: {weight_spread:.4f}")

    # Overall summary
    print("\n" + "=" * 80)
    print("CONVERGENCE SUMMARY")
    print("=" * 80)
    print(f"{'Scenario':25s} | {'CHR Δ':>8s} | {'Fatigue Δ':>10s} | {'TTA Δ':>7s} | {'Wt Spread':>9s}")
    print("-" * 70)
    for name, data in results.items():
        c = data["convergence"]
        print(f"{name:25s} | {c['chr_improvement']:+8.4f} | "
              f"{-c['fatigue_reduction']:+10.4f} | {c['tta_improvement']:+7.2f} | "
              f"{c.get('weight_spread', 0):9.4f}")

    output_path = Path(__file__).parent / "results_feedback_convergence.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
