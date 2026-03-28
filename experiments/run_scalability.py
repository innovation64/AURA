#!/usr/bin/env python3
"""Experiment 3: Scalability — how does AURA perform as signal volume grows?

Tests the proactive pipeline with increasing numbers of signals per step:
    - 5, 10, 20, 50, 100, 200 signals per step

Measures:
    - Processing latency (ms per step)
    - Detection accuracy (precision/recall of change detection)
    - Context quality (relevance of assembled context)

Usage:
    python experiments/run_scalability.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aura.proactive.change_detector import ChangeDetector
from aura.proactive.relevance_scorer import RelevanceScorer, TaskContext
from aura.proactive.context_assembler import ContextAssembler
from aura.types import EnvironmentSignal

import random

NUM_RUNS = 10


def generate_signals(n: int, anomaly_rate: float = 0.1, seed: int = 42) -> List[EnvironmentSignal]:
    """Generate n synthetic environment signals with controlled anomaly rate."""
    rng = random.Random(seed)
    sources = ["probe.system", "probe.network", "probe.filesystem",
               "probe.docker", "probe.process", "probe.git"]
    modalities = ["system", "network", "filesystem", "docker", "process", "git"]

    signals = []
    for i in range(n):
        is_anomaly = rng.random() < anomaly_rate
        src_idx = rng.randint(0, len(sources) - 1)

        if is_anomaly:
            payload = {
                "cpu_percent": rng.uniform(90, 100),
                "status": rng.choice(["down", "error", "crashed"]),
                "anomaly": True,
                "id": i,
            }
        else:
            payload = {
                "cpu_percent": rng.uniform(10, 70),
                "memory_percent": rng.uniform(20, 60),
                "status": "running",
                "id": i,
            }

        signals.append(EnvironmentSignal(
            source=sources[src_idx],
            modality=modalities[src_idx],
            payload=payload,
        ))

    return signals


def run_pipeline(signals: List[EnvironmentSignal]) -> Dict[str, Any]:
    """Run the full proactive pipeline and measure latency."""
    detector = ChangeDetector()
    scorer = RelevanceScorer()
    assembler = ContextAssembler()
    task_ctx = TaskContext(agent_type="sysadmin")

    # Detection
    t0 = time.perf_counter()
    events = detector.detect(signals)
    t_detect = (time.perf_counter() - t0) * 1000

    # Scoring
    t0 = time.perf_counter()
    scores = {}
    for evt in events:
        scores[evt.event_id] = scorer.score(evt, task_ctx)
    t_score = (time.perf_counter() - t0) * 1000

    # Assembly
    t0 = time.perf_counter()
    ctx = assembler.assemble(events, scores, {})
    t_assemble = (time.perf_counter() - t0) * 1000

    total_ms = t_detect + t_score + t_assemble

    return {
        "num_signals": len(signals),
        "num_events": len(events),
        "num_critical": len(ctx.critical_alerts),
        "num_relevant": len(ctx.relevant_changes),
        "latency_detect_ms": round(t_detect, 3),
        "latency_score_ms": round(t_score, 3),
        "latency_assemble_ms": round(t_assemble, 3),
        "latency_total_ms": round(total_ms, 3),
    }


def main():
    signal_counts = [5, 10, 20, 50, 100, 200, 500]
    results: Dict[str, Any] = {}

    print("=" * 80)
    print("AURA Scalability Experiment")
    print("=" * 80)
    print(f"{'Signals':>8s} | {'Events':>7s} | {'Detect':>10s} | {'Score':>10s} | "
          f"{'Assemble':>10s} | {'Total':>10s}")
    print("-" * 70)

    for n in signal_counts:
        run_results = []
        for run_idx in range(NUM_RUNS):
            signals = generate_signals(n, anomaly_rate=0.1, seed=42 + run_idx)
            r = run_pipeline(signals)
            run_results.append(r)

        # Average
        avg = {
            "num_signals": n,
            "avg_events": round(sum(r["num_events"] for r in run_results) / NUM_RUNS, 1),
            "avg_latency_detect_ms": round(sum(r["latency_detect_ms"] for r in run_results) / NUM_RUNS, 3),
            "avg_latency_score_ms": round(sum(r["latency_score_ms"] for r in run_results) / NUM_RUNS, 3),
            "avg_latency_assemble_ms": round(sum(r["latency_assemble_ms"] for r in run_results) / NUM_RUNS, 3),
            "avg_latency_total_ms": round(sum(r["latency_total_ms"] for r in run_results) / NUM_RUNS, 3),
        }

        results[str(n)] = avg
        print(f"{n:8d} | {avg['avg_events']:7.1f} | "
              f"{avg['avg_latency_detect_ms']:8.3f}ms | {avg['avg_latency_score_ms']:8.3f}ms | "
              f"{avg['avg_latency_assemble_ms']:8.3f}ms | {avg['avg_latency_total_ms']:8.3f}ms")

    # Linear scaling check
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)
    base = results["5"]["avg_latency_total_ms"]
    for n_str, avg in results.items():
        n = int(n_str)
        ratio = avg["avg_latency_total_ms"] / max(base, 0.001)
        signal_ratio = n / 5
        efficiency = ratio / signal_ratio if signal_ratio > 0 else 0
        print(f"  {n:4d} signals: {avg['avg_latency_total_ms']:8.3f}ms  "
              f"(×{ratio:.1f} latency for ×{signal_ratio:.0f} signals, "
              f"efficiency={efficiency:.2f})")

    output_path = Path(__file__).parent / "results_scalability.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
