#!/usr/bin/env python3
"""Experiment 2: Ablation study — measure contribution of each component.

Tests the collaborative paradigm with components removed one at a time:
    - Full system (all components)
    - No change detection (raw signals only)
    - No relevance scoring (push everything)
    - No attention feedback (static weights)
    - No push control (no throttling/batching)

Usage:
    python experiments/run_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aura.paradigm.base import (
    AgentObservation,
    AgentResponse,
    EnvironmentSimulator,
    EpisodeResult,
    InteractionParadigm,
    StepRecord,
)
from aura.paradigm.collaborative import CollaborativeParadigm, _state_to_signals, _agent_aware
from aura.proactive.change_detector import ChangeDetector, ChangeEvent
from aura.proactive.relevance_scorer import RelevanceScorer, TaskContext
from aura.proactive.context_assembler import ContextAssembler
from aura.proactive.push_controller import PushController
from aura.proactive.attention_tracker import AttentionTracker
from aura.types import EnvironmentSignal

from scenarios import all_scenarios
from agents import AdaptiveAgent

import time

MAX_STEPS = 20
NUM_RUNS = 5


class AblatedParadigm(InteractionParadigm):
    """Collaborative paradigm with optional component ablation."""

    def __init__(
        self,
        agent_type: str = "sysadmin",
        disable_detection: bool = False,
        disable_scoring: bool = False,
        disable_feedback: bool = False,
        disable_push_control: bool = False,
        label: str = "full",
    ):
        self.name = label
        self.agent_type = agent_type
        self.disable_detection = disable_detection
        self.disable_scoring = disable_scoring
        self.disable_feedback = disable_feedback
        self.disable_push_control = disable_push_control

        self.detector = ChangeDetector()
        self.scorer = RelevanceScorer()
        self.assembler = ContextAssembler()
        self.push_controller = PushController(min_push_interval=0.0, critical_override=True)
        self.attention_tracker = AttentionTracker(learning_rate=0.1)

    def run_episode(
        self,
        agent,
        env: EnvironmentSimulator,
        max_steps: int = 20,
        scenario_name: str = "",
    ) -> EpisodeResult:
        agent.reset()
        env.reset()

        steps = []
        total_reward = 0.0
        detected_at = -1
        pushes_made = 0
        pushes_used = 0
        pushes_ignored = 0

        task_ctx = TaskContext(agent_type=self.agent_type)

        for step_num in range(max_steps):
            state_before = dict(env.state)
            env_state = env.step(step_num)
            state_after = dict(env.state)

            signals = _state_to_signals(env_state)

            # --- Detection (or skip) ---
            if self.disable_detection:
                # Create synthetic events from raw signals
                events = []
                for sig in signals:
                    events.append(ChangeEvent(
                        event_type="raw",
                        source=sig.source,
                        severity=0.5,
                        description=str(sig.payload)[:100],
                        signals=[sig],
                        timestamp=time.time(),
                    ))
            else:
                events = self.detector.detect(signals)

            # --- Scoring (or uniform) ---
            relevance_scores: Dict[str, float] = {}
            if self.disable_scoring:
                for evt in events:
                    relevance_scores[evt.event_id] = 1.0  # everything is relevant
            else:
                if not self.disable_feedback:
                    self.scorer.update_source_weights(
                        self.attention_tracker.get_attention_weights().source_weights
                    )
                for evt in events:
                    base_score = self.scorer.score(evt, task_ctx)
                    if not self.disable_feedback:
                        kb = self.attention_tracker.get_keyword_boost(evt.description)
                        relevance_scores[evt.event_id] = min(1.0, base_score + kb * 0.1)
                    else:
                        relevance_scores[evt.event_id] = base_score

            assembled_ctx = self.assembler.assemble(events, relevance_scores, {})

            # --- Push control (or always push) ---
            pushed_context = None
            should_push = True
            if not self.disable_push_control:
                should_push = self.push_controller.should_push(assembled_ctx)

            if should_push and (assembled_ctx.critical_alerts or assembled_ctx.relevant_changes):
                pushed_context = {
                    "summary": assembled_ctx.summary,
                    "critical_alerts": [
                        {"type": e.event_type, "source": e.source,
                         "severity": e.severity, "description": e.description}
                        for e in assembled_ctx.critical_alerts
                    ],
                    "relevant_changes": [
                        {"type": e.event_type, "source": e.source,
                         "severity": e.severity, "description": e.description}
                        for e in assembled_ctx.relevant_changes
                    ],
                    "hints": assembled_ctx.agent_hints,
                }
                if not self.disable_push_control:
                    self.push_controller.record_push(
                        self.push_controller.classify_priority(assembled_ctx),
                        len(assembled_ctx.critical_alerts),
                    )
                pushes_made += 1

                if not self.disable_feedback:
                    self.attention_tracker.on_push(assembled_ctx)

            observation = AgentObservation(
                environment_state=env_state,
                pushed_context=pushed_context,
                available_tools=["system.snapshot", "git.status", "docker.status",
                                 "process.list", "service.check"],
                step_number=step_num,
            )

            response = agent.act(observation)

            for tc in response.tool_calls:
                env.execute_tool(tc.get("tool", ""), tc.get("args", {}))

            # Feedback
            if pushed_context:
                if not self.disable_feedback:
                    self.attention_tracker.on_agent_action(
                        response.action, used_context=response.used_pushed_context,
                    )
                if response.used_pushed_context:
                    pushes_used += 1
                    if not self.disable_push_control:
                        self.push_controller.record_acknowledgement()
                else:
                    pushes_ignored += 1

            if not self.disable_feedback:
                if response.action and response.action.lower() not in ("idle", "wait", "pass"):
                    self.attention_tracker.on_agent_query(response.action)

            if detected_at < 0 and _agent_aware(response, pushed_context):
                detected_at = step_num

            reward = 0.3
            if env.state.get("alerts") and detected_at >= 0:
                reward += 0.35
            if pushed_context and response.used_pushed_context:
                reward += 0.2
            if response.tool_calls:
                reward += 0.05
            reward = min(1.0, reward)
            total_reward += reward

            steps.append(StepRecord(
                step_number=step_num,
                observation=observation,
                response=response,
                env_state_before=state_before,
                env_state_after=state_after,
                reward=reward,
            ))

            if response.action.lower().strip() in ("done", "exit", "complete"):
                break

        return EpisodeResult(
            paradigm=self.name,
            scenario_name=scenario_name,
            steps=steps,
            total_reward=total_reward,
            detected_change_at_step=detected_at,
            task_completed=any(s.response.action.lower().strip() in ("done", "complete") for s in steps),
            metrics={
                "pushes_made": pushes_made,
                "pushes_used": pushes_used,
                "context_hit_rate": pushes_used / max(pushes_made, 1),
                "alert_fatigue": pushes_ignored / max(pushes_made, 1),
            },
        )


def main():
    scenarios = all_scenarios()

    configs = {
        "full_system": AblatedParadigm(label="full_system"),
        "no_detection": AblatedParadigm(disable_detection=True, label="no_detection"),
        "no_scoring": AblatedParadigm(disable_scoring=True, label="no_scoring"),
        "no_feedback": AblatedParadigm(disable_feedback=True, label="no_feedback"),
        "no_push_ctrl": AblatedParadigm(disable_push_control=True, label="no_push_ctrl"),
    }

    all_results: Dict[str, Dict[str, Any]] = {}

    print("=" * 80)
    print("AURA Ablation Study")
    print("=" * 80)

    for config_name, paradigm in configs.items():
        config_results: Dict[str, Any] = {}

        for env, meta in scenarios:
            scenario_name = meta["name"]
            inject_step = meta["inject_step"]

            episode_results = []
            for _ in range(NUM_RUNS):
                agent = AdaptiveAgent()
                result = paradigm.run_episode(agent, env, MAX_STEPS, scenario_name)
                episode_results.append(result)

            ttAs = []
            rewards = []
            hit_rates = []
            for r in episode_results:
                tta = r.detected_change_at_step - inject_step if r.detected_change_at_step >= 0 else MAX_STEPS
                ttAs.append(max(0, tta))
                rewards.append(r.total_reward)
                if "context_hit_rate" in r.metrics:
                    hit_rates.append(r.metrics["context_hit_rate"])

            metrics = {
                "avg_tta": round(sum(ttAs) / len(ttAs), 2),
                "detection_rate": round(sum(1 for t in ttAs if t < MAX_STEPS) / len(ttAs), 3),
                "avg_reward": round(sum(rewards) / len(rewards), 3),
                "avg_chr": round(sum(hit_rates) / len(hit_rates), 3) if hit_rates else 0,
            }
            config_results[scenario_name] = metrics

            print(f"  [{config_name:15s}] {scenario_name:25s} | "
                  f"TTA={metrics['avg_tta']:5.1f}  "
                  f"Det={metrics['detection_rate']:.2f}  "
                  f"Reward={metrics['avg_reward']:.2f}")

        # Aggregate
        all_metrics = list(config_results.values())
        config_results["_aggregate"] = {
            "avg_tta": round(sum(m["avg_tta"] for m in all_metrics) / len(all_metrics), 2),
            "detection_rate": round(sum(m["detection_rate"] for m in all_metrics) / len(all_metrics), 3),
            "avg_reward": round(sum(m["avg_reward"] for m in all_metrics) / len(all_metrics), 3),
            "avg_chr": round(sum(m.get("avg_chr", 0) for m in all_metrics) / len(all_metrics), 3),
        }

        all_results[config_name] = config_results
        print()

    # Summary table
    print("\n" + "=" * 80)
    print("ABLATION RESULTS (aggregate)")
    print("=" * 80)
    print(f"{'Configuration':20s} | {'TTA↓':>6s} | {'Det%↑':>6s} | {'Reward↑':>8s} | {'CHR↑':>6s} | {'Δ TTA':>6s}")
    print("-" * 70)

    full_tta = all_results["full_system"]["_aggregate"]["avg_tta"]
    for name in ["full_system", "no_detection", "no_scoring", "no_feedback", "no_push_ctrl"]:
        agg = all_results[name]["_aggregate"]
        delta = agg["avg_tta"] - full_tta
        print(f"{name:20s} | {agg['avg_tta']:6.1f} | {agg['detection_rate']:6.3f} | "
              f"{agg['avg_reward']:8.3f} | {agg.get('avg_chr', 0):6.3f} | {delta:+6.1f}")

    output_path = Path(__file__).parent / "results_ablation.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
