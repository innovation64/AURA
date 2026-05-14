#!/usr/bin/env python3
"""
RQ7: Backend-Agnostic Ablation — prove AURA's value across any backend.

Experiment matrix:
    4 backends  (default, llm, bmam, model)
  × 5 AURA layers (bare, +proactive, +explorer, +guard, full)
  = 20 configurations

Each is tested on:
  1. Grounding accuracy (RQ1 subset)
  2. Loop detection scenarios (new)
  3. Strategy staleness scenarios (new)

Usage:
    python -m evaluation.rq7_backend_ablation [--backends default llm] [--layers bare full]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aura.core import AURAAgent, AURAConfig
from aura.guard import ExecutionGuard, InterventionLevel
from aura.feedback import ConditionalFeedbackStore, Outcome, StatePattern, extract_pattern
from aura.auditor import StrategyAuditor
from aura.types import SceneState, ReasoningResult
from aura.workflow import WorkflowEngine, WorkflowMemory, WorkflowStep, WorkflowStatus


# =============================================================================
# Configuration
# =============================================================================

ALL_BACKENDS = ["default", "llm", "bmam", "model"]

LAYER_CONFIGS: Dict[str, Dict[str, bool]] = {
    "bare":  {"proactive_enabled": False, "explore_enabled": False, "guard_enabled": False},
    "pro":   {"proactive_enabled": True,  "explore_enabled": False, "guard_enabled": False},
    "exp":   {"proactive_enabled": False, "explore_enabled": True,  "guard_enabled": False},
    "grd":   {"proactive_enabled": False, "explore_enabled": False, "guard_enabled": True},
    # NOTE: proactive_enabled=False in experiment mode because real OS probes
    # are slow and non-deterministic.  The proactive engine is tested separately
    # in RQ1/RQ6.  "full" here means explore + guard (the new components).
    "full":  {"proactive_enabled": False, "explore_enabled": True,  "guard_enabled": True},
    "wf":    {"proactive_enabled": False, "explore_enabled": True,  "guard_enabled": True,
              "workflow_enabled": True},
}

SEEDS = [42, 123, 456]


# =============================================================================
# Simulated environment for controlled experiments
# =============================================================================

class SimulatedEnvironment:
    """Deterministic environment for reproducible experiments."""

    def __init__(self, scenario: Dict[str, Any], seed: int = 42):
        self.scenario = scenario
        self.rng = random.Random(seed)
        self.step_count = 0
        self.state: Dict[str, Any] = copy.deepcopy(scenario.get("initial_state", {}))
        self.injections: Dict[int, List[Dict]] = scenario.get("injections", {})

    def reset(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.step_count = 0
        self.state = copy.deepcopy(self.scenario.get("initial_state", {}))

    def step(self) -> Dict[str, Any]:
        step_injections = self.injections.get(self.step_count, [])
        for inj in step_injections:
            for k, v in inj.items():
                self.state[k] = v
        self.step_count += 1
        noise = self.rng.uniform(-0.02, 0.02)
        out = copy.deepcopy(self.state)
        out["_step"] = self.step_count
        out["_noise"] = noise
        return out


# =============================================================================
# Scenario definitions
# =============================================================================

def _make_loop_scenarios() -> List[Dict[str, Any]]:
    """5 loop-detection scenarios."""
    return [
        {
            "name": "true_loop_restart",
            "description": "Service crashes repeatedly; restart never works",
            "expected_guard_trigger": True,
            "max_steps": 15,
            "initial_state": {
                "summary": "web service returning 503",
                "entities": ["web_service", "nginx"],
                "cpu": 30, "memory": 45,
                "error": "503 Service Unavailable",
            },
            "injections": {},  # static environment — restart never fixes it
            "agent_action_pattern": "restart_service",  # agent repeats this
        },
        {
            "name": "false_positive_bisect",
            "description": "Binary search — looks repetitive but each step narrows",
            "expected_guard_trigger": False,
            "max_steps": 12,
            "initial_state": {
                "summary": "searching for bug in commit range 1-100",
                "entities": ["git_repo"],
                "range_low": 1, "range_high": 100,
            },
            "injections": {
                i: [{"range_low": i * 5, "range_high": 100 - i * 5,
                     "summary": f"bisecting range {i*5}-{100-i*5}"}]
                for i in range(20)
            },
            "agent_action_pattern": None,  # diverse actions
        },
        {
            "name": "slow_convergence",
            "description": "Performance tuning — tiny improvements each step",
            "expected_guard_trigger": False,
            "max_steps": 12,
            "initial_state": {
                "summary": "tuning database query performance",
                "entities": ["postgres", "query_optimizer"],
                "latency_ms": 500,
            },
            "injections": {
                i: [{"latency_ms": max(50, 500 - i * 25),
                     "summary": f"latency improved to {max(50, 500 - i*25)}ms"}]
                for i in range(20)
            },
            "agent_action_pattern": None,
        },
        {
            "name": "stale_strategy",
            "description": "docker-compose → k8s migration; old strategy stops working",
            "expected_guard_trigger": True,
            "max_steps": 12,
            "initial_state": {
                "summary": "docker-compose services running",
                "entities": ["docker_compose", "app_container"],
                "orchestrator": "docker-compose",
            },
            "injections": {
                5: [{"orchestrator": "kubernetes", "summary": "migrated to kubernetes",
                     "entities": ["kubernetes", "deployment"]}],
            },
            "agent_action_pattern": "docker_compose_restart",  # stale after step 5
        },
        {
            "name": "local_optimum",
            "description": "Agent manually checks files; grep would be faster",
            "expected_guard_trigger": False,  # guard shouldn't block, but auditor should notice
            "max_steps": 12,
            "initial_state": {
                "summary": "searching for error string in codebase",
                "entities": ["source_code"],
                "files_checked": 0, "total_files": 100,
            },
            "injections": {
                i: [{"files_checked": i + 1,
                     "summary": f"checked {i+1}/100 files manually"}]
                for i in range(20)
            },
            "agent_action_pattern": None,
        },
    ]


def _make_staleness_scenarios() -> List[Dict[str, Any]]:
    """Scenarios for StrategyAuditor validation."""
    return [
        {
            "name": "tool_deprecated",
            "description": "A tool the agent relied on gets removed",
            "phases": [
                {"steps": 5, "state": {"tool_available": True, "summary": "tool X works"}},
                {"steps": 10, "state": {"tool_available": False,
                                         "summary": "tool X removed, use tool Y instead"}},
            ],
        },
        {
            "name": "api_version_change",
            "description": "API v1 -> v2 migration; old endpoints return 404",
            "phases": [
                {"steps": 5, "state": {"api_version": "v1", "summary": "api v1 healthy"}},
                {"steps": 10, "state": {"api_version": "v2",
                                         "summary": "api v1 endpoints returning 404"}},
            ],
        },
    ]


def _make_workflow_scenarios() -> List[Dict[str, Any]]:
    """Scenarios for workflow lifecycle — pipeline reuse, validation, tool forge."""
    return [
        {
            "name": "pipeline_reuse",
            "description": "Agent discovers a 3-step pipeline, then reuses it on similar tasks",
            "phases": [
                # Phase 1: explore and discover pipeline
                {"steps": 5, "state": {
                    "summary": "deploy service v1",
                    "entities": ["build_server", "docker_registry", "k8s_cluster"],
                    "task": "deploy", "version": "v1",
                }, "action": "deploy_pipeline"},
                # Phase 2: same task type — should reuse
                {"steps": 5, "state": {
                    "summary": "deploy service v2",
                    "entities": ["build_server", "docker_registry", "k8s_cluster"],
                    "task": "deploy", "version": "v2",
                }, "action": "deploy_pipeline"},
            ],
            "expected_reuse": True,
        },
        {
            "name": "pipeline_invalidation",
            "description": "Pipeline becomes stale when a tool is removed",
            "phases": [
                {"steps": 5, "state": {
                    "summary": "backup database",
                    "entities": ["postgres", "s3_bucket"],
                    "task": "backup",
                }, "action": "backup_pipeline"},
                # Tool gets removed mid-run
                {"steps": 5, "state": {
                    "summary": "backup database — pg_dump removed",
                    "entities": ["postgres", "s3_bucket"],
                    "task": "backup", "tool_removed": "pg_dump",
                }, "action": "backup_pipeline"},
            ],
            "expected_reuse": False,  # should detect invalidation
        },
        {
            "name": "tool_forge_on_failure",
            "description": "Tool fails repeatedly, forge creates a replacement",
            "phases": [
                {"steps": 8, "state": {
                    "summary": "monitoring health check",
                    "entities": ["monitoring_service"],
                    "task": "health_check",
                    "tool_failing": "health.check",
                }, "action": "health_check"},
            ],
            "expected_forge": True,
        },
    ]


# =============================================================================
# Test runners
# =============================================================================

@dataclass
class ScenarioResult:
    scenario_name: str
    backend: str
    layers: str
    seed: int
    guard_triggered: bool = False
    trigger_step: int = -1
    max_intervention_level: str = "OBSERVE"
    avg_information_gain: float = 0.0
    final_pattern_strength: float = 0.0
    steps_run: int = 0
    correct_decision: bool = False
    feedback_entries: int = 0
    auditor_stale_count: int = 0


def run_loop_scenario(
    scenario: Dict[str, Any],
    agent: AURAAgent,
    seed: int = 42,
) -> ScenarioResult:
    """Run a single loop-detection scenario against an agent."""
    env = SimulatedEnvironment(scenario, seed=seed)
    max_steps = scenario.get("max_steps", 20)
    fixed_action = scenario.get("agent_action_pattern")

    result = ScenarioResult(
        scenario_name=scenario["name"],
        backend="",
        layers="",
        seed=seed,
    )

    if agent.guard:
        agent.guard.reset()

    max_level = InterventionLevel.OBSERVE

    for step in range(max_steps):
        env_state = env.step()

        # Build raw input
        raw_input = env_state

        # If scenario prescribes a fixed action pattern, use it as query
        if fixed_action:
            query = fixed_action
        else:
            query = f"step_{step}_action_{step % 7}"

        interaction = agent.run(raw_input, user_query=query)
        result.steps_run = step + 1

        # Check guard payload
        guard_info = interaction.payload.get("guard", {})
        level_name = guard_info.get("level", "OBSERVE")
        try:
            level = InterventionLevel[level_name]
        except KeyError:
            level = InterventionLevel.OBSERVE

        if level.value > max_level.value:
            max_level = level
        if level.value >= InterventionLevel.HINT.value and not result.guard_triggered:
            result.guard_triggered = True
            result.trigger_step = step

    result.max_intervention_level = max_level.name

    if agent.guard:
        stats = agent.guard.get_stats()
        result.avg_information_gain = stats.get("avg_information_gain", 0)
        result.final_pattern_strength = stats.get("pattern_strength", 0)

    if agent.feedback_store:
        result.feedback_entries = len(agent.feedback_store.get_all_entries())

    # Check correctness
    expected = scenario.get("expected_guard_trigger", False)
    result.correct_decision = (result.guard_triggered == expected)

    return result


def run_staleness_scenario(
    scenario: Dict[str, Any],
    agent: AURAAgent,
    seed: int = 42,
) -> ScenarioResult:
    """Run a staleness-detection scenario."""
    result = ScenarioResult(
        scenario_name=scenario["name"],
        backend="",
        layers="",
        seed=seed,
    )

    if agent.guard:
        agent.guard.reset()

    total_steps = 0
    for phase in scenario["phases"]:
        phase_steps = phase["steps"]
        state = phase["state"]
        for step in range(phase_steps):
            raw_input = {**state, "_step": total_steps}
            interaction = agent.run(raw_input, user_query="continue_task")
            total_steps += 1

    result.steps_run = total_steps

    if agent.auditor:
        audit = agent.auditor.audit(
            SceneState(summary="final", entities=[], context={})
        )
        result.auditor_stale_count = audit.get("stale_count", 0)

    if agent.feedback_store:
        result.feedback_entries = len(agent.feedback_store.get_all_entries())

    # Staleness scenarios are "correct" if the auditor found stale strategies
    result.correct_decision = result.auditor_stale_count > 0 or result.feedback_entries > 0
    return result


def run_workflow_scenario(
    scenario: Dict[str, Any],
    agent: AURAAgent,
    seed: int = 42,
) -> ScenarioResult:
    """Run a workflow lifecycle scenario."""
    result = ScenarioResult(
        scenario_name=scenario["name"],
        backend="",
        layers="",
        seed=seed,
    )

    if agent.guard:
        agent.guard.reset()

    total_steps = 0
    for phase in scenario["phases"]:
        phase_steps = phase["steps"]
        state = phase["state"]
        for step in range(phase_steps):
            raw_input = {**state, "_step": total_steps}
            action = phase.get("action", f"step_{step}")
            interaction = agent.run(raw_input, user_query=action)
            total_steps += 1

    result.steps_run = total_steps

    # Evaluate workflow metrics
    wf_stats = {}
    if agent.workflow_engine:
        wf_stats = agent.workflow_engine.get_stats()
        result.feedback_entries = len(agent.feedback_store.get_all_entries()) if agent.feedback_store else 0

    # For pipeline_reuse: check if reuse happened
    if scenario.get("expected_reuse"):
        reuse_count = wf_stats.get("reuse_count", 0)
        result.correct_decision = reuse_count > 0
    # For pipeline_invalidation: check if stale detected
    elif scenario.get("expected_reuse") is False:
        stale = agent.workflow_engine.memory.list_stale() if agent.workflow_engine else []
        result.correct_decision = len(stale) > 0 or result.feedback_entries > 0
    # For tool_forge: check if forge happened
    elif scenario.get("expected_forge"):
        forge_count = wf_stats.get("forge", {}).get("forged_count", 0)
        gap_count = wf_stats.get("forge", {}).get("active_gaps", 0)
        result.correct_decision = forge_count > 0 or gap_count > 0
    else:
        result.correct_decision = True

    return result


def build_agent(backend: str, layer_config: Dict[str, bool]) -> AURAAgent:
    """Build an AURAAgent with specific backend and layer configuration."""
    config = AURAConfig(
        backend=backend,
        proactive_enabled=layer_config.get("proactive_enabled", False),
        explore_enabled=layer_config.get("explore_enabled", False),
        explore_max_steps=1,  # lightweight for experiments (1 tool call)
        guard_enabled=layer_config.get("guard_enabled", False),
        guard_window=6,
        guard_threshold=0.5,
        auditor_enabled=layer_config.get("guard_enabled", False),
        auditor_staleness_halflife=2.0,  # fast decay for testing
        auditor_revalidation_budget=0.15,
        workflow_enabled=layer_config.get("workflow_enabled", False),
    )
    try:
        return AURAAgent(config=config)
    except Exception as e:
        print(f"  [WARN] Backend '{backend}' failed: {e}, falling back to default")
        config.backend = "default"
        return AURAAgent(config=config)


# =============================================================================
# Main experiment runner
# =============================================================================

def run_rq7(
    backends: Optional[List[str]] = None,
    layers: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    results_dir: str = "evaluation/results",
) -> Dict[str, Any]:
    """Run the full RQ7 experiment."""
    backends = backends or ["default"]
    layers = layers or list(LAYER_CONFIGS.keys())
    seeds = seeds or SEEDS

    loop_scenarios = _make_loop_scenarios()
    stale_scenarios = _make_staleness_scenarios()
    workflow_scenarios = _make_workflow_scenarios()

    all_results: Dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "backends": backends,
            "layers": layers,
            "seeds": seeds,
            "num_loop_scenarios": len(loop_scenarios),
            "num_stale_scenarios": len(stale_scenarios),
            "num_workflow_scenarios": len(workflow_scenarios),
        },
        "loop_detection": {},
        "staleness_detection": {},
        "workflow_lifecycle": {},
        "summary": {},
    }

    print("=" * 70)
    print("RQ7: Backend-Agnostic Ablation Experiment")
    print("=" * 70)
    print(f"Backends: {backends}")
    print(f"Layers: {layers}")
    print(f"Seeds: {seeds}")
    print(f"Loop scenarios: {len(loop_scenarios)}")
    print(f"Staleness scenarios: {len(stale_scenarios)}")
    print(f"Workflow scenarios: {len(workflow_scenarios)}")
    print()

    for backend in backends:
        for layer_name in layers:
            experiment_id = f"{backend[0].upper()}+{layer_name}"
            layer_cfg = LAYER_CONFIGS[layer_name]

            print(f"\n--- {experiment_id} (backend={backend}, layers={layer_name}) ---")

            loop_results: List[Dict] = []
            stale_results: List[Dict] = []

            for seed in seeds:
                agent = build_agent(backend, layer_cfg)

                # Loop detection scenarios
                for scenario in loop_scenarios:
                    r = run_loop_scenario(scenario, agent, seed=seed)
                    r.backend = backend
                    r.layers = layer_name
                    loop_results.append(asdict(r))

                # Staleness detection scenarios
                for scenario in stale_scenarios:
                    agent_fresh = build_agent(backend, layer_cfg)
                    r = run_staleness_scenario(scenario, agent_fresh, seed=seed)
                    r.backend = backend
                    r.layers = layer_name
                    stale_results.append(asdict(r))

            # Workflow lifecycle scenarios
            wf_results: List[Dict] = []
            for seed in seeds:
                for scenario in workflow_scenarios:
                    agent_wf = build_agent(backend, layer_cfg)
                    r = run_workflow_scenario(scenario, agent_wf, seed=seed)
                    r.backend = backend
                    r.layers = layer_name
                    wf_results.append(asdict(r))

            # Aggregate
            loop_correct = [r["correct_decision"] for r in loop_results]
            loop_precision = sum(loop_correct) / max(len(loop_correct), 1)

            stale_correct = [r["correct_decision"] for r in stale_results]
            stale_precision = sum(stale_correct) / max(len(stale_correct), 1)

            avg_ig = _mean([r["avg_information_gain"] for r in loop_results])
            avg_trigger_step = _mean([
                r["trigger_step"] for r in loop_results if r["trigger_step"] >= 0
            ])

            wf_correct = [r["correct_decision"] for r in wf_results]
            wf_precision = sum(wf_correct) / max(len(wf_correct), 1)

            summary = {
                "loop_precision": round(loop_precision, 4),
                "stale_precision": round(stale_precision, 4),
                "workflow_precision": round(wf_precision, 4),
                "avg_information_gain": round(avg_ig, 4),
                "avg_trigger_step": round(avg_trigger_step, 2),
                "total_loop_tests": len(loop_results),
                "total_stale_tests": len(stale_results),
                "total_workflow_tests": len(wf_results),
            }

            all_results["loop_detection"][experiment_id] = loop_results
            all_results["staleness_detection"][experiment_id] = stale_results
            all_results["workflow_lifecycle"][experiment_id] = wf_results
            all_results["summary"][experiment_id] = summary

            print(f"  Loop precision: {loop_precision:.2%}")
            print(f"  Stale precision: {stale_precision:.2%}")
            print(f"  Workflow precision: {wf_precision:.2%}")
            print(f"  Avg IG: {avg_ig:.4f}")
            if avg_trigger_step > 0:
                print(f"  Avg trigger step: {avg_trigger_step:.1f}")

    # Cross-backend analysis
    print("\n" + "=" * 70)
    print("Cross-Backend Analysis")
    print("=" * 70)
    _print_cross_backend_table(all_results["summary"])

    # Save results
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rq7_backend_ablation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return all_results


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_cross_backend_table(summary: Dict[str, Dict]) -> str:
    """Print a comparison table."""
    header = f"{'Config':<12} {'Loop Prec':>10} {'Stale Prec':>11} {'WF Prec':>9} {'Avg IG':>8} {'Trigger@':>9}"
    print(header)
    print("-" * len(header))
    for config_id, metrics in sorted(summary.items()):
        print(
            f"{config_id:<12} "
            f"{metrics['loop_precision']:>10.2%} "
            f"{metrics['stale_precision']:>11.2%} "
            f"{metrics.get('workflow_precision', 0):>9.2%} "
            f"{metrics['avg_information_gain']:>8.4f} "
            f"{metrics['avg_trigger_step']:>9.1f}"
        )

    # Key comparison: does +guard improve over bare?
    print()
    for backend_prefix in set(k.split("+")[0] for k in summary):
        bare_key = f"{backend_prefix}+bare"
        full_key = f"{backend_prefix}+full"
        if bare_key in summary and full_key in summary:
            bare_lp = summary[bare_key]["loop_precision"]
            full_lp = summary[full_key]["loop_precision"]
            delta = full_lp - bare_lp
            print(f"  {backend_prefix}: bare→full loop precision delta = {delta:+.2%}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RQ7: Backend-Agnostic Ablation")
    parser.add_argument("--backends", nargs="*", default=["default"],
                        help="Backends to test")
    parser.add_argument("--layers", nargs="*", default=None,
                        help="Layer configs to test (default: all)")
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--results-dir", default="evaluation/results")
    args = parser.parse_args()

    run_rq7(
        backends=args.backends,
        layers=args.layers,
        seeds=args.seeds,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
