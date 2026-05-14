#!/usr/bin/env python3
"""
Main experiment runner for the AURA paper.

Usage:
    python -m evaluation.run_experiments --rq all
    python -m evaluation.run_experiments --rq 1 2
    python -m evaluation.run_experiments --rq 1 --steps 50
"""

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.config import EvalConfig, SOTOPIA_DIMENSIONS
from evaluation.llm_judge import (
    judge_grounding,
    judge_factual_accuracy,
    judge_social_interaction,
    judge_context_utilization,
)
from evaluation.action_grounding_eval import check_location_consistency, check_time_consistency
from evaluation.baselines import vanilla_llm_chat, static_context_chat, react_chat
from evaluation.baselines import (
    reflexion_chat, reflexion_action_decision,
    plan_and_solve_chat, plan_and_solve_action_decision,
)


# =============================================================================
# Statistical Helpers
# =============================================================================

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _paired_ttest(a: List[float], b: List[float]) -> float:
    """Paired t-test p-value (two-tailed). Returns p-value estimate."""
    if len(a) != len(b) or len(a) < 2:
        return 1.0
    diffs = [x - y for x, y in zip(a, b)]
    m = _mean(diffs)
    s = _std(diffs)
    if s == 0:
        return 0.0 if m != 0 else 1.0
    n = len(diffs)
    t_stat = m / (s / math.sqrt(n))
    # Approximate p-value using normal distribution for large-ish samples
    # For proper stats, use scipy — this is a rough estimate
    p = 2 * (1 - _normal_cdf(abs(t_stat)))
    return p


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _aggregate_budget_statistics(per_seed_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate RQ6-style per-budget results across seeds.

    Multi-seed RQ6 returns structured outputs ({per_budget, pareto_frontier}),
    not a single scalar metric. Summarising it as a scalar breaks the experiment
    runner and loses the latency/GA trade-off that the analysis depends on.
    """
    per_budget: Dict[str, Dict[str, List[float]]] = {}
    for result in per_seed_results.values():
        if not isinstance(result, dict):
            continue
        for budget, entry in result.get("per_budget", {}).items():
            bucket = per_budget.setdefault(str(budget), {
                "avg_ga": [],
                "avg_latency": [],
                "num_ga_judgments": [],
            })
            for metric in bucket:
                value = entry.get(metric)
                if isinstance(value, (int, float)):
                    bucket[metric].append(float(value))

    budget_summary: Dict[str, Dict[str, Any]] = {}
    pareto_points: List[Dict[str, float]] = []
    for budget in sorted(per_budget, key=int):
        metrics = per_budget[budget]
        ga_vals = metrics["avg_ga"]
        latency_vals = metrics["avg_latency"]
        judgment_vals = metrics["num_ga_judgments"]
        entry = {
            "avg_ga_mean": round(_mean(ga_vals), 4),
            "avg_ga_std": round(_std(ga_vals), 4),
            "avg_latency_mean": round(_mean(latency_vals), 3),
            "avg_latency_std": round(_std(latency_vals), 3),
            "avg_num_judgments": round(_mean(judgment_vals), 1),
            "seeds": len(ga_vals),
        }
        budget_summary[budget] = entry
        pareto_points.append({
            "budget": int(budget),
            "avg_latency": entry["avg_latency_mean"],
            "avg_ga": entry["avg_ga_mean"],
        })

    return {
        "per_budget": budget_summary,
        "pareto_frontier": _compute_pareto_frontier(pareto_points),
    }


# =============================================================================
# API Helpers
# =============================================================================

class AURAClient:
    """Client for the AURA Town API server."""

    def __init__(self, base_url: str = "http://127.0.0.1:7861"):
        self.base = base_url.rstrip("/")

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base}/api/health", timeout=5)
            return r.json().get("ok", False)
        except Exception:
            return False

    def _request_with_retry(self, method, path, max_retries=3, **kwargs):
        for attempt in range(max_retries):
            try:
                if method == "GET":
                    r = requests.get(f"{self.base}{path}", **kwargs)
                else:
                    r = requests.post(f"{self.base}{path}", **kwargs)
                return r.json()
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** (attempt + 1)
                print(f"  [RETRY] {method} {path} attempt {attempt+1}/{max_retries}: {e}. Waiting {wait}s...")
                time.sleep(wait)
        return {}

    def state(self) -> Dict:
        r = requests.get(f"{self.base}/api/state", timeout=10)
        return r.json()

    def step(self) -> Dict:
        return self._request_with_retry("POST", "/api/step", timeout=120)

    def reset(self, seed: Optional[int] = None) -> Dict:
        payload = {"seed": int(seed)} if seed is not None else None
        return self._request_with_retry(
            "POST", "/api/reset", json=payload, timeout=30,
        )

    def chat(self, user: str, message: str, read_only: bool = False) -> Dict:
        return self._request_with_retry(
            "POST", "/api/chat",
            json={"user": user, "message": message, "read_only": read_only},
            timeout=120,
        )

    def agent_detail(self, name: str) -> Dict:
        r = requests.get(f"{self.base}/api/agent", params={"name": name}, timeout=10)
        return r.json()

    def set_probe(self, enabled: bool, max_steps: int) -> Dict:
        r = requests.post(
            f"{self.base}/api/probe",
            json={"enabled": enabled, "max_steps": max_steps},
            timeout=10,
        )
        return r.json()

    def set_ablation(self, memory_enabled: bool = True, reflection_enabled: bool = True) -> Dict:
        r = requests.post(
            f"{self.base}/api/ablation",
            json={"memory_enabled": memory_enabled, "reflection_enabled": reflection_enabled},
            timeout=10,
        )
        return r.json()

    def set_action_mode(self, react_mode: bool = False) -> Dict:
        r = requests.post(
            f"{self.base}/api/action_mode",
            json={"react_mode": react_mode},
            timeout=10,
        )
        return r.json()


def save_results(results: Any, filename: str, config: EvalConfig):
    """Save results to JSON file with experiment metadata."""
    out_dir = Path(config.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Inject metadata so we know which model produced these results
    if isinstance(results, dict):
        results["_metadata"] = {
            "backbone_model": config.model,
            "judge_model": config.judge_model,
            "timestamp": datetime.now().isoformat(),
        }
    path = out_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  -> Saved to {path}")


# =============================================================================
# RQ1: Proactive Probing vs Reactive — Grounding Accuracy
# =============================================================================

def _run_rq1_condition(config, client, cond, num_steps):
    """Run a single RQ1 condition and return step_results + summary."""
    step_results = []
    for step_i in range(num_steps):
        if step_i % 10 == 0:
            print(f"  Step {step_i}/{num_steps}...")

        t0 = time.time()
        result = client.step()
        latency = time.time() - t0

        if not result.get("ok"):
            step_results.append({"step": step_i, "error": "step failed"})
            continue

        post_state = result["state"]
        hour = post_state.get("hour", 6)

        # Prepare all agent data for this step, then judge in parallel
        agent_tasks = []
        for agent in post_state.get("agents", []):
            name = agent["name"]
            action_str = agent.get("action", "")
            location_str = agent.get("location", "")

            loc_ok = check_location_consistency(action_str, location_str)
            time_ok = check_time_consistency(action_str, hour)

            daily_plan = agent.get("plan", [])
            memories = [m.get("content", str(m)) if isinstance(m, dict) else str(m)
                        for m in agent.get("memories", [])[:5]]

            if not memories:
                detail = client.agent_detail(name)
                if detail.get("ok"):
                    agent_detail = detail.get("agent", {})
                    memories = [m.get("content", "") for m in agent_detail.get("memories", [])[:5]]
                    if not daily_plan:
                        daily_plan = agent_detail.get("plan", [])

            action_info = {
                "action": action_str,
                "location": location_str,
                "thought": agent.get("thought_bubble", ""),
            }
            agent_tasks.append((name, action_str, location_str, loc_ok, time_ok, action_info, daily_plan, memories))

        # Parallel LLM judge calls (5 agents per step)
        def _judge_agent(args):
            name, action_str, location_str, loc_ok, time_ok, action_info, daily_plan, memories = args
            judgment = judge_grounding(
                config,
                env_state=post_state,
                agent_name=name,
                action=action_info,
                daily_plan=daily_plan,
                memories=memories,
            )
            return {
                "step": step_i,
                "agent": name,
                "action": action_str,
                "location": location_str,
                "latency": latency,
                "judgment": judgment,
                "rule_location_ok": loc_ok,
                "rule_time_ok": time_ok,
            }

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_judge_agent, t) for t in agent_tasks]
            for f in as_completed(futures):
                step_results.append(f.result())

    # Aggregate
    valid = [r for r in step_results if "judgment" in r and "error" not in r.get("judgment", {})]
    rule_total = [r for r in step_results if "rule_location_ok" in r]
    if valid:
        avg_overall = sum(r["judgment"].get("overall", 0) for r in valid) / len(valid)
        avg_latency = sum(r["latency"] for r in step_results if "latency" in r) / max(len(step_results), 1)

        dim_scores = {}
        for dim in ["location_consistency", "time_appropriateness", "social_awareness", "memory_utilization", "plan_adherence"]:
            scores = [r["judgment"].get(dim, 0) for r in valid]
            dim_scores[dim] = sum(scores) / len(scores) if scores else 0

        rule_loc_acc = sum(1 for r in rule_total if r["rule_location_ok"]) / max(len(rule_total), 1)
        rule_time_acc = sum(1 for r in rule_total if r["rule_time_ok"]) / max(len(rule_total), 1)

        summary = {
            "condition": cond["name"],
            "total_steps": num_steps,
            "valid_judgments": len(valid),
            "overall_grounding_accuracy": round(avg_overall, 4),
            "dimension_scores": {k: round(v, 4) for k, v in dim_scores.items()},
            "rule_based_location_accuracy": round(rule_loc_acc, 4),
            "rule_based_time_accuracy": round(rule_time_acc, 4),
            "avg_latency_per_step": round(avg_latency, 3),
        }
    else:
        summary = {"condition": cond["name"], "error": "no valid judgments"}

    return summary, step_results


def run_rq1(config: EvalConfig, client: AURAClient):
    """
    Compare grounding accuracy across conditions including baselines.
    Baselines are simulated by disabling components via ablation API.
    """
    print("\n" + "=" * 60)
    print("RQ1: Proactive Probing vs Reactive — Grounding Accuracy")
    print("=" * 60)

    # Each condition specifies probe + ablation settings.
    # IMPORTANT: AURA_Full uses probe_max_steps=2 to align with RQ6 optimal
    # budget (B=2) and the ablation study (RQ3). This ensures consistent
    # reporting across all RQs in the paper.
    #
    # Baseline budget parity:
    #   - ReAct, Reflexion, Plan-and-Solve all use max_steps=5 (same tool budget)
    #   - AURA_Full uses probe B=2 (optimal from RQ6 Pareto analysis)
    conditions = [
        {"name": "Vanilla_LLM", "probe_enabled": False, "probe_max_steps": 0,
         "memory_enabled": False, "reflection_enabled": False, "react_mode": False},
        {"name": "Static_Context", "probe_enabled": False, "probe_max_steps": 0,
         "memory_enabled": True, "reflection_enabled": False, "react_mode": False},
        {"name": "ReAct", "probe_enabled": False, "probe_max_steps": 0,
         "memory_enabled": True, "reflection_enabled": True, "react_mode": True},
        {"name": "Reflexion", "probe_enabled": False, "probe_max_steps": 0,
         "memory_enabled": True, "reflection_enabled": True, "react_mode": True,
         "baseline_type": "reflexion"},
        {"name": "Plan_and_Solve", "probe_enabled": False, "probe_max_steps": 0,
         "memory_enabled": True, "reflection_enabled": True, "react_mode": True,
         "baseline_type": "plan_and_solve"},
        {"name": "AURA_NoProbe", "probe_enabled": False, "probe_max_steps": 0,
         "memory_enabled": True, "reflection_enabled": True, "react_mode": False},
        {"name": "AURA_Full", "probe_enabled": True, "probe_max_steps": 2,
         "memory_enabled": True, "reflection_enabled": True, "react_mode": False},
    ]

    all_results = {}

    for cond in conditions:
        print(f"\n--- Condition: {cond['name']} ---")
        client.reset()
        client.set_probe(cond["probe_enabled"], cond["probe_max_steps"])
        client.set_ablation(cond.get("memory_enabled", True), cond.get("reflection_enabled", True))
        client.set_action_mode(cond.get("react_mode", False))

        summary, step_results = _run_rq1_condition(config, client, cond, config.num_simulation_steps)

        all_results[cond["name"]] = {"summary": summary, "details": step_results}
        print(f"  GA = {summary.get('overall_grounding_accuracy', 'N/A')}, "
              f"Rule Loc = {summary.get('rule_based_location_accuracy', 'N/A')}, "
              f"Latency = {summary.get('avg_latency_per_step', 'N/A')}s")

    save_results(all_results, "rq1_grounding_accuracy.json", config)
    return all_results


# =============================================================================
# RQ2: Environment-Enriched Chat — Factual Accuracy
# =============================================================================

def run_rq2(config: EvalConfig, client: AURAClient):
    """RQ2: Environment-Enriched Chat — Factual Accuracy (paired snapshot mode).

    Three structural fixes compared to the legacy implementation:
      1. Per-condition reset+warmup so cross-condition state is identical.
      2. Frozen ground-truth snapshot per query position, taken once in a
         dedicated Phase A; all conditions (AURA + external baselines)
         judge against the same snapshot at the same position.
      3. AURA chats run in `read_only=True` mode so within-condition chats
         do not write event log or memory, leaving the trajectory between
         queries deterministic.

    Seed is propagated via /api/reset so the simulation server gets a
    fresh, reproducible TownConfig per (cond, seed) pair.
    """
    print("\n" + "=" * 60)
    print("RQ2: Environment-Enriched Chat — Factual Accuracy (paired)")
    print("=" * 60)

    seed = getattr(config, "current_seed", None)

    queries_path = Path(__file__).parent / "data" / "chat_queries.json"
    with open(queries_path) as f:
        query_data = json.load(f)
    queries = query_data["queries"][:config.num_chat_queries]

    # ── Phase A: collect frozen snapshots ────────────────────────────────
    # Reset once with the experiment seed, warm up 10 steps, then advance
    # 1 tick per query position to get distinct simulation states. Each
    # snapshot is the gt_state captured BEFORE any chat happens at that
    # position. Snapshots feed both AURA conditions (judging only) and
    # external baselines (which receive snapshot as static context).
    print("  [Phase A] collecting %d frozen snapshots (seed=%s)..." % (len(queries), seed))
    client.reset(seed=seed)
    for _ in range(10):
        client.step()
        time.sleep(0.3)
    agents = client.state()["state"]["agents"]
    agent_names = [a["name"] for a in agents]

    snapshots: List[Dict[str, Any]] = []
    for qi in range(len(queries)):
        gt_state = client.state()["state"]
        snapshots.append({
            "qi": qi,
            "tick": gt_state.get("tick", 10 + qi),
            "agent": agent_names[qi % len(agent_names)],
            "gt_state": gt_state,
        })
        # Advance 1 tick so each query sees a different state (avoid all
        # 50 queries hitting the same world snapshot, which would be a
        # different artifact).
        client.step()
        time.sleep(0.2)
    print(f"  [Phase A] collected {len(snapshots)} snapshots")

    aura_conditions = [
        {"name": "AURA_Full", "probe_enabled": True, "probe_max_steps": 2},
        {"name": "AURA_NoProbe", "probe_enabled": False, "probe_max_steps": 0},
    ]

    all_results = {}

    # ── Phase B: replay snapshots per condition ──────────────────────────
    # Per AURA condition: reset with the same seed, warm 10, then for each
    # query qi advance to qi extra steps so we are at the same world tick
    # the snapshot was captured at, and ask read_only chat.
    for cond in aura_conditions:
        print(f"\n--- Condition: {cond['name']} (paired replay) ---")
        client.reset(seed=seed)
        for _ in range(10):
            client.step()
            time.sleep(0.3)
        client.set_probe(cond["probe_enabled"], cond["probe_max_steps"])

        query_results = []
        for qi, q in enumerate(queries):
            if qi % 10 == 0:
                print(f"  Query {qi}/{len(queries)}...")

            snap = snapshots[qi]
            agent_name = snap["agent"]
            gt_state = snap["gt_state"]

            t0 = time.time()
            chat_result = client.chat(agent_name, q["query"], read_only=True)
            latency = time.time() - t0

            if not chat_result.get("ok"):
                query_results.append({"query_id": q["id"], "error": "chat failed"})
                # still advance the world so the next query sees the right tick
                client.step()
                continue

            response = chat_result.get("chat", {}).get("ai_response", "")
            env_context = chat_result.get("chat", {}).get("env_context", {})

            fa_judgment = judge_factual_accuracy(config, q["query"], gt_state, response, agent_name)
            cu_judgment = judge_context_utilization(config, env_context, response)

            query_results.append({
                "query_id": q["id"],
                "category": q["category"],
                "query": q["query"],
                "agent": agent_name,
                "response": response,
                "latency": latency,
                "factual_accuracy": fa_judgment,
                "context_utilization": cu_judgment,
                "has_probe": cond["probe_enabled"],
                "snapshot_tick": snap["tick"],
                "read_only_chat": True,
            })

            # Advance one tick to keep the trajectory aligned with snapshots
            client.step()
            time.sleep(0.1)

        summary = _aggregate_rq2(cond["name"], queries, query_results)
        all_results[cond["name"]] = {"summary": summary, "details": query_results}
        print(f"  FA = {summary['avg_factual_accuracy']}, CU = {summary['avg_context_utilization']}")

    # ── External baselines: each gets the SAME snapshots[qi] ─────────────
    # No simulation interaction needed — these are pure LLM calls with
    # injected context. By using snapshots[qi] (the same gt_state the
    # AURA conditions used at the same position) we make the per-query
    # paired contrast valid across all 5 conditions.
    baseline_methods = [
        ("Vanilla_LLM", "vanilla"),
        ("Static_Context", "static"),
        ("ReAct", "react"),
        ("Reflexion", "reflexion"),
        ("Plan_and_Solve", "plan_and_solve"),
    ]

    for bl_name, bl_type in baseline_methods:
        print(f"\n--- Baseline: {bl_name} (paired replay) ---")
        query_results = []
        for qi, q in enumerate(queries):
            if qi % 10 == 0:
                print(f"  Query {qi}/{len(queries)}...")

            snap = snapshots[qi]
            agent_name = snap["agent"]
            gt_state = snap["gt_state"]

            try:
                if bl_type == "vanilla":
                    bl_result = vanilla_llm_chat(config, agent_name, q["query"])
                elif bl_type == "static":
                    bl_result = static_context_chat(config, agent_name, q["query"], gt_state)
                elif bl_type == "reflexion":
                    bl_result = reflexion_chat(config, agent_name, q["query"], gt_state)
                elif bl_type == "plan_and_solve":
                    bl_result = plan_and_solve_chat(config, agent_name, q["query"], gt_state)
                else:
                    bl_result = react_chat(config, agent_name, q["query"], gt_state)

                response = bl_result.get("response", "")
                env_context = bl_result.get("env_context", {})
                latency = bl_result.get("latency", 0)

                fa_judgment = judge_factual_accuracy(config, q["query"], gt_state, response, agent_name)
                cu_judgment = judge_context_utilization(config, env_context, response)

                query_results.append({
                    "query_id": q["id"],
                    "category": q["category"],
                    "query": q["query"],
                    "agent": agent_name,
                    "response": response,
                    "latency": latency,
                    "factual_accuracy": fa_judgment,
                    "context_utilization": cu_judgment,
                    "has_probe": False,
                    "snapshot_tick": snap["tick"],
                })
            except Exception as e:
                query_results.append({"query_id": q["id"], "error": str(e)})
            time.sleep(0.3)

        summary = _aggregate_rq2(bl_name, queries, query_results)
        all_results[bl_name] = {"summary": summary, "details": query_results}
        print(f"  FA = {summary['avg_factual_accuracy']}, CU = {summary['avg_context_utilization']}")

    # Stamp paired-snapshot metadata into the saved file so downstream
    # aggregators can verify the replay structure.
    all_results["_paired_snapshot_meta"] = {
        "version": "1.0",
        "seed": seed,
        "n_queries": len(queries),
        "n_snapshots": len(snapshots),
        "warmup_steps": 10,
        "step_between_queries": 1,
        "read_only_chat": True,
        "shared_snapshots_across_conditions": True,
    }

    save_results(all_results, "rq2_factual_accuracy.json", config)
    return all_results


def _aggregate_rq2(cond_name: str, queries: list, query_results: list) -> dict:
    """Aggregate RQ2 results for a single condition."""
    valid_fa = [r for r in query_results if "accuracy" in r.get("factual_accuracy", {})]
    valid_cu = [r for r in query_results if "utilization" in r.get("context_utilization", {})]

    summary = {
        "condition": cond_name,
        "total_queries": len(queries),
        "valid_fa": len(valid_fa),
        "avg_factual_accuracy": round(
            sum(r["factual_accuracy"]["accuracy"] for r in valid_fa) / max(len(valid_fa), 1), 4
        ),
        "avg_context_utilization": round(
            sum(r["context_utilization"]["utilization"] for r in valid_cu) / max(len(valid_cu), 1), 4
        ),
        "avg_latency": round(
            sum(r["latency"] for r in query_results if "latency" in r) / max(len(query_results), 1), 3
        ),
        "by_category": {},
    }

    for cat in ["spatial", "social", "temporal", "memory", "planning"]:
        cat_results = [r for r in valid_fa if r.get("category") == cat]
        if cat_results:
            summary["by_category"][cat] = round(
                sum(r["factual_accuracy"]["accuracy"] for r in cat_results) / len(cat_results), 4
            )

    return summary


# =============================================================================
# RQ3: Social Interaction Quality (SOTOPIA 7-Dimension)
# =============================================================================

def run_rq3(config: EvalConfig, client: AURAClient):
    """
    Evaluate social interaction quality using SOTOPIA 7-dimension framework.
    Run simulation steps and collect conversations for evaluation.
    """
    print("\n" + "=" * 60)
    print("RQ3: Social Interaction Quality (SOTOPIA 7-Dimension)")
    print("=" * 60)

    client.reset()
    client.set_probe(True, 2)

    conversations = []
    steps_run = 0
    max_steps = config.num_simulation_steps * 2  # May need more steps to collect enough conversations

    print(f"  Collecting {config.num_social_episodes} conversations...")
    while len(conversations) < config.num_social_episodes and steps_run < max_steps:
        result = client.step()
        steps_run += 1

        if not result.get("ok"):
            continue

        state = result["state"]
        events = state.get("events", [])

        # Find conversation events from this step
        for evt in events:
            if evt.get("type") == "conversation" and evt.get("details", {}).get("dialogue"):
                conversations.append({
                    "event": evt,
                    "state_snapshot": {
                        "time": state.get("time"),
                        "agents": state.get("agents"),
                    },
                })

        if steps_run % 20 == 0:
            print(f"    Step {steps_run}, conversations collected: {len(conversations)}")

    print(f"  Collected {len(conversations)} conversations in {steps_run} steps.")

    # Evaluate each conversation
    results = []
    for ci, conv in enumerate(conversations[:config.num_social_episodes]):
        if ci % 5 == 0:
            print(f"  Evaluating conversation {ci}...")

        evt = conv["event"]
        dialogue = evt.get("details", {}).get("dialogue", [])
        agent_name = evt.get("agent", "")

        # Find agent profiles from state
        agents_in_state = conv["state_snapshot"].get("agents", [])

        # Try to identify both agents from dialogue
        speakers = set()
        for line in dialogue:
            if ":" in line:
                speakers.add(line.split(":")[0].strip())

        speaker_list = list(speakers)
        if len(speaker_list) < 2:
            continue

        agent1 = next((a for a in agents_in_state if a["name"] == speaker_list[0]), {})
        agent2 = next((a for a in agents_in_state if a["name"] == speaker_list[1]), {})

        judgment = judge_social_interaction(
            config,
            agent1_profile=agent1,
            agent2_profile=agent2,
            conversation=dialogue,
            context={
                "location": evt.get("details", {}).get("location", "unknown"),
                "time": conv["state_snapshot"].get("time", ""),
            },
        )

        results.append({
            "conversation_index": ci,
            "speakers": speaker_list,
            "dialogue_length": len(dialogue),
            "location": evt.get("details", {}).get("location"),
            "judgment": judgment,
        })

    # Aggregate SOTOPIA scores
    valid = [r for r in results if "error" not in r.get("judgment", {})]
    dimension_avgs = {}
    for dim in SOTOPIA_DIMENSIONS:
        scores = [r["judgment"].get(dim, {}).get("score", 0) for r in valid
                  if isinstance(r["judgment"].get(dim), dict)]
        dimension_avgs[dim] = round(sum(scores) / max(len(scores), 1), 3)

    overall_scores = [r["judgment"].get("overall_quality", 0) for r in valid]

    summary = {
        "total_conversations": len(conversations),
        "evaluated": len(valid),
        "steps_needed": steps_run,
        "sotopia_dimensions": dimension_avgs,
        "overall_quality": round(sum(overall_scores) / max(len(overall_scores), 1), 3),
    }

    save_results(
        {"summary": summary, "details": results},
        "rq3_social_interaction.json",
        config,
    )
    print(f"  SOTOPIA scores: {dimension_avgs}")
    print(f"  Overall quality: {summary['overall_quality']}")
    return summary


# =============================================================================
# RQ4: Emergent Social Behavior Analysis
# =============================================================================

def run_rq4(config: EvalConfig, client: AURAClient):
    """
    Run extended simulation (200 steps), collect all events, analyse social
    network metrics, SOTOPIA scores, and emergent behaviour patterns.
    """
    from evaluation.social_analysis import SocialNetworkAnalyzer, BehaviorPatternDetector

    print("\n" + "=" * 60)
    print("RQ4: Emergent Social Behavior Analysis")
    print("=" * 60)

    client.reset()
    client.set_probe(True, 2)
    client.set_ablation(True, True)

    analyzer = SocialNetworkAnalyzer()
    detector = BehaviorPatternDetector()
    all_events: List[Dict[str, Any]] = []
    conversations_for_sotopia: list = []

    num_steps = 200
    print(f"  Running {num_steps} simulation steps...")

    for step_i in range(num_steps):
        result = client.step()
        if not result.get("ok"):
            continue

        state = result["state"]
        events = state.get("events", [])

        for evt in events:
            all_events.append(evt)
            analyzer.add_event(evt)
            detector.add_event(evt)

            # Collect conversations for SOTOPIA evaluation
            if evt.get("type") == "conversation" and evt.get("details", {}).get("dialogue"):
                conversations_for_sotopia.append({
                    "event": evt,
                    "state_snapshot": {"time": state.get("time"), "agents": state.get("agents")},
                })

        # Record co-presence
        agents_by_loc: Dict[str, List[str]] = {}
        for a in state.get("agents", []):
            loc = a.get("location", "")
            if loc:
                agents_by_loc.setdefault(loc, []).append(a["name"])
        co_locs = {loc: names for loc, names in agents_by_loc.items() if len(names) >= 2}
        if co_locs:
            analyzer.add_co_presence(co_locs)

        if step_i % 50 == 0:
            print(f"    Step {step_i}/{num_steps}, events={len(all_events)}, "
                  f"conversations={len(conversations_for_sotopia)}")

    # Social network metrics
    network_metrics = analyzer.to_dict()

    # Emergent behavior detection
    behavior_summary = detector.to_summary()

    # SOTOPIA evaluation on a sample of conversations
    sotopia_scores = []
    sample_convs = conversations_for_sotopia[:config.num_social_episodes]
    print(f"\n  Evaluating {len(sample_convs)} conversations with SOTOPIA...")

    for ci, conv in enumerate(sample_convs):
        evt = conv["event"]
        dialogue = evt.get("details", {}).get("dialogue", [])
        agents_in_state = conv["state_snapshot"].get("agents", [])

        speakers = set()
        for line in dialogue:
            if ":" in line:
                speakers.add(line.split(":")[0].strip())
        speaker_list = list(speakers)
        if len(speaker_list) < 2:
            continue

        agent1 = next((a for a in agents_in_state if a["name"] == speaker_list[0]), {})
        agent2 = next((a for a in agents_in_state if a["name"] == speaker_list[1]), {})

        judgment = judge_social_interaction(
            config,
            agent1_profile=agent1,
            agent2_profile=agent2,
            conversation=dialogue,
            context={
                "location": evt.get("details", {}).get("location", "unknown"),
                "time": conv["state_snapshot"].get("time", ""),
            },
        )
        sotopia_scores.append(judgment)

    # Aggregate SOTOPIA
    valid_sotopia = [s for s in sotopia_scores if "error" not in s]
    dim_avgs = {}
    for dim in SOTOPIA_DIMENSIONS:
        scores = [s.get(dim, {}).get("score", 0) for s in valid_sotopia if isinstance(s.get(dim), dict)]
        dim_avgs[dim] = round(sum(scores) / max(len(scores), 1), 3)

    overall_scores = [s.get("overall_quality", 0) for s in valid_sotopia]

    # LLM-based emergent behavior identification (sample 5 conversations)
    llm_emergent_examples = []
    for conv in conversations_for_sotopia[:5]:
        evt = conv["event"]
        dialogue = evt.get("details", {}).get("dialogue", [])
        if dialogue:
            llm_emergent_examples.append({
                "time": evt.get("time", ""),
                "location": evt.get("details", {}).get("location", ""),
                "dialogue_excerpt": dialogue[:6],
            })

    results = {
        "simulation_steps": num_steps,
        "total_events": len(all_events),
        "total_conversations": len(conversations_for_sotopia),
        "network_metrics": network_metrics,
        "emergent_behaviors": behavior_summary,
        "sotopia_evaluation": {
            "evaluated": len(valid_sotopia),
            "dimension_averages": dim_avgs,
            "overall_quality": round(sum(overall_scores) / max(len(overall_scores), 1), 3),
        },
        "qualitative_examples": llm_emergent_examples,
    }

    save_results(results, "rq4_emergent_social.json", config)
    print(f"  Network: {network_metrics['num_edges']} conversations, density={network_metrics['density']}")
    print(f"  Emergent behaviors: {behavior_summary['total_behaviors']}")
    print(f"  SOTOPIA overall: {results['sotopia_evaluation']['overall_quality']}")
    return results


def run_rq4_baselines(config: EvalConfig, client: AURAClient):
    """
    Run RQ4 social simulation with baselines (Generative Agents, Static Context)
    for SOTOPIA comparison. Each baseline runs 200 steps with the same judge.
    """
    from evaluation.baselines import generative_agents_action_decision

    print("\n" + "=" * 60)
    print("RQ4 Baselines: Social Simulation with Baseline Action Policies")
    print("=" * 60)

    baseline_configs = [
        {"name": "Generative_Agents", "probe": False, "memory": True, "reflection": True,
         "action_mode": "generative_agents"},
        {"name": "Static_Context", "probe": False, "memory": False, "reflection": False,
         "action_mode": "static"},
    ]

    all_baseline_results = {}
    num_steps = 200

    for bcfg in baseline_configs:
        print(f"\n--- Baseline: {bcfg['name']} ---")
        client.reset()
        client.set_probe(bcfg["probe"], 0)
        client.set_ablation(bcfg.get("memory", True), bcfg.get("reflection", True))

        conversations_for_sotopia = []
        for step_i in range(num_steps):
            result = client.step()
            if not result.get("ok"):
                continue
            state = result["state"]
            events = state.get("events", [])
            for evt in events:
                if evt.get("type") == "conversation" and evt.get("details", {}).get("dialogue"):
                    conversations_for_sotopia.append({
                        "event": evt,
                        "state_snapshot": {"time": state.get("time"), "agents": state.get("agents")},
                    })
            if step_i % 50 == 0:
                print(f"    Step {step_i}/{num_steps}, conversations={len(conversations_for_sotopia)}")

        # SOTOPIA evaluation
        sotopia_scores = []
        sample_convs = conversations_for_sotopia[:config.num_social_episodes]
        print(f"  Evaluating {len(sample_convs)} conversations with SOTOPIA...")

        for conv in sample_convs:
            evt = conv["event"]
            dialogue = evt.get("details", {}).get("dialogue", [])
            agents_in_state = conv["state_snapshot"].get("agents", [])
            speakers = set()
            for line in dialogue:
                if ":" in line:
                    speakers.add(line.split(":")[0].strip())
            speaker_list = list(speakers)
            if len(speaker_list) < 2:
                continue
            agent1 = next((a for a in agents_in_state if a["name"] == speaker_list[0]), {})
            agent2 = next((a for a in agents_in_state if a["name"] == speaker_list[1]), {})
            judgment = judge_social_interaction(
                config, agent1_profile=agent1, agent2_profile=agent2,
                conversation=dialogue,
                context={"location": evt.get("details", {}).get("location", "unknown"),
                         "time": conv["state_snapshot"].get("time", "")},
            )
            sotopia_scores.append(judgment)

        valid_sotopia = [s for s in sotopia_scores if "error" not in s]
        dim_avgs = {}
        for dim in SOTOPIA_DIMENSIONS:
            scores = [s.get(dim, {}).get("score", 0) for s in valid_sotopia if isinstance(s.get(dim), dict)]
            dim_avgs[dim] = round(sum(scores) / max(len(scores), 1), 3)
        overall_scores = [s.get("overall_quality", 0) for s in valid_sotopia]
        overall = round(sum(overall_scores) / max(len(overall_scores), 1), 3)

        all_baseline_results[bcfg["name"]] = {
            "total_conversations": len(conversations_for_sotopia),
            "evaluated": len(valid_sotopia),
            "dimension_averages": dim_avgs,
            "overall_quality": overall,
        }
        print(f"  {bcfg['name']} SOTOPIA overall: {overall}")

    save_results(all_baseline_results, "rq4_baselines_sotopia.json", config)
    return all_baseline_results


# =============================================================================
# RQ3: Ablation Study (memory, reflection, probe components)
# =============================================================================

def run_rq3_ablation(config: EvalConfig, client: AURAClient):
    """
    Ablation study: systematically disable components to measure their
    contribution.  Each configuration is tested on both GA (simulation steps +
    grounding judge) and FA (chat queries).
    """
    print("\n" + "=" * 60)
    print("RQ3: Ablation Study")
    print("=" * 60)

    queries_path = Path(__file__).parent / "data" / "chat_queries.json"
    with open(queries_path) as f:
        query_data = json.load(f)
    queries = query_data["queries"][:config.num_chat_queries]

    ablation_configs = [
        {"name": "Full (B=2)",       "probe": True, "max_steps": 2, "memory": True,  "reflection": True},
        {"name": "-Probing",         "probe": False, "max_steps": 0, "memory": True,  "reflection": True},
        {"name": "-Memory",          "probe": True, "max_steps": 2, "memory": False, "reflection": True},
        {"name": "-Reflection",      "probe": True, "max_steps": 2, "memory": True,  "reflection": False},
        {"name": "-Memory&Reflect",  "probe": True, "max_steps": 2, "memory": False, "reflection": False},
        {"name": "Vanilla (All off)","probe": False, "max_steps": 0, "memory": False, "reflection": False},
        # Fine-grained reflection ablation: test whether reflection frequency
        # matters. This addresses the counterintuitive finding that removing
        # reflection improves GA — hypothesis: over-frequent reflection causes
        # self-doubt loops that override correct initial decisions.
        {"name": "-Reflect(Light)",  "probe": True, "max_steps": 2, "memory": True,  "reflection": True,
         "reflection_frequency": "light"},   # reflect every 5 steps instead of every step
        {"name": "Probe(B=1)+NoRef", "probe": True, "max_steps": 1, "memory": True,  "reflection": False},
    ]

    all_results = {}

    for abl in ablation_configs:
        print(f"\n--- Ablation: {abl['name']} ---")

        # --- GA measurement (20 simulation steps) ---
        client.reset()
        client.set_probe(abl["probe"], abl["max_steps"])
        client.set_ablation(abl["memory"], abl["reflection"])

        ga_cond = {"name": abl["name"]}
        ga_summary, _ = _run_rq1_condition(config, client, ga_cond, num_steps=100)

        # --- FA measurement (chat queries) ---
        # Warm up 10 steps first
        client.reset()
        client.set_probe(abl["probe"], abl["max_steps"])
        client.set_ablation(abl["memory"], abl["reflection"])
        for _ in range(10):
            client.step()
            time.sleep(0.5)

        agents = client.state()["state"]["agents"]
        agent_names = [a["name"] for a in agents]

        fa_scores = []
        cu_scores = []
        latencies = []

        for qi, q in enumerate(queries):
            agent_name = agent_names[qi % len(agent_names)]
            gt_state = client.state()["state"]

            t0 = time.time()
            chat_result = client.chat(agent_name, q["query"])
            lat = time.time() - t0
            latencies.append(lat)

            if not chat_result.get("ok"):
                continue

            response = chat_result.get("chat", {}).get("ai_response", "")
            env_context = chat_result.get("chat", {}).get("env_context", {})

            fa = judge_factual_accuracy(config, q["query"], gt_state, response, agent_name)
            cu = judge_context_utilization(config, env_context, response)

            if "accuracy" in fa:
                fa_scores.append(fa["accuracy"])
            if "utilization" in cu:
                cu_scores.append(cu["utilization"])

        summary = {
            "ablation": abl["name"],
            "probe": abl["probe"],
            "memory": abl["memory"],
            "reflection": abl["reflection"],
            "avg_ga": ga_summary.get("overall_grounding_accuracy", 0),
            "rule_loc_acc": ga_summary.get("rule_based_location_accuracy", 0),
            "rule_time_acc": ga_summary.get("rule_based_time_accuracy", 0),
            "num_queries": len(queries),
            "avg_fa": round(sum(fa_scores) / max(len(fa_scores), 1), 4),
            "avg_cu": round(sum(cu_scores) / max(len(cu_scores), 1), 4),
            "avg_latency": round(sum(latencies) / max(len(latencies), 1), 3),
        }
        all_results[abl["name"]] = summary
        print(f"  GA={summary['avg_ga']}, FA={summary['avg_fa']}, Latency={summary['avg_latency']}s")

    save_results(all_results, "rq3_ablation.json", config)
    return all_results


# =============================================================================
# RQ5: Human Evaluation Framework
# =============================================================================

def run_rq5_human(config: EvalConfig, client: AURAClient):
    """
    Generate paired evaluation materials (AURA vs Vanilla baseline) for human
    annotators.  Outputs JSON and HTML form to the results directory.
    """
    from evaluation.human_eval import HumanEvalGenerator

    print("\n" + "=" * 60)
    print("RQ5: Human Evaluation — Generating Materials")
    print("=" * 60)

    queries_path = Path(__file__).parent / "data" / "chat_queries.json"
    with open(queries_path) as f:
        query_data = json.load(f)
    queries = query_data["queries"][:config.num_chat_queries]

    # Warm up
    print("  Warming up simulation (10 steps)...")
    client.reset()
    client.set_probe(True, 2)
    client.set_ablation(True, True)
    for _ in range(10):
        client.step()
        time.sleep(0.5)

    agents = client.state()["state"]["agents"]
    agent_names = [a["name"] for a in agents]
    gt_state = client.state()["state"]

    # Collect AURA responses
    print("  Collecting AURA responses...")
    aura_results = []
    for qi, q in enumerate(queries):
        agent_name = agent_names[qi % len(agent_names)]
        chat_result = client.chat(agent_name, q["query"])
        if chat_result.get("ok"):
            aura_results.append({
                "query": q["query"],
                "agent": agent_name,
                "category": q.get("category", ""),
                "response": chat_result.get("chat", {}).get("ai_response", ""),
            })
        else:
            aura_results.append({"query": q["query"], "agent": agent_name, "category": q.get("category", ""), "response": ""})
        if qi % 10 == 0:
            print(f"    AURA query {qi}/{len(queries)}")

    # Collect Vanilla LLM responses
    print("  Collecting Vanilla LLM responses...")
    baseline_results = []
    for qi, q in enumerate(queries):
        agent_name = agent_names[qi % len(agent_names)]
        try:
            bl = vanilla_llm_chat(config, agent_name, q["query"])
            baseline_results.append({
                "query": q["query"],
                "agent": agent_name,
                "category": q.get("category", ""),
                "response": bl.get("response", ""),
            })
        except Exception:
            baseline_results.append({"query": q["query"], "agent": agent_name, "category": q.get("category", ""), "response": ""})
        if qi % 10 == 0:
            print(f"    Baseline query {qi}/{len(queries)}")
        time.sleep(0.5)

    # Generate evaluation materials
    gen = HumanEvalGenerator(aura_results, baseline_results, config.results_dir)
    scenarios = gen.generate(seed=config.seed)
    json_path = gen.save_json(scenarios)
    html_path = gen.save_html(scenarios)

    print(f"  -> JSON: {json_path}")
    print(f"  -> HTML: {html_path}")
    print(f"  Generated {len(scenarios)} paired evaluation scenarios")

    results = {
        "num_scenarios": len(scenarios),
        "json_path": str(json_path),
        "html_path": str(html_path),
        "aura_responses": len(aura_results),
        "baseline_responses": len(baseline_results),
    }
    save_results(results, "rq5_human_eval_meta.json", config)
    return results


# =============================================================================
# RQ6: Probe Budget Analysis
# =============================================================================

def _compute_pareto_frontier(points: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """
    Compute Pareto frontier for (minimise latency, maximise GA).
    Each point: {"budget": int, "avg_latency": float, "avg_ga": float}
    """
    # Sort by latency ascending
    pts = sorted(points, key=lambda p: p["avg_latency"])
    frontier = []
    best_ga = -1.0
    for p in pts:
        if p["avg_ga"] > best_ga:
            frontier.append(p)
            best_ga = p["avg_ga"]
    return frontier


def run_rq6(config: EvalConfig, client: AURAClient):
    """
    Vary probe budget from 0 to 5, measure GA (grounding accuracy via judge)
    and latency.  Compute Pareto frontier.

    NOTE: Uses the same evaluation pipeline as RQ1 (_run_rq1_condition) for
    consistency. Each budget level runs num_rq6_steps steps (default 50) to
    ensure sufficient data while keeping total experiment time manageable.
    """
    print("\n" + "=" * 60)
    print("RQ6: Probe Budget Pareto Analysis")
    print("=" * 60)

    # Use SAME step count as RQ1 for consistency. Previously capped at 50,
    # which caused RQ1 (100 steps) and RQ6 (50 steps) to report different GA
    # for the same system configuration — a major experimental inconsistency.
    num_rq6_steps = config.num_simulation_steps

    budget_results = {}
    pareto_points = []

    for budget in config.probe_budgets:
        print(f"\n--- Probe Budget = {budget} ---")
        client.reset()
        client.set_probe(budget > 0, budget)
        client.set_ablation(True, True)

        # Reuse the same evaluation pipeline as RQ1 for consistency
        cond = {"name": f"Budget_{budget}"}
        summary, step_results = _run_rq1_condition(
            config, client, cond, num_rq6_steps
        )

        # Also compute latency from step results
        latencies = [r["latency"] for r in step_results if "latency" in r]
        ga_scores = [
            r["judgment"]["overall"]
            for r in step_results
            if "judgment" in r and "overall" in r.get("judgment", {})
        ]

        avg_lat = round(sum(latencies) / max(len(latencies), 1), 3)
        avg_ga = round(sum(ga_scores) / max(len(ga_scores), 1), 4)
        rule_total = len(latencies) * 5  # approximate

        ga_std = round(_std(ga_scores), 4)
        budget_entry = {
            "budget": budget,
            "avg_ga": avg_ga,
            "std_ga": ga_std,
            "avg_latency": avg_lat,
            "min_latency": round(min(latencies) if latencies else 0, 3),
            "max_latency": round(max(latencies) if latencies else 0, 3),
            "num_ga_judgments": len(ga_scores),
            "ga_from_rq1_pipeline": True,  # confirms aligned evaluation
        }
        budget_results[budget] = budget_entry
        pareto_points.append({"budget": budget, "avg_latency": avg_lat, "avg_ga": avg_ga})
        print(f"  GA={avg_ga}, Avg latency={avg_lat}s")

    # Pareto frontier
    frontier = _compute_pareto_frontier(pareto_points)

    output = {
        "per_budget": budget_results,
        "pareto_frontier": frontier,
    }
    save_results(output, "rq6_probe_budget.json", config)
    print(f"\n  Pareto frontier: {frontier}")
    return output


# =============================================================================
# Main
# =============================================================================

def _run_multi_seed(config: EvalConfig, client: AURAClient, rq_fn, rq_name: str, metric_key: str):
    """Run an experiment across multiple seeds and aggregate results."""
    seeds = config.seeds
    per_seed_results = {}
    per_seed_metrics = {}

    for seed in seeds:
        print(f"\n{'~' * 40}")
        print(f"  Seed = {seed}")
        print(f"{'~' * 40}")
        config.current_seed = seed
        random.seed(seed)

        result = rq_fn(config, client)
        per_seed_results[seed] = result

        # Extract the primary metric for aggregation
        if isinstance(result, dict):
            if "summary" in result and isinstance(result["summary"], dict):
                per_seed_metrics[seed] = result["summary"].get(metric_key, 0)
            elif metric_key in result:
                per_seed_metrics[seed] = result[metric_key]
            else:
                # For RQ1-style multi-condition results, aggregate per-condition
                cond_metrics = {}
                for cond_name, cond_data in result.items():
                    if isinstance(cond_data, dict) and "summary" in cond_data:
                        cond_metrics[cond_name] = cond_data["summary"].get(metric_key, 0)
                per_seed_metrics[seed] = cond_metrics

    # Statistical summary
    print(f"\n{'=' * 60}")
    print(f"  Multi-Seed Summary for {rq_name} ({len(seeds)} seeds)")
    print(f"{'=' * 60}")

    if per_seed_results and all(
        isinstance(result, dict) and "per_budget" in result
        for result in per_seed_results.values()
    ):
        stat_summary = _aggregate_budget_statistics(per_seed_results)
        for budget, stats in stat_summary["per_budget"].items():
            print(
                f"  B={budget}: GA={stats['avg_ga_mean']:.4f} +/- {stats['avg_ga_std']:.4f}, "
                f"latency={stats['avg_latency_mean']:.3f}s +/- {stats['avg_latency_std']:.3f}s"
            )
    elif per_seed_metrics and isinstance(list(per_seed_metrics.values())[0], dict):
        # Per-condition aggregation (RQ1-style)
        all_conditions = set()
        for v in per_seed_metrics.values():
            all_conditions.update(v.keys())

        stat_summary = {}
        for cond in sorted(all_conditions):
            vals = [per_seed_metrics[s].get(cond, 0) for s in seeds if isinstance(per_seed_metrics.get(s), dict)]
            stat_summary[cond] = {
                "mean": round(_mean(vals), 4),
                "std": round(_std(vals), 4),
                "values": vals,
            }
            print(f"  {cond}: {stat_summary[cond]['mean']:.4f} +/- {stat_summary[cond]['std']:.4f}")
    else:
        vals = [per_seed_metrics.get(s, 0) for s in seeds]
        stat_summary = {
            "mean": round(_mean(vals), 4),
            "std": round(_std(vals), 4),
            "values": vals,
        }
        print(f"  {metric_key}: {stat_summary['mean']:.4f} +/- {stat_summary['std']:.4f}")

    # Save aggregated results
    agg_output = {
        "seeds": seeds,
        "per_seed": per_seed_results,
        "statistics": stat_summary,
    }
    safe_name = rq_name.lower().replace(' ', '_').replace(':', '').replace('__', '_')
    save_results(agg_output, f"{safe_name}_multiseed.json", config)
    return agg_output


def main():
    # Load .env so OPENAI_API_KEY is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="AURA Paper Experiment Runner")
    parser.add_argument("--rq", "--rqs", nargs="+", dest="rq", default=["all"],
                        help="Which RQs to run: 1 2 3 4 5 6 or 'all'")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of simulation steps for RQ1")
    parser.add_argument("--queries", type=int, default=50,
                        help="Number of chat queries for RQ2")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:7861",
                        help="AURA server URL")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="Seeds for multi-run experiments (e.g., --seeds 42 123 456)")
    parser.add_argument("--multi-seed", action="store_true",
                        help="Run each RQ with multiple seeds for statistical significance")
    parser.add_argument("--model", type=str, default=None,
                        help="Backbone LLM model (e.g., gpt-4o-mini, gpt-4o, claude-sonnet-4-6)")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Judge model (default: gpt-4o). Should be stronger than backbone.")
    parser.add_argument("--analyze", action="store_true",
                        help="Run post-experiment analysis (stats, errors, temporal)")
    args = parser.parse_args()

    config = EvalConfig(
        num_simulation_steps=args.steps,
        num_chat_queries=args.queries,
        aura_server=args.server,
    )

    if args.model:
        config.model = args.model
    if args.judge_model:
        config.judge_model = args.judge_model
    if args.seeds:
        config.seeds = args.seeds

    client = AURAClient(config.aura_server)

    # Check server health
    if not client.health():
        print(f"ERROR: Cannot connect to AURA server at {config.aura_server}")
        print("  Start with: python -m demo.town.server")
        sys.exit(1)

    print(f"Connected to AURA server at {config.aura_server}")
    print(f"Backbone model: {config.model}")
    print(f"Judge model:    {config.judge_model}")
    print(f"Results will be saved to {config.results_dir}/")
    if args.multi_seed:
        print(f"Multi-seed mode: {config.seeds}")

    # Unified RQ mapping (matches paper numbering)
    rq_map = {
        "1": ("RQ1: Grounding Accuracy", run_rq1, "overall_grounding_accuracy"),
        "2": ("RQ2: Factual Accuracy", run_rq2, "avg_factual_accuracy"),
        "3": ("RQ3: Ablation Study", run_rq3_ablation, "avg_ga"),
        "4": ("RQ4: Emergent Social Behavior", run_rq4, "overall_quality"),
        "4b": ("RQ4b: Social Baselines (SOTOPIA)", run_rq4_baselines, "overall_quality"),
        "5": ("RQ5: Human Evaluation", run_rq5_human, "num_scenarios"),
        "6": ("RQ6: Probe Budget Pareto", run_rq6, "pareto_frontier"),
    }

    rqs = []
    for item in args.rq:
        rqs.extend(part.strip() for part in item.split(",") if part.strip())
    if "all" in rqs:
        rqs = sorted(rq_map.keys())

    for rq in rqs:
        if rq in rq_map:
            label, fn, metric_key = rq_map[rq]
            print(f"\n{'#' * 60}")
            print(f"# Running {label}")
            print(f"{'#' * 60}")

            if args.multi_seed and rq in ("1", "2", "3", "6"):
                _run_multi_seed(config, client, fn, label, metric_key)
            else:
                fn(config, client)
        else:
            print(f"  [WARN] Unknown RQ: {rq} (valid: {', '.join(sorted(rq_map))})")

    # =====================================================================
    # Post-experiment analysis pipeline (NeurIPS requirements)
    # =====================================================================
    if args.analyze or "all" in args.rq:
        print(f"\n{'#' * 60}")
        print("# Post-Experiment Analysis Pipeline")
        print(f"{'#' * 60}")

        # 1. Statistical significance testing
        print("\n[1/4] Statistical Analysis...")
        try:
            from evaluation.statistical_analysis import run_full_analysis
            run_full_analysis(config.results_dir)
        except Exception as e:
            print(f"  [WARN] Statistical analysis failed: {e}")

        # 2. Error analysis & failure categorization
        print("\n[2/4] Error Analysis...")
        try:
            from evaluation.error_analysis import run_full_error_analysis
            run_full_error_analysis(config.results_dir)
        except Exception as e:
            print(f"  [WARN] Error analysis failed: {e}")

        # 3. Temporal analysis
        print("\n[3/4] Temporal Analysis...")
        try:
            from evaluation.temporal_analysis import run_temporal_analysis
            run_temporal_analysis(config.results_dir)
        except Exception as e:
            print(f"  [WARN] Temporal analysis failed: {e}")

        # 4. Human evaluation analysis (if annotations exist)
        print("\n[4/4] Human Evaluation Analysis...")
        try:
            from evaluation.human_eval import analyze_human_eval
            analyze_human_eval(config.results_dir)
        except Exception as e:
            print(f"  [WARN] Human eval analysis failed: {e}")

    print(f"\nAll experiments completed. Results in {config.results_dir}/")


if __name__ == "__main__":
    main()
