#!/usr/bin/env python3
"""
Appendix Experiment: BMAM Backend Comparison

Compares four memory backends on controlled memory tasks:
  1. EphemeralMemory (default)  — in-memory TF-IDF
  2. PersistentMemory (llm)     — SQLite + LLM importance
  3. BMAMMemory (bmam)          — 5-brain-region distributed retrieval
  4. ModelMemory (model)        — SQLite + neural plasticity

Tests:
  A. Retrieval Precision@k — store N items, query, measure relevance
  B. Memory Consolidation  — test if BMAM consolidation improves recall
  C. Plasticity Adaptation — test if ModelMemory adapts over repeated queries
  D. Feedback Loop         — test if AURA→BMAM feedback improves results
  E. Forgetting & Capacity — test graceful degradation under capacity pressure

Usage:
    python -m evaluation.appendix_bmam_eval
"""

from __future__ import annotations

import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "AURA" / "src"))

from aura.core import AURAAgent, AURAConfig
from aura.memory import EphemeralMemory
from aura.types import SceneState, MemoryItem

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load env
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


# =============================================================================
# Test Scenarios — controlled memory tasks
# =============================================================================

MEMORY_SCENARIOS = [
    {
        "id": "daily_routine",
        "description": "Agent daily activities — test temporal memory",
        "memories": [
            ("Alice woke up at 7am and had breakfast", ["Alice", "breakfast"]),
            ("Alice walked to the library at 9am", ["Alice", "library"]),
            ("Alice met Bob at the library and discussed the project", ["Alice", "Bob", "library", "project"]),
            ("Alice had lunch at the cafe at noon", ["Alice", "cafe", "lunch"]),
            ("Bob told Alice about the upcoming deadline on Friday", ["Bob", "Alice", "deadline", "Friday"]),
            ("Alice worked on the report in the afternoon", ["Alice", "report"]),
            ("Alice received an urgent email about budget cuts", ["Alice", "email", "budget"]),
            ("Alice attended the team meeting at 3pm", ["Alice", "meeting", "team"]),
            ("During the meeting, the manager announced a new project lead", ["meeting", "manager", "project lead"]),
            ("Alice went home at 6pm feeling tired", ["Alice", "home"]),
        ],
        "queries": [
            ("What did Alice discuss with Bob?", ["project", "deadline"]),
            ("What happened at the library?", ["Alice", "Bob", "library", "project"]),
            ("What was the urgent news?", ["email", "budget", "deadline"]),
            ("What happened during the meeting?", ["manager", "project lead", "team"]),
            ("What did Alice do in the morning?", ["breakfast", "library", "woke"]),
        ],
    },
    {
        "id": "social_network",
        "description": "Multi-agent social interactions — test entity linking",
        "memories": [
            ("Charlie and Diana had a heated argument about politics", ["Charlie", "Diana", "argument", "politics"]),
            ("Eve helped Charlie move to a new apartment", ["Eve", "Charlie", "apartment"]),
            ("Diana apologized to Charlie the next day", ["Diana", "Charlie", "apologized"]),
            ("Frank invited everyone to his birthday party", ["Frank", "birthday", "party"]),
            ("At the party, Charlie and Diana reconciled", ["Charlie", "Diana", "party", "reconciled"]),
            ("Eve announced she was moving to another city", ["Eve", "moving", "city"]),
            ("George joined the friend group through Frank", ["George", "Frank"]),
            ("George and Eve started dating", ["George", "Eve", "dating"]),
            ("The group planned a farewell dinner for Eve", ["group", "farewell", "Eve", "dinner"]),
            ("Charlie gave a touching speech at Eve's farewell", ["Charlie", "Eve", "farewell", "speech"]),
        ],
        "queries": [
            ("What is the relationship between Charlie and Diana?", ["argument", "apologized", "reconciled"]),
            ("What happened at Frank's party?", ["birthday", "reconciled", "Charlie", "Diana"]),
            ("Why is Eve leaving?", ["moving", "city"]),
            ("Who is George connected to?", ["Frank", "Eve", "dating"]),
            ("What happened at the farewell?", ["dinner", "speech", "Charlie", "Eve"]),
        ],
    },
    {
        "id": "task_context",
        "description": "Work context with evolving state — test memory update",
        "memories": [
            ("The server CPU is at 95% utilization", ["server", "CPU", "95%"]),
            ("Database query response time increased to 500ms", ["database", "response time", "500ms"]),
            ("Deployed hotfix v2.1.3 to production", ["hotfix", "v2.1.3", "production"]),
            ("After hotfix, CPU dropped to 60%", ["CPU", "60%", "hotfix"]),
            ("Database response time normalized to 50ms", ["database", "50ms"]),
            ("User reports indicate the login page is slow", ["login", "slow", "user reports"]),
            ("Found memory leak in the authentication service", ["memory leak", "authentication"]),
            ("Scheduled maintenance window for Saturday 2am", ["maintenance", "Saturday"]),
            ("New feature flag enabled: dark mode", ["feature flag", "dark mode"]),
            ("Monitoring alert: disk usage at 85%", ["disk", "85%", "alert"]),
        ],
        "queries": [
            ("What caused the performance issue?", ["CPU", "95%", "memory leak"]),
            ("What was deployed recently?", ["hotfix", "v2.1.3", "production"]),
            ("What is the current system status?", ["CPU", "60%", "disk", "85%"]),
            ("What is planned for maintenance?", ["maintenance", "Saturday"]),
            ("What user issues were reported?", ["login", "slow"]),
        ],
    },
]


# =============================================================================
# Helper: build agent with specific backend
# =============================================================================

def build_agent(backend: str) -> AURAAgent:
    """Build an AURAAgent with the specified backend."""
    config = AURAConfig(
        backend=backend,
        llm_api_key=os.environ.get("OPENAI_API_KEY", ""),
        llm_model="gpt-4o-mini",
        memory_limit=100,
        explore_enabled=False,
        proactive_enabled=False,
        guard_enabled=False,
        workflow_enabled=False,
    )
    return AURAAgent(config=config)


def evaluate_recall(memories_returned: List[MemoryItem], expected_keywords: List[str]) -> Dict[str, float]:
    """Evaluate recall quality: how many expected keywords appear in retrieved memories."""
    if not memories_returned or not expected_keywords:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hit_rate": 0.0}

    combined_text = " ".join(
        m.content.lower() if hasattr(m, "content") else str(m).lower()
        for m in memories_returned
    )

    hits = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
    recall = hits / len(expected_keywords)

    # How many retrieved items contain at least one keyword
    items_with_hits = 0
    for m in memories_returned:
        text = (m.content if hasattr(m, "content") else str(m)).lower()
        if any(kw.lower() in text for kw in expected_keywords):
            items_with_hits += 1
    precision = items_with_hits / len(memories_returned) if memories_returned else 0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "hit_rate": round(hits / len(expected_keywords), 4),
    }


# =============================================================================
# Test A: Retrieval Precision
# =============================================================================

def test_retrieval_precision(backend: str) -> Dict[str, Any]:
    """Store memories, query, measure retrieval quality."""
    logger.info("  [%s] Test A: Retrieval Precision", backend)
    agent = build_agent(backend)

    all_results = []
    for scenario in MEMORY_SCENARIOS:
        # Store all memories
        for content, entities in scenario["memories"]:
            scene = SceneState(summary=content, entities=entities, context={})
            agent.memory.update(scene)

        # Query and evaluate
        query_results = []
        for query, expected_kw in scenario["queries"]:
            t0 = time.time()
            recalled = agent.memory.recall(query=query, limit=5)
            latency = time.time() - t0

            metrics = evaluate_recall(recalled, expected_kw)
            metrics["latency_ms"] = round(latency * 1000, 1)
            metrics["query"] = query
            metrics["num_recalled"] = len(recalled)
            query_results.append(metrics)

        avg_f1 = sum(r["f1"] for r in query_results) / max(len(query_results), 1)
        avg_recall = sum(r["recall"] for r in query_results) / max(len(query_results), 1)
        avg_latency = sum(r["latency_ms"] for r in query_results) / max(len(query_results), 1)

        all_results.append({
            "scenario": scenario["id"],
            "avg_f1": round(avg_f1, 4),
            "avg_recall": round(avg_recall, 4),
            "avg_latency_ms": round(avg_latency, 1),
            "per_query": query_results,
        })

    overall_f1 = sum(r["avg_f1"] for r in all_results) / max(len(all_results), 1)
    overall_recall = sum(r["avg_recall"] for r in all_results) / max(len(all_results), 1)
    overall_latency = sum(r["avg_latency_ms"] for r in all_results) / max(len(all_results), 1)

    return {
        "test": "retrieval_precision",
        "backend": backend,
        "overall_f1": round(overall_f1, 4),
        "overall_recall": round(overall_recall, 4),
        "overall_latency_ms": round(overall_latency, 1),
        "per_scenario": all_results,
    }


# =============================================================================
# Test B: Memory Adaptation (Plasticity / Consolidation)
# =============================================================================

def test_adaptation(backend: str) -> Dict[str, Any]:
    """Test if repeated queries improve retrieval (plasticity effect)."""
    logger.info("  [%s] Test B: Adaptation over repeated queries", backend)
    agent = build_agent(backend)

    scenario = MEMORY_SCENARIOS[0]  # daily_routine
    for content, entities in scenario["memories"]:
        scene = SceneState(summary=content, entities=entities, context={})
        agent.memory.update(scene)

    # Query the same set multiple times and track improvement
    rounds_results = []
    for round_num in range(5):
        round_metrics = []
        for query, expected_kw in scenario["queries"]:
            recalled = agent.memory.recall(query=query, limit=5)
            metrics = evaluate_recall(recalled, expected_kw)
            round_metrics.append(metrics)

        avg_f1 = sum(r["f1"] for r in round_metrics) / max(len(round_metrics), 1)
        rounds_results.append({
            "round": round_num + 1,
            "avg_f1": round(avg_f1, 4),
            "avg_recall": round(
                sum(r["recall"] for r in round_metrics) / max(len(round_metrics), 1), 4
            ),
        })

    # Measure improvement from round 1 to round 5
    f1_delta = rounds_results[-1]["avg_f1"] - rounds_results[0]["avg_f1"]

    return {
        "test": "adaptation",
        "backend": backend,
        "rounds": rounds_results,
        "f1_improvement": round(f1_delta, 4),
        "has_adaptation": f1_delta > 0.01,
    }


# =============================================================================
# Test C: BMAM-specific — Consolidation & Forgetting
# =============================================================================

def test_bmam_operations(backend: str) -> Dict[str, Any]:
    """Test BMAM-specific operations: consolidation, forgetting, feedback."""
    logger.info("  [%s] Test C: BMAM operations", backend)

    if backend != "bmam":
        return {"test": "bmam_operations", "backend": backend, "skipped": True,
                "reason": "Only applicable to bmam backend"}

    agent = build_agent("bmam")
    memory = agent.memory

    results = {}

    # Store memories
    scenario = MEMORY_SCENARIOS[0]
    for content, entities in scenario["memories"]:
        scene = SceneState(summary=content, entities=entities, context={})
        memory.update(scene)

    # Baseline recall
    baseline_items = memory.recall("What did Alice discuss with Bob?", limit=5)
    results["baseline_recall_count"] = len(baseline_items)

    # Test consolidation
    if hasattr(memory, "consolidate"):
        try:
            consolidation_result = memory.consolidate()
            results["consolidation"] = {
                "success": True,
                "result": str(consolidation_result)[:200],
            }
        except Exception as e:
            results["consolidation"] = {"success": False, "error": str(e)}

    # Post-consolidation recall
    post_items = memory.recall("What did Alice discuss with Bob?", limit=5)
    results["post_consolidation_recall_count"] = len(post_items)

    # Test feedback
    if hasattr(memory, "feedback"):
        try:
            fb_result = memory.feedback(
                query="What did Alice discuss with Bob?",
                response="Alice discussed the project with Bob at the library",
                reward=0.9,
            )
            results["feedback"] = {"success": True, "result": str(fb_result)[:200]}
        except Exception as e:
            results["feedback"] = {"success": False, "error": str(e)}

    # Post-feedback recall
    post_fb_items = memory.recall("What did Alice discuss with Bob?", limit=5)
    results["post_feedback_recall_count"] = len(post_fb_items)

    # Test forgetting
    if hasattr(memory, "forget"):
        try:
            forget_result = memory.forget(threshold=0.8)
            results["forgetting"] = {"success": True, "result": str(forget_result)[:200]}
        except Exception as e:
            results["forgetting"] = {"success": False, "error": str(e)}

    # BMAM availability check
    if hasattr(memory, "is_available"):
        results["bmam_available"] = memory.is_available()

    return {"test": "bmam_operations", "backend": backend, **results}


# =============================================================================
# Test D: Capacity Pressure
# =============================================================================

def test_capacity_pressure(backend: str, n_items: int = 200) -> Dict[str, Any]:
    """Test memory quality under capacity pressure (many items)."""
    logger.info("  [%s] Test D: Capacity pressure (%d items)", backend, n_items)
    agent = build_agent(backend)

    # Store N filler memories + 5 important target memories
    target_memories = [
        ("CRITICAL: The server room caught fire at 3am, all data backed up successfully", ["fire", "server", "backup"]),
        ("IMPORTANT: The CEO announced the company is going public next month", ["CEO", "public", "IPO"]),
        ("URGENT: Security breach detected in the payment system", ["security", "breach", "payment"]),
        ("KEY MEETING: Board approved the $50M acquisition of TechCorp", ["board", "acquisition", "TechCorp"]),
        ("BREAKING: Lead engineer resigned, taking 3 team members", ["engineer", "resigned"]),
    ]

    # Store filler
    for i in range(n_items):
        scene = SceneState(
            summary=f"Regular observation #{i}: nothing notable happened at step {i}",
            entities=[f"step_{i}"],
            context={},
        )
        agent.memory.update(scene)

    # Store targets interspersed
    for j, (content, entities) in enumerate(target_memories):
        scene = SceneState(summary=content, entities=entities, context={})
        agent.memory.update(scene)
        # Add more filler after each target
        for i in range(10):
            filler = SceneState(
                summary=f"Filler after target {j}, item {i}: routine check passed",
                entities=[],
                context={},
            )
            agent.memory.update(filler)

    # Query for the important memories
    queries = [
        ("What happened with the server?", ["fire", "backup"]),
        ("What did the CEO announce?", ["public", "IPO"]),
        ("Was there a security incident?", ["breach", "payment"]),
        ("What acquisition was approved?", ["acquisition", "TechCorp"]),
        ("Who left the company?", ["engineer", "resigned"]),
    ]

    query_results = []
    for query, expected_kw in queries:
        recalled = agent.memory.recall(query=query, limit=5)
        metrics = evaluate_recall(recalled, expected_kw)
        query_results.append(metrics)

    avg_recall = sum(r["recall"] for r in query_results) / max(len(query_results), 1)
    avg_f1 = sum(r["f1"] for r in query_results) / max(len(query_results), 1)

    return {
        "test": "capacity_pressure",
        "backend": backend,
        "total_items_stored": n_items + len(target_memories) * 11,
        "avg_recall": round(avg_recall, 4),
        "avg_f1": round(avg_f1, 4),
        "per_query": query_results,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    logger.info("=" * 60)
    logger.info("Appendix: BMAM Backend Comparison Experiment")
    logger.info("=" * 60)

    backends_to_test = []
    for b in ["default", "llm", "bmam", "model"]:
        try:
            agent = build_agent(b)
            backends_to_test.append(b)
            logger.info("Backend '%s' available: memory=%s", b, type(agent.memory).__name__)
        except Exception as e:
            logger.warning("Backend '%s' not available: %s", b, e)

    all_results = {"metadata": {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "backends_tested": backends_to_test,
    }}

    # Test A: Retrieval Precision
    logger.info("\n=== Test A: Retrieval Precision ===")
    retrieval_results = {}
    for backend in backends_to_test:
        try:
            retrieval_results[backend] = test_retrieval_precision(backend)
            logger.info("  %s: F1=%.3f, Recall=%.3f, Latency=%.1fms",
                        backend,
                        retrieval_results[backend]["overall_f1"],
                        retrieval_results[backend]["overall_recall"],
                        retrieval_results[backend]["overall_latency_ms"])
        except Exception as e:
            logger.error("  %s failed: %s", backend, e)
            retrieval_results[backend] = {"error": str(e)}
    all_results["retrieval_precision"] = retrieval_results

    # Test B: Adaptation
    logger.info("\n=== Test B: Adaptation ===")
    adaptation_results = {}
    for backend in backends_to_test:
        try:
            adaptation_results[backend] = test_adaptation(backend)
            r = adaptation_results[backend]
            logger.info("  %s: F1 improvement=%.4f, has_adaptation=%s",
                        backend, r.get("f1_improvement", 0), r.get("has_adaptation", False))
        except Exception as e:
            logger.error("  %s failed: %s", backend, e)
            adaptation_results[backend] = {"error": str(e)}
    all_results["adaptation"] = adaptation_results

    # Test C: BMAM Operations
    logger.info("\n=== Test C: BMAM Operations ===")
    if "bmam" in backends_to_test:
        try:
            all_results["bmam_operations"] = test_bmam_operations("bmam")
            logger.info("  BMAM operations completed")
        except Exception as e:
            logger.error("  BMAM operations failed: %s", e)
            all_results["bmam_operations"] = {"error": str(e)}
    else:
        all_results["bmam_operations"] = {"skipped": True, "reason": "BMAM not available"}

    # Test D: Capacity Pressure
    logger.info("\n=== Test D: Capacity Pressure ===")
    capacity_results = {}
    for backend in backends_to_test:
        try:
            capacity_results[backend] = test_capacity_pressure(backend)
            logger.info("  %s: Recall=%.3f, F1=%.3f",
                        backend,
                        capacity_results[backend]["avg_recall"],
                        capacity_results[backend]["avg_f1"])
        except Exception as e:
            logger.error("  %s failed: %s", backend, e)
            capacity_results[backend] = {"error": str(e)}
    all_results["capacity_pressure"] = capacity_results

    # Summary table
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("%-12s | %-8s | %-8s | %-10s | %-8s | %-10s",
                "Backend", "Ret.F1", "Ret.Rec", "Latency", "Adapt", "Cap.Rec")
    logger.info("-" * 70)
    for b in backends_to_test:
        ret = retrieval_results.get(b, {})
        ada = adaptation_results.get(b, {})
        cap = capacity_results.get(b, {})
        logger.info("%-12s | %-8.3f | %-8.3f | %-10.1f | %-8.4f | %-10.3f",
                     b,
                     ret.get("overall_f1", 0),
                     ret.get("overall_recall", 0),
                     ret.get("overall_latency_ms", 0),
                     ada.get("f1_improvement", 0),
                     cap.get("avg_recall", 0))

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "appendix_bmam_comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    logger.info("\nResults saved to %s", out_path)


if __name__ == "__main__":
    main()
