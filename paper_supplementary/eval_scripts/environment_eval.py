"""
Environment Grounding Evaluation for AURA Town.

Core idea: Run simulation → capture ground truth state → ask questions → verify answers.
No GPT-4 judge needed for factual questions — answers are compared directly against simulation state.
"""

import json
import math
import random
import re
import time
import requests
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from evaluation.config import EvalConfig


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GroundTruth:
    """A single verifiable fact extracted from simulation state."""
    question: str
    category: str
    template_id: str
    expected: Any                       # ground truth value
    agent_queried: str                  # which agent we ask
    tick: int


@dataclass
class EvalResult:
    """Result of evaluating one question."""
    question: str
    category: str
    template_id: str
    expected: Any
    agent_queried: str
    agent_response: str
    correct: bool
    tick: int
    condition: str                      # AURA-Full / AURA-NoProbe / etc.
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Ground truth extractor
# ---------------------------------------------------------------------------

class GroundTruthExtractor:
    """Extracts verifiable ground truth from a simulation state snapshot."""

    AGENTS = ["Lin Wei", "Zhang Hao", "Chen Mei", "Liu Yang", "Wang Jun"]

    AGENT_HOMES = {
        "Lin Wei": "Lin Wei's Home",
        "Zhang Hao": "Zhang Hao's Home",
        "Chen Mei": "Chen Mei's Home",
        "Liu Yang": "Liu Yang's Home",
        "Wang Jun": "Wang Jun's Home",
    }

    def __init__(self, state: Dict[str, Any]):
        self.state = state
        self.time = state.get("time", "")
        self.day = state.get("day", 1)
        self.hour = state.get("hour", 6)
        self.minute = state.get("minute", 0)
        self.agents = {a["name"]: a for a in state.get("agents", [])}
        self.locations = {loc["name"]: loc for loc in state.get("locations", [])}
        self.events = state.get("events", [])

    # --- helpers ---

    def _time_period(self) -> str:
        if self.hour < 12:
            return "morning"
        elif self.hour < 17:
            return "afternoon"
        else:
            return "evening"

    def _agents_at_location(self, loc_name: str) -> List[str]:
        return [a["name"] for a in self.agents.values()
                if a.get("location", "") == loc_name]

    def _is_at_home(self, agent_name: str) -> bool:
        loc = self.agents.get(agent_name, {}).get("location", "")
        home = self.AGENT_HOMES.get(agent_name, "")
        return loc == home

    def _distance(self, x1, y1, x2, y2) -> float:
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def _nearest_agent(self, agent_name: str) -> Optional[str]:
        a = self.agents.get(agent_name)
        if not a:
            return None
        ax, ay = a["x"], a["y"]
        best, best_d = None, float("inf")
        for other_name, other in self.agents.items():
            if other_name == agent_name:
                continue
            d = self._distance(ax, ay, other["x"], other["y"])
            if d < best_d:
                best, best_d = other_name, d
        return best

    def _recent_events(self, minutes: int = 60) -> List[str]:
        """Events in the last N simulation minutes."""
        return [e["description"] for e in self.events[-10:]]

    def _conversations_today(self, agent_name: str) -> List[str]:
        return [
            e["description"] for e in self.events
            if e.get("event_type") == "conversation"
            and agent_name in e.get("description", "")
        ]

    def _recently_moved(self) -> List[str]:
        moved = set()
        for e in self.events[-5:]:
            if e.get("event_type") == "movement":
                agent = e.get("agent", "")
                if agent in self.AGENTS:
                    moved.add(agent)
        return list(moved)

    # --- question generators ---

    def generate_questions(self, queried_agent: str, tick: int,
                           num: int = 10) -> List[GroundTruth]:
        """Generate N verifiable questions from current state."""
        pool: List[GroundTruth] = []

        # --- Location Awareness ---
        # LOC-01: Where is [agent]?
        for agent_name in self.AGENTS:
            loc = self.agents.get(agent_name, {}).get("location", "Unknown")
            pool.append(GroundTruth(
                question=f"Where is {agent_name} right now?",
                category="location_awareness",
                template_id="LOC-01",
                expected=loc,
                agent_queried=queried_agent,
                tick=tick,
            ))

        # LOC-03: Who is at [location]?
        for loc_name in random.sample(list(self.locations.keys()),
                                       min(5, len(self.locations))):
            agents_here = self._agents_at_location(loc_name)
            pool.append(GroundTruth(
                question=f"Who is at {loc_name} right now?",
                category="location_awareness",
                template_id="LOC-03",
                expected=agents_here if agents_here else ["nobody"],
                agent_queried=queried_agent,
                tick=tick,
            ))

        # LOC-05: How many people at [location]?
        for loc_name in random.sample(list(self.locations.keys()),
                                       min(3, len(self.locations))):
            count = len(self._agents_at_location(loc_name))
            pool.append(GroundTruth(
                question=f"How many people are at {loc_name}?",
                category="location_awareness",
                template_id="LOC-05",
                expected=count,
                agent_queried=queried_agent,
                tick=tick,
            ))

        # --- Temporal Awareness ---
        # TIME-01: What time is it?
        pool.append(GroundTruth(
            question="What time is it right now?",
            category="temporal_awareness",
            template_id="TIME-01",
            expected=self.time,
            agent_queried=queried_agent,
            tick=tick,
        ))

        # TIME-02: morning/afternoon/evening?
        pool.append(GroundTruth(
            question="Is it morning, afternoon, or evening?",
            category="temporal_awareness",
            template_id="TIME-02",
            expected=self._time_period(),
            agent_queried=queried_agent,
            tick=tick,
        ))

        # TIME-03: What day?
        pool.append(GroundTruth(
            question="What day of the simulation is it?",
            category="temporal_awareness",
            template_id="TIME-03",
            expected=self.day,
            agent_queried=queried_agent,
            tick=tick,
        ))

        # --- Action Grounding ---
        # ACT-01: What is [agent] doing?
        for agent_name in self.AGENTS:
            action = self.agents.get(agent_name, {}).get("action", "Unknown")
            pool.append(GroundTruth(
                question=f"What is {agent_name} doing right now?",
                category="action_grounding",
                template_id="ACT-01",
                expected=action,
                agent_queried=queried_agent,
                tick=tick,
            ))

        # ACT-02: Is [agent] at home?
        for agent_name in self.AGENTS:
            at_home = self._is_at_home(agent_name)
            pool.append(GroundTruth(
                question=f"Is {agent_name} at home or out right now?",
                category="action_grounding",
                template_id="ACT-02",
                expected="at home" if at_home else "out",
                agent_queried=queried_agent,
                tick=tick,
            ))

        # --- Social Awareness ---
        # SOC-01: Who is closest to [agent]?
        for agent_name in self.AGENTS:
            nearest = self._nearest_agent(agent_name)
            if nearest:
                pool.append(GroundTruth(
                    question=f"Who is physically closest to {agent_name} right now?",
                    category="social_awareness",
                    template_id="SOC-01",
                    expected=nearest,
                    agent_queried=queried_agent,
                    tick=tick,
                ))

        # SOC-02: Has [agent] talked to anyone today?
        for agent_name in self.AGENTS:
            convos = self._conversations_today(agent_name)
            pool.append(GroundTruth(
                question=f"Has {agent_name} talked to anyone today?",
                category="social_awareness",
                template_id="SOC-02",
                expected="yes" if convos else "no",
                agent_queried=queried_agent,
                tick=tick,
            ))

        # --- Change Detection ---
        # CHG-02: Who moved recently?
        moved = self._recently_moved()
        pool.append(GroundTruth(
            question="Which agents have moved to a new location recently?",
            category="change_detection",
            template_id="CHG-02",
            expected=moved if moved else ["none"],
            agent_queried=queried_agent,
            tick=tick,
        ))

        # --- Event Tracking ---
        # EVT-01: What happened recently?
        recent = self._recent_events(60)
        if recent:
            pool.append(GroundTruth(
                question="What has happened in town in the last hour?",
                category="event_tracking",
                template_id="EVT-01",
                expected=recent,
                agent_queried=queried_agent,
                tick=tick,
            ))

        # Shuffle and sample
        random.shuffle(pool)
        return pool[:num]


# ---------------------------------------------------------------------------
# Answer verifier
# ---------------------------------------------------------------------------

class AnswerVerifier:
    """Compare agent's natural language response against ground truth."""

    @staticmethod
    def verify(response: str, expected: Any, eval_type: str = "auto") -> bool:
        resp_lower = response.lower().strip()

        if isinstance(expected, str):
            return AnswerVerifier._check_string(resp_lower, expected.lower())

        elif isinstance(expected, bool):
            return AnswerVerifier._check_boolean(resp_lower, expected)

        elif isinstance(expected, int):
            return AnswerVerifier._check_numeric(resp_lower, expected)

        elif isinstance(expected, list):
            return AnswerVerifier._check_list(resp_lower, expected)

        return False

    @staticmethod
    def _check_string(response: str, expected: str) -> bool:
        # Check if expected value appears in response
        if expected in response:
            return True
        # Fuzzy: check key words
        words = expected.split()
        if len(words) <= 3:
            return all(w.lower() in response for w in words)
        # At least 60% of words match
        matches = sum(1 for w in words if w.lower() in response)
        return matches / len(words) >= 0.6

    @staticmethod
    def _check_boolean(response: str, expected: bool) -> bool:
        positive = {"yes", "true", "correct", "indeed", "at home", "has"}
        negative = {"no", "false", "not", "hasn't", "nobody", "none", "out"}
        if expected:
            return any(w in response for w in positive)
        else:
            return any(w in response for w in negative)

    @staticmethod
    def _check_numeric(response: str, expected: int) -> bool:
        # Extract numbers from response
        numbers = re.findall(r'\b(\d+)\b', response)
        # Also check word numbers
        word_nums = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "no one": 0, "nobody": 0, "none": 0,
        }
        for word, num in word_nums.items():
            if word in response and num == expected:
                return True
        return str(expected) in numbers

    @staticmethod
    def _check_list(response: str, expected: List[str]) -> bool:
        if expected == ["nobody"] or expected == ["none"]:
            neg = {"nobody", "no one", "none", "empty", "no agents", "没有人"}
            return any(w in response.lower() for w in neg)
        # Check if at least half of expected items are mentioned
        if not expected:
            return True
        matches = sum(1 for item in expected
                      if item.lower() in response.lower())
        return matches / len(expected) >= 0.5


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

class EnvironmentEvaluator:
    """
    Runs environment grounding evaluation against a live AURA Town server.

    Protocol:
      1. Reset simulation
      2. For each sampled tick:
         a. Step simulation to that tick
         b. Capture state (ground truth)
         c. Generate questions from state
         d. Ask agent via /api/chat
         e. Compare response to ground truth
      3. Aggregate results
    """

    def __init__(self, config: EvalConfig):
        self.config = config
        self.server = config.aura_server
        self.results: List[EvalResult] = []

    def _api(self, method: str, endpoint: str,
             json_data: Optional[dict] = None) -> dict:
        url = f"{self.server}{endpoint}"
        try:
            if method == "GET":
                r = requests.get(url, timeout=30)
            else:
                r = requests.post(url, json=json_data or {}, timeout=60)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def get_state(self) -> dict:
        return self._api("GET", "/api/state")

    def step(self) -> dict:
        return self._api("POST", "/api/step")

    def reset(self) -> dict:
        return self._api("POST", "/api/reset")

    def set_probe(self, enabled: bool, max_steps: int) -> dict:
        return self._api("POST", "/api/probe",
                         {"enabled": enabled, "max_steps": max_steps})

    def chat(self, agent: str, message: str) -> Tuple[str, float]:
        """Send chat, return (response_text, latency_ms)."""
        t0 = time.time()
        resp = self._api("POST", "/api/chat",
                         {"agent": agent, "message": message})
        latency = (time.time() - t0) * 1000
        reply = resp.get("reply", resp.get("response", str(resp)))
        return reply, latency

    # ------------------------------------------------------------------

    def run(self,
            conditions: Optional[List[dict]] = None,
            ticks_to_sample: int = 20,
            queries_per_tick: int = 10,
            seed: int = 42) -> Dict[str, Any]:
        """
        Main evaluation entry point.

        Args:
            conditions: List of {"name", "probe_enabled", "probe_max_steps"}
            ticks_to_sample: How many ticks to sample questions from
            queries_per_tick: Questions per tick
            seed: Random seed
        """
        random.seed(seed)

        if conditions is None:
            conditions = [
                {"name": "AURA-Full", "probe_enabled": True, "probe_max_steps": 2},
                {"name": "AURA-NoProbe", "probe_enabled": False, "probe_max_steps": 0},
            ]

        # Decide which ticks to sample (spread across a day: ticks 1-34)
        max_ticks = 34  # one full day 6am-11pm at 30min intervals
        sample_ticks = sorted(random.sample(
            range(1, max_ticks + 1),
            min(ticks_to_sample, max_ticks),
        ))

        all_results = []

        for cond in conditions:
            cond_name = cond["name"]
            print(f"\n{'='*60}")
            print(f"Condition: {cond_name}")
            print(f"  probe_enabled={cond['probe_enabled']}, "
                  f"max_steps={cond['probe_max_steps']}")
            print(f"{'='*60}")

            # Reset and configure
            self.reset()
            time.sleep(1)
            self.set_probe(cond["probe_enabled"], cond["probe_max_steps"])
            time.sleep(0.5)

            current_tick = 0
            for target_tick in sample_ticks:
                # Advance to target tick
                steps_needed = target_tick - current_tick
                for _ in range(steps_needed):
                    self.step()
                    time.sleep(0.3)  # don't overwhelm the server
                current_tick = target_tick

                # Capture state
                state = self.get_state()
                if "error" in state:
                    print(f"  [tick {target_tick}] Error getting state: {state['error']}")
                    continue

                extractor = GroundTruthExtractor(state)

                # Pick a random agent to query
                queried_agent = random.choice(GroundTruthExtractor.AGENTS)

                # Generate questions
                questions = extractor.generate_questions(
                    queried_agent, target_tick, num=queries_per_tick)

                print(f"  [tick {target_tick}] time={state.get('time', '?')}, "
                      f"asking {queried_agent}, {len(questions)} questions")

                for q in questions:
                    reply, latency = self.chat(queried_agent, q.question)
                    correct = AnswerVerifier.verify(reply, q.expected)

                    result = EvalResult(
                        question=q.question,
                        category=q.category,
                        template_id=q.template_id,
                        expected=q.expected,
                        agent_queried=q.agent_queried,
                        agent_response=reply[:500],
                        correct=correct,
                        tick=q.tick,
                        condition=cond_name,
                        latency_ms=latency,
                    )
                    all_results.append(result)

                    status = "✓" if correct else "✗"
                    print(f"    {status} [{q.template_id}] {q.question[:50]}...")

        self.results = all_results
        return self._aggregate(all_results)

    # ------------------------------------------------------------------

    def _aggregate(self, results: List[EvalResult]) -> Dict[str, Any]:
        """Aggregate results into summary statistics."""
        summary = {
            "total": len(results),
            "conditions": {},
            "categories": {},
            "per_condition_per_category": {},
        }

        # Per condition
        by_cond: Dict[str, List[EvalResult]] = {}
        for r in results:
            by_cond.setdefault(r.condition, []).append(r)

        for cond, rs in by_cond.items():
            correct = sum(1 for r in rs if r.correct)
            total = len(rs)
            avg_latency = sum(r.latency_ms for r in rs) / total if total else 0
            summary["conditions"][cond] = {
                "accuracy": correct / total if total else 0,
                "correct": correct,
                "total": total,
                "avg_latency_ms": round(avg_latency, 1),
            }

            # Per category within condition
            by_cat: Dict[str, List[EvalResult]] = {}
            for r in rs:
                by_cat.setdefault(r.category, []).append(r)

            cond_cats = {}
            for cat, cat_rs in by_cat.items():
                c = sum(1 for r in cat_rs if r.correct)
                t = len(cat_rs)
                cond_cats[cat] = {
                    "accuracy": c / t if t else 0,
                    "correct": c,
                    "total": t,
                }
            summary["per_condition_per_category"][cond] = cond_cats

        # Per category (all conditions)
        by_cat_all: Dict[str, List[EvalResult]] = {}
        for r in results:
            by_cat_all.setdefault(r.category, []).append(r)

        for cat, rs in by_cat_all.items():
            correct = sum(1 for r in rs if r.correct)
            total = len(rs)
            summary["categories"][cat] = {
                "accuracy": correct / total if total else 0,
                "correct": correct,
                "total": total,
            }

        return summary

    # ------------------------------------------------------------------

    def save_results(self, path: Optional[str] = None):
        """Save all results to JSON."""
        if path is None:
            path = Path(self.config.results_dir) / "environment_grounding.json"
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        output = {
            "summary": self._aggregate(self.results),
            "results": [asdict(r) for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to {path}")

    def print_summary(self, summary: Dict[str, Any]):
        """Pretty-print evaluation summary."""
        print(f"\n{'='*60}")
        print("ENVIRONMENT GROUNDING EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Total evaluations: {summary['total']}")

        print("\n--- Per Condition ---")
        for cond, stats in summary["conditions"].items():
            acc = stats["accuracy"] * 100
            print(f"  {cond:20s}: {acc:5.1f}% "
                  f"({stats['correct']}/{stats['total']}) "
                  f"latency={stats['avg_latency_ms']:.0f}ms")

        print("\n--- Per Category (all conditions) ---")
        for cat, stats in summary["categories"].items():
            acc = stats["accuracy"] * 100
            print(f"  {cat:25s}: {acc:5.1f}% "
                  f"({stats['correct']}/{stats['total']})")

        print("\n--- Per Condition × Category ---")
        for cond, cats in summary["per_condition_per_category"].items():
            print(f"\n  [{cond}]")
            for cat, stats in sorted(cats.items()):
                acc = stats["accuracy"] * 100
                print(f"    {cat:25s}: {acc:5.1f}% "
                      f"({stats['correct']}/{stats['total']})")

    def generate_latex_table(self, summary: Dict[str, Any]) -> str:
        """Generate LaTeX table for the paper."""
        conditions = list(summary["conditions"].keys())
        categories = sorted(set(
            cat for cats in summary["per_condition_per_category"].values()
            for cat in cats
        ))

        cat_labels = {
            "location_awareness": "Location",
            "temporal_awareness": "Temporal",
            "action_grounding": "Action",
            "social_awareness": "Social",
            "change_detection": "Change Det.",
            "event_tracking": "Event Track.",
        }

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Environment Grounding Accuracy (\%). "
            r"Higher is better.}",
            r"\label{tab:grounding}",
            r"\begin{tabular}{l" + "c" * len(conditions) + "}",
            r"\toprule",
            "Category & " + " & ".join(conditions) + r" \\",
            r"\midrule",
        ]

        for cat in categories:
            label = cat_labels.get(cat, cat)
            vals = []
            for cond in conditions:
                stats = summary["per_condition_per_category"].get(
                    cond, {}).get(cat, {})
                acc = stats.get("accuracy", 0) * 100
                vals.append(f"{acc:.1f}")
            lines.append(f"{label} & " + " & ".join(vals) + r" \\")

        lines.append(r"\midrule")
        # Overall
        vals = []
        for cond in conditions:
            acc = summary["conditions"][cond]["accuracy"] * 100
            vals.append(f"\\textbf{{{acc:.1f}}}")
        lines.append(r"\textbf{Overall} & " + " & ".join(vals) + r" \\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run AURA Environment Grounding Evaluation")
    parser.add_argument("--server", default="http://127.0.0.1:7861",
                        help="AURA Town server URL")
    parser.add_argument("--ticks", type=int, default=20,
                        help="Number of ticks to sample")
    parser.add_argument("--queries", type=int, default=10,
                        help="Queries per tick")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    parser.add_argument("--conditions", default="full,noprobe",
                        help="Comma-separated: full,noprobe,minprobe,maxprobe")

    args = parser.parse_args()

    cond_map = {
        "full": {"name": "AURA-Full",
                 "probe_enabled": True, "probe_max_steps": 2},
        "noprobe": {"name": "AURA-NoProbe",
                    "probe_enabled": False, "probe_max_steps": 0},
        "minprobe": {"name": "AURA-MinProbe",
                     "probe_enabled": True, "probe_max_steps": 1},
        "maxprobe": {"name": "AURA-MaxProbe",
                     "probe_enabled": True, "probe_max_steps": 5},
    }
    conditions = [cond_map[c.strip()]
                  for c in args.conditions.split(",")
                  if c.strip() in cond_map]

    config = EvalConfig(aura_server=args.server)
    evaluator = EnvironmentEvaluator(config)

    print("Starting Environment Grounding Evaluation...")
    print(f"  Server:     {args.server}")
    print(f"  Ticks:      {args.ticks}")
    print(f"  Queries:    {args.queries}/tick")
    print(f"  Conditions: {[c['name'] for c in conditions]}")
    print(f"  Seed:       {args.seed}")

    summary = evaluator.run(
        conditions=conditions,
        ticks_to_sample=args.ticks,
        queries_per_tick=args.queries,
        seed=args.seed,
    )

    evaluator.print_summary(summary)
    evaluator.save_results(args.output)

    latex = evaluator.generate_latex_table(summary)
    print(f"\n--- LaTeX Table ---\n{latex}")
