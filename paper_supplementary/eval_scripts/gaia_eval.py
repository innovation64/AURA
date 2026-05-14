"""
GAIA Benchmark Evaluation for AURA's Probe Mechanism.

Demonstrates that AURA's multi-step tool-calling (probe) architecture
generalizes beyond environment probing to general AI assistant tasks.

GAIA tasks require: web search, calculation, code execution, reasoning.
We adapt AURA's probe loop with general-purpose tools.
"""

import json
import os
import re
import time
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Load .env so OPENAI_API_KEY is available when run as __main__
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from evaluation.config import EvalConfig


# ---------------------------------------------------------------------------
# Tool definitions for GAIA
# ---------------------------------------------------------------------------

GAIA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information. Returns top search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, log, sin, cos, pi, e.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '(225623 / 13.1) / 1000'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": "Execute a Python code snippet and return the output. Useful for data processing, string manipulation, conversions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Use print() for output.",
                    }
                },
                "required": ["code"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tool_web_search(query: str) -> str:
    """Real web search via OpenAI Responses API's built-in web_search tool.

    Returns the model's synthesized answer grounded in live search results.
    Falls back to an LLM-simulated placeholder if the Responses API path
    is unavailable (older SDK, network error, etc.) so callers never
    get None.
    """
    client = _get_client()
    if client is None:
        return "Search unavailable: no API client."

    # Preferred path: Responses API with the built-in web_search tool.
    err_primary: Optional[Exception] = None
    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=(
                f"Search the web for: {query}\n\n"
                "Provide a concise factual answer (2-4 sentences) with key numbers, "
                "names, and dates. Cite the most relevant source inline."
            ),
            tools=[{"type": "web_search"}],
        )
        text = getattr(resp, "output_text", None)
        if text:
            return text.strip()[:2000]
        err_primary = RuntimeError("Responses API returned empty output_text")
    except Exception as e_primary:
        # Fall through to legacy stub so the probe loop doesn't break.
        err_primary = e_primary

    # Fallback: legacy LLM-as-search-engine stub (no real retrieval).
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a web search engine. Given a query, provide factual search results with key information. Be concise and factual."},
                {"role": "user", "content": f"Search query: {query}\n\nProvide the top 3-5 search results with key factual information."},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e_fallback:
        return f"Search error: primary={err_primary!r} fallback={e_fallback!r}"


def tool_calculator(expression: str) -> str:
    """Safe math expression evaluator."""
    allowed_names = {
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e, "abs": abs, "round": round,
        "int": int, "float": float, "pow": pow,
    }
    try:
        # Only allow safe math operations
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


def tool_python_exec(code: str) -> str:
    """Execute Python code in a restricted environment."""
    import io
    import contextlib

    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {"__builtins__": __builtins__,
                        "math": math, "re": re, "json": json})
        result = output.getvalue()
        return result if result else "(no output)"
    except Exception as e:
        return f"Execution error: {e}"


TOOL_DISPATCH = {
    "web_search": lambda args: tool_web_search(args.get("query", "")),
    "calculator": lambda args: tool_calculator(args.get("expression", "")),
    "python_exec": lambda args: tool_python_exec(args.get("code", "")),
}


# ---------------------------------------------------------------------------
# Client helper
# ---------------------------------------------------------------------------

_client_instance = None


def _get_client() -> Optional[Any]:
    global _client_instance
    if _client_instance is None and OpenAI is not None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if api_key:
            _client_instance = OpenAI(api_key=api_key, base_url=base_url)
    return _client_instance


# ---------------------------------------------------------------------------
# GAIA solver using probe-style loop
# ---------------------------------------------------------------------------

@dataclass
class GaiaResult:
    task_id: str
    question: str
    level: int
    expected_answer: str
    predicted_answer: str
    correct: bool
    num_tool_calls: int
    tools_used: List[str]
    reasoning_trace: str
    latency_s: float
    condition: str              # "probe" or "direct"


class GaiaSolver:
    """
    Solves GAIA tasks using AURA's probe-style multi-step tool-calling loop.

    Two conditions:
    - "probe": Uses iterative tool-calling loop (like AURA probe)
    - "direct": Single LLM call without tools (baseline)
    """

    def __init__(self, config: EvalConfig, max_steps: int = 5, seed: Optional[int] = None):
        self.config = config
        self.max_steps = max_steps
        self.seed = seed
        self.client = _get_client()

    def solve_with_probe(self, question: str) -> Tuple[str, int, List[str], str]:
        """
        Solve using probe-style iterative tool-calling.
        Returns: (answer, num_tool_calls, tools_used, reasoning_trace)
        """
        if not self.client:
            return "No API client", 0, [], "Error: No OpenAI client"

        messages = [
            {"role": "system", "content": (
                "You are a precise AI assistant solving a benchmark task. "
                "Use the provided tools to search for information, calculate, "
                "or run code as needed. After gathering enough information, "
                "provide your FINAL ANSWER as concisely as possible — "
                "just the answer value, no explanation."
            )},
            {"role": "user", "content": question},
        ]

        tool_calls_count = 0
        tools_used = []
        trace_lines = []

        for step in range(self.max_steps):
            try:
                kwargs = dict(
                    model=self.config.model,
                    messages=messages,
                    tools=GAIA_TOOLS,
                    tool_choice="auto",
                    temperature=0.1,
                    max_tokens=1024,
                )
                if self.seed is not None:
                    kwargs["seed"] = self.seed
                resp = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                trace_lines.append(f"[Step {step}] API error: {e}")
                break

            msg = resp.choices[0].message

            # If model wants to call tools
            if msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    func_name = tc.function.name
                    try:
                        func_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        func_args = {}

                    tool_calls_count += 1
                    tools_used.append(func_name)

                    # Execute tool
                    handler = TOOL_DISPATCH.get(func_name)
                    if handler:
                        result = handler(func_args)
                    else:
                        result = f"Unknown tool: {func_name}"

                    trace_lines.append(
                        f"[Step {step}] {func_name}({json.dumps(func_args, ensure_ascii=False)[:100]}) "
                        f"→ {result[:200]}"
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result[:2000],
                    })
            else:
                # Model is done — extract final answer
                answer = msg.content or ""
                trace_lines.append(f"[Final] {answer[:200]}")
                return answer.strip(), tool_calls_count, tools_used, "\n".join(trace_lines)

        # Fell through — ask for final answer
        messages.append({"role": "user", "content": "Based on what you've found, provide your FINAL ANSWER concisely."})
        try:
            kwargs = dict(
                model=self.config.model,
                messages=messages,
                temperature=0.1,
                max_tokens=256,
            )
            if self.seed is not None:
                kwargs["seed"] = self.seed
            resp = self.client.chat.completions.create(**kwargs)
            answer = resp.choices[0].message.content or ""
            trace_lines.append(f"[Forced Final] {answer[:200]}")
            return answer.strip(), tool_calls_count, tools_used, "\n".join(trace_lines)
        except Exception as e:
            return str(e), tool_calls_count, tools_used, "\n".join(trace_lines)

    def solve_direct(self, question: str) -> Tuple[str, str]:
        """
        Solve with a single LLM call, no tools (baseline).
        Returns: (answer, reasoning)
        """
        if not self.client:
            return "No API client", "Error"

        try:
            kwargs = dict(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": (
                        "Answer the following question as precisely as possible. "
                        "Provide ONLY the final answer value, nothing else."
                    )},
                    {"role": "user", "content": question},
                ],
                temperature=0.1,
                max_tokens=256,
            )
            if self.seed is not None:
                kwargs["seed"] = self.seed
            resp = self.client.chat.completions.create(**kwargs)
            answer = resp.choices[0].message.content or ""
            return answer.strip(), "Direct answer (no tools)"
        except Exception as e:
            return str(e), "Error"


# ---------------------------------------------------------------------------
# Answer matching (GAIA uses exact match / normalized match)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    s = s.strip().lower()
    # Remove common prefixes
    for prefix in ["the answer is", "final answer:", "answer:", "the final answer is"]:
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    # Remove trailing period
    s = s.rstrip(".")
    # Remove quotes
    s = s.strip("'\"")
    return s


def check_answer(predicted: str, expected: str) -> bool:
    """Check if predicted answer matches expected (GAIA-style)."""
    pred = normalize_answer(predicted)
    exp = normalize_answer(expected)

    if not exp:
        return False

    # Exact match
    if pred == exp:
        return True

    # Check if expected is contained in predicted
    if exp in pred:
        return True

    # Numeric comparison
    try:
        p_num = float(pred.replace(",", ""))
        e_num = float(exp.replace(",", ""))
        return abs(p_num - e_num) < 0.01 * max(abs(e_num), 1)
    except ValueError:
        pass

    return False


# ---------------------------------------------------------------------------
# GAIA Evaluator
# ---------------------------------------------------------------------------

class GaiaEvaluator:
    """
    Run GAIA benchmark evaluation.

    Compares:
    - Probe-style (multi-step tool calling) — AURA's approach
    - Direct (single LLM call, no tools) — baseline
    """

    def __init__(self, config: EvalConfig, max_probe_steps: int = 5, seed: Optional[int] = None):
        self.config = config
        self.seed = seed
        self.solver = GaiaSolver(config, max_steps=max_probe_steps, seed=seed)
        self.results: List[GaiaResult] = []

    def load_gaia(self, split: str = "validation",
                  max_questions: Optional[int] = None,
                  levels: Optional[List[int]] = None) -> List[dict]:
        """Load GAIA questions from saved dataset."""
        try:
            from datasets import load_from_disk
            ds = load_from_disk(
                str(Path(__file__).parent.parent / "datasets" / "gaia"))
            data = ds[split]
        except Exception:
            # Fallback: try loading from Arrow directly
            gaia_path = Path(__file__).parent.parent / "datasets" / "gaia"
            if not gaia_path.exists():
                raise FileNotFoundError(f"GAIA dataset not found at {gaia_path}")
            from datasets import load_from_disk
            data = load_from_disk(str(gaia_path))[split]

        questions = []
        level_set = {str(lvl) for lvl in levels} if levels else None
        for row in data:
            row_level = str(row["Level"])
            if level_set and row_level not in level_set:
                continue
            if not row.get("Final answer"):
                continue  # skip test set (no answers)
            questions.append({
                "task_id": row["task_id"],
                "question": row["Question"],
                "level": row_level,
                "answer": row["Final answer"],
            })

        if max_questions:
            questions = questions[:max_questions]

        return questions

    def run(self, questions: Optional[List[dict]] = None,
            max_questions: int = 50,
            levels: Optional[List[int]] = None,
            run_direct: bool = True) -> Dict[str, Any]:
        """
        Run GAIA evaluation.

        Args:
            questions: Pre-loaded questions (or None to load from dataset)
            max_questions: Max questions to evaluate
            levels: Filter by GAIA levels [1, 2, 3]
            run_direct: Whether to also run the direct (no-tool) baseline
        """
        if questions is None:
            questions = self.load_gaia("validation", max_questions, levels)

        print(f"\nGAIA Evaluation: {len(questions)} questions")
        if levels:
            print(f"  Levels: {levels}")

        all_results = []
        conditions = ["probe"]
        if run_direct:
            conditions.append("direct")

        for cond in conditions:
            print(f"\n{'='*60}")
            print(f"Condition: {cond}")
            print(f"{'='*60}")

            for i, q in enumerate(questions):
                t0 = time.time()

                if cond == "probe":
                    answer, n_tools, tools, trace = self.solver.solve_with_probe(
                        q["question"])
                else:
                    answer, trace = self.solver.solve_direct(q["question"])
                    n_tools = 0
                    tools = []

                latency = time.time() - t0
                correct = check_answer(answer, q["answer"])

                result = GaiaResult(
                    task_id=q["task_id"],
                    question=q["question"][:200],
                    level=q["level"],
                    expected_answer=q["answer"],
                    predicted_answer=answer[:200],
                    correct=correct,
                    num_tool_calls=n_tools,
                    tools_used=tools,
                    reasoning_trace=trace[:500],
                    latency_s=latency,
                    condition=cond,
                )
                all_results.append(result)

                status = "✓" if correct else "✗"
                print(f"  [{i+1}/{len(questions)}] {status} L{q['level']} "
                      f"tools={n_tools} {latency:.1f}s | "
                      f"expected='{q['answer']}' got='{answer[:50]}'")

                # Rate limiting
                time.sleep(0.5)

        self.results = all_results
        return self._aggregate(all_results)

    def _aggregate(self, results: List[GaiaResult]) -> Dict[str, Any]:
        summary = {"total": len(results), "conditions": {}}

        by_cond: Dict[str, List[GaiaResult]] = {}
        for r in results:
            by_cond.setdefault(r.condition, []).append(r)

        for cond, rs in by_cond.items():
            correct = sum(1 for r in rs if r.correct)
            total = len(rs)
            avg_latency = sum(r.latency_s for r in rs) / total if total else 0
            avg_tools = sum(r.num_tool_calls for r in rs) / total if total else 0

            # Per level
            by_level: Dict[int, List[GaiaResult]] = {}
            for r in rs:
                by_level.setdefault(r.level, []).append(r)

            per_level = {}
            for level, lrs in sorted(by_level.items()):
                lc = sum(1 for r in lrs if r.correct)
                lt = len(lrs)
                per_level[f"level_{level}"] = {
                    "accuracy": lc / lt if lt else 0,
                    "correct": lc,
                    "total": lt,
                }

            summary["conditions"][cond] = {
                "accuracy": correct / total if total else 0,
                "correct": correct,
                "total": total,
                "avg_latency_s": round(avg_latency, 2),
                "avg_tool_calls": round(avg_tools, 1),
                "per_level": per_level,
            }

        return summary

    def save_results(self, path: Optional[str] = None):
        if path is None:
            path = Path(self.config.results_dir) / "gaia_results.json"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        output = {
            "summary": self._aggregate(self.results),
            "results": [asdict(r) for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to {path}")

    def print_summary(self, summary: Dict[str, Any]):
        print(f"\n{'='*60}")
        print("GAIA BENCHMARK RESULTS")
        print(f"{'='*60}")

        for cond, stats in summary["conditions"].items():
            acc = stats["accuracy"] * 100
            print(f"\n  [{cond}] Overall: {acc:.1f}% "
                  f"({stats['correct']}/{stats['total']})")
            print(f"    Avg latency: {stats['avg_latency_s']:.1f}s, "
                  f"Avg tools: {stats['avg_tool_calls']:.1f}")
            for level_key, lstats in stats.get("per_level", {}).items():
                lacc = lstats["accuracy"] * 100
                print(f"    {level_key}: {lacc:.1f}% "
                      f"({lstats['correct']}/{lstats['total']})")

    def generate_latex_table(self, summary: Dict[str, Any]) -> str:
        conditions = list(summary["conditions"].keys())
        all_levels = set()
        for cond_data in summary["conditions"].values():
            all_levels.update(cond_data.get("per_level", {}).keys())
        levels = sorted(all_levels)

        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{GAIA Benchmark Accuracy (\%). "
            r"Probe = AURA multi-step tool calling. "
            r"Direct = single LLM call without tools.}",
            r"\label{tab:gaia}",
            r"\begin{tabular}{l" + "c" * len(conditions) + "}",
            r"\toprule",
            "Level & " + " & ".join(
                c.capitalize() for c in conditions) + r" \\",
            r"\midrule",
        ]

        for level in levels:
            label = level.replace("_", " ").title()
            vals = []
            for cond in conditions:
                lstats = summary["conditions"][cond].get(
                    "per_level", {}).get(level, {})
                acc = lstats.get("accuracy", 0) * 100
                vals.append(f"{acc:.1f}")
            lines.append(f"{label} & " + " & ".join(vals) + r" \\")

        lines.append(r"\midrule")
        vals = []
        for cond in conditions:
            acc = summary["conditions"][cond]["accuracy"] * 100
            vals.append(f"\\textbf{{{acc:.1f}}}")
        lines.append(r"\textbf{Overall} & " + " & ".join(vals) + r" \\")

        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GAIA Benchmark Evaluation")
    parser.add_argument("--max-questions", type=int, default=50,
                        help="Max questions to evaluate")
    parser.add_argument("--levels", type=int, nargs="+", default=None,
                        help="GAIA levels to include (1, 2, 3)")
    parser.add_argument("--max-steps", type=int, default=5,
                        help="Max probe steps (tool calls) per question")
    parser.add_argument("--no-direct", action="store_true",
                        help="Skip direct (no-tool) baseline")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed passed to OpenAI API for reproducibility")
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    config = EvalConfig()
    evaluator = GaiaEvaluator(config, max_probe_steps=args.max_steps, seed=args.seed)

    print("GAIA Benchmark Evaluation")
    print(f"  Max questions: {args.max_questions}")
    print(f"  Levels: {args.levels or 'all'}")
    print(f"  Max probe steps: {args.max_steps}")
    print(f"  Seed: {args.seed}")

    summary = evaluator.run(
        max_questions=args.max_questions,
        levels=args.levels,
        run_direct=not args.no_direct,
    )

    evaluator.print_summary(summary)
    evaluator.save_results(args.output)
    print(f"\n{evaluator.generate_latex_table(summary)}")
