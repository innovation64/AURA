"""FANToM 5-question smoke: 3 conditions x 1 seed on gpt-4o-mini.

Conditions (parallel structure to scripts/run_implicit_intent_smoke.py):
  - literal   : single LLM call with conversation context only, no tools.
  - no_intent : LLM may issue ReAct-style tool calls up to fixed budget.
  - intent    : full AURA path -- LLMIntentInferrer -> dynamic budget ->
                directed probing -> answer.

Tools surfaced to LLM (defined in evaluation/fantom_eval.py):
  - get_utterance_history(speaker)
  - get_present_speakers_at_utterance(idx)
  - get_information_disclosed_in_window(start, end)

Output: evaluation/results/fantom_smoke.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

_REPO_ROOT = Path(__file__).resolve().parent.parent
_AURA_SRC = _REPO_ROOT / "AURA" / "src"
if str(_AURA_SRC) not in sys.path:
    sys.path.insert(0, str(_AURA_SRC))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openai import OpenAI

from aura.intent import LLMIntentInferrer

from evaluation.fantom_eval import (
    FantomQuestion,
    PROBE_TOOL_NAMES,
    PROBE_TOOL_SCHEMA,
    build_scene_and_memories,
    make_probe_executor,
    sample_questions,
    score_question,
)


BACKBONE_MODEL = os.environ.get("FANTOM_BACKBONE", "gpt-4o-mini")
N_QUESTIONS = int(os.environ.get("FANTOM_N", "5"))
CONFIGURED_BUDGET = int(os.environ.get("FANTOM_BUDGET", "3"))
SEED = int(os.environ.get("FANTOM_SEED", "1"))

# gpt-4o-mini pricing (USD per 1M tokens), as of 2024-2025
PRICE_IN_PER_1M = 0.150
PRICE_OUT_PER_1M = 0.600

OUT_PATH = _REPO_ROOT / "evaluation" / "results" / "fantom_smoke.json"


# ---------------------------------------------------------------------------
# Token / cost tracking helper
# ---------------------------------------------------------------------------


class TokenMeter:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, resp: Any) -> None:
        u = getattr(resp, "usage", None)
        if u is None:
            return
        self.input_tokens += int(getattr(u, "prompt_tokens", 0) or 0)
        self.output_tokens += int(getattr(u, "completion_tokens", 0) or 0)

    def cost_usd(self) -> float:
        return (
            self.input_tokens * PRICE_IN_PER_1M / 1_000_000
            + self.output_tokens * PRICE_OUT_PER_1M / 1_000_000
        )


# ---------------------------------------------------------------------------
# Conversation-context helper
# ---------------------------------------------------------------------------


def _build_system_prompt(q: FantomQuestion, condition: str, extra: str = "") -> str:
    base = (
        "You are answering a Theory-of-Mind question about a multi-party conversation.\n\n"
        f"=== CONVERSATION ===\n{q.full_context}\n=== END CONVERSATION ===\n\n"
        f"Participants: {', '.join(q.speakers)}.\n"
    )
    if condition == "literal":
        base += "Answer the question using ONLY the conversation above. Do not invent facts."
    else:
        base += extra
    return base


# ---------------------------------------------------------------------------
# Three conditions
# ---------------------------------------------------------------------------


def run_literal(client: OpenAI, q: FantomQuestion, meter: TokenMeter) -> Tuple[str, int, float]:
    t0 = time.time()
    resp = client.chat.completions.create(
        model=BACKBONE_MODEL,
        messages=[
            {"role": "system", "content": _build_system_prompt(q, "literal")},
            {"role": "user", "content": q.question},
        ],
        temperature=0.1,
        max_tokens=120,
    )
    meter.add(resp)
    return resp.choices[0].message.content or "", 0, time.time() - t0


def _react_loop(
    client: OpenAI,
    q: FantomQuestion,
    meter: TokenMeter,
    budget: int,
    system_prompt: str,
) -> Tuple[str, int, float]:
    t0 = time.time()
    tool_exec = make_probe_executor(q)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": q.question},
    ]
    calls = 0
    for _ in range(budget + 1):
        resp = client.chat.completions.create(
            model=BACKBONE_MODEL,
            messages=messages,
            tools=PROBE_TOOL_SCHEMA,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=200,
        )
        meter.add(resp)
        msg = resp.choices[0].message
        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                if calls < budget:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {}
                    out = tool_exec(tc.function.name, args)
                    calls += 1
                else:
                    out = json.dumps({"error": "budget exhausted"})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out[:3000]})
            if calls >= budget:
                messages.append({"role": "user", "content": "Based on what you have, answer now in the requested format."})
                final = client.chat.completions.create(
                    model=BACKBONE_MODEL,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=120,
                )
                meter.add(final)
                return final.choices[0].message.content or "", calls, time.time() - t0
            continue
        return msg.content or "", calls, time.time() - t0
    return "", calls, time.time() - t0


def run_no_intent(client: OpenAI, q: FantomQuestion, meter: TokenMeter) -> Tuple[str, int, float]:
    sys_prompt = _build_system_prompt(
        q,
        "no_intent",
        f"You may call up to {CONFIGURED_BUDGET} tools to inspect the conversation structure. "
        "Then answer in the requested format.",
    )
    return _react_loop(client, q, meter, CONFIGURED_BUDGET, sys_prompt)


def run_intent(
    client: OpenAI, q: FantomQuestion, meter: TokenMeter,
) -> Tuple[str, int, float, float, List[str]]:
    """Returns (answer, probes, latency, gap, recommended_probes)."""
    t0 = time.time()
    inferrer = LLMIntentInferrer(client=client, model=BACKBONE_MODEL)
    scene, memories = build_scene_and_memories(q)
    frame = inferrer.infer(q.question, scene, memories, available_tools=PROBE_TOOL_NAMES)
    # Note: LLMIntentInferrer makes its own LLM call internally; we don't
    # have access to its usage object. We approximate by the budget
    # being implicit. For accurate accounting, we instead time-and-count
    # only the answering loop here. (Documented limitation.)

    g = frame.gap or 0.0
    if g < 0.20:    dyn_budget = 0
    elif g < 0.40:  dyn_budget = 1
    elif g < 0.60:  dyn_budget = 2
    elif g < 0.80:  dyn_budget = 3
    else:           dyn_budget = 5
    dyn_budget = min(CONFIGURED_BUDGET, dyn_budget)
    recommended = list(frame.recommended_probes or [])

    if dyn_budget == 0:
        ans, _, _ = run_literal(client, q, meter)
        return ans, 0, time.time() - t0, g, recommended

    preferred = recommended or PROBE_TOOL_NAMES
    sys_prompt = _build_system_prompt(
        q,
        "intent",
        f"Inferred implicit need: {frame.implicit_need}. "
        f"You may call up to {dyn_budget} tools. Prefer these first if relevant: {sorted(set(preferred))}. "
        "Then answer in the requested format.",
    )
    ans, calls, _ = _react_loop(client, q, meter, dyn_budget, sys_prompt)
    return ans, calls, time.time() - t0, g, recommended


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> int:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERR: OPENAI_API_KEY not set", file=sys.stderr)
        return 2
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"Sampling {N_QUESTIONS} FANToM questions (seed={SEED}) ...")
    questions = sample_questions(N_QUESTIONS, seed=SEED)
    print(f"Got {len(questions)} questions across types: {[q.qtype for q in questions]}\n")

    conditions = ["literal", "no_intent", "intent"]
    meters = {c: TokenMeter() for c in conditions}
    results: Dict[str, Any] = {
        "meta": {
            "backbone": BACKBONE_MODEL,
            "n_questions": len(questions),
            "configured_budget": CONFIGURED_BUDGET,
            "seed": SEED,
            "pricing_per_1M_tokens": {"input_usd": PRICE_IN_PER_1M, "output_usd": PRICE_OUT_PER_1M},
        },
        "per_question": [],
    }

    wall_start = time.time()
    for q in questions:
        row: Dict[str, Any] = {
            "qid": q.qid,
            "qtype": q.qtype,
            "sub_qtype": q.sub_qtype,
            "agent_subject": q.agent_subject,
            "gold": list(q.gold) if isinstance(q.gold, set) else q.gold,
            "question_preview": q.question[:160],
            "by_condition": {},
        }
        print(f"--- {q.qid} [{q.qtype}/{q.sub_qtype}] subject={q.agent_subject} gold={row['gold']!r} ---")

        # Literal
        ans, calls, dur = run_literal(client, q, meters["literal"])
        ok = score_question(q, ans)
        row["by_condition"]["literal"] = {
            "answer": ans, "probes": calls, "latency_s": round(dur, 2), "correct": ok,
        }
        print(f"  literal   ok={ok} {dur:.2f}s -> {ans[:80]!r}")

        # No-intent
        ans, calls, dur = run_no_intent(client, q, meters["no_intent"])
        ok = score_question(q, ans)
        row["by_condition"]["no_intent"] = {
            "answer": ans, "probes": calls, "latency_s": round(dur, 2), "correct": ok,
        }
        print(f"  no_intent ok={ok} {dur:.2f}s probes={calls} -> {ans[:80]!r}")

        # Intent
        ans, calls, dur, gap, recommended = run_intent(client, q, meters["intent"])
        ok = score_question(q, ans)
        row["by_condition"]["intent"] = {
            "answer": ans, "probes": calls, "latency_s": round(dur, 2),
            "gap": gap, "recommended_probes": recommended, "correct": ok,
        }
        print(f"  intent    ok={ok} {dur:.2f}s probes={calls} gap={gap:.2f} -> {ans[:80]!r}")
        print()

        results["per_question"].append(row)

    wall_total = time.time() - wall_start

    # Aggregate
    agg: Dict[str, Any] = {}
    for cond in conditions:
        per = [r["by_condition"][cond]["correct"] for r in results["per_question"]]
        lat = [r["by_condition"][cond]["latency_s"] for r in results["per_question"]]
        m = meters[cond]
        agg[cond] = {
            "n": len(per),
            "correct": int(sum(per)),
            "accuracy": round(sum(per) / len(per), 3) if per else 0.0,
            "input_tokens": m.input_tokens,
            "output_tokens": m.output_tokens,
            "total_tokens": m.input_tokens + m.output_tokens,
            "cost_usd": round(m.cost_usd(), 4),
            "mean_latency_s": round(sum(lat) / len(lat), 2) if lat else 0.0,
        }
    total_tokens = sum(m.input_tokens + m.output_tokens for m in meters.values())
    total_cost = sum(m.cost_usd() for m in meters.values())

    # Project to ~1000-question full split
    scale = 1000.0 / max(1, len(questions))
    full_split_estimate = {
        "tokens": int(total_tokens * scale),
        "cost_usd": round(total_cost * scale, 2),
        "wall_clock_min": round(wall_total * scale / 60, 1),
    }

    results["aggregate"] = agg
    results["wall_clock_s_total"] = round(wall_total, 2)
    results["full_split_estimate_1000q"] = full_split_estimate

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n[write] {OUT_PATH}")
    print("=== Aggregate ===")
    print(f"{'cond':<10} {'acc':>5} {'in_tok':>8} {'out_tok':>8} {'cost($)':>8} {'lat(s)':>7}")
    for cond in conditions:
        a = agg[cond]
        print(f"{cond:<10} {a['accuracy']:>5.3f} {a['input_tokens']:>8} {a['output_tokens']:>8} {a['cost_usd']:>8.4f} {a['mean_latency_s']:>7.2f}")
    print(f"\nWall-clock total: {wall_total:.2f}s")
    print(f"Total tokens:     {total_tokens}")
    print(f"Total cost (smoke): ${total_cost:.4f}")
    print(f"Full-split est (1000q): {full_split_estimate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
