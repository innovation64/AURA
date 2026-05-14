"""LoCoMo smoke run: 30 questions stratified across 4 categories, 1 seed, gpt-4o-mini.

Three conditions: literal / no_intent / intent. Output JSON mirrors
fantom_full_seed42.json structure.

Sampling rule (seed=42):
  cat 1 (multi-hop)    : 7
  cat 2 (temporal)     : 8
  cat 3 (open-domain)  : 7
  cat 4 (single-hop)   : 8
  Total                : 30
  (cat 5 / adversarial is excluded from this smoke; it uses a different
   scoring regime.)

Output: evaluation/results/locomo_smoke.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

from aura.intent import LLMIntentInferrer, HeuristicIntentInferrer  # noqa: F401
from aura.types import IntentFrame, MemoryItem, SceneState

from evaluation.locomo_eval import (
    LoCoMoQuestion,
    PROBE_TOOL_NAMES,
    PROBE_TOOL_SCHEMA,
    build_scene_and_memories,
    make_probe_executor,
    score_question,
    stratified_sample,
    _CAT_LABEL,
    _session_brief,
)


BACKBONE_MODEL = os.environ.get("LOCOMO_BACKBONE", "gpt-4o-mini")
N_QUESTIONS = int(os.environ.get("LOCOMO_N", "30"))
CONFIGURED_BUDGET = int(os.environ.get("LOCOMO_BUDGET", "3"))
SEED = int(os.environ.get("LOCOMO_SEED", "42"))
N_WORKERS = int(os.environ.get("LOCOMO_WORKERS", "6"))

# gpt-4o-mini pricing (USD per 1M tokens)
PRICE_IN_PER_1M = 0.150
PRICE_OUT_PER_1M = 0.600

OUT_PATH = _REPO_ROOT / "evaluation" / "results" / "locomo_smoke.json"

CONDITIONS = ["literal", "no_intent", "intent"]


# ---------------------------------------------------------------------------
# Thread-safe meters
# ---------------------------------------------------------------------------


class TokenMeter:
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()

    def add(self, resp: Any) -> None:
        u = getattr(resp, "usage", None)
        if u is None:
            return
        ptok = int(getattr(u, "prompt_tokens", 0) or 0)
        ctok = int(getattr(u, "completion_tokens", 0) or 0)
        with self._lock:
            self.input_tokens += ptok
            self.output_tokens += ctok

    def cost_usd(self) -> float:
        return (
            self.input_tokens * PRICE_IN_PER_1M / 1_000_000
            + self.output_tokens * PRICE_OUT_PER_1M / 1_000_000
        )


class FallbackCounter:
    def __init__(self) -> None:
        self.calls = 0
        self.fallbacks = 0
        self._lock = threading.Lock()

    def record(self, fell_back: bool) -> None:
        with self._lock:
            self.calls += 1
            if fell_back:
                self.fallbacks += 1

    def rate(self) -> float:
        with self._lock:
            return self.fallbacks / self.calls if self.calls else 0.0


# ---------------------------------------------------------------------------
# Seed-aware chat completion wrapper
# ---------------------------------------------------------------------------


_SEED_SUPPORTED = True
_SEED_LOCK = threading.Lock()


def _chat_create(client: OpenAI, **kwargs: Any) -> Any:
    global _SEED_SUPPORTED
    with _SEED_LOCK:
        send_seed = _SEED_SUPPORTED
    if send_seed:
        try:
            return client.chat.completions.create(seed=SEED, **kwargs)
        except TypeError:
            with _SEED_LOCK:
                _SEED_SUPPORTED = False
        except Exception as e:
            msg = str(e).lower()
            if "seed" in msg and ("unsupported" in msg or "unknown" in msg or "not supported" in msg or "invalid" in msg):
                with _SEED_LOCK:
                    _SEED_SUPPORTED = False
            else:
                raise
    return client.chat.completions.create(**kwargs)


# ---------------------------------------------------------------------------
# Wrapped IntentInferrer that records fallback events
# ---------------------------------------------------------------------------


class _CountingIntentInferrer:
    """Wraps LLMIntentInferrer; tracks fallback rate and token usage via shared meter."""

    def __init__(self, client: OpenAI, model: str, counter: FallbackCounter, meter: TokenMeter) -> None:
        from aura.intent import _INTENT_SYSTEM_PROMPT, _build_user_message, _parse_intent_json
        self._client = client
        self._model = model
        self._counter = counter
        self._meter = meter
        self._fallback = HeuristicIntentInferrer()
        self._sys = _INTENT_SYSTEM_PROMPT
        self._build_user = _build_user_message
        self._parse = _parse_intent_json

    def infer(
        self,
        user_query: str,
        scene: SceneState,
        memories: Sequence[MemoryItem],
        user_profile: Optional[Dict[str, Any]] = None,
        available_tools: Optional[Sequence[str]] = None,
    ) -> IntentFrame:
        if not user_query or not user_query.strip():
            self._counter.record(True)
            return self._fallback.infer(user_query, scene, memories, user_profile, available_tools)
        if self._client is None:
            self._counter.record(True)
            return self._fallback.infer(user_query, scene, memories, user_profile, available_tools)

        user_message = self._build_user(user_query, scene, memories, user_profile, available_tools)
        kwargs = dict(
            model=self._model,
            messages=[
                {"role": "system", "content": self._sys},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        raw = ""
        try:
            resp = _chat_create(self._client, **kwargs, response_format={"type": "json_object"})
            self._meter.add(resp)
            raw = (resp.choices[0].message.content or "").strip()
        except Exception:
            raw = ""
        if not raw:
            try:
                resp = _chat_create(self._client, **kwargs)
                self._meter.add(resp)
                raw = (resp.choices[0].message.content or "").strip()
            except Exception:
                self._counter.record(True)
                return self._fallback.infer(user_query, scene, memories, user_profile, available_tools)

        frame = self._parse(raw, user_query)
        if frame is None:
            self._counter.record(True)
            return self._fallback.infer(user_query, scene, memories, user_profile, available_tools)

        if available_tools is not None and frame.recommended_probes:
            allowed = set(available_tools)
            frame.recommended_probes = [p for p in frame.recommended_probes if p in allowed]

        self._counter.record(False)
        return frame


# ---------------------------------------------------------------------------
# Conversation-context helper
# ---------------------------------------------------------------------------


def _scene_brief(q: LoCoMoQuestion) -> str:
    """Compact per-session catalog: 'Session N (date, K turns): preview...'"""
    return "\n".join(_session_brief(s, max_chars=240) for s in q.sessions)


def _build_system_prompt(q: LoCoMoQuestion, condition: str, extra: str = "") -> str:
    base = (
        "You are answering a question about a long-term conversation between two people, "
        "split across many sessions on different dates.\n\n"
        f"=== PARTICIPANTS ===\n{q.speaker_a} and {q.speaker_b}\n"
        f"Total sessions: {len(q.sessions)}.\n\n"
        f"=== SESSION CATALOG ===\n{_scene_brief(q)}\n=== END CATALOG ===\n\n"
        "Answer concisely; for factual questions reply with the answer phrase only "
        "(e.g. dates, names, short noun phrases). If you cannot determine the answer "
        "from the conversation, reply 'No information available.'\n\n"
    )
    if condition == "literal":
        base += "Answer using ONLY the session catalog above. Do not invent facts."
    else:
        base += extra
    return base


# ---------------------------------------------------------------------------
# Three conditions
# ---------------------------------------------------------------------------


def run_literal(client: OpenAI, q: LoCoMoQuestion, meter: TokenMeter) -> Tuple[str, int, float, str]:
    t0 = time.time()
    sys_prompt = _build_system_prompt(q, "literal")
    resp = _chat_create(
        client,
        model=BACKBONE_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q.question},
        ],
        temperature=0.1,
        max_tokens=120,
    )
    meter.add(resp)
    return resp.choices[0].message.content or "", 0, time.time() - t0, sys_prompt


def _react_loop(
    client: OpenAI,
    q: LoCoMoQuestion,
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
        resp = _chat_create(
            client,
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
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out[:3500]})
            if calls >= budget:
                messages.append({"role": "user",
                                 "content": "Based on what you have, answer the question concisely now."})
                final = _chat_create(
                    client,
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


def run_no_intent(client: OpenAI, q: LoCoMoQuestion, meter: TokenMeter) -> Tuple[str, int, float, str]:
    sys_prompt = _build_system_prompt(
        q,
        "no_intent",
        f"You may call up to {CONFIGURED_BUDGET} tools to inspect specific sessions, "
        "search by speaker, or list sessions on a date. Then answer concisely.",
    )
    ans, calls, dur = _react_loop(client, q, meter, CONFIGURED_BUDGET, sys_prompt)
    return ans, calls, dur, sys_prompt


def run_intent(
    client: OpenAI,
    q: LoCoMoQuestion,
    meter: TokenMeter,
    inferrer: _CountingIntentInferrer,
) -> Tuple[str, int, float, float, List[str], str]:
    t0 = time.time()
    scene, memories = build_scene_and_memories(q)
    frame = inferrer.infer(q.question, scene, memories, available_tools=PROBE_TOOL_NAMES)
    g = frame.gap or 0.0
    if g < 0.20:
        dyn_budget = 0
    elif g < 0.40:
        dyn_budget = 1
    elif g < 0.60:
        dyn_budget = 2
    elif g < 0.80:
        dyn_budget = 3
    else:
        dyn_budget = 5
    dyn_budget = min(CONFIGURED_BUDGET, dyn_budget)
    recommended = list(frame.recommended_probes or [])
    if dyn_budget == 0:
        ans, _, _, sp = run_literal(client, q, meter)
        return ans, 0, time.time() - t0, g, recommended, sp
    preferred = recommended or PROBE_TOOL_NAMES
    sys_prompt = _build_system_prompt(
        q,
        "intent",
        f"Inferred implicit need: {frame.implicit_need}. "
        f"You may call up to {dyn_budget} tools. Prefer these first if relevant: {sorted(set(preferred))}. "
        "Then answer concisely.",
    )
    ans, calls, _ = _react_loop(client, q, meter, dyn_budget, sys_prompt)
    return ans, calls, time.time() - t0, g, recommended, sys_prompt


# ---------------------------------------------------------------------------
# Per-question worker
# ---------------------------------------------------------------------------


def _process_question(
    idx: int,
    q: LoCoMoQuestion,
    client: OpenAI,
    meters: Dict[str, TokenMeter],
    inferrer: _CountingIntentInferrer,
) -> Tuple[int, Dict[str, Any]]:
    record: Dict[str, Any] = {
        "idx": idx,
        "qid": q.qid,
        "sample_id": q.sample_id,
        "category": q.category,
        "category_label": q.category_label,
        "question": q.question,
        "gold": q.gold,
        "evidence": q.evidence,
        "n_sessions": len(q.sessions),
        "by_condition": {},
    }
    # literal
    try:
        ans, calls, dur, sp = run_literal(client, q, meters["literal"])
        sc = score_question(q, ans)
        record["by_condition"]["literal"] = {
            "answer": ans, "probes": calls, "latency_s": round(dur, 3),
            "f1": round(sc["f1"], 4), "em": int(sc["em"]),
            "correct": int(sc["em"]),  # alias for cross-benchmark uniformity
            "system_prompt_chars": len(sp),
        }
    except Exception as e:
        record["by_condition"]["literal"] = {"error": str(e)[:240], "f1": 0.0, "em": 0,
                                             "correct": 0, "probes": 0, "latency_s": 0.0}

    # no_intent
    try:
        ans, calls, dur, sp = run_no_intent(client, q, meters["no_intent"])
        sc = score_question(q, ans)
        record["by_condition"]["no_intent"] = {
            "answer": ans, "probes": calls, "latency_s": round(dur, 3),
            "f1": round(sc["f1"], 4), "em": int(sc["em"]),
            "correct": int(sc["em"]),
            "system_prompt_chars": len(sp),
        }
    except Exception as e:
        record["by_condition"]["no_intent"] = {"error": str(e)[:240], "f1": 0.0, "em": 0,
                                               "correct": 0, "probes": 0, "latency_s": 0.0}

    # intent
    try:
        ans, calls, dur, gap, recommended, sp = run_intent(client, q, meters["intent"], inferrer)
        sc = score_question(q, ans)
        record["by_condition"]["intent"] = {
            "answer": ans, "probes": calls, "latency_s": round(dur, 3),
            "gap": round(gap, 3), "recommended_probes": recommended,
            "f1": round(sc["f1"], 4), "em": int(sc["em"]),
            "correct": int(sc["em"]),
            "system_prompt_chars": len(sp),
        }
    except Exception as e:
        record["by_condition"]["intent"] = {"error": str(e)[:240], "f1": 0.0, "em": 0,
                                            "correct": 0, "probes": 0, "latency_s": 0.0}

    return idx, record


# ---------------------------------------------------------------------------
# Stats: McNemar (on EM) + paired-t (on F1)
# ---------------------------------------------------------------------------


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _paired_t(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    n = len(a)
    if n != len(b) or n < 2:
        return 0.0, 1.0
    diffs = [a[i] - b[i] for i in range(n)]
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var == 0.0:
        return 0.0, 1.0
    se = math.sqrt(var / n)
    t = mean / se if se > 0 else 0.0
    p = 2.0 * (1.0 - _phi(abs(t)))
    return t, p


def _mcnemar(a: Sequence[int], b: Sequence[int]) -> Tuple[int, int, float]:
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    cum = 0.0
    for i in range(k + 1):
        cum += math.comb(n, i)
    cum /= 2 ** n
    return b01, b10, min(1.0, 2.0 * cum)


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

    print(f"[locomo-smoke] model={BACKBONE_MODEL} seed={SEED} workers={N_WORKERS} budget={CONFIGURED_BUDGET}")
    questions = stratified_sample(N_QUESTIONS, seed=SEED)
    from collections import Counter
    qctr = Counter(q.category_label for q in questions)
    print(f"[sample] got {len(questions)} questions; by category: {dict(qctr)}")

    meters = {c: TokenMeter() for c in CONDITIONS}
    fallback_counters = {c: FallbackCounter() for c in CONDITIONS}
    inferrer = _CountingIntentInferrer(client, BACKBONE_MODEL, fallback_counters["intent"], meters["intent"])

    details: List[Optional[Dict[str, Any]]] = [None] * len(questions)
    wall_start = time.time()
    completed = 0
    progress_lock = threading.Lock()

    def _on_done(idx: int, rec: Dict[str, Any]) -> None:
        nonlocal completed
        with progress_lock:
            completed += 1
            cur = completed
        n = len(questions)
        elapsed = time.time() - wall_start
        rate = cur / elapsed if elapsed > 0 else 0.0
        eta = (n - cur) / rate if rate > 0 else 0.0
        bc = rec["by_condition"]
        lit = bc.get("literal", {}).get("f1", 0.0)
        ni = bc.get("no_intent", {}).get("f1", 0.0)
        it = bc.get("intent", {}).get("f1", 0.0)
        print(
            f"[{cur:3d}/{n}] elapsed={elapsed/60:5.1f}m eta={eta/60:5.1f}m "
            f"cat={rec['category_label']:<11} L_f1={lit:.2f} N_f1={ni:.2f} I_f1={it:.2f}",
            flush=True,
        )

    if N_WORKERS <= 1:
        for i, q in enumerate(questions):
            _, rec = _process_question(i, q, client, meters, inferrer)
            details[i] = rec
            _on_done(i, rec)
    else:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = [
                ex.submit(_process_question, i, q, client, meters, inferrer)
                for i, q in enumerate(questions)
            ]
            for fut in as_completed(futures):
                idx, rec = fut.result()
                details[idx] = rec
                _on_done(idx, rec)

    wall_total = time.time() - wall_start

    # Aggregate
    n = len(questions)
    per_condition: Dict[str, Any] = {}
    for cond in CONDITIONS:
        f1_vec = [float(details[i]["by_condition"][cond].get("f1", 0.0)) for i in range(n)]
        em_vec = [int(details[i]["by_condition"][cond].get("em", 0)) for i in range(n)]
        lat_vec = [float(details[i]["by_condition"][cond].get("latency_s", 0.0)) for i in range(n)]
        probe_vec = [int(details[i]["by_condition"][cond].get("probes", 0)) for i in range(n)]
        # By category
        by_cat: Dict[str, Dict[str, Any]] = {}
        for cat in (1, 2, 3, 4):
            idxs = [i for i in range(n) if details[i]["category"] == cat]
            if not idxs:
                continue
            f1c = [f1_vec[i] for i in idxs]
            emc = [em_vec[i] for i in idxs]
            by_cat[_CAT_LABEL[cat]] = {
                "n": len(idxs),
                "f1": round(sum(f1c) / len(f1c), 4),
                "em": round(sum(emc) / len(emc), 4),
            }
        m = meters[cond]
        fb = fallback_counters[cond]
        per_condition[cond] = {
            "n_items": n,
            "f1": round(sum(f1_vec) / n, 4),
            "em": round(sum(em_vec) / n, 4),
            "by_category": by_cat,
            "mean_latency": round(sum(lat_vec) / n, 3),
            "mean_probes": round(sum(probe_vec) / n, 3),
            "input_tokens": m.input_tokens,
            "output_tokens": m.output_tokens,
            "cost_usd": round(m.cost_usd(), 4),
            "fallback_calls": fb.calls,
            "fallback_count": fb.fallbacks,
            "fallback_rate": round(fb.rate(), 4),
        }

    # Stats: paired tests on F1 (continuous) and McNemar on EM (binary)
    f1_lit = [float(details[i]["by_condition"]["literal"].get("f1", 0.0)) for i in range(n)]
    f1_ni = [float(details[i]["by_condition"]["no_intent"].get("f1", 0.0)) for i in range(n)]
    f1_it = [float(details[i]["by_condition"]["intent"].get("f1", 0.0)) for i in range(n)]
    em_lit = [int(details[i]["by_condition"]["literal"].get("em", 0)) for i in range(n)]
    em_ni = [int(details[i]["by_condition"]["no_intent"].get("em", 0)) for i in range(n)]
    em_it = [int(details[i]["by_condition"]["intent"].get("em", 0)) for i in range(n)]

    t_il, p_il = _paired_t(f1_it, f1_lit)
    t_in, p_in = _paired_t(f1_it, f1_ni)
    b01_il, b10_il, mc_il = _mcnemar(em_lit, em_it)
    b01_in, b10_in, mc_in = _mcnemar(em_ni, em_it)

    stats = {
        "intent_vs_literal": {
            "paired_t_on_f1": {"t": round(t_il, 4), "p": round(p_il, 6)},
            "mcnemar_on_em": {"b_lit_only": b10_il, "b_int_only": b01_il, "p": round(mc_il, 6)},
            "delta_f1": round(per_condition["intent"]["f1"] - per_condition["literal"]["f1"], 4),
            "delta_em": round(per_condition["intent"]["em"] - per_condition["literal"]["em"], 4),
        },
        "intent_vs_no_intent": {
            "paired_t_on_f1": {"t": round(t_in, 4), "p": round(p_in, 6)},
            "mcnemar_on_em": {"b_ni_only": b10_in, "b_int_only": b01_in, "p": round(mc_in, 6)},
            "delta_f1": round(per_condition["intent"]["f1"] - per_condition["no_intent"]["f1"], 4),
            "delta_em": round(per_condition["intent"]["em"] - per_condition["no_intent"]["em"], 4),
        },
    }

    total_in = sum(meters[c].input_tokens for c in CONDITIONS)
    total_out = sum(meters[c].output_tokens for c in CONDITIONS)
    total_cost = sum(meters[c].cost_usd() for c in CONDITIONS)

    out: Dict[str, Any] = {
        "meta": {
            "n_questions": n,
            "model": BACKBONE_MODEL,
            "seed": SEED,
            "configured_budget": CONFIGURED_BUDGET,
            "n_workers": N_WORKERS,
            "categories": [1, 2, 3, 4],
            "category_labels": _CAT_LABEL,
            "pricing_per_1M_tokens": {"input_usd": PRICE_IN_PER_1M, "output_usd": PRICE_OUT_PER_1M},
            "total_tokens_in": total_in,
            "total_tokens_out": total_out,
            "total_cost_usd": round(total_cost, 4),
            "wall_clock_s": round(wall_total, 2),
            "openai_seed_supported": _SEED_SUPPORTED,
        },
        "per_condition": per_condition,
        "stats": stats,
        "details": details,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[write] {OUT_PATH}")

    # Console summary
    print("\n=== Per-condition F1 / EM ===")
    print(f"{'cond':<10} {'F1':>6} {'EM':>6}  {'multi':>6} {'tempo':>6} {'open':>6} {'singl':>6}  {'lat(s)':>7} {'probes':>7} {'fb':>6}")
    cat_order = ["multi_hop", "temporal", "open_domain", "single_hop"]
    for cond in CONDITIONS:
        pc = per_condition[cond]
        bc = pc["by_category"]
        def _g(k): return f"{bc.get(k,{}).get('f1',0.0):>6.3f}"
        print(
            f"{cond:<10} {pc['f1']:>6.3f} {pc['em']:>6.3f}  "
            f"{_g('multi_hop')} {_g('temporal')} {_g('open_domain')} {_g('single_hop')}  "
            f"{pc['mean_latency']:>7.2f} {pc['mean_probes']:>7.2f} {pc['fallback_rate']:>6.3f}"
        )
    print(f"\nTotals: in={total_in} out={total_out} cost=${total_cost:.4f} wall={wall_total/60:.1f}min")
    print(f"\nStats:")
    print(f"  intent vs literal:    dF1={stats['intent_vs_literal']['delta_f1']:+.4f} "
          f"dEM={stats['intent_vs_literal']['delta_em']:+.4f}  "
          f"paired-t(F1) p={p_il:.4g}  McNemar(EM) p={mc_il:.4g}  "
          f"(lit_only={b10_il}, int_only={b01_il})")
    print(f"  intent vs no_intent:  dF1={stats['intent_vs_no_intent']['delta_f1']:+.4f} "
          f"dEM={stats['intent_vs_no_intent']['delta_em']:+.4f}  "
          f"paired-t(F1) p={p_in:.4g}  McNemar(EM) p={mc_in:.4g}  "
          f"(ni_only={b10_in}, int_only={b01_in})")
    print(f"\nIntent fallback: {fallback_counters['intent'].fallbacks}/{fallback_counters['intent'].calls} "
          f"({per_condition['intent']['fallback_rate']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
