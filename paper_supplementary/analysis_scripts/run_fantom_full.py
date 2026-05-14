"""FANToM paper-grade evaluation: 400-question stratified split, 1 seed, gpt-4o-mini.

Sampling rule (seed=42):
  80 questions per qtype across 5 types: beliefQA, answ_bin, answ_list, info_bin, info_list.
  Within beliefQA, balance 4 buckets x 20 each:
    - first-order accessible
    - first-order inaccessible
    - second-order accessible (any of: second-order:accessible, second-order:acyclic-accessible, second-order:cyclic-accessible)
    - second-order inaccessible (second-order:acyclic-inaccessible, second-order:cyclic-inaccessible)

Three conditions: literal / no_intent / intent.
- For intent: tracks LLMIntentInferrer fallback rate.
- All chat completions pass seed=42, gracefully retry without on rejection.

Concurrency: ThreadPoolExecutor with N_WORKERS parallel questions; ordering of
`details` is preserved (results list is pre-allocated and indexed).

Output: evaluation/results/fantom_full_seed42.json
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

import random

from openai import OpenAI

from aura.intent import LLMIntentInferrer, HeuristicIntentInferrer
from aura.types import IntentFrame, MemoryItem, SceneState

from evaluation.fantom_eval import (
    FantomQuestion,
    PROBE_TOOL_NAMES,
    PROBE_TOOL_SCHEMA,
    adapt_row_to_questions,
    build_scene_and_memories,
    load_fantom_dataframe,
    make_probe_executor,
    score_question,
)


BACKBONE_MODEL = os.environ.get("FANTOM_BACKBONE", "gpt-4o-mini")
N_PER_TYPE = int(os.environ.get("FANTOM_N_PER_TYPE", "80"))
CONFIGURED_BUDGET = int(os.environ.get("FANTOM_BUDGET", "3"))
SEED = int(os.environ.get("FANTOM_SEED", "42"))
N_WORKERS = int(os.environ.get("FANTOM_WORKERS", "8"))

# gpt-4o-mini pricing (USD per 1M tokens)
PRICE_IN_PER_1M = 0.150
PRICE_OUT_PER_1M = 0.600

OUT_PATH = _REPO_ROOT / "evaluation" / "results" / "fantom_full_seed42.json"

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


# Module-global flag: once a server tells us 'seed' is unsupported, stop sending
# it to avoid retry overhead on every call.
_SEED_SUPPORTED = True
_SEED_LOCK = threading.Lock()


def _chat_create(client: OpenAI, **kwargs: Any) -> Any:
    """OpenAI chat.completions.create with seed=SEED, falling back if rejected."""
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
    """Wraps LLMIntentInferrer; records whether the LLM JSON path succeeded.

    We re-implement infer() by mirroring LLMIntentInferrer.infer() but tracking
    when the heuristic fallback fires, and (separately) using the seed-aware
    _chat_create wrapper for determinism when supported.
    """

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


def run_literal(client: OpenAI, q: FantomQuestion, meter: TokenMeter) -> Tuple[str, int, float, str]:
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
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out[:3000]})
            if calls >= budget:
                messages.append({"role": "user", "content": "Based on what you have, answer now in the requested format."})
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


def run_no_intent(client: OpenAI, q: FantomQuestion, meter: TokenMeter) -> Tuple[str, int, float, str]:
    sys_prompt = _build_system_prompt(
        q,
        "no_intent",
        f"You may call up to {CONFIGURED_BUDGET} tools to inspect the conversation structure. "
        "Then answer in the requested format.",
    )
    ans, calls, dur = _react_loop(client, q, meter, CONFIGURED_BUDGET, sys_prompt)
    return ans, calls, dur, sys_prompt


def run_intent(
    client: OpenAI,
    q: FantomQuestion,
    meter: TokenMeter,
    inferrer: _CountingIntentInferrer,
) -> Tuple[str, int, float, float, List[str], str]:
    t0 = time.time()
    scene, memories = build_scene_and_memories(q)
    frame = inferrer.infer(q.question, scene, memories, available_tools=PROBE_TOOL_NAMES)
    g = frame.gap or 0.0
    if g < 0.20:    dyn_budget = 0
    elif g < 0.40:  dyn_budget = 1
    elif g < 0.60:  dyn_budget = 2
    elif g < 0.80:  dyn_budget = 3
    else:           dyn_budget = 5
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
        "Then answer in the requested format.",
    )
    ans, calls, _ = _react_loop(client, q, meter, dyn_budget, sys_prompt)
    return ans, calls, time.time() - t0, g, recommended, sys_prompt


# ---------------------------------------------------------------------------
# Stratified sampler
# ---------------------------------------------------------------------------


def _belief_bucket(q: FantomQuestion) -> Optional[str]:
    """Return one of: 'fo_acc', 'fo_inacc', 'so_acc', 'so_inacc'. None if unclassifiable."""
    tom = q.tom_order or ""
    acc = q.info_accessibility or ""
    if tom == "first-order":
        if acc == "accessible":
            return "fo_acc"
        if acc == "inaccessible":
            return "fo_inacc"
    elif tom.startswith("second-order"):
        if acc == "accessible":
            return "so_acc"
        if acc == "inaccessible":
            return "so_inacc"
    return None


def stratified_sample(seed: int, n_per_type: int) -> List[FantomQuestion]:
    df = load_fantom_dataframe()
    rng = random.Random(seed)
    all_qs: List[FantomQuestion] = []
    for _, row in df.iterrows():
        all_qs.extend(adapt_row_to_questions(row.to_dict(), rng))

    by_type: Dict[str, List[FantomQuestion]] = {t: [] for t in
                                                ("belief", "answ_bin", "answ_list", "info_bin", "info_list")}
    for q in all_qs:
        if q.qtype in by_type:
            by_type[q.qtype].append(q)

    chosen: List[FantomQuestion] = []

    # belief: 4 buckets x (n_per_type/4) each
    per_bucket = n_per_type // 4
    belief_buckets: Dict[str, List[FantomQuestion]] = {"fo_acc": [], "fo_inacc": [], "so_acc": [], "so_inacc": []}
    for q in by_type["belief"]:
        b = _belief_bucket(q)
        if b is not None:
            belief_buckets[b].append(q)
    for bname, pool in belief_buckets.items():
        rng.shuffle(pool)
        if len(pool) < per_bucket:
            print(f"[warn] belief bucket {bname} has only {len(pool)} (need {per_bucket}); taking all")
        chosen.extend(pool[:per_bucket])

    # other 4 qtypes: simple shuffle and slice
    for t in ("answ_bin", "answ_list", "info_bin", "info_list"):
        pool = list(by_type[t])
        rng.shuffle(pool)
        if len(pool) < n_per_type:
            print(f"[warn] qtype {t} has only {len(pool)} (need {n_per_type}); taking all")
        chosen.extend(pool[:n_per_type])

    rng.shuffle(chosen)
    return chosen


# ---------------------------------------------------------------------------
# Per-question worker
# ---------------------------------------------------------------------------


def _process_question(
    idx: int,
    q: FantomQuestion,
    client: OpenAI,
    meters: Dict[str, TokenMeter],
    fallback_counters: Dict[str, FallbackCounter],
    inferrer: _CountingIntentInferrer,
) -> Tuple[int, Dict[str, Any]]:
    record: Dict[str, Any] = {
        "idx": idx,
        "qid": q.qid,
        "qtype": q.qtype,
        "sub_qtype": q.sub_qtype,
        "tom_order": q.tom_order,
        "info_accessibility": q.info_accessibility,
        "agent_subject": q.agent_subject,
        "gold": list(q.gold) if isinstance(q.gold, set) else q.gold,
        "question": q.question,
        "by_condition": {},
    }
    # literal
    try:
        ans, calls, dur, sp = run_literal(client, q, meters["literal"])
        ok = score_question(q, ans)
        record["by_condition"]["literal"] = {
            "answer": ans, "probes": calls, "latency_s": round(dur, 3), "correct": int(ok),
            "system_prompt_chars": len(sp),
        }
    except Exception as e:
        record["by_condition"]["literal"] = {"error": str(e)[:240], "correct": 0, "probes": 0, "latency_s": 0.0}

    # no_intent
    try:
        ans, calls, dur, sp = run_no_intent(client, q, meters["no_intent"])
        ok = score_question(q, ans)
        record["by_condition"]["no_intent"] = {
            "answer": ans, "probes": calls, "latency_s": round(dur, 3), "correct": int(ok),
            "system_prompt_chars": len(sp),
        }
    except Exception as e:
        record["by_condition"]["no_intent"] = {"error": str(e)[:240], "correct": 0, "probes": 0, "latency_s": 0.0}

    # intent
    try:
        ans, calls, dur, gap, recommended, sp = run_intent(client, q, meters["intent"], inferrer)
        ok = score_question(q, ans)
        record["by_condition"]["intent"] = {
            "answer": ans, "probes": calls, "latency_s": round(dur, 3),
            "gap": round(gap, 3), "recommended_probes": recommended,
            "correct": int(ok), "system_prompt_chars": len(sp),
        }
    except Exception as e:
        record["by_condition"]["intent"] = {"error": str(e)[:240], "correct": 0, "probes": 0, "latency_s": 0.0}

    return idx, record


# ---------------------------------------------------------------------------
# Stats: McNemar + paired-t on per-question accuracy vector
# ---------------------------------------------------------------------------


def _paired_t(a: Sequence[int], b: Sequence[int]) -> Tuple[float, float]:
    """Returns (t_stat, two_sided_p)."""
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
    # Approximate two-sided p using survival via normal (n=400 large enough)
    p = 2.0 * (1.0 - _phi(abs(t)))
    return t, p


def _phi(x: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _mcnemar(a: Sequence[int], b: Sequence[int]) -> Tuple[int, int, float]:
    """Returns (b01_count, b10_count, two-sided p) of mid-p exact-binomial McNemar.

    a, b are 0/1 correctness vectors. b01 = a wrong & b right, b10 = a right & b wrong.
    """
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    # exact two-sided binomial test with p=0.5
    k = min(b01, b10)
    # cumulative P(X <= k | n, 0.5)
    cum = 0.0
    for i in range(k + 1):
        cum += math.comb(n, i)
    cum /= 2 ** n
    p = min(1.0, 2.0 * cum)
    return b01, b10, p


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

    print(f"[fantom-full] model={BACKBONE_MODEL} seed={SEED} workers={N_WORKERS} budget={CONFIGURED_BUDGET}")
    print(f"[sample] stratified: {N_PER_TYPE}/qtype x 5 = {N_PER_TYPE * 5} questions")
    questions = stratified_sample(SEED, N_PER_TYPE)
    print(f"[sample] got {len(questions)} questions")
    # Distribution sanity print
    from collections import Counter
    qctr = Counter(q.qtype for q in questions)
    print(f"[sample] by qtype: {dict(qctr)}")
    bctr = Counter(_belief_bucket(q) for q in questions if q.qtype == "belief")
    print(f"[sample] belief buckets: {dict(bctr)}")

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
        lit = bc.get("literal", {}).get("correct", 0)
        ni = bc.get("no_intent", {}).get("correct", 0)
        it = bc.get("intent", {}).get("correct", 0)
        print(
            f"[{cur:3d}/{n}] elapsed={elapsed/60:5.1f}m eta={eta/60:5.1f}m "
            f"qtype={rec['qtype']:<10} L={lit} N={ni} I={it}",
            flush=True,
        )

    if N_WORKERS <= 1:
        for i, q in enumerate(questions):
            _, rec = _process_question(i, q, client, meters, fallback_counters, inferrer)
            details[i] = rec
            _on_done(i, rec)
    else:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = [
                ex.submit(_process_question, i, q, client, meters, fallback_counters, inferrer)
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
        correct_vec = [int(details[i]["by_condition"][cond].get("correct", 0)) for i in range(n)]
        lat_vec = [float(details[i]["by_condition"][cond].get("latency_s", 0.0)) for i in range(n)]
        # By qtype
        acc_by_type: Dict[str, Dict[str, Any]] = {}
        for t in ("belief", "answ_bin", "answ_list", "info_bin", "info_list"):
            idxs = [i for i in range(n) if details[i]["qtype"] == t]
            if not idxs:
                continue
            cv = [correct_vec[i] for i in idxs]
            acc_by_type[t] = {"acc": round(sum(cv) / len(cv), 4), "n": len(cv), "correct": sum(cv)}
        # belief sub-buckets
        belief_subbuckets: Dict[str, Dict[str, Any]] = {}
        for sub in ("fo_acc", "fo_inacc", "so_acc", "so_inacc"):
            idxs = [
                i for i in range(n)
                if details[i]["qtype"] == "belief" and _belief_bucket(_q_from_record(details[i])) == sub
            ]
            if not idxs:
                continue
            cv = [correct_vec[i] for i in idxs]
            belief_subbuckets[sub] = {"acc": round(sum(cv) / len(cv), 4), "n": len(cv), "correct": sum(cv)}
        m = meters[cond]
        fb = fallback_counters[cond]
        per_condition[cond] = {
            "n_items": n,
            "correct": sum(correct_vec),
            "acc": round(sum(correct_vec) / n, 4),
            "acc_by_type": acc_by_type,
            "acc_by_belief_bucket": belief_subbuckets,
            "mean_latency": round(sum(lat_vec) / n, 3),
            "input_tokens": m.input_tokens,
            "output_tokens": m.output_tokens,
            "cost_usd": round(m.cost_usd(), 4),
            "fallback_calls": fb.calls,
            "fallback_count": fb.fallbacks,
            "fallback_rate": round(fb.rate(), 4),
        }

    # Stats
    correct_lit = [int(details[i]["by_condition"]["literal"].get("correct", 0)) for i in range(n)]
    correct_ni = [int(details[i]["by_condition"]["no_intent"].get("correct", 0)) for i in range(n)]
    correct_it = [int(details[i]["by_condition"]["intent"].get("correct", 0)) for i in range(n)]

    t_il, p_il = _paired_t(correct_it, correct_lit)
    t_in, p_in = _paired_t(correct_it, correct_ni)
    b01_il, b10_il, mc_il = _mcnemar(correct_lit, correct_it)
    b01_in, b10_in, mc_in = _mcnemar(correct_ni, correct_it)

    stats = {
        "intent_vs_literal": {
            "paired_t": {"t": round(t_il, 4), "p": round(p_il, 6)},
            "mcnemar": {"b_lit_only": b10_il, "b_int_only": b01_il, "p": round(mc_il, 6)},
            "delta_acc": round(per_condition["intent"]["acc"] - per_condition["literal"]["acc"], 4),
        },
        "intent_vs_no_intent": {
            "paired_t": {"t": round(t_in, 4), "p": round(p_in, 6)},
            "mcnemar": {"b_ni_only": b10_in, "b_int_only": b01_in, "p": round(mc_in, 6)},
            "delta_acc": round(per_condition["intent"]["acc"] - per_condition["no_intent"]["acc"], 4),
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
            "sampling_rule": {
                "n_per_type": N_PER_TYPE,
                "qtypes": ["belief", "answ_bin", "answ_list", "info_bin", "info_list"],
                "belief_strata": ["fo_acc", "fo_inacc", "so_acc", "so_inacc"],
                "belief_per_stratum": N_PER_TYPE // 4,
            },
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
    print("\n=== Per-condition accuracy ===")
    print(f"{'cond':<10} {'overall':>8} {'belief':>8} {'answ_b':>8} {'answ_l':>8} {'info_b':>8} {'info_l':>8} {'lat(s)':>7} {'fb_rate':>8}")
    for cond in CONDITIONS:
        pc = per_condition[cond]
        abt = pc["acc_by_type"]
        def _g(t): return f"{abt.get(t, {}).get('acc', 0.0):>8.3f}"
        print(
            f"{cond:<10} {pc['acc']:>8.3f} "
            f"{_g('belief')} {_g('answ_bin')} {_g('answ_list')} {_g('info_bin')} {_g('info_list')} "
            f"{pc['mean_latency']:>7.2f} {pc['fallback_rate']:>8.3f}"
        )
    print(f"\nTotals: in={total_in} out={total_out} cost=${total_cost:.4f} wall={wall_total/60:.1f}min")
    print(f"\nStats:")
    print(f"  intent vs literal:    dAcc={stats['intent_vs_literal']['delta_acc']:+.4f}  "
          f"paired-t p={p_il:.4g}  McNemar p={mc_il:.4g}  (lit_only={b10_il}, int_only={b01_il})")
    print(f"  intent vs no_intent:  dAcc={stats['intent_vs_no_intent']['delta_acc']:+.4f}  "
          f"paired-t p={p_in:.4g}  McNemar p={mc_in:.4g}  (ni_only={b10_in}, int_only={b01_in})")
    print(f"\nIntent fallback: {fallback_counters['intent'].fallbacks}/{fallback_counters['intent'].calls} "
          f"({per_condition['intent']['fallback_rate']:.3f})")
    return 0


def _q_from_record(rec: Dict[str, Any]) -> FantomQuestion:
    """Reconstruct a minimal FantomQuestion shim from a stored record (only fields _belief_bucket needs)."""
    return FantomQuestion(
        set_id="", qid=rec.get("qid",""), qtype=rec.get("qtype",""),
        sub_qtype=rec.get("sub_qtype",""), question="", agent_subject="",
        gold=None, full_context="", short_context="", joining_speaker="",
        missed_info="", speakers=[],
        tom_order=rec.get("tom_order","") or "",
        info_accessibility=rec.get("info_accessibility","") or "",
    )


if __name__ == "__main__":
    raise SystemExit(main())
