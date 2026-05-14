"""Full run: 25 implicit-intent queries x selected conditions x seeds + judge.

Scales the Day-8 smoke to paper-grade evidence. Each response is scored
by an independent judge call on two dimensions:

  literal_score  [0,1]: does the response correctly answer the surface
                       query? (Primary check: no fabricated facts vs the
                       scene / tool returns the response claims to use.)
  implicit_score [0,1]: does the response surface or address the
                       implicit_need that the query was actually about?
                       A response that answers the surface question but
                       misses the private-state context gets a low
                       implicit_score even with a high literal_score.

For each (condition, seed, query) we record:
  answer, probes_called, latency, gap (tom only), literal_score,
  implicit_score, judge_rationale.

Output: evaluation/results/rq2_implicit_intent_multiseed.json with
  per-seed x per-condition x per-query details AND a paired t-test
  block for the three condition contrasts (tom vs literal, tom vs
  no_intent, no_intent vs literal) on each score.

Default conditions are literal, no_intent, and tom. Optional extra
baselines are fixed_probe and oracle_intent. Budget estimate for the
default run: 25 queries x 3 conds x 3 seeds = 225 answer runs
(~2 LLM calls/run on avg) + 225 judge calls, about $2 on gpt-4o-mini.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
_AURA_SRC = ROOT / "AURA" / "src"
if str(_AURA_SRC) not in sys.path:
    sys.path.insert(0, str(_AURA_SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI

# Reuse the Day-8 smoke's condition implementations verbatim so the
# runtime behavior stays identical between smoke and full.
from scripts.run_implicit_intent_smoke import (
    SCENE_SUMMARY, PUBLIC_STATE, PRIVATE_STATE, BELIEFS_ABOUT_OTHERS,
    AVAILABLE_TOOLS, tool_exec, TOOL_SCHEMA,
    _literal_answer, _react_answer, _tom_answer,
)

from aura.intent import LLMIntentInferrer  # noqa: F401  (imported for symmetry)

JUDGE_MODEL = "gpt-4o-mini"
CONFIGURED_BUDGET = 3
DEFAULT_CONDITIONS = ["literal", "no_intent", "tom"]
VALID_CONDITIONS = DEFAULT_CONDITIONS + ["fixed_probe", "oracle_intent"]

JUDGE_SYSTEM = (
    "You are an expert evaluator for a theory-of-mind-aware assistant. "
    "Given a user's query, hidden ground-truth private state, the implicit "
    "need the query is really asking about, and an AI response, produce a "
    "JSON object with two independent scores in [0, 1]:\n\n"
    "  literal_score  : does the response correctly answer the surface "
    "question? Consistency with the scene / ground truth earns 1.0. "
    "Fabricating contradictory public facts earns 0.\n"
    "  implicit_score : does the response surface or address the implicit "
    "need? Explicit naming of the implicit concern earns 1.0. Ignoring it "
    "earns 0. Partial surfacing earns 0.4-0.6.\n\n"
    "SECOND-ORDER NOTE: when the ground-truth dict contains a "
    "'believers_belief_about_target' field, the query is about the "
    "BELIEVER'S BELIEF -- not the target's actual state. A correct "
    "second-order answer reflects the believer's belief (even if that "
    "belief is wrong compared to targets_actual_private_state). A response "
    "that conflates the believer's belief with the target's ground truth "
    "earns a LOWER implicit_score, because it fails to model the believer's "
    "mind.\n\n"
    "Output schema (JSON, no prose):\n"
    "{\n"
    '  "literal_score": 0.0,\n'
    '  "implicit_score": 0.0,\n'
    '  "rationale": "<one sentence>"\n'
    "}"
)


def judge_response(client: OpenAI, query: str, implicit_need: str,
                   response: str, agent_private_state: Dict[str, Any]) -> Dict[str, Any]:
    """Score a single response on literal + implicit. Independent of condition."""
    ground_truth = {
        "scene_summary": SCENE_SUMMARY,
        "public_state": PUBLIC_STATE,
        "private_state": agent_private_state,
        "implicit_need": implicit_need,
    }
    user_msg = (
        f"USER QUERY: {query}\n\n"
        f"GROUND TRUTH (scene + all private state): {json.dumps(ground_truth, ensure_ascii=False)}\n\n"
        f"IMPLICIT NEED THE QUERY IS REALLY AFTER: {implicit_need}\n\n"
        f"AI RESPONSE: {response}\n\n"
        "Score both dimensions and output the JSON."
    )
    try:
        r = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1, max_tokens=200,
            response_format={"type": "json_object"},
        )
        raw = (r.choices[0].message.content or "").strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return {"literal_score": None, "implicit_score": None,
                    "rationale": f"parse error: {raw[:80]}"}
        ls = float(obj.get("literal_score", 0) or 0)
        ims = float(obj.get("implicit_score", 0) or 0)
        return {
            "literal_score": max(0.0, min(1.0, ls)),
            "implicit_score": max(0.0, min(1.0, ims)),
            "rationale": str(obj.get("rationale", ""))[:400],
        }
    except Exception as e:
        return {"literal_score": None, "implicit_score": None,
                "rationale": f"judge error: {type(e).__name__}: {e}"}


def paired_t(diffs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    valid = [d for d in diffs if d is not None and not math.isnan(d)]
    if len(valid) < 2:
        return None, None
    md = statistics.mean(valid)
    sd = statistics.stdev(valid) if len(set(valid)) > 1 else 0.0
    if sd == 0:
        return (float("inf") if md != 0 else 0.0), None
    t = md / (sd / math.sqrt(len(valid)))
    try:
        from scipy import stats  # type: ignore
        p = float(stats.t.sf(abs(t), len(valid) - 1) * 2)
    except ImportError:
        p = None
    return t, p


def _load_queries() -> List[Dict[str, Any]]:
    src = Path(__file__).resolve().parent.parent / "evaluation" / "data" / "implicit_intent_queries.json"
    with open(src) as f:
        return json.load(f)["queries"]


def _subjects_for_query(q: Dict[str, Any], budget: int = CONFIGURED_BUDGET) -> List[str]:
    """Deterministic subject resolver for non-LLM baselines.

    The benchmark has a few group queries with no `agent_subject`. For those,
    we use the public scene to choose likely subjects, capped by the same probe
    budget, so fixed/oracle baselines do not get unbounded private-state access.
    """
    subject = q.get("agent_subject")
    if subject:
        return [subject]

    query = str(q.get("query", "")).lower()
    if "cafe" in query or "group" in query or "atmosphere" in query:
        cafe_agents = [
            name for name, state in PUBLIC_STATE.items()
            if state.get("location") == "Sunrise Cafe"
        ]
        return cafe_agents[:budget]
    return list(PUBLIC_STATE.keys())[:budget]


def _dedupe_probe_plan(plan: Iterable[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for step in plan:
        tool = step.get("tool")
        args = step.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        key = (tool, json.dumps(args, sort_keys=True, ensure_ascii=False))
        if tool not in AVAILABLE_TOOLS or key in seen:
            continue
        seen.add(key)
        deduped.append({
            "tool": tool,
            "args": args,
            "reason": step.get("reason", ""),
        })
        if len(deduped) >= budget:
            break
    return deduped


def _execute_probe_plan(plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    for step in plan:
        tool = step["tool"]
        args = step.get("args") or {}
        try:
            out = tool_exec(tool, args)
        except Exception as e:
            out = json.dumps({"error": f"{type(e).__name__}: {e}"})
        observations.append({
            "tool": tool,
            "args": args,
            "reason": step.get("reason", ""),
            "result": out,
        })
    return observations


def _synthesize_from_observations(
    client: OpenAI,
    query: str,
    implicit_need: str,
    observations: List[Dict[str, Any]],
) -> str:
    obs_block = (
        "\n".join(
            f"- {o['tool']}({json.dumps(o.get('args', {}), ensure_ascii=False)})"
            f" -> {o.get('result', '')}"
            for o in observations
        )
        if observations else "(no probes called)"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                f"Scene: {SCENE_SUMMARY}\n\n"
                f"Public state: {json.dumps(PUBLIC_STATE, ensure_ascii=False)}\n\n"
                "You are answering after a deterministic baseline has already "
                "gathered probe evidence. You MUST answer both parts:\n"
                "1. The user's literal surface question.\n"
                "2. The implicit private-state concern below, using the probe "
                "evidence when it supports it.\n\n"
                f"Implicit private-state concern to address: {implicit_need}\n\n"
                "Be concise; do not mention scoring, labels, or gold data.\n\n"
                f"Probe evidence:\n{obs_block}"
            )},
            {"role": "user", "content": query},
        ],
        temperature=0.1,
        max_tokens=400,
    )
    return resp.choices[0].message.content or ""


def _fixed_probe_answer(
    client: OpenAI, q: Dict[str, Any], budget: int
) -> Tuple[str, int, float, List[Dict[str, Any]]]:
    """Unconditional private-state probe baseline.

    Skips IntentInferrer. For first-order queries, probes private state for the
    resolved subject(s). For second-order queries, probes the believer's belief
    about the target. The total probe count is capped by the same budget.
    """
    t0 = time.time()
    if q.get("subcategory") == "second_order":
        plan = [{
            "tool": "get_agent_belief_about",
            "args": {
                "believer": q.get("agent_subject") or "",
                "target": q.get("target") or "",
            },
            "reason": "fixed second-order private-belief probe",
        }]
    else:
        plan = [
            {
                "tool": "get_agent_private_state",
                "args": {"agent_name": subject},
                "reason": "fixed private-state probe",
            }
            for subject in _subjects_for_query(q, budget)
        ]
    plan = _dedupe_probe_plan(plan, budget)
    observations = _execute_probe_plan(plan)
    answer = _synthesize_from_observations(
        client, q["query"], q.get("implicit_need", ""), observations,
    )
    return answer, len(observations), time.time() - t0, observations


def _oracle_probe_plan(q: Dict[str, Any], budget: int) -> List[Dict[str, Any]]:
    requirements = list(q.get("implicit_requires") or []) + list(q.get("literal_requires") or [])
    subjects = _subjects_for_query(q, budget)
    plan: List[Dict[str, Any]] = []

    def add(tool: str, args: Dict[str, Any], reason: str) -> None:
        plan.append({"tool": tool, "args": args, "reason": reason})

    for req in requirements:
        req_s = str(req)
        if req_s.startswith("beliefs_about_others"):
            add(
                "get_agent_belief_about",
                {
                    "believer": q.get("agent_subject") or "",
                    "target": q.get("target") or "",
                },
                f"oracle requirement: {req_s}",
            )
        elif req_s in {"availability", "mood", "emotional_state", "unspoken_goal", "secrets"}:
            for subject in subjects:
                add(
                    "get_agent_private_state",
                    {"agent_name": subject},
                    f"oracle requirement: {req_s}",
                )
        elif req_s == "agent_plan":
            for subject in subjects:
                add(
                    "get_agent_plan",
                    {"agent_name": subject},
                    f"oracle requirement: {req_s}",
                )
        else:
            add("get_all_agents", {}, f"oracle fallback requirement: {req_s}")

    if not plan:
        add("get_all_agents", {}, "oracle fallback: no requirements")
    return _dedupe_probe_plan(plan, budget)


def _oracle_intent_answer(
    client: OpenAI, q: Dict[str, Any], budget: int
) -> Tuple[str, int, float, List[Dict[str, Any]]]:
    """Gold-requirement routed baseline.

    Skips IntentInferrer and derives the probe plan from benchmark labels
    (`implicit_need`, `implicit_requires`, plus literal requirements for plan
    needs). This estimates the upper bound of deterministic intent routing.
    """
    t0 = time.time()
    plan = _oracle_probe_plan(q, budget)
    observations = _execute_probe_plan(plan)
    answer = _synthesize_from_observations(
        client, q["query"], q.get("implicit_need", ""), observations,
    )
    return answer, len(observations), time.time() - t0, observations


def run_one_condition(client: OpenAI, cond: str, q: Dict[str, Any]) -> Dict[str, Any]:
    query_text = q["query"]
    if cond == "literal":
        ans, calls, dur = _literal_answer(client, query_text)
        return {"answer": ans, "probes": calls, "latency": round(dur, 2), "gap": None}
    if cond == "no_intent":
        ans, calls, dur = _react_answer(client, query_text, CONFIGURED_BUDGET)
        return {"answer": ans, "probes": calls, "latency": round(dur, 2), "gap": None}
    if cond == "tom":
        ans, calls, dur, gap = _tom_answer(client, query_text, CONFIGURED_BUDGET)
        return {"answer": ans, "probes": calls, "latency": round(dur, 2), "gap": gap}
    if cond == "fixed_probe":
        ans, calls, dur, observations = _fixed_probe_answer(client, q, CONFIGURED_BUDGET)
        return {
            "answer": ans,
            "probes": calls,
            "latency": round(dur, 2),
            "gap": None,
            "probe_observations": observations,
        }
    if cond == "oracle_intent":
        ans, calls, dur, observations = _oracle_intent_answer(client, q, CONFIGURED_BUDGET)
        return {
            "answer": ans,
            "probes": calls,
            "latency": round(dur, 2),
            "gap": "oracle",
            "probe_observations": observations,
        }
    raise ValueError(f"unknown condition {cond!r}")


def _parse_conditions(raw: Optional[List[str]]) -> List[str]:
    if not raw:
        return list(DEFAULT_CONDITIONS)
    conditions: List[str] = []
    for item in raw:
        for part in item.split(","):
            cond = part.strip()
            if not cond:
                continue
            if cond == "all":
                for valid in VALID_CONDITIONS:
                    if valid not in conditions:
                        conditions.append(valid)
                continue
            if cond not in VALID_CONDITIONS:
                raise ValueError(
                    f"unknown condition {cond!r}; valid: {', '.join(VALID_CONDITIONS)}, all"
                )
            if cond not in conditions:
                conditions.append(cond)
    return conditions


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--limit", type=int, default=None,
                   help="Limit the number of queries (useful for partial runs)")
    p.add_argument("--conditions", nargs="+", default=None,
                   help=("Conditions to run. Valid: "
                         + ", ".join(VALID_CONDITIONS)
                         + ", all. Accepts spaces or comma-separated values."))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERR: OPENAI_API_KEY not set", file=sys.stderr)
        return 2
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    queries = _load_queries()
    if args.limit:
        queries = queries[:args.limit]
    try:
        conditions = _parse_conditions(args.conditions)
    except ValueError as e:
        print(f"ERR: {e}", file=sys.stderr)
        return 2
    print(f"Running {len(queries)} queries x {len(conditions)} conditions x {len(args.seeds)} seeds = "
          f"{len(queries) * len(conditions) * len(args.seeds)} answer runs "
          f"+ {len(queries) * len(conditions) * len(args.seeds)} judge calls")

    results: Dict[str, Any] = {
        "meta": {
            "scene": SCENE_SUMMARY,
            "configured_budget": CONFIGURED_BUDGET,
            "seeds": list(args.seeds),
            "model": "gpt-4o-mini",
            "judge_model": JUDGE_MODEL,
            "intent_prompt_variant": os.environ.get("AURA_INTENT_PROMPT_VARIANT", "clean"),
        },
        "per_seed": {str(s): {c: [] for c in conditions} for s in args.seeds},
    }

    t_start = time.time()
    for si, seed in enumerate(args.seeds):
        print(f"\n=== seed {seed} ({si + 1}/{len(args.seeds)}) ===")
        # OpenAI supports seed for near-deterministic replay; we pass it through
        # the client by setting a per-session 'seed' via env if the SDK honors it.
        # The LLM calls inside the condition helpers don't take seed directly
        # (simplification for this run), so effectively the only stochasticity
        # between seeds is temperature sampling at T=0.1 which is near-zero.
        # The 3 seeds give us paired variance estimates; if the variance is tiny
        # we will say so explicitly in the paper.

        for qi, q in enumerate(queries, 1):
            subject = q.get("agent_subject") or list(PRIVATE_STATE.keys())[0]
            private = dict(PRIVATE_STATE.get(subject, {}))
            # For second-order ToM queries, the relevant ground-truth for
            # the judge is the believer's BELIEFS about the target, not the
            # target's ground-truth state. Expose the belief explicitly.
            if q.get("subcategory") == "second_order":
                target = q.get("target") or ""
                believer_beliefs = BELIEFS_ABOUT_OTHERS.get(subject, {})
                private["beliefs_about_others"] = believer_beliefs
                private["target_in_question"] = target
                private["believers_belief_about_target"] = believer_beliefs.get(target, {})
                private["targets_actual_private_state"] = PRIVATE_STATE.get(target, {})
            print(f"  [{qi}/{len(queries)}] id={q['id']:<4} {q.get('subcategory'):<17} {q['query'][:60]}")

            for cond in conditions:
                t_run = time.time()
                try:
                    resp = run_one_condition(client, cond, q)
                except Exception as e:
                    resp = {"answer": "", "probes": 0, "latency": 0.0, "gap": None,
                            "error": f"{type(e).__name__}: {e}"}

                # Judge
                try:
                    score = judge_response(
                        client, q["query"], q.get("implicit_need", ""),
                        resp.get("answer", ""), private,
                    )
                except Exception as e:
                    score = {"literal_score": None, "implicit_score": None,
                             "rationale": f"judge exc: {e}"}

                row = {
                    "query_id": q["id"],
                    "subcategory": q.get("subcategory"),
                    "agent_subject": subject,
                    **resp,
                    **score,
                    "total_time": round(time.time() - t_run, 2),
                }
                results["per_seed"][str(seed)][cond].append(row)

                gstr = f" gap={resp.get('gap')}" if resp.get("gap") is not None else ""
                err = resp.get("error")
                err_s = f" ERR={err}" if err else ""
                print(f"    [{cond:<9}] probes={resp['probes']}{gstr}  "
                      f"lit={score.get('literal_score')} imp={score.get('implicit_score')}{err_s}")

    # Aggregate
    print("\n=== Aggregation ===")
    stats: Dict[str, Any] = {}
    for cond in conditions:
        stats[cond] = {"literal_score": [], "implicit_score": [], "probes": [], "latency": []}
        for seed in args.seeds:
            rows = results["per_seed"][str(seed)][cond]
            lit_vals = [r["literal_score"] for r in rows if r.get("literal_score") is not None]
            imp_vals = [r["implicit_score"] for r in rows if r.get("implicit_score") is not None]
            probe_vals = [r["probes"] for r in rows]
            lat_vals = [r["latency"] for r in rows]
            if lit_vals:
                stats[cond]["literal_score"].append(sum(lit_vals) / len(lit_vals))
            if imp_vals:
                stats[cond]["implicit_score"].append(sum(imp_vals) / len(imp_vals))
            if probe_vals:
                stats[cond]["probes"].append(sum(probe_vals) / len(probe_vals))
            if lat_vals:
                stats[cond]["latency"].append(sum(lat_vals) / len(lat_vals))

    # Summary print
    for cond in conditions:
        s = stats[cond]
        def mean_std(vals: List[float]) -> str:
            if not vals:
                return "n/a"
            if len(vals) == 1:
                return f"{vals[0]:.4f}"
            return f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"
        print(f"  {cond:<10}  literal={mean_std(s['literal_score'])}  "
              f"implicit={mean_std(s['implicit_score'])}  "
              f"probes={mean_std(s['probes'])}  "
              f"latency={mean_std(s['latency'])}")

    # Paired t-tests (query-level, pooled across seeds)
    def paired_rows(metric: str, cond_a: str, cond_b: str) -> List[float]:
        diffs = []
        for seed in args.seeds:
            a_rows = {r["query_id"]: r for r in results["per_seed"][str(seed)][cond_a]}
            b_rows = {r["query_id"]: r for r in results["per_seed"][str(seed)][cond_b]}
            for qid, ra in a_rows.items():
                rb = b_rows.get(qid)
                if not rb:
                    continue
                va, vb = ra.get(metric), rb.get(metric)
                if va is None or vb is None:
                    continue
                diffs.append(va - vb)
        return diffs

    pair_results: Dict[str, Any] = {}
    for metric in ("literal_score", "implicit_score"):
        pair_results[metric] = {}
        candidate_pairs = [
            ("tom", "literal"),
            ("tom", "no_intent"),
            ("tom", "fixed_probe"),
            ("oracle_intent", "tom"),
            ("oracle_intent", "no_intent"),
            ("oracle_intent", "fixed_probe"),
            ("fixed_probe", "no_intent"),
            ("fixed_probe", "literal"),
            ("oracle_intent", "literal"),
            ("no_intent", "literal"),
        ]
        for ca, cb in candidate_pairs:
            if ca not in conditions or cb not in conditions:
                continue
            diffs = paired_rows(metric, ca, cb)
            t, pv = paired_t(diffs)
            pair_results[metric][f"{ca}_vs_{cb}"] = {
                "n_pairs": len(diffs),
                "mean_delta": round(statistics.mean(diffs), 4) if diffs else None,
                "std": round(statistics.stdev(diffs), 4) if len(diffs) > 1 else None,
                "t": round(t, 3) if t is not None and t != float("inf") else None,
                "p_two_sided": round(pv, 5) if pv is not None else None,
            }

    print("\n=== Paired t-tests (query-level, pooled across seeds) ===")
    for metric, pairs in pair_results.items():
        print(f"  {metric}:")
        for key, s in pairs.items():
            star = " *" if s.get("p_two_sided") is not None and s["p_two_sided"] < 0.05 else ""
            print(f"    {key:<22}  n={s['n_pairs']:<3} Δ={s.get('mean_delta')} "
                  f"t={s.get('t')} p={s.get('p_two_sided')}{star}")

    results["statistics"] = stats
    results["paired_tests"] = pair_results
    results["meta"]["elapsed_sec"] = round(time.time() - t_start, 1)

    out = args.out or (
        Path(__file__).resolve().parent.parent / "evaluation" / "results"
        / "rq2_implicit_intent_multiseed.json"
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[write] {out}")
    print(f"Elapsed: {results['meta']['elapsed_sec']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
