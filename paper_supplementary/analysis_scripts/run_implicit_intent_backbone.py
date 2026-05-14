"""Run RQ-Intent on a configurable backbone (e.g., Claude 3.5 Haiku via
OpenRouter) while keeping the judge fixed on gpt-4o-mini for
cross-backbone score consistency.

Usage:
  python scripts/run_implicit_intent_backbone.py \\
    --answer-model anthropic/claude-3.5-haiku \\
    --answer-base-url https://openrouter.ai/api/v1 \\
    --answer-api-env OPENROUTER_API_KEY \\
    --seeds 42 \\
    --out evaluation/results/rq2_implicit_intent_claude35haiku.json
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
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

_AURA_SRC = Path(__file__).resolve().parent.parent / "AURA" / "src"
if str(_AURA_SRC) not in sys.path:
    sys.path.insert(0, str(_AURA_SRC))

from openai import OpenAI

import scripts.run_implicit_intent_smoke as smoke
from scripts.run_implicit_intent_smoke import (
    SCENE_SUMMARY, PUBLIC_STATE, PRIVATE_STATE, BELIEFS_ABOUT_OTHERS,
    AVAILABLE_TOOLS, TOOL_SCHEMA, tool_exec,
    _literal_answer, _react_answer, _tom_answer,
)

JUDGE_MODEL = "gpt-4o-mini"
CONFIGURED_BUDGET = 3

JUDGE_SYSTEM = (
    "You are an expert evaluator for an implicit-intent-aware assistant. "
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
    "earns a LOWER implicit_score.\n\n"
    "Output schema (JSON, no prose):\n"
    "{\n"
    '  "literal_score": 0.0,\n'
    '  "implicit_score": 0.0,\n'
    '  "rationale": "<one sentence>"\n'
    "}"
)


def judge_response(judge_client: OpenAI, query: str, implicit_need: str,
                   response: str, agent_private_state: Dict[str, Any]) -> Dict[str, Any]:
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
        r = judge_client.chat.completions.create(
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


def run_one_condition(client: OpenAI, cond: str, query_text: str) -> Dict[str, Any]:
    if cond == "literal":
        ans, calls, dur = _literal_answer(client, query_text)
        return {"answer": ans, "probes": calls, "latency": round(dur, 2), "gap": None}
    if cond == "no_intent":
        ans, calls, dur = _react_answer(client, query_text, CONFIGURED_BUDGET)
        return {"answer": ans, "probes": calls, "latency": round(dur, 2), "gap": None}
    if cond == "tom":
        ans, calls, dur, gap = _tom_answer(client, query_text, CONFIGURED_BUDGET)
        return {"answer": ans, "probes": calls, "latency": round(dur, 2), "gap": gap}
    raise ValueError(f"unknown condition {cond!r}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--answer-model", required=True,
                   help="Backbone model id, e.g. 'anthropic/claude-3.5-haiku'")
    p.add_argument("--answer-base-url", default=None,
                   help="Base URL for the answer client (default: OpenAI)")
    p.add_argument("--answer-api-env", default="OPENAI_API_KEY",
                   help="Env var name holding the answer-client API key")
    p.add_argument("--judge-model", default="gpt-4o-mini",
                   help="Judge model (kept on OpenAI for cross-backbone consistency)")
    p.add_argument("--judge-api-env", default="OPENAI_API_KEY")
    p.add_argument("--judge-base-url", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    answer_key = os.environ.get(args.answer_api_env, "")
    if not answer_key:
        print(f"ERR: {args.answer_api_env} not set", file=sys.stderr)
        return 2
    judge_key = os.environ.get(args.judge_api_env, "")
    if not judge_key:
        print(f"ERR: {args.judge_api_env} not set", file=sys.stderr)
        return 2

    answer_base = (args.answer_base_url or os.environ.get("OPENAI_BASE_URL")
                   or "https://api.openai.com/v1").strip().strip('"')
    judge_base = (args.judge_base_url or "https://api.openai.com/v1").strip().strip('"')

    answer_client = OpenAI(api_key=answer_key, base_url=answer_base)
    judge_client = OpenAI(api_key=judge_key, base_url=judge_base)

    smoke.set_backbone_model(args.answer_model)
    global JUDGE_MODEL
    JUDGE_MODEL = args.judge_model

    queries = _load_queries()
    if args.limit:
        queries = queries[:args.limit]
    conditions = ["literal", "no_intent", "tom"]
    print(f"Backbone: {args.answer_model} @ {answer_base}")
    print(f"Judge:    {args.judge_model} @ {judge_base}")
    print(f"Running {len(queries)} queries x {len(conditions)} conditions x {len(args.seeds)} seeds = "
          f"{len(queries) * len(conditions) * len(args.seeds)} answer runs "
          f"+ {len(queries) * len(conditions) * len(args.seeds)} judge calls")

    results: Dict[str, Any] = {
        "meta": {
            "scene": SCENE_SUMMARY,
            "configured_budget": CONFIGURED_BUDGET,
            "seeds": list(args.seeds),
            "model": args.answer_model,
            "answer_base_url": answer_base,
            "judge_model": args.judge_model,
        },
        "per_seed": {str(s): {c: [] for c in conditions} for s in args.seeds},
    }

    t_start = time.time()
    for si, seed in enumerate(args.seeds):
        print(f"\n=== seed {seed} ({si + 1}/{len(args.seeds)}) ===")
        for qi, q in enumerate(queries, 1):
            subject = q.get("agent_subject") or list(PRIVATE_STATE.keys())[0]
            private = dict(PRIVATE_STATE.get(subject, {}))
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
                    resp = run_one_condition(answer_client, cond, q["query"])
                except Exception as e:
                    resp = {"answer": "", "probes": 0, "latency": 0.0, "gap": None,
                            "error": f"{type(e).__name__}: {e}"}

                try:
                    score = judge_response(
                        judge_client, q["query"], q.get("implicit_need", ""),
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
        for ca, cb in [("tom", "literal"), ("tom", "no_intent"), ("no_intent", "literal")]:
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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[write] {out}")
    print(f"Elapsed: {results['meta']['elapsed_sec']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
