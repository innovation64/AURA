"""Cross-backbone v2 robustness check.

Re-runs the 100-query, four-scene RQ-Intent v2 benchmark on a different
answer-backbone while keeping the judge fixed on gpt-4o-mini for
cross-backbone score comparability. Mirrors the v1 cross-backbone
protocol (`run_implicit_intent_backbone.py`) but uses the v2 scene-keyed
loader so the cross-scene robustness claim is testable on a second
model family.

Default protocol mirrors v1 cross-backbone diagnostics: 3 conditions
(literal / no_intent / tom) on 1 seed (42). Use --seeds to widen.

Usage:
  python scripts/run_implicit_intent_v2_backbone.py \\
    --answer-model anthropic/claude-3.5-haiku \\
    --answer-base-url https://openrouter.ai/api/v1 \\
    --answer-api-env OPENROUTER_API_KEY \\
    --seeds 42 \\
    --out evaluation/results/rq_intent_v2_claudehaiku45_seed42.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_AURA_SRC = ROOT / "AURA" / "src"
if str(_AURA_SRC) not in sys.path:
    sys.path.insert(0, str(_AURA_SRC))

from openai import OpenAI

import scripts.run_implicit_intent_smoke as smoke
import scripts.run_implicit_intent_v2 as v2

CONFIGURED_BUDGET = 3
DEFAULT_QUERY_FILE = ROOT / "evaluation" / "data" / "implicit_intent_queries_v2.json"
DEFAULT_CONDITIONS = ["literal", "no_intent", "tom"]


def _condition_dispatch(
    answer_client: OpenAI, scene: Dict[str, Any], cond: str, q: Dict[str, Any]
) -> Dict[str, Any]:
    """Same surface as v2.run_one_condition but binds answer_client."""
    v2.CURRENT_TOOL_TRACE = []
    query_text = q["query"]
    if cond == "literal":
        ans, calls, dur = smoke._literal_answer(answer_client, query_text)
        observations = []
        gap = None
    elif cond == "no_intent":
        ans, calls, dur = smoke._react_answer(answer_client, query_text, CONFIGURED_BUDGET)
        observations = list(v2.CURRENT_TOOL_TRACE)
        gap = None
    elif cond == "tom":
        ans, calls, dur, gap, observations, _frame = v2._tom_routed_answer(
            answer_client, scene, q, CONFIGURED_BUDGET
        )
    elif cond == "fixed_probe":
        ans, calls, dur, observations = v2._fixed_probe_answer(
            answer_client, scene, q, CONFIGURED_BUDGET
        )
        gap = None
    elif cond == "oracle_intent":
        ans, calls, dur, observations = v2._oracle_intent_answer(
            answer_client, scene, q, CONFIGURED_BUDGET
        )
        gap = "oracle"
    else:
        raise ValueError(f"unknown condition {cond!r}")
    tools = [o["tool"] for o in observations]
    forbidden = set(q.get("forbidden_tools") or [])
    required = set(q.get("gold_required_tools") or [])
    return {
        "answer": ans,
        "probes": calls,
        "latency": round(dur, 2),
        "gap": gap,
        "tools_called": tools,
        "forbidden_violations": sorted(set(tools) & forbidden),
        "gold_tools_hit": sorted(set(tools) & required),
    }


def _paired_t(diffs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Same as v2.paired_t but local so we can import without re-running its main."""
    valid = [d for d in diffs if d is not None and not math.isnan(d)]
    if len(valid) < 2:
        return None, None
    md = statistics.mean(valid)
    sd = statistics.stdev(valid) if len(set(valid)) > 1 else 0.0
    if sd == 0:
        return (float("inf") if md != 0 else 0.0), None
    try:
        from scipy import stats as scst
    except Exception:
        return None, None
    n = len(valid)
    t = md / (sd / math.sqrt(n))
    p = 2 * (1 - scst.t.cdf(abs(t), df=n - 1))
    return float(t), float(p)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--query-file", default=str(DEFAULT_QUERY_FILE))
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--conditions", nargs="+", default=None)
    p.add_argument("--scenes", nargs="+", default=None,
                   help="Limit to specific scene keys (smoke/diagnostic)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit total queries (smoke)")
    p.add_argument("--answer-model", required=True,
                   help="Answer-model name (e.g. anthropic/claude-3.5-haiku for OpenRouter)")
    p.add_argument("--answer-base-url", default=None)
    p.add_argument("--answer-api-env", default="OPENROUTER_API_KEY")
    p.add_argument("--judge-model", default="gpt-4o-mini")
    p.add_argument("--judge-base-url", default=None)
    p.add_argument("--judge-api-env", default="OPENAI_API_KEY")
    p.add_argument("--out", required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    answer_key = os.environ.get(args.answer_api_env, "")
    judge_key = os.environ.get(args.judge_api_env, "")
    if not answer_key:
        print(f"ERR: {args.answer_api_env} not set", file=sys.stderr)
        return 2
    if not judge_key:
        print(f"ERR: {args.judge_api_env} not set", file=sys.stderr)
        return 2

    answer_base = args.answer_base_url or os.environ.get("OPENROUTER_BASE_URL") or "https://api.openai.com/v1"
    judge_base = args.judge_base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    answer_client = OpenAI(api_key=answer_key, base_url=answer_base)
    judge_client = OpenAI(api_key=judge_key, base_url=judge_base)

    # Point smoke / v2 internals at the answer backbone for the answering
    # path; judge stays on its own client and model via judge_response.
    smoke.set_backbone_model(args.answer_model)
    v2.JUDGE_MODEL = args.judge_model

    spec = json.load(open(args.query_file))
    errors = v2.validate_benchmark(spec)
    if errors:
        print("ERR: benchmark validation failed", file=sys.stderr)
        for err in errors[:20]:
            print(f"  - {err}", file=sys.stderr)
        return 2
    queries = v2._filter_queries(spec["queries"], args.scenes, None, args.limit)
    conditions = args.conditions or DEFAULT_CONDITIONS

    print(f"[setup] {len(queries)} queries x {len(conditions)} conditions x {len(args.seeds)} seeds = "
          f"{len(queries) * len(conditions) * len(args.seeds)} answer runs + judge calls")
    print(f"[setup] answer-model: {args.answer_model} via {answer_base}")
    print(f"[setup] judge-model: {args.judge_model} via {judge_base}")

    out_path = Path(args.out)
    if args.resume and out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        results.setdefault("per_seed", {})
        for seed in args.seeds:
            results["per_seed"].setdefault(str(seed), {})
            for cond in conditions:
                results["per_seed"][str(seed)].setdefault(cond, [])
        print(f"[resume] loaded partial output from {out_path}")
    else:
        results = {
            "meta": {
                "benchmark": "implicit_intent_v2",
                "benchmark_version": spec.get("version"),
                "configured_budget": CONFIGURED_BUDGET,
                "seeds": list(args.seeds),
                "conditions": list(conditions),
                "answer_model": args.answer_model,
                "answer_base_url": answer_base,
                "judge_model": args.judge_model,
                "intent_prompt_variant": os.environ.get("AURA_INTENT_PROMPT_VARIANT", "clean"),
                "selected_scenes": args.scenes,
            },
            "per_seed": {str(seed): {c: [] for c in conditions} for seed in args.seeds},
        }

    t_start = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for si, seed in enumerate(args.seeds, 1):
        print(f"\n=== seed {seed} ({si}/{len(args.seeds)}) ===")
        for qi, q in enumerate(queries, 1):
            sid = q["scene"]
            scene = spec["scenes"][sid]
            v2.apply_scene(scene)
            subject, private = v2._private_truth_for_judge(scene, q)
            print(f"  [{qi}/{len(queries)}] {sid} id={q['id']:<4} {q['subcategory']:<17} {q['query'][:60]}")

            for cond in conditions:
                done = {(r.get("scene"), r.get("query_id")) for r in results["per_seed"][str(seed)].get(cond, [])}
                if (sid, q["id"]) in done:
                    print(f"    [{cond:<13}] skip existing")
                    continue
                t_run = time.time()
                try:
                    resp = _condition_dispatch(answer_client, scene, cond, q)
                except Exception as e:
                    resp = {
                        "answer": "", "probes": 0, "latency": 0.0, "gap": None,
                        "tools_called": [], "forbidden_violations": [], "gold_tools_hit": [],
                        "error": f"{type(e).__name__}: {e}",
                    }

                try:
                    score = v2.judge_response(
                        judge_client, scene, q["query"], q.get("implicit_need", ""),
                        resp.get("answer", ""), private,
                    )
                except Exception as e:
                    score = {"literal_score": None, "implicit_score": None,
                             "rationale": f"judge error: {type(e).__name__}: {e}"}

                row = {
                    "query_id": q["id"],
                    "scene": sid,
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
                print(f"    [{cond:<13}] probes={resp['probes']}{gstr}  "
                      f"lit={score.get('literal_score')} imp={score.get('implicit_score')}{err_s}")

            if qi % 5 == 0 or qi == len(queries):
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  [seed {seed} saved] → {out_path}")

    print("\n=== Aggregation ===")
    for cond in conditions:
        rows_all = []
        for seed in args.seeds:
            rows_all.extend(results["per_seed"][str(seed)].get(cond, []))
        if not rows_all:
            continue
        imps = [r["implicit_score"] for r in rows_all if r.get("implicit_score") is not None]
        lits = [r["literal_score"] for r in rows_all if r.get("literal_score") is not None]
        probes = [r.get("probes", 0) for r in rows_all]
        print(f"  {cond:<13}  literal={statistics.mean(lits):.3f} ({len(lits)})  "
              f"implicit={statistics.mean(imps):.3f} ({len(imps)})  "
              f"probes_mean={statistics.mean(probes):.2f}")

    # Headline contrast: tom vs no_intent paired by query_id × seed
    if "tom" in conditions and "no_intent" in conditions:
        diffs: List[float] = []
        for seed in args.seeds:
            tom_map = {(r["scene"], r["query_id"]): r for r in results["per_seed"][str(seed)]["tom"]}
            no_map = {(r["scene"], r["query_id"]): r for r in results["per_seed"][str(seed)]["no_intent"]}
            for key, t in tom_map.items():
                n = no_map.get(key)
                if not n:
                    continue
                if t.get("implicit_score") is None or n.get("implicit_score") is None:
                    continue
                diffs.append(t["implicit_score"] - n["implicit_score"])
        if diffs:
            t_stat, p_val = _paired_t(diffs)
            print(f"\n  tom vs no_intent paired cell-level n={len(diffs)}  "
                  f"Δ={statistics.mean(diffs):+.4f}  t={t_stat}  p={p_val}")
            results["tom_vs_no_intent_overall"] = {
                "n": len(diffs),
                "delta": statistics.mean(diffs),
                "t": t_stat,
                "p": p_val,
            }

        # Per-scene breakdown
        print("\n=== Per-scene tom vs no_intent ===")
        scene_diffs: Dict[str, List[float]] = defaultdict(list)
        for seed in args.seeds:
            tom_map = {(r["scene"], r["query_id"]): r for r in results["per_seed"][str(seed)]["tom"]}
            no_map = {(r["scene"], r["query_id"]): r for r in results["per_seed"][str(seed)]["no_intent"]}
            for key, t in tom_map.items():
                n = no_map.get(key)
                if not n:
                    continue
                ti, ni = t.get("implicit_score"), n.get("implicit_score")
                if ti is None or ni is None:
                    continue
                scene_diffs[key[0]].append(ti - ni)
        for sid in sorted(scene_diffs):
            d = scene_diffs[sid]
            tt, pp = _paired_t(d)
            print(f"  {sid:<22}  n={len(d):>3}  Δ={statistics.mean(d):+.4f}  t={tt}  p={pp}")

    results["meta"]["elapsed_sec"] = round(time.time() - t_start, 1)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[write] {out_path}")
    print(f"Elapsed: {results['meta']['elapsed_sec']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
