"""Pull every headline number we cite in the paper from its source JSON.

Print one canonical value per claim, with the file, key-path, and the seeds
that produced it. The paper editor should grep these against main.tex and
appendix_experiments.tex to confirm there is no stale number left over.

This is read-only — does not modify any results.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict

try:
    from scipy import stats
except Exception:  # pragma: no cover - audit script fallback
    stats = None

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "evaluation" / "results"


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def fmt(v, fmt="{:.3f}"):
    if v is None:
        return "—"
    if isinstance(v, float):
        return fmt.format(v)
    return str(v)


def paired_t_from_query_maps(test_map: dict, ref_map: dict, metric: str) -> dict:
    qids = sorted(set(test_map) & set(ref_map))
    diffs = [test_map[q][metric] - ref_map[q][metric] for q in qids]
    if not diffs:
        return {"n": 0, "delta": None, "t": None, "p": None}
    delta = mean(diffs)
    if len(set(round(d, 12) for d in diffs)) <= 1:
        if abs(delta) < 1e-12:
            t_val, p_val = 0.0, None
        else:
            t_val, p_val = math.inf, 0.0
    elif stats is not None:
        t_val, p_val = stats.ttest_rel(
            [test_map[q][metric] for q in qids],
            [ref_map[q][metric] for q in qids],
        )
    else:
        sd = stdev(diffs)
        t_val = delta / (sd / math.sqrt(len(diffs)))
        p_val = None
    return {"n": len(qids), "delta": delta, "t": t_val, "p": p_val}


def privacy_query_maps(d: dict) -> dict:
    per_cond: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for seed_block in d.get("per_seed", {}).values():
        for cond, block in seed_block.items():
            for r in block.get("details", []):
                if not isinstance(r.get("factual_accuracy"), dict):
                    continue
                qid = r["query_id"]
                per_cond[cond][qid]["fa"].append(r["factual_accuracy"]["accuracy"])
                per_cond[cond][qid]["viol"].append(r.get("privacy_violation", 0))
                per_cond[cond][qid]["probes"].append(r.get("tool_calls", 0))
    out = {}
    for cond, qmap in per_cond.items():
        out[cond] = {
            qid: {metric: mean(vals) for metric, vals in metrics.items()}
            for qid, metrics in qmap.items()
        }
    return out


# =============================================================================
# RQ2 main multi-seed (5 conditions)
# =============================================================================
section("RQ2 main multi-seed (file: rq2_factual_accuracy_multiseed.json)")
d = load(RES / "rq2_factual_accuracy_multiseed.json")
seeds = sorted(d["per_seed"].keys())
print(f"seeds: {seeds}")

for cond in ["Vanilla_LLM", "Static_Context", "ReAct", "AURA_NoProbe", "AURA_Full"]:
    fa_per_seed = []
    cu_per_seed = []
    lat_per_seed = []
    for s in seeds:
        cd = d["per_seed"][s].get(cond, {})
        details = cd.get("details", [])
        fas = [r["factual_accuracy"]["accuracy"]
               for r in details
               if isinstance(r.get("factual_accuracy"), dict)
               and "accuracy" in r["factual_accuracy"]]
        if fas:
            fa_per_seed.append(mean(fas))
        cus = [r["context_utilization"]["utilization"]
               for r in details
               if isinstance(r.get("context_utilization"), dict)
               and "utilization" in r["context_utilization"]]
        if cus:
            cu_per_seed.append(mean(cus))
        lats = [r["latency"] for r in details if "latency" in r]
        if lats:
            lat_per_seed.append(mean(lats))
    if fa_per_seed:
        print(f"  {cond:18s}  FA = {mean(fa_per_seed):.4f} ± {stdev(fa_per_seed):.4f}    "
              f"CU = {mean(cu_per_seed):.3f}  Lat = {mean(lat_per_seed):.2f}s   "
              f"per-seed FA: {[round(x, 3) for x in fa_per_seed]}")


# =============================================================================
# RQ2 extra baselines (Reflexion + PnS)
# =============================================================================
section("RQ2 extra baselines (file: rq2_extra_baselines_multiseed.json)")
de = load(RES / "rq2_extra_baselines_multiseed.json")
print(f"seeds: {de.get('seeds')}")
for cond in ["Reflexion", "Plan_and_Solve"]:
    mss = de.get("multi_seed_summary", {}).get(cond)
    if mss:
        print(f"  {cond:18s}  FA = {mss['fa_mean']:.4f} ± {mss['fa_std']:.4f}    "
              f"Lat = {mss['lat_mean']:.2f}s   per-seed FA: {mss['fa_per_seed']}")
print(f"wall_clock_s: {de.get('wall_clock_s')}")


# =============================================================================
# RQ2 GapRouted + Fixed-Probe Pareto controls
# =============================================================================
section("RQ2 GapRouted / Fixed-Probe Pareto controls")
dg_path = RES / "rq2_aura_gap_routed_multiseed.json"
df_path = RES / "rq2_fixed_probe_multiseed.json"
dp_path = RES / "rq2_pareto_analysis.json"
if dg_path.exists():
    dg = load(dg_path)
    s = dg.get("multi_seed_summary", {}).get("AURA_GapRouted", {})
    print(
        f"  AURA_GapRouted  FA = {s.get('fa_mean')} ± {s.get('fa_std')}    "
        f"probes = {s.get('probes_mean')}  gap = {s.get('gap_mean')}  "
        f"Lat = {s.get('lat_mean')}s   per-seed FA: {s.get('fa_per_seed')}"
    )
if df_path.exists():
    df = load(df_path)
    s = df.get("multi_seed_summary", {}).get("Fixed_Probe", {})
    print(
        f"  Fixed_Probe     FA = {s.get('fa_mean')} ± {s.get('fa_std')}    "
        f"Lat = {s.get('lat_mean')}s   per-seed FA: {s.get('fa_per_seed')}"
    )
if dp_path.exists():
    dp2 = load(dp_path)
    pc = dp2.get("per_condition", {})
    for cond in ["AURA_GapRouted", "Fixed_Probe", "Plan_and_Solve"]:
        if cond in pc:
            a = pc[cond]
            print(
                f"  Pareto {cond:16s} FA={a['fa_mean']:.4f} "
                f"probes={a['probes_mean']:.2f} disclosure={a['disclosure_mean']:.2f} "
                f"lat={a['latency_mean']:.2f}s n={a['n_queries']}"
            )


# =============================================================================
# RQ2 privacy-sensitive distractor slice
# =============================================================================
section("RQ2 privacy-sensitive distractor slice (file: rq2_privacy_distractor_multiseed.json)")
dpriv_path = RES / "rq2_privacy_distractor_multiseed.json"
if dpriv_path.exists():
    dpriv = load(dpriv_path)
    print(f"seeds: {dpriv.get('seeds')}  n_queries={dpriv.get('n_queries')}")
    for cond in ["Fixed_Probe", "AURA_GapRouted", "Plan_and_Solve", "ReAct", "Static_Context", "Vanilla_LLM"]:
        s = dpriv.get("multi_seed_summary", {}).get(cond)
        if not s:
            continue
        print(
            f"  {cond:16s} FA={s['fa_mean']:.4f}±{s['fa_std']:.4f}  "
            f"viol={s['viol_rate_mean']:.4f}±{s['viol_rate_std']:.4f}  "
            f"probes={s['probes_mean']:.2f}  lat={s['lat_mean']:.2f}s"
        )
    qmaps = privacy_query_maps(dpriv)
    if "AURA_GapRouted" in qmaps:
        print("  paired vs AURA_GapRouted (query-level, mean over seeds):")
        for cond in ["Fixed_Probe", "Plan_and_Solve", "ReAct", "Static_Context"]:
            if cond not in qmaps:
                continue
            fa = paired_t_from_query_maps(qmaps[cond], qmaps["AURA_GapRouted"], "fa")
            viol = paired_t_from_query_maps(qmaps[cond], qmaps["AURA_GapRouted"], "viol")
            print(
                f"    {cond:16s} ΔFA={fa['delta']:+.4f} p={fa['p']}  "
                f"Δviol={viol['delta']:+.4f} p={viol['p']}"
            )


# =============================================================================
# RQ2 strict rescore
# =============================================================================
section("RQ2 strict precision rescore (file: rq2_strict_rescore.json)")
ds = load(RES / "rq2_strict_rescore.json")
for cond, agg in ds.get("per_condition", {}).items():
    print(f"  {cond:18s}  StrictP = {agg['strict_precision_mean']:.3f}    "
          f"Halluc% = {agg['hallucination_rate']*100:.1f}    "
          f"Perfect% = {agg['perfect_rate']*100:.1f}    "
          f"LenientFA = {agg['lenient_accuracy_mean']:.3f}")
print(f"\nPaired contrasts (AURA_Full vs ...):")
for c in ds.get("paired_contrasts_AURA_Full_vs", []):
    print(f"  vs {c['vs']:18s}  ΔStrictP = {c['strict_precision_delta']:+.3f}  p={c['strict_precision_p']:.4f}")


# =============================================================================
# RQ2 unified 7-condition
# =============================================================================
section("RQ2 unified 7-cond (file: rq2_unified_multiseed.json)")
du = load(RES / "rq2_unified_multiseed.json")
for cond, agg in du.get("per_condition", {}).items():
    print(f"  {cond:18s}  StrictP = {agg['strict_precision_mean']:.3f}    "
          f"LenientFA = {agg['lenient_FA_mean']:.3f}    "
          f"Lat = {agg['mean_latency_s']:.2f}s    n_cells = {agg['n_cells']}")
for c in du.get("paired_AURA_Full_vs", []):
    print(f"  AURA_Full vs {c['vs']:18s}  ΔStrictP={c['delta_strict_p']:+.3f}, p={c['p_strict']:.4f}    "
          f"ΔLenient={c['delta_lenient_FA']:+.3f}, p={c['p_lenient']:.4f}    n_pairs={c['n_pairs_strict']}")


# =============================================================================
# RQ-Intent main (3 conditions: literal / no_intent / tom)
# =============================================================================
section("RQ-Intent main (file: rq_intent_clean_prompt_multiseed.json)")
# Clean prompt is the canonical final system. The older
# rq2_implicit_intent_v2_with_second_order.json is retained only as the
# leaked-prompt audit row in the prompt-ablation table.
dr = load(RES / "rq_intent_clean_prompt_multiseed.json")
seeds_r = sorted(dr.get("per_seed", {}).keys())
print(f"seeds: {seeds_r}")
for cond in ["literal", "no_intent", "tom"]:
    lit_per_seed, imp_per_seed, lat_per_seed, probes_per_seed = [], [], [], []
    for s in seeds_r:
        entries = dr["per_seed"][s].get(cond, [])
        lits = [e["literal_score"] for e in entries if e.get("literal_score") is not None]
        imps = [e["implicit_score"] for e in entries if e.get("implicit_score") is not None]
        lats = [e.get("latency", 0) for e in entries if e.get("latency") is not None]
        probes = [
            e.get("probes", e.get("probes_called", 0))
            for e in entries
            if e.get("probes", e.get("probes_called")) is not None
        ]
        if lits:
            lit_per_seed.append(mean(lits))
        if imps:
            imp_per_seed.append(mean(imps))
        if lats:
            lat_per_seed.append(mean(lats))
        if probes:
            probes_per_seed.append(mean(probes))
    if imp_per_seed:
        probes_str = f"{mean(probes_per_seed):.2f}" if probes_per_seed else "—"
        lat_str = f"{mean(lat_per_seed):.2f}s" if lat_per_seed else "—"
        print(f"  {cond:14s}  literal={mean(lit_per_seed):.3f}  implicit={mean(imp_per_seed):.3f}    "
              f"probes={probes_str}  lat={lat_str}   "
              f"per-seed implicit: {[round(x, 3) for x in imp_per_seed]}")


# =============================================================================
# RQ-Intent v2 expanded cross-scene check
# =============================================================================
section("RQ-Intent v2 expanded check (file: rq_intent_v2_multiseed.json)")
dv2_path = RES / "rq_intent_v2_multiseed.json"
if dv2_path.exists():
    dv2 = load(dv2_path)
    stats_v2 = dv2.get("statistics", {})
    for cond in ["literal", "no_intent", "tom"]:
        s = stats_v2.get(cond, {})
        print(
            f"  {cond:14s} literal={s.get('literal_score', {}).get('mean'):.3f}  "
            f"implicit={s.get('implicit_score', {}).get('mean'):.3f}  "
            f"probes={s.get('probes', {}).get('mean'):.2f}  "
            f"viol={s.get('forbidden_violation_rate', {}).get('mean'):.3f}  "
            f"per-seed implicit={[round(x, 3) for x in s.get('implicit_score', {}).get('per_seed', [])]}"
        )
    pt_v2 = dv2.get("paired_tests", {}).get("overall", {}).get("implicit_score", {})
    for key, vals in pt_v2.items():
        print(
            f"  paired {key:<22} n={vals.get('n_pairs')} "
            f"Δ={vals.get('mean_delta')} t={vals.get('t')} "
            f"p={vals.get('p_two_sided')}"
        )
    print("  by-scene tom_vs_no_intent:")
    for scene, block in dv2.get("paired_tests", {}).get("by_scene", {}).items():
        vals = block.get("implicit_score", {}).get("tom_vs_no_intent", {})
        print(
            f"    {scene:<24} n={vals.get('n_pairs')} "
            f"Δ={vals.get('mean_delta')} p={vals.get('p_two_sided')}"
        )
    print("  by-subcategory tom_vs_no_intent:")
    for cat, block in dv2.get("paired_tests", {}).get("by_subcategory", {}).items():
        vals = block.get("implicit_score", {}).get("tom_vs_no_intent", {})
        print(
            f"    {cat:<18} n={vals.get('n_pairs')} "
            f"Δ={vals.get('mean_delta')} p={vals.get('p_two_sided')}"
        )
else:
    print(f"  [missing] {dv2_path}")


# =============================================================================
# RQ-Intent prompt ablation
# =============================================================================
section("RQ-Intent prompt ablation (leaked / clean / no-few-shot)")
for label, fname in [
    ("Leaked few-shot", "rq2_implicit_intent_v2_with_second_order.json"),
    ("Clean few-shot", "rq_intent_clean_prompt_multiseed.json"),
    ("No few-shot", "rq_intent_no_fewshot_multiseed.json"),
]:
    dpa = load(RES / fname)
    stats_pa = dpa.get("statistics", {})
    def m(cond: str, key: str) -> float:
        vals = stats_pa.get(cond, {}).get(key, [])
        return mean(vals) if vals else float("nan")
    pt = dpa.get("paired_tests", {}).get("implicit_score", {}).get("tom_vs_no_intent", {})
    print(
        f"  {label:17s}  Intent={m('tom', 'implicit_score'):.3f}  "
        f"NoIntent={m('no_intent', 'implicit_score'):.3f}  "
        f"Literal={m('literal', 'implicit_score'):.3f}  "
        f"cell Δ={pt.get('mean_delta')} p={pt.get('p_two_sided')}  "
        f"file={fname}"
    )


# =============================================================================
# RQ-Intent + Plan-and-Solve
# =============================================================================
section("RQ-Intent Plan-and-Solve (file: rq_intent_pns_multiseed.json)")
dp = load(RES / "rq_intent_pns_multiseed.json")
mss = dp.get("multi_seed_summary", {})
print(f"  Plan_and_Solve  literal = {mss.get('literal_score_mean')} ± {mss.get('literal_score_std')}")
print(f"  Plan_and_Solve  implicit = {mss.get('implicit_score_mean')} ± {mss.get('implicit_score_std')}    "
      f"per-seed: {mss.get('implicit_per_seed')}")
print(f"  mean_latency: {mss.get('mean_latency')}s   wall_clock: {dp.get('wall_clock_s')}s")


# =============================================================================
# RQ-Intent oracle / fixed-private controls
# =============================================================================
section("RQ-Intent oracle/fixed controls (file: rq_intent_oracle_fixed_multiseed.json)")
dc = load(RES / "rq_intent_oracle_fixed_multiseed.json")
for cond in ["fixed_probe", "oracle_intent"]:
    s = dc.get("statistics", {}).get(cond, {})
    print(
        f"  {cond:14s} literal={mean(s.get('literal_score', [])):.3f}  "
        f"implicit={mean(s.get('implicit_score', [])):.3f}  "
        f"probes={mean(s.get('probes', [])):.2f}  "
        f"lat={mean(s.get('latency', [])):.2f}s  "
        f"per-seed implicit={[round(x, 3) for x in s.get('implicit_score', [])]}"
    )
for metric, pairs in dc.get("paired_tests", {}).items():
    print(f"  paired {metric}:")
    for key, vals in pairs.items():
        print(
            f"    {key:<30} n={vals.get('n_pairs')} "
            f"Δ={vals.get('mean_delta')} t={vals.get('t')} "
            f"p={vals.get('p_two_sided')}"
        )


# =============================================================================
# RQ-Intent IAA
# =============================================================================
section("RQ-Intent IAA (file: iaa_implicit_intent.json)")
iaa_path = RES / "iaa_implicit_intent.json"
if iaa_path.exists():
    di = load(iaa_path)
    print(f"  n_queries={di.get('n_queries')}  annotators={di.get('annotators')}")
    for pair, vals in di.get("pairwise_kappa", {}).items():
        print(
            f"  human-human {pair}: κ={vals.get('kappa'):+.4f} "
            f"agree={vals.get('agree')}/{vals.get('n')} "
            f"p_o={vals.get('p_o'):.3f}"
        )
    if di.get("vs_gold_kappa"):
        avg_gold = mean(v["kappa"] for v in di["vs_gold_kappa"].values())
        print(f"  annotator-vs-gold avg κ={avg_gold:+.4f}")
        for name, vals in di["vs_gold_kappa"].items():
            print(
                f"    {name}: κ={vals.get('kappa'):+.4f} "
                f"agree={vals.get('agree')}/{vals.get('n')}"
            )


# =============================================================================
# Cross-backbone (single seed)
# =============================================================================
section("Cross-backbone single-seed (files: rq2_implicit_intent_*_seed42.json)")
for backbone, fname in [
    ("gpt-4o-mini (multi-seed mean above)", None),
    ("claude-haiku-4-5", "rq2_implicit_intent_claudehaiku45_seed42.json"),
    ("qwen-plus", "rq2_implicit_intent_qwenplus_seed42.json"),
    ("gemini-2.5-flash", "rq2_implicit_intent_gemini25flash_seed42.json"),
]:
    if fname is None:
        continue
    p = RES / fname
    if not p.exists():
        print(f"  {backbone}: FILE MISSING {p}")
        continue
    d = load(p)
    # Best-effort: locate condition-level summaries
    found = False
    for cond_key in ["tom", "no_intent"]:
        # Some files store per_seed structure, some store flat list
        if "per_seed" in d:
            for s, conds in d["per_seed"].items():
                entries = conds.get(cond_key, [])
                imps = [e.get("implicit_score") for e in entries
                        if e.get("implicit_score") is not None]
                if imps:
                    print(f"  {backbone} {cond_key} (seed {s}): "
                          f"implicit = {mean(imps):.3f}  n={len(imps)}")
                    found = True
        elif cond_key in d:
            entries = d[cond_key]
            imps = [e.get("implicit_score") for e in entries
                    if e.get("implicit_score") is not None]
            if imps:
                print(f"  {backbone} {cond_key}: implicit = {mean(imps):.3f}  n={len(imps)}")
                found = True
    if not found:
        print(f"  {backbone}: structure unknown — top keys = {list(d.keys())[:5]}")


# =============================================================================
# FANToM 400Q
# =============================================================================
section("FANToM 400Q (file: fantom_full_seed42.json)")
df = load(RES / "fantom_full_seed42.json")
meta = df.get("meta", {})
print(f"  n_questions={meta.get('n_questions')}  model={meta.get('model')}  "
      f"seed={meta.get('seed')}  cost=${meta.get('total_cost_usd')}  "
      f"wall_clock={round(meta.get('wall_clock_s', 0)/60, 1)} min")
for cond in ["literal", "no_intent", "intent"]:
    cd = df.get("per_condition", {}).get(cond, {})
    print(f"  {cond:12s}  acc = {cd.get('acc'):.4f}  "
          f"fallback = {cd.get('fallback_rate', 0):.3f}  "
          f"lat = {cd.get('mean_latency'):.2f}s  n = {cd.get('n_items')}")
print(f"\n  Stats:")
for k, v in df.get("stats", {}).items():
    print(f"    {k}: ΔAcc = {v.get('delta_acc'):+.4f}, "
          f"paired-t p = {v['paired_t']['p']:.4g}, "
          f"McNemar p = {v['mcnemar']['p']:.4g}")


# =============================================================================
# LoCoMo 200Q
# =============================================================================
section("LoCoMo 200Q (file: locomo_smoke.json)")
dl = load(RES / "locomo_smoke.json")
meta = dl.get("meta", {})
print(f"  n_questions={meta.get('n_questions')}  model={meta.get('model')}  "
      f"seed={meta.get('seed')}  cost=${meta.get('total_cost_usd')}  "
      f"wall_clock={round(meta.get('wall_clock_s', 0)/60, 1)} min")
for cond in ["literal", "no_intent", "intent"]:
    cd = dl.get("per_condition", {}).get(cond, {})
    print(f"  {cond:12s}  F1 = {cd.get('f1'):.4f}  EM = {cd.get('em'):.4f}  "
          f"fallback = {cd.get('fallback_rate', 0):.3f}  "
          f"lat = {cd.get('mean_latency'):.2f}s  n = {cd.get('n_items')}")
print(f"\n  Stats:")
for k, v in dl.get("stats", {}).items():
    print(f"    {k}: ΔF1 = {v.get('delta_f1'):+.4f}, "
          f"paired-t p = {v['paired_t_on_f1']['p']:.4g}")


# =============================================================================
# RQ5 rater-aggregated
# =============================================================================
section("RQ5 rater-aggregated (file: rq5_rater_aggregated.json)")
d5 = load(RES / "rq5_rater_aggregated.json")
print(f"  n_raters={d5['n_raters']}")
for dim, agg in d5.get("per_dimension", {}).items():
    print(f"  {dim:28s}  Δ̄ = {agg['delta_mean_of_means']:+.3f}    "
          f"d_z = {agg['cohens_dz_rater_level']:+.2f}    "
          f"Wilcoxon p = {agg['wilcoxon_p_rater']:.4f}    "
          f"CI = [{agg['cluster_bootstrap_lo95']:+.3f}, {agg['cluster_bootstrap_hi95']:+.3f}]")


# =============================================================================
# RQ3 multi-seed ablation
# =============================================================================
section("RQ3 multi-seed ablation (file: rq3_ablation_study_multiseed.json)")
d3 = load(RES / "rq3_ablation_study_multiseed.json")
seeds_3 = sorted(d3.get("per_seed", {}).keys())
print(f"  seeds: {seeds_3}")
agg = defaultdict(lambda: {"ga": [], "fa": [], "lat": []})
for s in seeds_3:
    for c, v in d3["per_seed"][s].items():
        if isinstance(v, dict) and "avg_ga" in v:
            agg[c]["ga"].append(v["avg_ga"])
            agg[c]["fa"].append(v["avg_fa"])
            agg[c]["lat"].append(v["avg_latency"])
for c, m in agg.items():
    if m["ga"]:
        print(f"  {c:25s}  GA = {mean(m['ga']):.4f}  FA = {mean(m['fa']):.4f}  Lat = {mean(m['lat']):.2f}s")


# =============================================================================
# RQ6 probe budget Pareto
# =============================================================================
section("RQ6 probe budget multiseed (file: rq6_probe_budget_pareto_multiseed.json)")
d6 = load(RES / "rq6_probe_budget_pareto_multiseed.json")
# best-effort; structure may be different
for k in ["multi_seed_summary", "summary", "pareto"]:
    if k in d6:
        print(f"  [{k}]:")
        v = d6[k]
        if isinstance(v, dict):
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        break
else:
    print(f"  top keys: {list(d6.keys())[:8]}")


print(f"\n\n{'=' * 80}\nDone. Cross-check these numbers against paper text.\n{'=' * 80}")
