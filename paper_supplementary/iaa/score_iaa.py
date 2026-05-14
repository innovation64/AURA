"""Score IAA on the implicit-intent annotator returns.

Reads every iaa_*.json under evaluation/iaa_implicit_intent/responses/ and
the gold subcategory labels from the canonical query file. Computes:

  - Cohen's κ between every pair of annotators (categorical, 6-class)
  - Cohen's κ between each annotator and the gold subcategory label
  - Confusion matrices
  - Per-category accuracy

Output:
  evaluation/results/iaa_implicit_intent.json
  Console summary suitable for paste-into Limitations
"""

from __future__ import annotations

import json
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESPONSES_DIR = Path(__file__).parent / "responses"
QUERIES_PATH = ROOT / "evaluation" / "data" / "implicit_intent_queries.json"
OUTPUT_PATH = ROOT / "evaluation" / "results" / "iaa_implicit_intent.json"

# 6 categories the form asks for
CATS = ["availability", "mood", "appropriateness", "latent_goal", "second_order", "literal"]


def cohens_kappa(rater_a: list[str], rater_b: list[str], categories: list[str]) -> tuple[float, dict]:
    """Cohen's kappa for two equal-length categorical lists."""
    assert len(rater_a) == len(rater_b)
    n = len(rater_a)
    if n == 0:
        return 0.0, {}
    # observed agreement
    agree = sum(1 for a, b in zip(rater_a, rater_b) if a == b)
    p_o = agree / n
    # expected agreement under chance
    counts_a = Counter(rater_a)
    counts_b = Counter(rater_b)
    p_e = sum((counts_a.get(c, 0) / n) * (counts_b.get(c, 0) / n) for c in categories)
    if p_e >= 1.0:
        return 1.0, {"agree": agree, "n": n, "p_o": p_o, "p_e": p_e}
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa, {"agree": agree, "n": n, "p_o": p_o, "p_e": p_e}


def load_gold() -> dict[int, str]:
    with open(QUERIES_PATH) as f:
        d = json.load(f)
    queries = d.get("queries", d if isinstance(d, list) else [])
    return {int(q["id"]): q.get("subcategory", "?") for q in queries}


def load_responses() -> dict[str, dict[int, str]]:
    """{annotator_name: {query_id: category}}"""
    if not RESPONSES_DIR.exists():
        return {}
    out: dict[str, dict[int, str]] = {}
    for p in sorted(RESPONSES_DIR.glob("iaa_*.json")):
        with open(p) as f:
            d = json.load(f)
        name = d.get("annotator") or p.stem.replace("iaa_", "")
        ratings = {int(k): v for k, v in d.get("ratings", {}).items()}
        out[name] = ratings
    return out


def main() -> int:
    gold = load_gold()
    if not gold:
        print(f"ERROR: no queries in {QUERIES_PATH}")
        return 1

    responses = load_responses()
    if not responses:
        print(f"No annotator responses found in {RESPONSES_DIR}/")
        print(f"Drop iaa_<annotator>.json files there and rerun.")
        return 1

    qids = sorted(gold.keys())
    print(f"loaded {len(gold)} queries, {len(responses)} annotators: {sorted(responses.keys())}")

    # Build aligned rating arrays per annotator + gold
    aligned: dict[str, list[str]] = {}
    aligned["__gold__"] = [gold[qid] for qid in qids]
    missing: dict[str, list[int]] = {}
    for name, ratings in responses.items():
        aligned[name] = []
        miss = []
        for qid in qids:
            if qid in ratings:
                aligned[name].append(ratings[qid])
            else:
                aligned[name].append("__MISSING__")
                miss.append(qid)
        if miss:
            missing[name] = miss

    # Pairwise Cohen's kappa between every pair of human annotators
    print("\n=== Pairwise Cohen's κ (human ↔ human) ===")
    pairwise = {}
    for a, b in combinations(sorted(responses.keys()), 2):
        # exclude __MISSING__ rows
        a_arr, b_arr = [], []
        for ai, bi in zip(aligned[a], aligned[b]):
            if ai == "__MISSING__" or bi == "__MISSING__":
                continue
            a_arr.append(ai)
            b_arr.append(bi)
        kappa, info = cohens_kappa(a_arr, b_arr, CATS)
        pairwise[f"{a} ↔ {b}"] = {"kappa": round(kappa, 4), **info}
        print(f"  {a} ↔ {b}: κ = {kappa:+.3f}    "
              f"agree {info['agree']}/{info['n']}  ({info['p_o']*100:.1f}%)")

    # Each annotator vs gold
    print("\n=== Cohen's κ vs gold (annotator ↔ author labels) ===")
    vs_gold = {}
    for name in sorted(responses.keys()):
        a_arr, g_arr = [], []
        for ai, gi in zip(aligned[name], aligned["__gold__"]):
            if ai == "__MISSING__":
                continue
            a_arr.append(ai)
            g_arr.append(gi)
        kappa, info = cohens_kappa(a_arr, g_arr, CATS)
        vs_gold[name] = {"kappa": round(kappa, 4), **info}
        print(f"  {name} ↔ gold: κ = {kappa:+.3f}    "
              f"agree {info['agree']}/{info['n']}  ({info['p_o']*100:.1f}%)")

    # Per-category breakdown vs gold (which categories each annotator hits)
    print("\n=== Per-category agreement with gold ===")
    per_cat: dict[str, dict[str, dict]] = {}
    for name in sorted(responses.keys()):
        cat_counts: dict[str, dict] = {c: {"correct": 0, "total": 0} for c in CATS}
        for qid in qids:
            g = gold[qid]
            r = responses[name].get(qid)
            if r is None:
                continue
            cat_counts[g]["total"] += 1
            if r == g:
                cat_counts[g]["correct"] += 1
        per_cat[name] = cat_counts
        line = f"  {name}: "
        for c in CATS:
            cc = cat_counts[c]
            if cc["total"] > 0:
                line += f"{c}={cc['correct']}/{cc['total']}  "
        print(line)

    # Save
    output = {
        "schema_version": "1.0",
        "n_queries": len(gold),
        "annotators": sorted(responses.keys()),
        "missing": missing,
        "pairwise_kappa": pairwise,
        "vs_gold_kappa": vs_gold,
        "per_category_vs_gold": per_cat,
        "gold_subcategory_distribution": Counter(gold.values()),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=lambda o: dict(o) if isinstance(o, Counter) else str(o))
    print(f"\nSaved: {OUTPUT_PATH}")

    # Suggested paper sentence
    print("\n=== Suggested paper sentence ===")
    if pairwise:
        avg_pw = sum(v["kappa"] for v in pairwise.values()) / len(pairwise)
        avg_g = sum(v["kappa"] for v in vs_gold.values()) / len(vs_gold)
        print(f"  Inter-annotator agreement on the 25 implicit-intent queries' subcategory")
        print(f"  labels (6-class: availability / mood / appropriateness / latent_goal /")
        print(f"  second_order / literal) was Cohen's κ = {avg_pw:+.2f} between {len(responses)} independent")
        print(f"  annotators (range {min(v['kappa'] for v in pairwise.values()):+.2f}–{max(v['kappa'] for v in pairwise.values()):+.2f}); each annotator's labels matched")
        print(f"  the author gold at κ = {avg_g:+.2f} on average. We retain the author labels")
        print(f"  as gold but report this IAA in the limitations section.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
