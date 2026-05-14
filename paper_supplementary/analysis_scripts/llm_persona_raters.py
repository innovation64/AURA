"""LLM-simulated rater study: 14 diverse personas rate the same 50 scenarios
that the 6 human raters did. Treated as a robustness probe, NOT as a
substitute for human data.

Model choice rationale: AURA and Vanilla responses were both generated with
gpt-4o-mini, so using gpt-4o-mini as judge would be self-evaluation. We use
GPT-4o (same vendor, different model, much stronger) to reduce that bias.

Output: evaluation/results/llm_raters/<persona_id>.json — same shape as
human annotation files so the same analyzer can consume them.

Budget: 14 personas × 50 scenarios = 700 calls, ~$8 on GPT-4o.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from openai import OpenAI

ROOT      = Path(__file__).resolve().parent.parent
RESULTS   = ROOT / "evaluation" / "results"
SCEN_PATH = RESULTS / "human_eval_forms.json"
OUT_DIR   = RESULTS / "llm_raters"
OUT_DIR.mkdir(exist_ok=True)

DIMS = [
    "response_helpfulness",
    "environmental_awareness",
    "agent_believability",
    "factual_accuracy",
]

# 14 diverse personas. Each is a real-world archetype the AURA system might
# serve: mix of professions, ages, dispositions, and reading styles. The
# persona description seeds the rater's preferences without anchoring them
# on AURA-vs-Vanilla.
PERSONAS = [
    ("llm_software_engineer",
     "You are a 31-year-old software engineer at a mid-sized tech company. "
     "You value precision and concrete, factual responses; you find vague "
     "or hedge-heavy answers frustrating. You read carefully and you weight "
     "factual specificity over conversational warmth."),

    ("llm_school_teacher",
     "You are a 45-year-old elementary school teacher with 20 years of "
     "experience. You value warmth, clarity, and answers that meet the "
     "user where they are. You find overly clinical or dry answers off-putting "
     "and prefer responses that feel humane and grounded in real settings."),

    ("llm_retired_senior",
     "You are a 72-year-old retired civil engineer. You appreciate plain, "
     "respectful language and answers that don't talk down to you. You are "
     "skeptical of marketing-speak and excessively friendly tones; you "
     "prefer direct, substantive replies."),

    ("llm_medical_doctor",
     "You are a 38-year-old internal-medicine physician. You evaluate "
     "responses by the same standard as patient communication: factually "
     "rigorous, calibrated to uncertainty, and free of hallucinated detail. "
     "An answer that confidently invents specifics is worse to you than "
     "one that admits limited information."),

    ("llm_marketing_manager",
     "You are a 29-year-old marketing manager at a consumer-goods brand. "
     "You favor responses that feel engaging, narratively coherent, and "
     "rich in specific scene detail. A bland, generic reply is the worst "
     "kind in your book."),

    ("llm_freelance_artist",
     "You are a 34-year-old freelance illustrator. You appreciate atmospheric, "
     "sensory detail and dislike sterile, list-like answers. You read "
     "responses for whether they evoke the place and the people; "
     "factual hedging is fine if the texture is rich."),

    ("llm_undergraduate_student",
     "You are a 20-year-old undergraduate student. You read fast and prefer "
     "responses that get to the point. You don't like long flowery passages "
     "but you also penalize answers that say 'I don't know' when they could "
     "have inferred from context."),

    ("llm_corporate_lawyer",
     "You are a 41-year-old corporate lawyer. You evaluate responses by "
     "their precision and logical structure. You strongly penalize claims "
     "that aren't supported by the given context, but you also penalize "
     "evasive answers. Be even-handed; reward demonstrably correct content."),

    ("llm_investigative_journalist",
     "You are a 36-year-old investigative journalist. You are skeptical of "
     "any claim that could be unverifiable; you reward responses that "
     "carefully ground their assertions in named, checkable detail. You "
     "don't like hand-waving on either side."),

    ("llm_construction_foreman",
     "You are a 50-year-old construction foreman. You value short, useful, "
     "actionable replies. Long-winded answers irritate you regardless of "
     "their factual quality. Favor responses that say the useful thing first."),

    ("llm_clinical_therapist",
     "You are a 39-year-old licensed clinical therapist. You weight "
     "responses by emotional attunement and the impression of being "
     "genuinely heard. A factually sparse but humanly warm reply may rate "
     "above a colder factually-rich one."),

    ("llm_organic_farmer",
     "You are a 47-year-old organic farmer. You like grounded, plain-spoken "
     "replies that respect your time. You distrust language that sounds "
     "like it was written by a committee or a corporate AI assistant. "
     "Authenticity of voice matters a lot to you."),

    ("llm_michelin_chef",
     "You are a 44-year-old chef who runs a small fine-dining restaurant. "
     "You read responses for sensory specificity — colors, textures, named "
     "people and places. Generic responses bore you. But you also penalize "
     "fabricated specifics that would be embarrassing if checked."),

    ("llm_travel_writer",
     "You are a 32-year-old travel writer who has visited 60 countries. "
     "You value vivid, place-specific language and reward responses that "
     "evoke a real location. You penalize responses that could have been "
     "written about anywhere — they feel lazy. But you don't credit "
     "fabricated place names."),
]

ENV_BRIEF = (
    "AURATown is a small-town social simulation with 5 named characters and "
    "20 named locations:\n"
    "  Characters: Lin Wei (cafe owner), Zhang Hao (writer), Chen Mei "
    "(shop owner), Liu Yang (student), Wang Jun (retired professor).\n"
    "  Locations include: Sunrise Cafe, Tea House, Golden Wheat Bakery, "
    "Wellness Pharmacy, Chen's General Store, Art Gallery, Town Hall, "
    "Town Library, Temple, AURA Academy, Town Park, Town Square, Community "
    "Garden, Flower Garden, Riverside Walk, plus the five characters' homes.\n"
    "Each scenario contains a query a user (playing as one of the characters) "
    "asked an AI assistant; you see two anonymous responses, A and B."
)

JUDGE_INSTRUCTIONS = (
    "You will rate two responses on four dimensions. For each dimension and "
    "each side, output an integer 1 (poor) to 5 (excellent).\n\n"
    "  response_helpfulness   : Does it actually address the query?\n"
    "  environmental_awareness: Does it use AURATown specifics (named "
    "places/characters) vs being generic?\n"
    "  agent_believability    : Does the voice match the asking character's "
    "profile?\n"
    "  factual_accuracy       : Is it consistent with the AURATown setup? "
    "Penalise invented people/places.\n\n"
    "Rate from YOUR PERSONA's perspective consistent with your stated "
    "preferences. Do not split the difference; if you genuinely prefer one "
    "side strongly, reflect that. But also do not give 5/1 for everything; "
    "the scale exists.\n\n"
    "Output strictly this JSON object, no prose:\n"
    "{\n"
    '  "a_response_helpfulness": <1-5>,\n'
    '  "a_environmental_awareness": <1-5>,\n'
    '  "a_agent_believability": <1-5>,\n'
    '  "a_factual_accuracy": <1-5>,\n'
    '  "b_response_helpfulness": <1-5>,\n'
    '  "b_environmental_awareness": <1-5>,\n'
    '  "b_agent_believability": <1-5>,\n'
    '  "b_factual_accuracy": <1-5>\n'
    "}"
)


def rate_one(client: OpenAI, model: str, persona_prompt: str,
             scenario: Dict[str, Any]) -> Dict[str, int]:
    """Rate one scenario from one persona's POV. Returns 8 integer scores."""
    system = persona_prompt + "\n\n" + ENV_BRIEF + "\n\n" + JUDGE_INSTRUCTIONS
    user = (
        f"Query: {scenario['query']}\n"
        f"Asking character: {scenario['agent']}\n"
        f"Category: {scenario['category']}\n\n"
        f"Response A:\n{scenario['response_a']}\n\n"
        f"Response B:\n{scenario['response_b']}\n\n"
        "Output the JSON now."
    )
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.4,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            raw = (r.choices[0].message.content or "").strip()
            obj = json.loads(raw)
            out: Dict[str, int] = {}
            for side in ("a", "b"):
                for d in DIMS:
                    k = f"{side}_{d}"
                    v = obj.get(k)
                    if v is None:
                        raise ValueError(f"missing {k}")
                    iv = int(round(float(v)))
                    if iv < 1 or iv > 5:
                        raise ValueError(f"out-of-range {k}={iv}")
                    out[k] = iv
            return out
        except Exception as e:
            if attempt == 2:
                print(f"      WARN: rate failed: {type(e).__name__}: {e}")
                return {}
            time.sleep(1.5 * (attempt + 1))
    return {}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--limit-personas", type=int, default=None,
                   help="For testing, limit to first N personas")
    p.add_argument("--limit-scenarios", type=int, default=None)
    p.add_argument("--persona", default=None,
                   help="Run only this persona ID (for retries)")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERR: OPENAI_API_KEY not set", file=sys.stderr)
        return 2
    client = OpenAI(api_key=api_key,
                    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))

    scenarios = json.loads(SCEN_PATH.read_text())
    if args.limit_scenarios:
        scenarios = scenarios[:args.limit_scenarios]

    personas = PERSONAS
    if args.persona:
        personas = [(pid, prompt) for pid, prompt in PERSONAS if pid == args.persona]
        if not personas:
            print(f"ERR: persona {args.persona} not found", file=sys.stderr)
            return 2
    elif args.limit_personas:
        personas = personas[:args.limit_personas]

    print(f"Running {len(personas)} personas × {len(scenarios)} scenarios = "
          f"{len(personas) * len(scenarios)} calls on {args.model}")
    print(f"Out: {OUT_DIR}/")

    t_start = time.time()
    for pi, (pid, prompt) in enumerate(personas, 1):
        out_path = OUT_DIR / f"{pid}.json"
        ratings: Dict[str, int] = {}
        if out_path.exists():
            existing = json.loads(out_path.read_text())
            ratings = existing.get("ratings", {})
            print(f"\n[{pi}/{len(personas)}] {pid:<32} resuming from {len(ratings)} ratings")
        else:
            print(f"\n[{pi}/{len(personas)}] {pid:<32} starting fresh")

        for si, scen in enumerate(scenarios, 1):
            sid = scen["id"]
            # Skip if already rated this scenario
            already = all(f"s{sid}_{side}_{d}" in ratings
                          for side in ("a", "b") for d in DIMS)
            if already:
                continue
            res = rate_one(client, args.model, prompt, scen)
            if not res:
                continue
            for side in ("a", "b"):
                for d in DIMS:
                    ratings[f"s{sid}_{side}_{d}"] = res[f"{side}_{d}"]

            # Save incrementally so we don't lose progress on interruption
            record = {
                "annotator_id": pid,
                "ratings": ratings,
                "completed_scenarios": len({k.split("_")[0] for k in ratings.keys()}),
                "total_scenarios": len(scenarios),
                "is_complete": len(ratings) == len(scenarios) * 8,
                "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "rater_kind": "llm_persona",
                "model": args.model,
            }
            out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))

            if si % 10 == 0 or si == len(scenarios):
                elapsed = time.time() - t_start
                done = si + (pi - 1) * len(scenarios)
                total = len(personas) * len(scenarios)
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{si:2d}/{len(scenarios)}] sid={sid} done. "
                      f"overall {done}/{total} | elapsed {elapsed:.0f}s | "
                      f"rate {rate:.2f}/s | eta {eta:.0f}s")

    print(f"\nTotal wall time: {time.time() - t_start:.1f}s")
    print(f"Output dir: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
