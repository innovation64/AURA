"""Translate the 50 human-eval scenarios (query + response_a + response_b) to
Chinese using gpt-4o-mini and inject the translations into the HTML form as
data-zh-* attributes, so the rater can toggle language without losing the
original English artifact being rated.

Input:
    evaluation/results/human_eval_forms.json    (scenarios with EN content)
    evaluation/results/human_eval_forms.html    (form, already polished)

Output:
    evaluation/results/human_eval_forms.json    (augmented with *_zh fields)
    evaluation/results/human_eval_forms.html    (patched with data-zh-* attrs)

Budget: ~$0.05 on gpt-4o-mini (50 scenarios * 3 strings * short paragraphs).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from openai import OpenAI

RESULTS_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "results"
JSON_PATH = RESULTS_DIR / "human_eval_forms.json"
HTML_PATH = RESULTS_DIR / "human_eval_forms.html"

MODEL = "gpt-4o-mini"

SYSTEM = (
    "You translate English assistant-generated conversational text into natural, "
    "fluent Chinese (simplified). Keep character names (Lin Wei, Zhang Hao, Chen Mei, "
    "Liu Yang, Wang Jun) and place names (Sunrise Cafe, Tea House, Golden Wheat "
    "Bakery, Town Hall, Town Library, Wellness Pharmacy, Town Park, Chen's General "
    "Store, Art Gallery, Town Square, AURA Academy, Community Garden, Flower Garden, "
    "Riverside Walk, Temple) in their original English form inside the Chinese "
    "translation. Do not paraphrase, do not add content, do not remove content. "
    "Preserve the tone and any conversational quirks. Output only the Chinese "
    "translation as plain text, no quotes, no prefix."
)


def translate_one(client: OpenAI, text: str) -> str:
    if not text or not text.strip():
        return ""
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": text},
                ],
                temperature=0.2,
                max_tokens=800,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt == 2:
                print(f"      WARN: translate failed: {e}")
                return ""
            time.sleep(1.5 * (attempt + 1))
    return ""


def html_escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&#x27;"))


def patch_html(scenarios: list) -> None:
    """Inject data-zh attributes and rewire JS to toggle response/query text."""
    html = HTML_PATH.read_text(encoding="utf-8")

    for s in scenarios:
        sid = s["id"]
        q_zh  = html_escape(s.get("query_zh", "") or "")
        ra_zh = html_escape(s.get("response_a_zh", "") or "")
        rb_zh = html_escape(s.get("response_b_zh", "") or "")

        # Tag the <p><b>Query:</b> ...</p> block (upgradeDOM rewrites to .query-block
        # at runtime, but we need to put the translation on the *original* paragraph
        # so upgradeDOM picks it up and carries it over).
        # Simpler: tag the scenario div with data-query-zh and the two resp-box divs
        # with data-resp-zh. upgradeDOM + applyLang are patched separately to read these.

        scen_pattern = f'<div class="scenario" id="scenario-{sid}" data-sid="{sid}">'
        scen_replacement = (
            f'<div class="scenario" id="scenario-{sid}" data-sid="{sid}" '
            f'data-query-zh="{q_zh}" data-resp-a-zh="{ra_zh}" data-resp-b-zh="{rb_zh}">'
        )
        if scen_pattern in html and scen_replacement not in html:
            html = html.replace(scen_pattern, scen_replacement, 1)

    HTML_PATH.write_text(html, encoding="utf-8")


def main() -> int:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERR: OPENAI_API_KEY not set", file=sys.stderr)
        return 2
    client = OpenAI(api_key=api_key,
                    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))

    with open(JSON_PATH) as f:
        scenarios = json.load(f)

    print(f"Translating {len(scenarios)} scenarios (3 strings each)...")
    t0 = time.time()
    for i, s in enumerate(scenarios):
        q  = s.get("query", "")
        ra = s.get("response_a", "")
        rb = s.get("response_b", "")
        s["query_zh"]      = translate_one(client, q)
        s["response_a_zh"] = translate_one(client, ra)
        s["response_b_zh"] = translate_one(client, rb)
        elapsed = time.time() - t0
        print(f"  [{i+1:2d}/{len(scenarios)}] done  (total {elapsed:.1f}s)")

    # Save augmented JSON
    with open(JSON_PATH, "w") as f:
        json.dump(scenarios, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {JSON_PATH}")

    # Patch HTML with data-zh attributes
    patch_html(scenarios)
    print(f"Patched {HTML_PATH}")

    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
