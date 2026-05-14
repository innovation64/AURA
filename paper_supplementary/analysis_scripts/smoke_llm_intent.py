"""Real-API smoke test for LLMIntentInferrer.

One live query against the OpenAI Responses/Chat API to verify the
IntentFrame JSON schema parses cleanly against the current production
response format. Cost: ~1 gpt-4o-mini call (<$0.001).

Usage:
    python -m scripts.smoke_llm_intent
    # or to try several queries:
    python -m scripts.smoke_llm_intent --queries 3

Exits 0 on success, non-zero on parse/API failure. Intended as a
pre-deployment check before running the multi-seed intent ablation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from openai import OpenAI

# Import from the AURA src directly (no editable install required)
_AURA_SRC = Path(__file__).resolve().parent.parent / "AURA" / "src"
if str(_AURA_SRC) not in sys.path:
    sys.path.insert(0, str(_AURA_SRC))

from aura.intent import LLMIntentInferrer  # noqa: E402
from aura.types import MemoryItem, SceneState  # noqa: E402


# Three queries stratified by expected gap level.
# These mirror the implicit-intent query shape planned for the
# Day 8-9 experiment so we catch prompt drift early.
SMOKE_QUERIES = [
    ("what time is it right now?", "expect LOW gap (literal)"),
    ("is Lin Wei free to chat?",  "expect MEDIUM-HIGH gap (availability probe)"),
    ("should I go to the cafe now?", "expect MEDIUM gap (plan/context-dependent)"),
]


def _build_fake_scene() -> SceneState:
    return SceneState(
        summary="morning, Sunrise Cafe, 3 agents present",
        entities=["Lin Wei", "Zhang Hao", "Chen Mei"],
        context={"time": "10:15 AM", "location": "Sunrise Cafe"},
    )


def _build_fake_memories():
    return [
        MemoryItem(content="saw Lin Wei chatting with Zhang Hao at 10:00"),
        MemoryItem(content="earlier today, user mentioned wanting to catch up with Lin Wei"),
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--queries", type=int, default=len(SMOKE_QUERIES),
                   help="How many of the staged queries to run")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERR: OPENAI_API_KEY not set", file=sys.stderr)
        return 2
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    client = OpenAI(api_key=api_key, base_url=base_url)
    inferrer = LLMIntentInferrer(client=client)

    scene = _build_fake_scene()
    mems = _build_fake_memories()

    n = min(max(1, args.queries), len(SMOKE_QUERIES))
    total_start = time.time()
    failures = 0
    for i, (q, note) in enumerate(SMOKE_QUERIES[:n], 1):
        print(f"\n[{i}/{n}] {q}  ({note})")
        t0 = time.time()
        try:
            frame = inferrer.infer(q, scene, mems)
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}", file=sys.stderr)
            failures += 1
            continue
        dur = time.time() - t0
        print(f"  latency    : {dur:.2f}s")
        print(f"  literal    : {frame.literal_need}")
        print(f"  implicit   : {frame.implicit_need}")
        print(f"  gap        : {frame.gap}")
        print(f"  probes     : {frame.recommended_probes}")
        print(f"  should_alert: {frame.should_alert}")
        print(f"  confidence : {frame.confidence}")
        print(f"  rationale  : {frame.rationale}")

        # Basic sanity assertions (would fail if the LLM schema regresses)
        if not (0.0 <= frame.gap <= 1.0):
            print(f"  WARN: gap out of range: {frame.gap}")
            failures += 1
        if not frame.literal_need:
            print("  WARN: empty literal_need (fell back to heuristic)")

    total_dur = time.time() - total_start
    print(f"\n=== Summary: {n - failures}/{n} ok in {total_dur:.2f}s ===")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
