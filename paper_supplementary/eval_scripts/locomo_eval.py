"""LoCoMo benchmark adapter for AURA.

Bridges the LoCoMo long-term conversational memory benchmark
(Maharana et al., ACL 2024) into AURA's 3-condition evaluation harness
(Literal / NoIntent / Intent). Tests whether AURA's intent-conditioned
probing helps a model recall facts buried across many sessions of a
multi-session dialogue, complementing the FANToM null result.

LoCoMo schema (from datasets/locomo/data/locomo10.json):
  10 conversations, each with:
    sample_id           : 'conv-26' etc.
    conversation        : { speaker_a, speaker_b,
                            session_<N>_date_time : str,
                            session_<N>           : list of {speaker, dia_id, text} }
                          where N runs 1..K (K in 19-32 sessions).
    qa                  : list of QA dicts:
                            question : str
                            answer   : str (or absent for category 5)
                            adversarial_answer : str (cat 5 only)
                            evidence : list[str] of dia_ids like 'D1:3'
                            category : 1|2|3|4|5
    event_summary, observation, session_summary : annotation aux.

Category labels (from upstream evaluate_qa.py):
  1 = multi-hop    : answer is a comma-separated phrase split across sessions
  2 = temporal     : when did X happen
  3 = open-domain  : reasoning beyond literal text (gold has '; ' alternatives)
  4 = single-hop   : factual recall from one utterance
  5 = adversarial  : not-answerable; correct response is "no information available"

Mapping (LoCoMo -> AURA):
  full multi-session dialogue   -> SceneState.summary + per-session MemoryItem
  question                      -> user_query
  available_tools               -> probe tools that look up sessions / search by speaker / fetch events on a date
  evidence dia_ids              -> NOT shown to model; used only for analysis

Scoring (this adapter; matches LoCoMo's published F1 metric in evaluate_qa.py):
  cat 2,3,4 : token-F1 (Porter-stemmed, normalized)
  cat 1     : multi-answer F1 (split prediction & gold by ','; mean of per-gold max F1)
  cat 5     : 1 if 'no information' / 'not mentioned' / 'cannot' in pred else 0
  EM        : strict normalize+set-equal on tokens (categorical 0/1)
"""

from __future__ import annotations

import json
import random
import re
import string
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make AURA importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
_AURA_SRC = _REPO_ROOT / "AURA" / "src"
if str(_AURA_SRC) not in sys.path:
    sys.path.insert(0, str(_AURA_SRC))

from aura.types import MemoryItem, SceneState  # noqa: E402

LOCOMO_DIR = _REPO_ROOT / "datasets" / "locomo"
LOCOMO_JSON = LOCOMO_DIR / "data" / "locomo10.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


_CAT_LABEL = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}


@dataclass
class LoCoMoQuestion:
    """One canonicalized LoCoMo question."""

    sample_id: str
    qid: str
    category: int
    category_label: str
    question: str
    gold: str
    evidence: List[str]
    speaker_a: str
    speaker_b: str
    sessions: List[Dict[str, Any]]   # [{n, date_time, turns:[{speaker,dia_id,text}]}]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_locomo() -> List[Dict[str, Any]]:
    if not LOCOMO_JSON.exists():
        raise FileNotFoundError(f"LoCoMo JSON not found at {LOCOMO_JSON}")
    with open(LOCOMO_JSON) as f:
        return json.load(f)


def _extract_sessions(conv: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Pull session_1..session_K (with date_time) into a sorted list."""
    keys = [k for k in conv.keys() if re.fullmatch(r"session_\d+", k)]
    sessions = []
    for k in sorted(keys, key=lambda x: int(x.split("_")[1])):
        n = int(k.split("_")[1])
        dt = conv.get(f"session_{n}_date_time", "")
        turns = conv[k] or []
        # Each turn already has speaker/dia_id/text
        sessions.append({"n": n, "date_time": dt, "turns": turns})
    return sessions


def adapt_sample_to_questions(sample: Dict[str, Any]) -> List[LoCoMoQuestion]:
    """Flatten one LoCoMo conversation into one LoCoMoQuestion per qa item."""
    sample_id = sample["sample_id"]
    conv = sample["conversation"]
    sessions = _extract_sessions(conv)
    speaker_a = conv.get("speaker_a", "")
    speaker_b = conv.get("speaker_b", "")
    out: List[LoCoMoQuestion] = []
    for i, qa in enumerate(sample.get("qa") or []):
        cat = int(qa.get("category", 0))
        if cat == 5:
            gold = qa.get("adversarial_answer", "no information available")
        else:
            gold = str(qa.get("answer", ""))
        out.append(LoCoMoQuestion(
            sample_id=sample_id,
            qid=f"{sample_id}:q{i}",
            category=cat,
            category_label=_CAT_LABEL.get(cat, f"cat_{cat}"),
            question=qa["question"],
            gold=gold,
            evidence=list(qa.get("evidence") or []),
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            sessions=sessions,
        ))
    return out


# ---------------------------------------------------------------------------
# Scene + Memory: per-session compressed view to keep prompt sane
# ---------------------------------------------------------------------------


def _session_to_text(sess: Dict[str, Any], max_chars: int = 1500) -> str:
    """Render one session as 'Speaker: text' lines, truncated to max_chars."""
    lines = []
    for t in sess["turns"]:
        sp = t.get("speaker", "")
        tx = t.get("text", "")
        lines.append(f"{sp}: {tx}")
    s = "\n".join(lines)
    if len(s) > max_chars:
        s = s[:max_chars - 5] + " ..."
    return s


def _session_brief(sess: Dict[str, Any], max_chars: int = 320) -> str:
    """Tight 1-2 line precis: list of speakers seen + date + first-utterance preview."""
    speakers = sorted({t.get("speaker", "") for t in sess["turns"] if t.get("speaker")})
    first_utt = ""
    if sess["turns"]:
        ftext = sess["turns"][0].get("text", "")
        first_utt = ftext[:120]
    txt = (
        f"Session {sess['n']} ({sess.get('date_time','?')}, "
        f"{len(sess['turns'])} turns, speakers={speakers}). "
        f"Opening: \"{first_utt}\""
    )
    if len(txt) > max_chars:
        txt = txt[:max_chars - 5] + " ..."
    return txt


def build_scene_and_memories(q: LoCoMoQuestion) -> Tuple[SceneState, List[MemoryItem]]:
    """One MemoryItem per session (brief). Probe tools surface the full text."""
    memories = [
        MemoryItem(
            content=_session_brief(sess),
            metadata={
                "session_n": sess["n"],
                "date_time": sess.get("date_time", ""),
                "n_turns": len(sess["turns"]),
            },
        )
        for sess in q.sessions
    ]
    summary = (
        f"Long-term conversation between {q.speaker_a} and {q.speaker_b} across "
        f"{len(q.sessions)} sessions spanning multiple dates. "
        f"Question asks the model to recall, infer, or admit ignorance about a fact "
        f"that may be buried in any session."
    )
    scene = SceneState(
        summary=summary,
        entities=[q.speaker_a, q.speaker_b],
        context={
            "n_sessions": len(q.sessions),
            "sample_id": q.sample_id,
            "category": q.category_label,
        },
    )
    return scene, memories


# ---------------------------------------------------------------------------
# Probe tools
# ---------------------------------------------------------------------------


def make_probe_executor(q: LoCoMoQuestion):
    """Tools: get_session(n), search_by_speaker(speaker, query), list_sessions_on_date(date)."""

    by_n = {s["n"]: s for s in q.sessions}
    all_sessions = q.sessions

    def _exec(name: str, args: Dict[str, Any]) -> str:
        args = args or {}
        if name == "get_session":
            try:
                n = int(args.get("n", -1))
            except Exception:
                n = -1
            sess = by_n.get(n)
            if sess is None:
                return json.dumps({"error": f"session {n} not found",
                                   "available_sessions": sorted(by_n.keys())})
            return json.dumps({
                "n": n,
                "date_time": sess.get("date_time", ""),
                "transcript": _session_to_text(sess, max_chars=1500),
            })

        if name == "search_by_speaker":
            sp = str(args.get("speaker", "")).strip()
            query = str(args.get("query", "")).strip().lower()
            if not sp:
                return json.dumps({"error": "missing speaker"})
            hits: List[Dict[str, Any]] = []
            for sess in all_sessions:
                for t in sess["turns"]:
                    tsp = t.get("speaker", "")
                    if tsp.lower() != sp.lower():
                        continue
                    text = t.get("text", "")
                    if query and query not in text.lower():
                        continue
                    hits.append({
                        "session_n": sess["n"],
                        "date_time": sess.get("date_time", ""),
                        "dia_id": t.get("dia_id", ""),
                        "text": text,
                    })
                    if len(hits) >= 25:
                        break
                if len(hits) >= 25:
                    break
            return json.dumps({"speaker": sp, "query": query, "hits": hits})

        if name == "list_sessions_on_date":
            target = str(args.get("date", "")).strip().lower()
            if not target:
                return json.dumps({"error": "missing date"})
            matches = []
            for sess in all_sessions:
                dt = (sess.get("date_time") or "").lower()
                if target in dt:
                    matches.append({"n": sess["n"], "date_time": sess.get("date_time", ""),
                                    "n_turns": len(sess["turns"])})
            return json.dumps({"query_date": target, "sessions": matches})

        return json.dumps({"error": f"unknown tool {name!r}"})

    return _exec


PROBE_TOOL_NAMES = [
    "get_session",
    "search_by_speaker",
    "list_sessions_on_date",
]

PROBE_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_session",
            "description": (
                "Fetch the full transcript of session N as a 'Speaker: text' string. "
                "Use to inspect a specific session's contents."
            ),
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer", "description": "Session number (1-based)."}},
                "required": ["n"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_speaker",
            "description": (
                "Return all utterances by a given speaker, optionally filtered by a substring "
                "(case-insensitive). Useful to recover what one character disclosed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string", "description": "Speaker name."},
                    "query":   {"type": "string", "description": "Optional substring to filter utterances."},
                },
                "required": ["speaker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sessions_on_date",
            "description": (
                "List sessions whose date_time string matches a substring (e.g. '2023', "
                "'May', '5/7'). Useful to localize temporal evidence."
            ),
            "parameters": {
                "type": "object",
                "properties": {"date": {"type": "string", "description": "Date substring."}},
                "required": ["date"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Scoring (mirrors datasets/locomo/task_eval/evaluation.py F1 metric)
# ---------------------------------------------------------------------------

# Lightweight Porter stemmer fallback: try nltk; if unavailable, no stem.
try:
    from nltk.stem.porter import PorterStemmer  # type: ignore
    _ps = PorterStemmer()
    def _stem(w: str) -> str: return _ps.stem(w)
except Exception:
    def _stem(w: str) -> str: return w


_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES_RE.sub(" ", s)
    s = " ".join(s.split())
    return s


def _f1_score(prediction: str, ground_truth: str) -> float:
    p_toks = [_stem(w) for w in _normalize(prediction).split()]
    g_toks = [_stem(w) for w in _normalize(ground_truth).split()]
    if not p_toks or not g_toks:
        return 0.0
    common = Counter(p_toks) & Counter(g_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_toks)
    recall = num_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def _multi_f1(prediction: str, ground_truth: str) -> float:
    """Multi-answer F1: split both on commas; mean over golds of max-over-preds f1."""
    preds = [p.strip() for p in prediction.split(",") if p.strip()]
    golds = [g.strip() for g in ground_truth.split(",") if g.strip()]
    if not preds or not golds:
        return 0.0
    return sum(max(_f1_score(p, g) for p in preds) for g in golds) / len(golds)


def _exact_match(prediction: str, ground_truth: str) -> int:
    p = set(_normalize(prediction).split())
    g = set(_normalize(ground_truth).split())
    return int(p == g and len(p) > 0)


_NO_INFO_MARKERS = (
    "no information available",
    "not mentioned",
    "cannot answer",
    "cannot determine",
    "no information",
    "not enough information",
    "do not know",
    "don't know",
    "i don't know",
    "unknown",
    "no record",
)


def score_question(q: LoCoMoQuestion, model_answer: str) -> Dict[str, float]:
    """Return {'f1': float, 'em': int} matching LoCoMo's metric per category."""
    pred = (model_answer or "").strip()
    if not pred:
        return {"f1": 0.0, "em": 0}

    # Cat 5: adversarial. Correct iff the model declines.
    if q.category == 5:
        low = pred.lower()
        ok = int(any(m in low for m in _NO_INFO_MARKERS))
        return {"f1": float(ok), "em": ok}

    gold = q.gold or ""
    # Cat 3 (open-domain): gold may have '; '-separated alternatives;
    #   upstream takes only the first.
    if q.category == 3 and ";" in gold:
        gold = gold.split(";")[0].strip()

    if q.category == 1:
        f1 = _multi_f1(pred, gold)
    else:
        f1 = _f1_score(pred, gold)

    em = _exact_match(pred, gold)
    return {"f1": float(f1), "em": int(em)}


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


def all_questions() -> List[LoCoMoQuestion]:
    out: List[LoCoMoQuestion] = []
    for sample in load_locomo():
        out.extend(adapt_sample_to_questions(sample))
    return out


def stratified_sample(
    n: int = 30,
    seed: int = 42,
    category_quota: Optional[Dict[int, int]] = None,
) -> List[LoCoMoQuestion]:
    """Stratified across LoCoMo's 4 evaluation categories (1,2,3,4); skip cat 5
    by default (it's a different scoring regime; we keep the 30Q smoke focused
    on the 4 RAG-style categories the paper highlights).

    Default quota for n=30: cat 1 -> 7, cat 2 -> 8, cat 3 -> 7, cat 4 -> 8.
    """
    if category_quota is None:
        category_quota = {1: 7, 2: 8, 3: 7, 4: 8}
        # tweak so the sum equals n
        s = sum(category_quota.values())
        if s != n:
            # adjust last
            category_quota[4] += (n - s)
    rng = random.Random(seed)
    pool: Dict[int, List[LoCoMoQuestion]] = {c: [] for c in category_quota}
    for q in all_questions():
        if q.category in pool:
            pool[q.category].append(q)
    chosen: List[LoCoMoQuestion] = []
    for cat, k in category_quota.items():
        items = list(pool[cat])
        rng.shuffle(items)
        if len(items) < k:
            print(f"[warn] category {cat} only has {len(items)} (wanted {k}); taking all")
        chosen.extend(items[:k])
    rng.shuffle(chosen)
    return chosen


if __name__ == "__main__":
    qs = stratified_sample(30, seed=42)
    from collections import Counter as _C
    print("Sampled", len(qs), "questions; by category:", _C(q.category_label for q in qs))
    for q in qs[:3]:
        print(f"\n[{q.qid}] cat={q.category_label} sessions={len(q.sessions)}")
        print(f"  Q: {q.question}")
        print(f"  gold: {q.gold!r}")
