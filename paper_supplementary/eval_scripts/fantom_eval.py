"""FANToM benchmark adapter for AURA.

Bridges the FANToM ToM benchmark (Kim et al., EMNLP 2023) into AURA's
3-condition evaluation harness (Literal / NoIntent / Intent). This is
deliberately NOT a re-implementation of the upstream eval_fantom.py:
we do exact-match / binary scoring against gold labels rather than the
upstream's sentence-embedding-based judge, so the score depends only
on the model's answer and the ground-truth field.

Mapping (FANToM -> AURA):
  full_context (multi-turn dialogue)  -> SceneState.summary + per-utterance MemoryItem
  joining_speaker / target character  -> scene metadata + question target
  beliefQAs question                  -> user_query (multiple-choice)
  answerabilityQAs_binary             -> user_query (yes/no)
  infoAccessibilityQAs_binary         -> user_query (yes/no)
  answerabilityQA_list                -> user_query (list of names)
  infoAccessibilityQA_list            -> user_query (list of names)

Schema for each FANToM conversation row (from datasets/fantom/data/fantom/fantom_v1.json):
  set_id, part_id, conv_id          # identifiers
  full_context, short_context       # the dialogue, str
  missed_info                       # short summary of what was missed when joining_speaker rejoined
  joining_speaker                   # the character whose belief is "false" wrt missed info
  factQA                            # {question, correct_answer, wrong_answer}
  beliefQAs                         # list of {question, question_type, tom_type,
                                    #          correct_answer, wrong_answer,
                                    #          missed_info_accessibility}
  infoAccessibilityQA_list          # {question, correct_answer:list, wrong_answer:list}
  answerabilityQA_list              # {question, correct_answer:list, wrong_answer:list}
  infoAccessibilityQAs_binary       # list of {question, correct_answer:'yes'|'no'|'no:long'}
  answerabilityQAs_binary           # list of {question, correct_answer:'yes'|'no'|'no:long'}

Scoring (this adapter, NOT FANToM's upstream judge):
  * BeliefQ          -> 2-choice (a)/(b); randomized order with fixed seed; exact letter match.
  * AnswerabilityQ   binary  -> y/n match; treats 'no:long' as 'no'.
  * InfoAccessQ      binary  -> y/n match; treats 'no:long' as 'no'.
  * AnswerabilityQ   list    -> exact set-equal of character names (binary 1/0).
  * InfoAccessQ      list    -> exact set-equal of character names (binary 1/0).

This adapter does NOT cover FactQ (token-F1) -- that's a non-ToM control
and not the focus of the AURA paper.
"""

from __future__ import annotations

import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make AURA importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
_AURA_SRC = _REPO_ROOT / "AURA" / "src"
if str(_AURA_SRC) not in sys.path:
    sys.path.insert(0, str(_AURA_SRC))

from aura.types import MemoryItem, SceneState  # noqa: E402

FANTOM_DIR = _REPO_ROOT / "datasets" / "fantom"
FANTOM_JSON = FANTOM_DIR / "data" / "fantom" / "fantom_v1.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FantomQuestion:
    """One canonicalized FANToM question, ready to feed any condition runner."""

    set_id: str
    qid: str                              # globally unique id like '0-0-0:belief:0'
    qtype: str                            # 'belief' | 'answ_bin' | 'info_bin' | 'answ_list' | 'info_list'
    sub_qtype: str                        # FANToM's own question_type, for slicing
    question: str                         # what to ask the model (already includes choices for belief)
    agent_subject: str                    # the character whose mental state is being asked about
    gold: Any                             # gold answer in scoring format ('a'/'b' for belief, 'yes'/'no' for binary, set[str] for list)
    full_context: str                     # raw conversation text
    short_context: str                    # short context
    joining_speaker: str
    missed_info: str
    speakers: List[str]                   # all distinct speakers in full_context
    tom_order: str = ""                   # belief only: 'first-order' | 'second-order:accessible' | 'second-order:acyclic' | 'second-order:cyclic'
    info_accessibility: str = ""          # belief only: 'accessible' | 'inaccessible'


# ---------------------------------------------------------------------------
# Loader and adapter
# ---------------------------------------------------------------------------


def load_fantom_dataframe():
    """Load the FANToM dataframe; downloads if missing."""
    if not FANTOM_JSON.exists():
        # Trigger upstream loader (downloads tar.gz from GCS)
        loader_path = FANTOM_DIR / "task"
        sys.path.insert(0, str(loader_path))
        try:
            from dataset_loader import load as _load  # type: ignore
            return _load()
        finally:
            sys.path.remove(str(loader_path))
    import pandas as pd  # noqa: WPS433
    return pd.read_json(FANTOM_JSON)


def _extract_speakers(context: str) -> List[str]:
    """Pull "Speaker:" prefixes out of a FANToM context string, in order, deduped."""
    seen: List[str] = []
    for line in context.split("\n"):
        m = re.match(r"^([A-Z][A-Za-z\-]+(?: [A-Z][A-Za-z\-]+)*):", line.strip())
        if m:
            name = m.group(1)
            if name not in seen:
                seen.append(name)
    return seen


def _extract_subject(question: str, speakers: List[str], joining: str) -> str:
    """Best-effort: find the named character the question is asking about.

    BeliefQ: 'What does X believe ...' -> X
    AnswerabilityQ binary: 'Does X know ...' -> X
    Falls back to joining_speaker.
    """
    for sp in speakers + [joining]:
        if sp and sp in question:
            return sp
    return joining


def _make_belief_choices(qa: Dict[str, Any], rng: random.Random) -> Tuple[str, str]:
    """Format BeliefQ as a 2-choice MCQ. Returns (choices_text, gold_letter).

    Mirrors eval_fantom.py:set_beliefQA_multiple_choices — option_a is wrong,
    option_b is correct; we randomize who goes first using `rng`.
    """
    option_a = qa["wrong_answer"]
    option_b = qa["correct_answer"]
    if rng.random() < 0.5:
        choices = [option_b, option_a]
        gold = "a"
    else:
        choices = [option_a, option_b]
        gold = "b"
    text = f"(a) {choices[0]}\n(b) {choices[1]}"
    return text, gold


def adapt_row_to_questions(
    row: Dict[str, Any],
    rng: random.Random,
) -> List[FantomQuestion]:
    """Flatten one FANToM conversation into a list of FantomQuestion."""
    set_id = row["set_id"]
    full_context = row["full_context"]
    short_context = row["short_context"]
    joining = row["joining_speaker"]
    missed_info = row.get("missed_info", "") or ""
    speakers = _extract_speakers(full_context)

    # Pull fact-question and fact-answer; upstream eval_fantom.py prepends these
    # to answ/info questions so that "this question / this information" resolves.
    fact_qa = row.get("factQA") or {}
    fact_q = fact_qa.get("question", "") or ""
    fact_a = fact_qa.get("correct_answer", "") or ""

    out: List[FantomQuestion] = []

    # BeliefQAs -> 2-choice MCQ
    for i, qa in enumerate(row.get("beliefQAs") or []):
        choices_text, gold_letter = _make_belief_choices(qa, rng)
        prompt = (
            f"{qa['question']}\n\n"
            f"Choose the most plausible option:\n{choices_text}\n\n"
            f"Respond with ONLY the letter '(a)' or '(b)'."
        )
        out.append(FantomQuestion(
            set_id=set_id,
            qid=f"{set_id}:belief:{i}",
            qtype="belief",
            sub_qtype=qa.get("question_type", "tom:belief"),
            question=prompt,
            agent_subject=_extract_subject(qa["question"], speakers, joining),
            gold=gold_letter,
            full_context=full_context,
            short_context=short_context,
            joining_speaker=joining,
            missed_info=missed_info,
            speakers=speakers,
            tom_order=str(qa.get("tom_type", "") or ""),
            info_accessibility=str(qa.get("missed_info_accessibility", "") or ""),
        ))

    # AnswerabilityQ binary
    for i, qa in enumerate(row.get("answerabilityQAs_binary") or []):
        gold_raw = qa["correct_answer"]
        gold = "yes" if gold_raw == "yes" else "no"  # collapse 'no:long' -> 'no'
        prompt = (
            f"Target question: {fact_q}\n\n"
            f"{qa['question']}\n\n"
            f"Answer with ONLY 'yes' or 'no'."
        )
        out.append(FantomQuestion(
            set_id=set_id,
            qid=f"{set_id}:answ_bin:{i}",
            qtype="answ_bin",
            sub_qtype=qa.get("question_type", "tom:answerability:binary"),
            question=prompt,
            agent_subject=_extract_subject(qa["question"], speakers, joining),
            gold=gold,
            full_context=full_context,
            short_context=short_context,
            joining_speaker=joining,
            missed_info=missed_info,
            speakers=speakers,
        ))

    # InfoAccessibilityQ binary
    for i, qa in enumerate(row.get("infoAccessibilityQAs_binary") or []):
        gold_raw = qa["correct_answer"]
        gold = "yes" if gold_raw == "yes" else "no"
        prompt = (
            f"Information: {fact_q} {fact_a}\n\n"
            f"{qa['question']}\n\n"
            f"Answer with ONLY 'yes' or 'no'."
        )
        out.append(FantomQuestion(
            set_id=set_id,
            qid=f"{set_id}:info_bin:{i}",
            qtype="info_bin",
            sub_qtype=qa.get("question_type", "tom:info_accessibility:binary"),
            question=prompt,
            agent_subject=_extract_subject(qa["question"], speakers, joining),
            gold=gold,
            full_context=full_context,
            short_context=short_context,
            joining_speaker=joining,
            missed_info=missed_info,
            speakers=speakers,
        ))

    # AnswerabilityQ list
    qa = row.get("answerabilityQA_list")
    if qa:
        prompt = (
            f"Target question: {fact_q}\n\n"
            f"{qa['question']}\n\n"
            f"Reply with a comma-separated list of character names. Example: 'Alice, Bob'."
        )
        out.append(FantomQuestion(
            set_id=set_id,
            qid=f"{set_id}:answ_list",
            qtype="answ_list",
            sub_qtype=qa.get("question_type", "tom:answerability:list"),
            question=prompt,
            agent_subject=joining,
            gold=set(qa["correct_answer"]),
            full_context=full_context,
            short_context=short_context,
            joining_speaker=joining,
            missed_info=missed_info,
            speakers=speakers,
        ))

    # InfoAccessibilityQ list
    qa = row.get("infoAccessibilityQA_list")
    if qa:
        prompt = (
            f"Information: {fact_q} {fact_a}\n\n"
            f"{qa['question']}\n\n"
            f"Reply with a comma-separated list of character names. Example: 'Alice, Bob'."
        )
        out.append(FantomQuestion(
            set_id=set_id,
            qid=f"{set_id}:info_list",
            qtype="info_list",
            sub_qtype=qa.get("question_type", "tom:info_accessibility:list"),
            question=prompt,
            agent_subject=joining,
            gold=set(qa["correct_answer"]),
            full_context=full_context,
            short_context=short_context,
            joining_speaker=joining,
            missed_info=missed_info,
            speakers=speakers,
        ))

    return out


def build_scene_and_memories(q: FantomQuestion) -> Tuple[SceneState, List[MemoryItem]]:
    """Convert a FANToM question's full_context into AURA Scene + Memory.

    Each utterance becomes one MemoryItem (preserving turn order).
    SceneState.summary is a 2-3 line precis; entities are the speaker list.
    """
    utterances = [u.strip() for u in q.full_context.split("\n") if u.strip()]
    memories = [
        MemoryItem(
            content=utt,
            metadata={"turn_idx": i, "speaker": (utt.split(":", 1)[0] if ":" in utt else "")},
        )
        for i, utt in enumerate(utterances)
    ]
    summary = (
        f"Multi-party conversation among {', '.join(q.speakers)}. "
        f"At some point {q.joining_speaker} stepped away and rejoined. "
        f"Question is about what one of these characters knows or believes."
    )
    scene = SceneState(
        summary=summary,
        entities=q.speakers,
        context={
            "joining_speaker": q.joining_speaker,
            "n_utterances": len(utterances),
            "agent_subject": q.agent_subject,
        },
    )
    return scene, memories


# ---------------------------------------------------------------------------
# AURA-flavor probe tools that surface FANToM-style structure
# ---------------------------------------------------------------------------


def make_probe_executor(q: FantomQuestion):
    """Build a tool_exec(name, args)->json-string closure scoped to one question.

    Tools:
      - get_utterance_history(speaker)            : every utterance by a given speaker, in order.
      - get_present_speakers_at_utterance(idx)    : which speakers had spoken by turn <idx>.
      - get_information_disclosed_in_window(s,e)  : raw utterances in [s, e).
    """
    utterances = [u.strip() for u in q.full_context.split("\n") if u.strip()]

    def _speaker_of(line: str) -> str:
        return line.split(":", 1)[0].strip() if ":" in line else ""

    def _exec(name: str, args: Dict[str, Any]) -> str:
        if name == "get_utterance_history":
            sp = (args or {}).get("speaker", "")
            hist = [
                {"turn_idx": i, "text": u}
                for i, u in enumerate(utterances)
                if _speaker_of(u) == sp
            ]
            return json.dumps({"speaker": sp, "utterances": hist})
        if name == "get_present_speakers_at_utterance":
            idx = int((args or {}).get("idx", 0))
            seen: List[str] = []
            for u in utterances[: max(0, idx)]:
                s = _speaker_of(u)
                if s and s not in seen:
                    seen.append(s)
            return json.dumps({"idx": idx, "speakers_so_far": seen})
        if name == "get_information_disclosed_in_window":
            s = int((args or {}).get("start", 0))
            e = int((args or {}).get("end", len(utterances)))
            window = utterances[max(0, s): max(0, e)]
            return json.dumps({"start": s, "end": e, "utterances": window})
        return json.dumps({"error": f"unknown tool {name!r}"})

    return _exec


PROBE_TOOL_NAMES = [
    "get_utterance_history",
    "get_present_speakers_at_utterance",
    "get_information_disclosed_in_window",
]

PROBE_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_utterance_history",
            "description": "Return every utterance spoken by a given character, in chronological order. Useful to recover what one character could observe.",
            "parameters": {
                "type": "object",
                "properties": {"speaker": {"type": "string", "description": "Character name."}},
                "required": ["speaker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_present_speakers_at_utterance",
            "description": "List which characters had spoken at least once before turn <idx>. Useful to figure out who was present in a conversation window.",
            "parameters": {
                "type": "object",
                "properties": {"idx": {"type": "integer", "description": "Turn index (0-based)."}},
                "required": ["idx"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_information_disclosed_in_window",
            "description": "Return raw conversation utterances between turns [start, end). Useful to inspect what was said while a character was absent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                },
                "required": ["start", "end"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


_LETTER_RE = re.compile(r"\(?\s*([ab])\s*\)?", re.IGNORECASE)
_YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_belief_letter(answer: str) -> Optional[str]:
    """Pull the first '(a)' or '(b)' (or stray 'a'/'b') out of a model answer."""
    if not answer:
        return None
    m = _LETTER_RE.search(answer.strip())
    return m.group(1).lower() if m else None


def parse_yesno(answer: str) -> Optional[str]:
    if not answer:
        return None
    m = _YESNO_RE.search(answer.strip())
    return m.group(1).lower() if m else None


def parse_name_list(answer: str, candidates: List[str]) -> set:
    """Best-effort: extract a comma-separated name list, restrict to known speakers."""
    if not answer:
        return set()
    # Split on commas, semicolons, newlines, ' and '
    raw = re.split(r"[,;\n]| and ", answer)
    out: set = set()
    for token in raw:
        t = token.strip(" .'\"()[]")
        if not t:
            continue
        for c in candidates:
            if t.lower() == c.lower():
                out.add(c)
                break
    return out


def score_question(q: FantomQuestion, model_answer: str) -> int:
    """Return 1 if correct, 0 otherwise."""
    if q.qtype == "belief":
        pred = parse_belief_letter(model_answer)
        return int(pred is not None and pred == q.gold)
    if q.qtype in ("answ_bin", "info_bin"):
        pred = parse_yesno(model_answer)
        return int(pred is not None and pred == q.gold)
    if q.qtype in ("answ_list", "info_list"):
        pred = parse_name_list(model_answer, q.speakers)
        return int(pred == q.gold)
    return 0


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


def sample_questions(
    n: int,
    seed: int = 0,
    qtypes: Optional[List[str]] = None,
) -> List[FantomQuestion]:
    """Draw `n` questions uniformly across conversations and question types."""
    df = load_fantom_dataframe()
    rng = random.Random(seed)
    all_questions: List[FantomQuestion] = []
    for _, row in df.iterrows():
        all_questions.extend(adapt_row_to_questions(row.to_dict(), rng))
    if qtypes:
        all_questions = [q for q in all_questions if q.qtype in qtypes]
    rng.shuffle(all_questions)
    return all_questions[:n]


if __name__ == "__main__":
    # Quick sanity check
    qs = sample_questions(3, seed=1)
    for q in qs:
        print(f"[{q.qid}] qtype={q.qtype} subject={q.agent_subject} gold={q.gold!r}")
        print(f"  Q: {q.question[:120]}")
        print(f"  speakers: {q.speakers}")
        print()
