"""Intent inference: AURA's environment-mediated ToM stage.

The agentic core of AURA. Sits between Scene/Memory and Explore/Act,
and directs downstream control flow: it decides whether the literal
query suffices, what hidden information need may exist, and how much
probe/memory budget the gap justifies.

An IntentInferrer takes the user's surface query plus the agent's
current scene and memory view, and returns an IntentFrame that
downstream stages (Memory retrieval, Explore, Interact) consult to
branch. This makes the LLM, not the config file, the controller
of per-query pipeline shape -- the property that distinguishes the
agentic AURA from a pure context-engineering workflow.

Two concrete implementations:

- HeuristicIntentInferrer: no LLM, for unit tests and as the default
  when no backbone is configured. Uses simple surface cues (question
  words, category keywords) to estimate gap.

- LLMIntentInferrer: asks the backbone model to produce an IntentFrame
  via a structured JSON call, with a strict schema and a fallback to
  the heuristic on parse errors.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Protocol, Sequence

from .types import IntentFrame, MemoryItem, SceneState

logger = logging.getLogger(__name__)


class IntentInferrer(Protocol):
    """Interface any intent backend must satisfy."""

    def infer(
        self,
        user_query: str,
        scene: SceneState,
        memories: Sequence[MemoryItem],
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> IntentFrame:
        ...


# ---------------------------------------------------------------------------
# Heuristic fallback — no LLM, deterministic
# ---------------------------------------------------------------------------

# Literal question surface cues. When a query matches one of these patterns
# without any nearby-agent / social / private-state vocabulary, the heuristic
# classifies it as low-gap and will recommend a literal response.
_LITERAL_PATTERNS = (
    "what time", "where am i", "what day", "how many",
)

# Vocabulary that flags likely implicit / social / private-state intent.
_IMPLICIT_MARKERS = (
    "available", "busy", "free", "mood", "feeling", "welcome",
    "chat", "talking", "together", "with whom", "who is",
    "appropriate", "should i", "good time", "interrupt",
)


class HeuristicIntentInferrer:
    """Deterministic baseline intent inference.

    Used for unit tests and as the fallback when LLMIntentInferrer
    fails to produce valid JSON. Never makes an LLM call.
    """

    def infer(
        self,
        user_query: str,
        scene: SceneState,
        memories: Sequence[MemoryItem],
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> IntentFrame:
        q = (user_query or "").strip().lower()
        if not q:
            return IntentFrame(
                literal_need="(empty query)",
                implicit_need=[],
                gap=0.0,
                confidence=1.0,
                rationale="empty input",
            )

        has_literal = any(p in q for p in _LITERAL_PATTERNS)
        implicit_hits = [m for m in _IMPLICIT_MARKERS if m in q]
        gap = min(0.2 * len(implicit_hits), 1.0)
        if has_literal and not implicit_hits:
            gap = 0.0

        implicit_need: List[str] = []
        probes: List[str] = []
        alert = False
        if implicit_hits:
            implicit_need.append(
                "user may be asking about social availability or emotional state"
            )
            probes.extend(["get_nearby_agents", "get_recent_events"])
            alert = gap >= 0.4

        return IntentFrame(
            literal_need=user_query.strip(),
            implicit_need=implicit_need,
            gap=gap,
            recommended_probes=probes,
            should_alert=alert,
            confidence=0.5,
            rationale=f"heuristic: {len(implicit_hits)} implicit marker(s)",
        )


# ---------------------------------------------------------------------------
# LLM-backed inference
# ---------------------------------------------------------------------------

_INTENT_SYSTEM_PROMPT = (
    "You are the intent-inference stage of an environment-aware agent. "
    "Given the user's surface query, the current scene snapshot, and a few "
    "recent memories, output a JSON object describing the user's likely "
    "information need -- both literal and implicit. Think about what the "
    "user would actually want to know beyond the surface words, given what "
    "is observable in the scene. Your output controls how much environment "
    "probing the downstream agent will perform, so be calibrated: do NOT "
    "inflate implicit needs when the literal query is self-contained.\n\n"
    "Output schema (JSON, no prose):\n"
    "{\n"
    '  "literal_need": "<one-sentence restatement>",\n'
    '  "implicit_need": ["<plausible hidden need>", ...],\n'
    '  "gap": 0.0,  // [0,1], 0=literal is sufficient, 1=orthogonal\n'
    '  "recommended_probes": ["<tool_name>", ...],\n'
    '  "should_alert": false,  // add proactive info to the reply\n'
    '  "confidence": 0.0,  // [0,1]\n'
    '  "rationale": "<one sentence>"\n'
    "}"
)


class LLMIntentInferrer:
    """Production path: asks the backbone to produce an IntentFrame.

    Falls back to HeuristicIntentInferrer on JSON-parse failure or when
    the client is missing, so callers never receive None.
    """

    def __init__(
        self,
        client: Any,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 320,
        fallback: Optional[IntentInferrer] = None,
    ):
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._fallback: IntentInferrer = fallback or HeuristicIntentInferrer()

    def infer(
        self,
        user_query: str,
        scene: SceneState,
        memories: Sequence[MemoryItem],
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> IntentFrame:
        if not user_query or not user_query.strip():
            return self._fallback.infer(user_query, scene, memories, user_profile)

        if self._client is None:
            return self._fallback.infer(user_query, scene, memories, user_profile)

        user_message = _build_user_message(user_query, scene, memories, user_profile)

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"},
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning("LLMIntentInferrer call failed: %s — falling back", e)
            return self._fallback.infer(user_query, scene, memories, user_profile)

        frame = _parse_intent_json(raw, user_query)
        if frame is None:
            logger.warning("LLMIntentInferrer produced unparseable JSON; falling back")
            return self._fallback.infer(user_query, scene, memories, user_profile)
        return frame


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_user_message(
    user_query: str,
    scene: SceneState,
    memories: Sequence[MemoryItem],
    user_profile: Optional[Dict[str, Any]],
) -> str:
    mem_preview = "\n".join(
        f"- {m.content[:160]}" for m in list(memories)[:5]
    ) or "(none)"
    scene_preview = f"summary: {scene.summary}\nentities: {scene.entities[:10]}"
    profile_preview = json.dumps(user_profile or {}, ensure_ascii=False)[:400]
    return (
        f"USER QUERY: {user_query}\n\n"
        f"CURRENT SCENE:\n{scene_preview}\n\n"
        f"RECENT MEMORIES:\n{mem_preview}\n\n"
        f"USER PROFILE: {profile_preview}\n\n"
        "Infer the user's intent per the schema. Output JSON only."
    )


def _parse_intent_json(raw: str, user_query: str) -> Optional[IntentFrame]:
    """Parse an LLM JSON response into an IntentFrame.

    Accepts a best-effort JSON object and coerces fields with safe
    defaults. Returns None only when the text is not parseable as JSON
    or lacks a literal_need (which indicates the model misunderstood the
    task rather than produced a low-confidence but valid answer).
    """
    text = raw.strip()
    if text.startswith("```"):
        # Strip ``` or ```json fences if the model adds them despite
        # response_format="json_object".
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        obj = json.loads(text)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(obj, dict):
        return None

    literal = str(obj.get("literal_need", "") or "").strip() or user_query.strip()
    implicit_raw = obj.get("implicit_need") or []
    if isinstance(implicit_raw, str):
        implicit_list = [implicit_raw]
    elif isinstance(implicit_raw, list):
        implicit_list = [str(x) for x in implicit_raw if x]
    else:
        implicit_list = []

    probes_raw = obj.get("recommended_probes") or []
    if isinstance(probes_raw, list):
        probes = [str(p) for p in probes_raw if p]
    else:
        probes = []

    try:
        gap = float(obj.get("gap", 0.0))
    except (TypeError, ValueError):
        gap = 0.0
    try:
        confidence = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    gap = max(0.0, min(1.0, gap))
    confidence = max(0.0, min(1.0, confidence))

    should_alert = bool(obj.get("should_alert", False))
    rationale = str(obj.get("rationale", "") or "")[:400]

    return IntentFrame(
        literal_need=literal,
        implicit_need=implicit_list,
        gap=gap,
        recommended_probes=probes,
        should_alert=should_alert,
        confidence=confidence,
        rationale=rationale,
    )


def intent_frame_to_dict(frame: IntentFrame) -> Dict[str, Any]:
    """Convenience for logging and downstream metadata."""
    return asdict(frame)
