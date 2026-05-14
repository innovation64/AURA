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
import os
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
        available_tools: Optional[Sequence[str]] = None,
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
        available_tools: Optional[Sequence[str]] = None,
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
    "You are the intent-inference stage of an environment-aware agent that "
    "performs theory-of-mind style alignment for a human user in a social "
    "environment. Given the user's surface query, the scene, and recent "
    "memories, output a JSON object describing the user's likely information "
    "need -- both literal AND implicit.\n\n"
    "Calibration rules for `gap` (MOST IMPORTANT):\n"
    "- gap < 0.20: the literal query is self-contained. Facts-of-the-world "
    "  questions ('what time is it?', 'how many agents are at home?') with "
    "  no social or private-state subtext.\n"
    "- 0.20 <= gap < 0.40: literal is sufficient but user would benefit "
    "  from a small amount of situational context.\n"
    "- 0.40 <= gap < 0.60: literal answer is a fact visible in the scene, "
    "  but the user is likely after something that REQUIRES probing private "
    "  agent state (availability, mood, emotional_state, unspoken_goal) -- "
    "  e.g., 'where is X?' when they actually want to know if X is free.\n"
    "- gap >= 0.60: implicit need dominates. Queries about appropriateness "
    "  ('is this a good time to...?'), latent goals ('what is X up to?'), "
    "  or relational state ('is anyone avoiding anyone?').\n\n"
    "Calibration rules for `recommended_probes`:\n"
    "- ONLY use tool names from the AVAILABLE_TOOLS list in the user message.\n"
    "- Do NOT invent tool names. If no available tool matches the implicit "
    "  need, leave this list empty.\n"
    "- Prefer tools that return structured state (nearby_agents, agent plan, "
    "  private state) over free-form search.\n\n"
    "Output schema (JSON object, no prose, no markdown fences):\n"
    "{\n"
    '  "literal_need": "<one-sentence restatement>",\n'
    '  "implicit_need": ["<plausible hidden need>", ...],\n'
    '  "gap": 0.0,\n'
    '  "recommended_probes": ["<tool_name>", ...],\n'
    '  "should_alert": false,\n'
    '  "confidence": 0.0,\n'
    '  "rationale": "<one sentence>"\n'
    "}\n\n"
    "FEW-SHOT EXAMPLES (illustrative only -- names, locations, and topics here\n"
    "are deliberately disjoint from any evaluation benchmark to avoid prompt\n"
    "leakage; calibrate from the gap-bucket structure, not the surface tokens):\n\n"
    "Q: \"how many participants are signed up for the standup?\"\n"
    "A: {\"literal_need\":\"Count of registered standup participants\",\"implicit_need\":[],\"gap\":0.1,\"recommended_probes\":[],\"should_alert\":false,\"confidence\":0.9,\"rationale\":\"self-contained factual query, no social subtext\"}\n\n"
    "Q: \"is Diego still in the lab?\" (scene shows Diego at the lab)\n"
    "A: {\"literal_need\":\"Diego's current location\",\"implicit_need\":[\"the asker likely wants to know whether Diego is reachable now\",\"workload/mood may bear on whether to approach\"],\"gap\":0.5,\"recommended_probes\":[\"get_nearby_agents\",\"get_agent_plan\"],\"should_alert\":true,\"confidence\":0.8,\"rationale\":\"surface location query; real need is reachability assessment\"}\n\n"
    "Q: \"would now be a bad moment to drop by Priya's desk?\"\n"
    "A: {\"literal_need\":\"Appropriateness of approaching Priya now\",\"implicit_need\":[\"her current workload\",\"whether she is in deep focus\",\"any deadline pressure that would make an interruption costly\"],\"gap\":0.75,\"recommended_probes\":[\"get_nearby_agents\",\"get_agent_plan\",\"get_recent_events\"],\"should_alert\":true,\"confidence\":0.85,\"rationale\":\"explicit appropriateness query -- literal answer requires integrating private state\"}\n\n"
    "Q: \"what has Marcus been focused on this week?\"\n"
    "A: {\"literal_need\":\"Marcus's recent activity pattern\",\"implicit_need\":[\"his private priorities or latent plans beyond the visible task list\"],\"gap\":0.7,\"recommended_probes\":[\"get_recent_memories\",\"get_agent_plan\"],\"should_alert\":true,\"confidence\":0.75,\"rationale\":\"'focused on' signals interest in latent priorities, not surface task tickets\"}\n\n"
    "Q: \"does Aiko expect Tomas to ship before Friday?\" (available tools include get_agent_belief_about)\n"
    "A: {\"literal_need\":\"Aiko's belief about Tomas's delivery timeline\",\"implicit_need\":[\"what Aiko currently BELIEVES about Tomas -- not Tomas's actual schedule\"],\"gap\":0.8,\"recommended_probes\":[\"get_agent_belief_about\"],\"should_alert\":true,\"confidence\":0.85,\"rationale\":\"second-order ToM: question is about the believer's mental model of another agent, not the target's actual state\"}"
)


def _get_intent_system_prompt() -> str:
    """Return the configured intent prompt variant.

    Default is the production clean few-shot prompt. Set
    AURA_INTENT_PROMPT_VARIANT=no_fewshot for prompt-ablation runs.
    """
    variant = os.environ.get("AURA_INTENT_PROMPT_VARIANT", "clean").strip().lower()
    if variant in {"no_fewshot", "no-few-shot", "nofewshot", "zero_shot", "none"}:
        base = _INTENT_SYSTEM_PROMPT.split("FEW-SHOT EXAMPLES", 1)[0].rstrip()
        return (
            base
            + "\n\nZERO-SHOT PROMPT: Apply only the calibration rules above; "
            "do not rely on example-specific names, locations, or topics."
        )
    return _INTENT_SYSTEM_PROMPT


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
        available_tools: Optional[Sequence[str]] = None,
    ) -> IntentFrame:
        if not user_query or not user_query.strip():
            return self._fallback.infer(user_query, scene, memories, user_profile, available_tools)

        if self._client is None:
            return self._fallback.infer(user_query, scene, memories, user_profile, available_tools)

        user_message = _build_user_message(user_query, scene, memories, user_profile, available_tools)

        # Try with response_format=json_object first (OpenAI-native); on
        # providers that either reject this parameter (Anthropic returns
        # 400) or silently return empty content for it (Gemini OpenAI-
        # compat), retry without it and rely on the system prompt +
        # fence-stripping in _parse_intent_json. Use a generous max_tokens
        # cap so the JSON is not truncated for verbose backbones.
        kwargs = dict(
            model=self._model,
            messages=[
                {"role": "system", "content": _get_intent_system_prompt()},
                {"role": "user", "content": user_message},
            ],
            temperature=self._temperature,
            max_tokens=max(self._max_tokens, 512),
        )
        raw = ""
        try:
            resp = self._client.chat.completions.create(
                **kwargs, response_format={"type": "json_object"},
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as e_first:
            e_first_msg = str(e_first)
        else:
            e_first_msg = None
        if not raw:
            try:
                resp = self._client.chat.completions.create(**kwargs)
                raw = (resp.choices[0].message.content or "").strip()
            except Exception as e2:
                logger.warning(
                    "LLMIntentInferrer call failed: %s / %s — falling back",
                    e_first_msg, e2,
                )
                return self._fallback.infer(user_query, scene, memories, user_profile, available_tools)

        frame = _parse_intent_json(raw, user_query)
        if frame is None:
            logger.warning("LLMIntentInferrer produced unparseable JSON; falling back")
            return self._fallback.infer(user_query, scene, memories, user_profile, available_tools)

        # Enforce tool-name whitelist when available_tools is provided:
        # drop any probe the model invented that is not in the registry.
        if available_tools is not None and frame.recommended_probes:
            allowed = set(available_tools)
            kept = [p for p in frame.recommended_probes if p in allowed]
            if len(kept) != len(frame.recommended_probes):
                logger.debug(
                    "Filtered %d invalid probe names: %s",
                    len(frame.recommended_probes) - len(kept),
                    [p for p in frame.recommended_probes if p not in allowed],
                )
            frame.recommended_probes = kept

        return frame


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_user_message(
    user_query: str,
    scene: SceneState,
    memories: Sequence[MemoryItem],
    user_profile: Optional[Dict[str, Any]],
    available_tools: Optional[Sequence[str]] = None,
) -> str:
    mem_preview = "\n".join(
        f"- {m.content[:160]}" for m in list(memories)[:5]
    ) or "(none)"
    scene_preview = f"summary: {scene.summary}\nentities: {scene.entities[:10]}"
    profile_preview = json.dumps(user_profile or {}, ensure_ascii=False)[:400]
    tools_preview = (
        ", ".join(sorted(set(available_tools))) if available_tools else "(none provided)"
    )
    return (
        f"USER QUERY: {user_query}\n\n"
        f"CURRENT SCENE:\n{scene_preview}\n\n"
        f"RECENT MEMORIES:\n{mem_preview}\n\n"
        f"USER PROFILE: {profile_preview}\n\n"
        f"AVAILABLE_TOOLS: {tools_preview}\n\n"
        "Infer the user's intent per the schema. Recommended_probes MUST only "
        "contain names from AVAILABLE_TOOLS (or be empty). Output JSON only."
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
