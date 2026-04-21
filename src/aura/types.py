from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class EnvironmentSignal:
    source: str
    payload: Dict[str, Any]
    modality: str = "generic"
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass
class SceneState:
    summary: str
    entities: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryItem:
    content: str
    timestamp: datetime = field(default_factory=_utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    intent: str
    rationale: str
    actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    type: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Interaction:
    message: str
    payload: Dict[str, Any] = field(default_factory=dict)
    halted: bool = False  # True when ExecutionGuard issued REDIRECT


@dataclass
class IntentFrame:
    """Agent's inferred model of what the user is actually asking for.

    AURA's env-mediated ToM contribution: the agentic component decides
    what to retrieve, probe, and how to respond based on the gap between
    literal_need (what the user said) and implicit_need (what they likely
    want given scene + memory + user history).

    Fields:
        literal_need: one-sentence restatement of the user's surface query.
        implicit_need: list of plausible hidden information needs the
            literal query does not directly express.
        gap: scalar in [0, 1]. 0 => literal == implicit (answer literally);
            1 => user's real need is orthogonal to the surface query.
        recommended_probes: tool-name hints the IntentInferrer would like
            Explore to prioritize. Empty list means 'no proactive probing
            needed'.
        should_alert: True when the gap is large enough that a purely
            literal answer would miss something the user would want to
            know (controls whether Interact issues a proactive addition
            to the reply).
        confidence: IntentInferrer's self-reported confidence in the
            inference, in [0, 1]. Low confidence falls back to literal.
        rationale: short explanation used for logging and ablation plots.
    """
    literal_need: str
    implicit_need: List[str] = field(default_factory=list)
    gap: float = 0.0
    recommended_probes: List[str] = field(default_factory=list)
    should_alert: bool = False
    confidence: float = 0.0
    rationale: str = ""
