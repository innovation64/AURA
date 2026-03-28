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
