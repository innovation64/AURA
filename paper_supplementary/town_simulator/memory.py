"""Memory system for AURA Town, extending AURA's MemoryStore with importance scoring and reflection."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from aura import EphemeralMemory, MemoryItem, MemoryStore, SceneState

from .config import TownConfig

logger = logging.getLogger(__name__)


@dataclass
class TownMemoryItem:
    """Extended memory item with importance and access tracking."""

    content: str
    timestamp: float  # simulation time in minutes from start
    importance: int = 5  # 1-10
    access_count: int = 0
    last_accessed: float = 0.0
    kind: str = "observation"  # observation, reflection, conversation, plan
    metadata: Dict[str, Any] = field(default_factory=dict)


class TownMemory(MemoryStore):
    """
    Memory stream inspired by Stanford Generative Agents.
    Extends AURA's MemoryStore interface.

    Features:
    - Timestamped observation stream
    - Importance scoring (via LLM)
    - Reflection synthesis (every N observations)
    - Retrieval weighted by recency + importance + relevance
    """

    def __init__(
        self,
        config: TownConfig,
        llm_engine: Any = None,
        max_items: int = 200,
    ) -> None:
        self._items: List[TownMemoryItem] = []
        self._max_items = max_items
        self._config = config
        self._llm = llm_engine
        self._observations_since_reflection = 0

    # ── AURA MemoryStore interface ──────────────────────────────────

    def update(self, scene: SceneState) -> None:
        """Implement AURA MemoryStore.update — store scene as observation."""
        self.add_observation(scene.summary, 0.0)

    def recall(self, query: Optional[str] = None, limit: int = 5) -> List[MemoryItem]:
        """Implement AURA MemoryStore.recall — return as AURA MemoryItem."""
        items = self.retrieve(query or "", current_time=0.0, limit=limit)
        return [
            MemoryItem(
                content=item.content,
                timestamp=datetime.fromtimestamp(item.timestamp, tz=timezone.utc),
                metadata={"importance": item.importance, "kind": item.kind},
            )
            for item in items
        ]

    # ── Town-specific memory operations ─────────────────────────────

    def add_observation(
        self,
        content: str,
        sim_time: float,
        kind: str = "observation",
        importance: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TownMemoryItem:
        """Add a new memory with importance scoring."""
        if importance is None:
            importance = self._score_importance(content)

        item = TownMemoryItem(
            content=content,
            timestamp=sim_time,
            importance=importance,
            kind=kind,
            metadata=metadata or {},
        )
        self._items.append(item)

        # Trim if over capacity
        if len(self._items) > self._max_items:
            # Remove oldest low-importance items
            self._items.sort(key=lambda m: (m.importance, m.timestamp))
            self._items = self._items[len(self._items) - self._max_items :]

        self._observations_since_reflection += 1
        return item

    def add_conversation(
        self,
        content: str,
        sim_time: float,
        partner: str,
        importance: Optional[int] = None,
    ) -> TownMemoryItem:
        """Record a conversation."""
        return self.add_observation(
            content=content,
            sim_time=sim_time,
            kind="conversation",
            importance=importance or 6,
            metadata={"partner": partner},
        )

    def retrieve(
        self,
        query: str,
        current_time: float,
        limit: int = 5,
    ) -> List[TownMemoryItem]:
        """Retrieve memories weighted by recency + importance + relevance."""
        if not self._items:
            return []

        scored: List[tuple[float, TownMemoryItem]] = []
        query_lower = query.lower()
        max_time = max(m.timestamp for m in self._items) if self._items else 1.0

        for item in self._items:
            # Recency: exponential decay
            time_diff = current_time - item.timestamp if current_time > 0 else 0
            recency = math.exp(-0.01 * max(time_diff, 0))

            # Importance: normalized 0-1
            importance = item.importance / 10.0

            # Relevance: simple keyword overlap
            if query_lower:
                words = set(query_lower.split())
                content_words = set(item.content.lower().split())
                overlap = len(words & content_words)
                relevance = min(overlap / max(len(words), 1), 1.0)
            else:
                relevance = 0.5

            score = (
                self._config.recency_weight * recency
                + self._config.importance_weight * importance
                + self._config.relevance_weight * relevance
            )
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [item for _, item in scored[:limit]]

        # Update access counts
        for item in results:
            item.access_count += 1
            item.last_accessed = current_time

        return results

    def get_recent(self, limit: int = 10) -> List[TownMemoryItem]:
        """Get the most recent memories."""
        return sorted(self._items, key=lambda m: m.timestamp, reverse=True)[:limit]

    def get_reflections(self) -> List[TownMemoryItem]:
        """Get all reflection-type memories."""
        return [m for m in self._items if m.kind == "reflection"]

    def should_reflect(self) -> bool:
        """Check if it's time for a reflection."""
        return self._observations_since_reflection >= self._config.reflection_threshold

    def add_reflection(self, content: str, sim_time: float) -> TownMemoryItem:
        """Add a reflection and reset the counter."""
        item = self.add_observation(
            content=content,
            sim_time=sim_time,
            kind="reflection",
            importance=8,
        )
        self._observations_since_reflection = 0
        return item

    def format_recent_for_prompt(self, limit: int = 10) -> str:
        """Format recent memories as text for LLM prompts."""
        recent = self.get_recent(limit)
        if not recent:
            return "No memories yet."
        lines = []
        for m in recent:
            kind_tag = f"[{m.kind}]" if m.kind != "observation" else ""
            lines.append(f"- {kind_tag} {m.content}")
        return "\n".join(lines)

    def _score_importance(self, content: str) -> int:
        """Score importance using LLM or fallback heuristic."""
        if self._llm is not None:
            try:
                return self._llm.score_importance(content)
            except Exception:
                pass

        # Heuristic fallback
        high_keywords = ["fight", "love", "death", "secret", "surprise", "important", "emergency"]
        low_keywords = ["walk", "sit", "stand", "routine", "usual"]
        lower = content.lower()
        for kw in high_keywords:
            if kw in lower:
                return 7
        for kw in low_keywords:
            if kw in lower:
                return 3
        return 5

    @property
    def count(self) -> int:
        return len(self._items)
