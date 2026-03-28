"""Persistent memory with importance scoring and weighted retrieval."""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..memory import MemoryStore
from ..types import MemoryItem, SceneState

logger = logging.getLogger(__name__)


@dataclass
class ScoredMemory:
    """Internal memory representation with scoring metadata."""

    id: int
    content: str
    timestamp: float
    importance: float  # 0.0 - 1.0
    access_count: int
    kind: str  # observation, reflection, conversation
    entities: List[str]
    metadata: Dict[str, Any]


class PersistentMemory(MemoryStore):
    """
    Production memory store with:
    - SQLite persistence (survives restart)
    - Importance scoring (LLM or heuristic)
    - Weighted retrieval (recency + importance + relevance)
    - Reflection synthesis
    - Capacity management with forgetting
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        max_items: int = 500,
        llm: Any = None,
        recency_weight: float = 0.3,
        importance_weight: float = 0.4,
        relevance_weight: float = 0.3,
    ) -> None:
        self._db_path = db_path
        self._max_items = max_items
        self._llm = llm
        self._recency_w = recency_weight
        self._importance_w = importance_weight
        self._relevance_w = relevance_weight
        self._observations_since_reflection = 0
        self._conn = self._init_db()

    def _init_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                kind TEXT DEFAULT 'observation',
                entities TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_ts ON memories(timestamp DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC)")
        conn.commit()
        return conn

    # ── AURA MemoryStore interface ────────────────────────────────

    def update(self, scene: SceneState) -> None:
        importance = self._score_importance(scene.summary)
        self._insert(
            content=scene.summary,
            importance=importance,
            kind="observation",
            entities=scene.entities,
            metadata={"context_type": scene.context.get("context_type", "unknown")},
        )
        self._observations_since_reflection += 1
        self._enforce_capacity()

    def recall(self, query: Optional[str] = None, limit: int = 5) -> List[MemoryItem]:
        scored = self._retrieve_scored(query or "", limit)
        return [
            MemoryItem(
                content=m.content,
                metadata={
                    "importance": m.importance,
                    "kind": m.kind,
                    "entities": m.entities,
                    "access_count": m.access_count,
                },
            )
            for m in scored
        ]

    # ── Extended operations ───────────────────────────────────────

    def store(
        self,
        content: str,
        importance: Optional[float] = None,
        kind: str = "observation",
        entities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        if importance is None:
            importance = self._score_importance(content)
        return self._insert(content, importance, kind, entities or [], metadata or {})

    def should_reflect(self, threshold: int = 10) -> bool:
        return self._observations_since_reflection >= threshold

    def add_reflection(self, content: str) -> int:
        rid = self._insert(content, importance=0.8, kind="reflection", entities=[], metadata={})
        self._observations_since_reflection = 0
        return rid

    def get_stats(self) -> Dict[str, Any]:
        row = self._conn.execute("SELECT COUNT(*), AVG(importance) FROM memories").fetchone()
        return {"total": row[0], "avg_importance": round(row[1] or 0, 3), "db_path": self._db_path}

    # ── Internal ──────────────────────────────────────────────────

    def _insert(
        self, content: str, importance: float, kind: str,
        entities: List[str], metadata: Dict[str, Any],
    ) -> int:
        cursor = self._conn.execute(
            "INSERT INTO memories (content, timestamp, importance, kind, entities, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (content, time.time(), importance, kind, json.dumps(entities), json.dumps(metadata)),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def _retrieve_scored(self, query: str, limit: int) -> List[ScoredMemory]:
        rows = self._conn.execute(
            "SELECT id, content, timestamp, importance, access_count, kind, entities, metadata FROM memories"
        ).fetchall()
        if not rows:
            return []

        now = time.time()
        query_lower = query.lower()
        query_words = set(query_lower.split()) if query_lower else set()

        scored: List[tuple[float, ScoredMemory]] = []
        for row in rows:
            mem = ScoredMemory(
                id=row[0], content=row[1], timestamp=row[2], importance=row[3],
                access_count=row[4], kind=row[5],
                entities=json.loads(row[6]), metadata=json.loads(row[7]),
            )
            # Recency: exponential decay
            age = now - mem.timestamp
            recency = math.exp(-0.001 * max(age, 0))

            # Relevance: keyword overlap
            if query_words:
                content_words = set(mem.content.lower().split())
                entity_words = set(e.lower() for e in mem.entities)
                overlap = len(query_words & (content_words | entity_words))
                relevance = min(overlap / max(len(query_words), 1), 1.0)
            else:
                relevance = 0.5

            score = (
                self._recency_w * recency
                + self._importance_w * mem.importance
                + self._relevance_w * relevance
            )
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [mem for _, mem in scored[:limit]]

        # Update access counts
        for mem in results:
            self._conn.execute(
                "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
                (mem.id,),
            )
        self._conn.commit()
        return results

    def _enforce_capacity(self) -> None:
        count = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if count <= self._max_items:
            return
        excess = count - self._max_items
        self._conn.execute(
            "DELETE FROM memories WHERE id IN (SELECT id FROM memories ORDER BY importance ASC, timestamp ASC LIMIT ?)",
            (excess,),
        )
        self._conn.commit()
        logger.debug("Forgot %d low-importance memories (capacity: %d)", excess, self._max_items)

    def _score_importance(self, content: str) -> float:
        """Score importance 0.0-1.0 using LLM or heuristic."""
        if self._llm:
            try:
                messages = [
                    {"role": "system", "content": "Rate the importance of this observation for long-term memory on a scale of 1-10. Respond with ONLY a single integer."},
                    {"role": "user", "content": content},
                ]
                result = self._llm.chat(messages, temperature=0.0, max_tokens=8)
                score = int(result.strip().split()[0])
                return max(0.1, min(1.0, score / 10.0))
            except Exception:
                pass

        # Heuristic fallback
        lower = content.lower()
        high = ["important", "critical", "emergency", "love", "death", "secret", "decision", "problem", "error"]
        low = ["routine", "usual", "normal", "walk", "sit", "stand"]
        for kw in high:
            if kw in lower:
                return 0.7
        for kw in low:
            if kw in lower:
                return 0.3
        return 0.5

    def close(self) -> None:
        self._conn.close()


def build_llm_memory(**kwargs: Any) -> PersistentMemory:
    return PersistentMemory(
        db_path=kwargs.get("db_path", ":memory:"),
        max_items=kwargs.get("memory_limit", 500),
        llm=kwargs.get("llm"),
    )
