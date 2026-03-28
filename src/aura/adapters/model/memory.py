"""Neural model memory — MemoryStore with embedded plasticity hooks."""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from typing import Any, Dict, List, Optional

from ...memory import MemoryStore
from ...types import MemoryItem, SceneState
from .plasticity import PlasticityEngine

logger = logging.getLogger(__name__)


class ModelMemory(MemoryStore):
    """
    Memory store backed by neural plasticity model.

    Combines persistent storage with dynamic plasticity:
    - Every store triggers on_store hooks (weight initialization)
    - Every recall triggers on_recall hooks (Hebbian, contrastive, forgetting)
    - Retrieval scores are modulated by plasticity weights

    This is what makes AURA's memory 'alive' — it learns and adapts
    from every interaction, just like biological memory systems.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        max_items: int = 500,
        plasticity: Optional[PlasticityEngine] = None,
    ) -> None:
        self._db_path = db_path
        self._max_items = max_items
        self.plasticity = plasticity or PlasticityEngine()
        self._conn = self._init_db()

    def _init_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                importance REAL DEFAULT 0.5,
                kind TEXT DEFAULT 'observation',
                entities TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.commit()
        return conn

    def update(self, scene: SceneState) -> None:
        cursor = self._conn.execute(
            "INSERT INTO memories (content, timestamp, importance, kind, entities, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (scene.summary, time.time(), 0.5, "observation", json.dumps(scene.entities), "{}"),
        )
        self._conn.commit()
        mem_id = str(cursor.lastrowid)

        # Trigger plasticity on_store
        self.plasticity.on_store(mem_id, scene.summary, {"entities": scene.entities})

    def recall(self, query: Optional[str] = None, limit: int = 5) -> List[MemoryItem]:
        rows = self._conn.execute(
            "SELECT id, content, timestamp, importance, kind, entities, metadata FROM memories"
        ).fetchall()
        if not rows:
            return []

        now = time.time()
        query_words = set((query or "").lower().split()) if query else set()

        scored: List[tuple[float, Any]] = []
        for row in rows:
            mid = str(row[0])
            content = row[1]
            ts = row[2]
            importance = row[3]
            entities = json.loads(row[5])

            # Base recency score
            recency = math.exp(-0.001 * (now - ts))

            # Relevance
            if query_words:
                content_words = set(content.lower().split())
                entity_words = set(e.lower() for e in entities)
                overlap = len(query_words & (content_words | entity_words))
                relevance = min(overlap / max(len(query_words), 1), 1.0)
            else:
                relevance = 0.5

            # Plasticity modulation — this is the key differentiator
            plasticity_weight = self.plasticity.get_weight(mid)

            score = (0.2 * recency + 0.3 * importance + 0.2 * relevance) * plasticity_weight
            scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = scored[:limit]

        # Build MemoryItems
        items = []
        retrieved_ids = []
        relevance_scores = []
        for score, row in results:
            mid = str(row[0])
            retrieved_ids.append(mid)
            relevance_scores.append(score)
            items.append(MemoryItem(
                content=row[1],
                metadata={
                    "importance": row[3],
                    "kind": row[4],
                    "entities": json.loads(row[5]),
                    "plasticity_weight": self.plasticity.get_weight(mid),
                    "model_backed": True,
                },
            ))

        # Trigger plasticity on_recall — this is where learning happens
        if retrieved_ids:
            self.plasticity.on_recall(query or "", retrieved_ids, relevance_scores)

        return items

    def get_stats(self) -> Dict[str, Any]:
        count = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        return {
            "total_memories": count,
            "db_path": self._db_path,
            "plasticity": self.plasticity.get_stats(),
        }


def build_model_memory(**kwargs: Any) -> ModelMemory:
    return ModelMemory(
        db_path=kwargs.get("db_path", ":memory:"),
        max_items=kwargs.get("memory_limit", 500),
    )
