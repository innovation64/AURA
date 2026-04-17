from __future__ import annotations

import math
import re
import time
from collections import Counter, deque
from typing import Deque, Dict, List, Optional, Sequence

from .types import MemoryItem, SceneState


class MemoryStore:
    def update(self, scene: SceneState) -> None:
        raise NotImplementedError

    def recall(self, query: Optional[str] = None, limit: int = 5) -> List[MemoryItem]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Lightweight TF-IDF helpers (no external deps)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in on at to for "
    "with by from as into about between through and or but not no nor "
    "so yet both each all any such that this these those it its i me "
    "my we our you your he him his she her they them their what which "
    "who whom how when where why am is are".split()
)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP_WORDS and len(w) > 1]


def _cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse term vectors."""
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class _TFIDFIndex:
    """Lightweight inverted index for TF-IDF scoring over memory items."""

    def __init__(self) -> None:
        self._doc_freq: Counter = Counter()  # term -> num docs containing it
        self._doc_vectors: List[Dict[str, float]] = []  # per-doc TF vectors
        self._num_docs: int = 0

    def add(self, tokens: List[str]) -> None:
        tf: Dict[str, float] = {}
        total = len(tokens) if tokens else 1
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1.0
        for t in tf:
            tf[t] /= total
        self._doc_vectors.append(tf)
        # Update document frequency
        for t in set(tokens):
            self._doc_freq[t] += 1
        self._num_docs += 1

    def query(self, tokens: List[str]) -> List[float]:
        """Return similarity scores for all documents."""
        if not tokens or self._num_docs == 0:
            return [0.0] * self._num_docs

        # Build query TF-IDF vector
        q_tf: Dict[str, float] = {}
        total = len(tokens)
        for t in tokens:
            q_tf[t] = q_tf.get(t, 0) + 1.0
        for t in q_tf:
            q_tf[t] /= total

        q_tfidf: Dict[str, float] = {}
        for t, tf_val in q_tf.items():
            df = self._doc_freq.get(t, 0)
            idf = math.log((self._num_docs + 1) / (df + 1)) + 1
            q_tfidf[t] = tf_val * idf

        # Score each document
        scores = []
        for doc_tf in self._doc_vectors:
            doc_tfidf: Dict[str, float] = {}
            for t, tf_val in doc_tf.items():
                df = self._doc_freq.get(t, 0)
                idf = math.log((self._num_docs + 1) / (df + 1)) + 1
                doc_tfidf[t] = tf_val * idf
            scores.append(_cosine_similarity(q_tfidf, doc_tfidf))

        return scores


class EphemeralMemory(MemoryStore):
    """Improved memory with TF-IDF semantic retrieval + recency + importance weighting."""

    def __init__(
        self,
        max_items: int = 100,
        recency_decay: float = 0.95,
        weight_semantic: float = 0.5,
        weight_recency: float = 0.3,
        weight_importance: float = 0.2,
    ) -> None:
        self._items: Deque[MemoryItem] = deque(maxlen=max_items)
        self._index = _TFIDFIndex()
        self._recency_decay: float = recency_decay
        self._w_sem: float = weight_semantic
        self._w_rec: float = weight_recency
        self._w_imp: float = weight_importance

    def update(self, scene: SceneState) -> None:
        content = scene.summary
        importance = self._estimate_importance(content, scene.entities)
        item = MemoryItem(
            content=content,
            metadata={
                "entities": scene.entities,
                "importance": importance,
            },
        )
        self._items.appendleft(item)
        tokens = _tokenize(content)
        for entity in scene.entities:
            tokens.extend(_tokenize(entity))
        self._index.add(tokens)

    def recall(self, query: Optional[str] = None, limit: int = 5) -> List[MemoryItem]:
        if not self._items:
            return []

        if not query:
            return list(self._items)[:limit]

        query_tokens = _tokenize(query)
        items_list = list(self._items)

        # TF-IDF similarity scores (index order matches deque insertion order)
        # Index is append-only; items deque may have dropped old items if full.
        # We score only the most recent len(items_list) entries from the index.
        all_scores = self._index.query(query_tokens)
        # Take only the last len(items_list) scores (most recent)
        offset = len(all_scores) - len(items_list)
        # Items are in reverse order (newest first), index is oldest first
        # Reverse to align: index[offset:] corresponds to items in reverse
        semantic_scores = list(reversed(all_scores[offset:])) if offset >= 0 else list(reversed(all_scores))

        if len(semantic_scores) < len(items_list):
            semantic_scores.extend([0.0] * (len(items_list) - len(semantic_scores)))

        # Combined scoring: semantic + recency + importance
        scored = []
        for i, item in enumerate(items_list):
            sem = semantic_scores[i] if i < len(semantic_scores) else 0.0
            recency = self._recency_decay ** i  # newest = 1.0, decays with position
            importance = item.metadata.get("importance", 0.5)

            # Weighted combination
            combined = self._w_sem * sem + self._w_rec * recency + self._w_imp * importance
            scored.append((combined, i, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, _, item in scored[:limit]]

    @staticmethod
    def _estimate_importance(content: str, entities: List[str]) -> float:
        """Heuristic importance score (0-1)."""
        score = 0.3  # baseline

        content_lower = content.lower()

        # Conversations and social interactions are important
        if any(kw in content_lower for kw in ("conversation", "talked", "discussed", "met", "greeted")):
            score += 0.2

        # Plans and decisions
        if any(kw in content_lower for kw in ("plan", "decide", "schedule", "goal", "intend")):
            score += 0.15

        # Emotional or significant events
        if any(kw in content_lower for kw in ("important", "urgent", "critical", "surprised", "angry", "happy")):
            score += 0.15

        # More entities = richer context
        if len(entities) >= 3:
            score += 0.1
        elif len(entities) >= 1:
            score += 0.05

        # Length indicates detail
        if len(content) > 200:
            score += 0.1

        return min(score, 1.0)
