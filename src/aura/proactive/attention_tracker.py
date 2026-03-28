"""Attention tracker — learns what the agent pays attention to.

Implements a simple online learning algorithm that adjusts source and
keyword weights based on whether the agent uses or ignores pushed context.
These weights feed back into the RelevanceScorer for personalization.
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .context_assembler import EnvironmentContext

logger = logging.getLogger(__name__)


@dataclass
class AttentionPattern:
    """Learned attention weights for personalized relevance scoring."""
    source_weights: Dict[str, float] = field(default_factory=dict)
    keyword_weights: Dict[str, float] = field(default_factory=dict)
    total_interactions: int = 0
    total_used: int = 0
    total_ignored: int = 0
    last_updated: float = field(default_factory=time.time)


class AttentionTracker:
    """Tracks which pushed contexts the agent actually uses.

    Learning algorithm:
    - When agent uses context from source X → increase source_weights[X]
    - When agent ignores context from source X → decrease source_weights[X]
    - Asymmetric learning: larger penalty for ignoring (to differentiate quickly)
    - Per-source EMA with configurable learning rate
    - Keywords extracted from agent queries boost keyword_weights
    - Wider weight range [0.05, 3.0] enables meaningful score differentiation
    """

    def __init__(self, learning_rate: float = 0.15, decay_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self._pattern = AttentionPattern()
        self._pending_sources: Set[str] = set()
        self._pending_keywords: Set[str] = set()
        # Per-source interaction counts for confidence-weighted updates
        self._source_used: Dict[str, int] = defaultdict(int)
        self._source_total: Dict[str, int] = defaultdict(int)

    def on_push(self, context: EnvironmentContext) -> None:
        """Record what was pushed — tracks sources for later feedback."""
        self._pending_sources.clear()
        for event in (context.critical_alerts or []) + (context.relevant_changes or []):
            self._pending_sources.add(event.source)

    def on_agent_action(self, action: str, used_context: bool) -> None:
        """Record whether the agent used the pushed context.

        Per-source differentiation: when the agent uses context, only
        reinforce sources whose domain keywords appear in the agent's action.
        Sources that were pushed but not referenced get a mild penalty.
        When the agent ignores context entirely, all sources are penalized.
        """
        self._pattern.total_interactions += 1
        action_lower = action.lower()

        if used_context:
            self._pattern.total_used += 1

            # Determine which sources the agent actually responded to
            addressed: Set[str] = set()
            for source in self._pending_sources:
                source_kws = _source_domain_keywords(source)
                if any(kw in action_lower for kw in source_kws):
                    addressed.add(source)

            # Fallback: if no specific source matched, reinforce all
            if not addressed:
                addressed = set(self._pending_sources)

            for source in self._pending_sources:
                self._source_total[source] += 1
                current = self._pattern.source_weights.get(source, 1.0)
                if source in addressed:
                    self._source_used[source] += 1
                    delta = self.learning_rate
                else:
                    # Pushed but not what the agent cared about → mild penalty
                    delta = -self.learning_rate * 0.4
                self._pattern.source_weights[source] = max(0.05, min(3.0, current + delta))
        else:
            self._pattern.total_ignored += 1
            for source in self._pending_sources:
                self._source_total[source] += 1
                current = self._pattern.source_weights.get(source, 1.0)
                self._pattern.source_weights[source] = max(
                    0.05, min(3.0, current - self.learning_rate * 0.8)
                )

        self._pattern.last_updated = time.time()

    def on_agent_query(self, query: str) -> None:
        """Extract keywords from agent query to learn what it's looking for."""
        words = re.findall(r'\b[a-z]{3,}\b', query.lower())
        for word in words:
            # Skip common stopwords
            if word in _STOPWORDS:
                continue
            current = self._pattern.keyword_weights.get(word, 0.0)
            self._pattern.keyword_weights[word] = min(2.0, current + self.learning_rate)

        # Decay old keywords
        for kw in list(self._pattern.keyword_weights):
            self._pattern.keyword_weights[kw] = max(
                0.0, self._pattern.keyword_weights[kw] - self.decay_rate
            )
            if self._pattern.keyword_weights[kw] <= 0:
                del self._pattern.keyword_weights[kw]

    def get_attention_weights(self) -> AttentionPattern:
        """Return current learned attention pattern."""
        return self._pattern

    def get_source_weight(self, source: str) -> float:
        """Get learned weight for a specific source."""
        return self._pattern.source_weights.get(source, 1.0)

    def get_keyword_boost(self, text: str) -> float:
        """Get keyword-based boost for a piece of text.

        Returns a value in [0, 1.5] — enables meaningful relevance amplification
        when accumulated keywords match strongly.
        """
        words = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
        if not words:
            return 0.0
        boost = 0.0
        matches = 0
        for word in words:
            w = self._pattern.keyword_weights.get(word, 0.0)
            if w > 0:
                boost += w
                matches += 1
        if matches == 0:
            return 0.0
        # Normalize by match count to reward concentrated relevance
        return min(1.5, boost / max(matches, 1) * matches * 0.3)

    def get_source_use_rate(self, source: str) -> float:
        """Return the historical use rate for a specific source."""
        total = self._source_total.get(source, 0)
        if total == 0:
            return 0.5  # unknown source — neutral
        return self._source_used[source] / total

    def get_stats(self) -> dict:
        return {
            "total_interactions": self._pattern.total_interactions,
            "total_used": self._pattern.total_used,
            "total_ignored": self._pattern.total_ignored,
            "use_rate": self._pattern.total_used / max(self._pattern.total_interactions, 1),
            "tracked_sources": len(self._pattern.source_weights),
            "tracked_keywords": len(self._pattern.keyword_weights),
            "source_weights": dict(self._pattern.source_weights),
            "top_keywords": dict(sorted(
                self._pattern.keyword_weights.items(),
                key=lambda x: x[1], reverse=True,
            )[:10]),
        }


# Domain keyword mapping: source name fragment → action keywords that indicate
# the agent is responding to that source's events.
_SOURCE_DOMAIN_MAP: Dict[str, List[str]] = {
    "network": ["service", "down", "latency", "connect", "restart", "port", "network", "status"],
    "system": ["cpu", "memory", "disk", "resource", "spike", "utilization", "load", "gpu"],
    "process": ["process", "pid", "suspicious", "miner", "kill", "terminate", "activity"],
    "filesystem": ["file", "conflict", "merge", "diff", "resolve", "git", "modified"],
    "security": ["security", "unauthorized", "scan", "compromise", "breach", "intrusion"],
    "service_down": ["service", "down", "restart", "failure", "connect", "refused"],
    "resource_spike": ["spike", "resource", "cpu", "memory", "gpu", "exhaustion"],
    "file_conflict": ["file", "conflict", "merge", "resolve"],
}


def _source_domain_keywords(source: str) -> List[str]:
    """Extract domain keywords for a source name.

    Uses word-boundary matching on the source name parts to avoid
    false matches like "system" matching "filesystem".
    """
    # Split source into distinct parts: "probe.file_conflict" → {"probe", "file", "conflict"}
    source_parts = set(re.split(r'[._]', source.lower()))
    keywords: List[str] = []
    for domain, kws in _SOURCE_DOMAIN_MAP.items():
        # Domain must match a whole part, not a substring
        domain_parts = set(domain.split("_"))
        if domain_parts & source_parts:
            keywords.extend(kws)
    if not keywords:
        keywords = [p for p in source_parts if len(p) > 2 and p != "probe"]
    return keywords


_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all",
    "can", "had", "her", "was", "one", "our", "out", "has",
    "have", "been", "some", "them", "than", "its", "over",
    "into", "just", "about", "could", "with", "this", "that",
    "what", "when", "where", "which", "who", "how", "from",
    "each", "will", "they", "been", "said", "many", "most",
    "like", "more", "also", "very", "much", "does", "did",
    "get", "let", "may", "any", "use", "show", "check",
})
