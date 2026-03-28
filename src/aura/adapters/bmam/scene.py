"""BMAM Scene adapter — enriches scene with BMAM memory search context."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from ...scene import SceneModel
from ...types import EnvironmentSignal, SceneState
from .client import BMAMClient, BMAMConfig

logger = logging.getLogger(__name__)


class BMAMScene(SceneModel):
    """
    Enriches scene building with BMAM's memory search.

    For each signal's text content, queries BMAM's semantic search
    to find related memories and context.
    """

    def __init__(self, client: Optional[BMAMClient] = None) -> None:
        self._client = client

    def build(self, signals: Sequence[EnvironmentSignal]) -> SceneState:
        entities: List[str] = []
        signal_texts: List[str] = []

        for sig in signals:
            payload = sig.payload
            if isinstance(payload, dict):
                if "entities" in payload and isinstance(payload["entities"], list):
                    entities.extend(str(e) for e in payload["entities"])
                text = payload.get("text") or payload.get("value") or payload.get("output")
                if text:
                    signal_texts.append(str(text))

        unique_entities = sorted(set(entities))

        # If BMAM available, enrich with memory context
        related_memories: List[Dict] = []
        if self._client and signal_texts:
            combined_query = " ".join(signal_texts[:3])[:500]
            try:
                result = self._client.search(
                    query=combined_query,
                    limit=3,
                    use_brain_retrieval=False,  # fast semantic search
                )
                related_memories = result.get("results", [])
            except Exception as e:
                logger.debug("BMAM scene enrichment skipped: %s", e)

        # Build summary
        summary_parts = [f"{len(signals)} signals observed"]
        if unique_entities:
            summary_parts.append(f"entities: {', '.join(unique_entities[:10])}")
        if related_memories:
            summary_parts.append(f"{len(related_memories)} related memories found")

        return SceneState(
            summary=", ".join(summary_parts),
            entities=unique_entities,
            context={
                "signals": [{"source": s.source, "modality": s.modality} for s in signals],
                "related_memories": related_memories,
                "context_type": "bmam_enriched" if related_memories else "basic",
            },
        )


def build_bmam_scene(**kwargs: Any) -> BMAMScene:
    config = BMAMConfig.from_env()
    if kwargs.get("bmam_base_url"):
        config.base_url = kwargs["bmam_base_url"]
    return BMAMScene(client=BMAMClient(config))
