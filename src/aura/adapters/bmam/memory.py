"""BMAM Memory adapter — bridges AURA MemoryStore to BMAM's REST API."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ...memory import MemoryStore
from ...types import MemoryItem, SceneState
from .client import BMAMClient, BMAMConfig

logger = logging.getLogger(__name__)


class BMAMMemory(MemoryStore):
    """
    Bridges AURA's MemoryStore to BMAM via HTTP REST API.

    Store → POST /v1/memories
    Recall → POST /v1/brain/retrieve (5-brain-region distributed retrieval)

    BMAM handles all the complexity internally:
    - Hippocampus (episodic), TemporalLobe (semantic+KG), Amygdala (emotion)
    - Fast path (BasalGanglia pattern match) vs Slow path (Prefrontal feedback loop)
    - LLM calls, embeddings, caching — all managed by BMAM's own infrastructure
    """

    def __init__(
        self,
        client: Optional[BMAMClient] = None,
        user_id: str = "default",
        use_brain_retrieval: bool = True,
    ) -> None:
        self._client = client or BMAMClient()
        self._user_id = user_id
        self._use_brain = use_brain_retrieval

    def update(self, scene: SceneState) -> None:
        """Store scene observation via BMAM API."""
        if not self._client:
            return

        context_type = scene.context.get("context_type", "observation")
        importance = 0.6 if context_type in ("task", "conversation") else 0.4

        try:
            self._client.store_memory(
                content=scene.summary,
                user_id=self._user_id,
                importance=importance,
                metadata={
                    "entities": scene.entities,
                    "context_type": context_type,
                    "source": "aura",
                },
            )
        except ConnectionError as e:
            logger.warning("BMAM store failed (service unavailable): %s", e)
        except Exception as e:
            logger.error("BMAM store failed: %s", e)

    def recall(self, query: Optional[str] = None, limit: int = 5) -> List[MemoryItem]:
        """Retrieve from BMAM via brain-distributed or semantic search."""
        if not self._client or not query:
            return []

        try:
            if self._use_brain:
                result = self._client.brain_retrieve(
                    query=query, k=limit,
                    context={"user_id": self._user_id, "source": "aura"},
                )
                memories = result.get("memories", [])
                path_type = result.get("path_type", "unknown")
                confidence = result.get("confidence", 0.0)
            else:
                result = self._client.search(
                    query=query, limit=limit,
                    context={"user_id": self._user_id},
                )
                memories = result.get("results", [])
                path_type = "semantic"
                confidence = 1.0

            return [
                MemoryItem(
                    content=mem.get("content", ""),
                    metadata={
                        "id": mem.get("id", ""),
                        "score": mem.get("score", 0.0),
                        "memory_type": mem.get("memory_type", "episodic"),
                        "brain_region": mem.get("brain_region", ""),
                        "path_type": path_type,
                        "confidence": confidence,
                        "bmam": True,
                    },
                )
                for mem in memories
            ]
        except ConnectionError as e:
            logger.warning("BMAM recall failed (service unavailable): %s", e)
            return []
        except Exception as e:
            logger.error("BMAM recall failed: %s", e)
            return []

    # ── Extended BMAM operations ──────────────────────────────

    def process(self, user_input: str) -> Dict[str, Any]:
        """Full BMAM pipeline: store + retrieve + reason. Returns processed result."""
        try:
            return self._client.process_input(user_input, user_id=self._user_id)
        except Exception as e:
            logger.error("BMAM process failed: %s", e)
            return {"error": str(e)}

    def consolidate(self) -> Dict[str, Any]:
        try:
            return self._client.consolidate()
        except Exception as e:
            return {"error": str(e)}

    def forget(self, threshold: float = 0.8) -> Dict[str, Any]:
        try:
            return self._client.forget(threshold)
        except Exception as e:
            return {"error": str(e)}

    def feedback(self, query: str, response: str, reward: float) -> Dict[str, Any]:
        try:
            return self._client.feedback(query, response, reward)
        except Exception as e:
            return {"error": str(e)}

    def get_preferences(self, query: str, k: int = 5) -> List[str]:
        """Retrieve user preferences relevant to a query."""
        try:
            result = self._client.get_preferences(query, user_id=self._user_id, k=k)
            return result.get("preferences", [])
        except Exception as e:
            logger.error("BMAM get_preferences failed: %s", e)
            return []

    def get_persona_portrait(self) -> Dict[str, Any]:
        """Get synthesized user persona portrait."""
        try:
            return self._client.get_persona_portrait(user_id=self._user_id)
        except Exception as e:
            logger.error("BMAM get_persona_portrait failed: %s", e)
            return {}

    def is_available(self) -> bool:
        return self._client.is_available() if self._client else False


def build_bmam_memory(**kwargs: Any) -> BMAMMemory:
    config = BMAMConfig.from_env()
    if kwargs.get("bmam_base_url"):
        config.base_url = kwargs["bmam_base_url"]
    return BMAMMemory(
        client=BMAMClient(config),
        user_id=kwargs.get("user_id", "default"),
    )
