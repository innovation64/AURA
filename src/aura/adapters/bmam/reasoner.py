"""BMAM Reasoner adapter — delegates reasoning to BMAM's brain/process endpoint."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from ...reason import Reasoner
from ...types import MemoryItem, ReasoningResult, SceneState
from .client import BMAMClient, BMAMConfig

logger = logging.getLogger(__name__)


class BMAMReasoner(Reasoner):
    """
    Delegates reasoning to BMAM's full processing pipeline.

    POST /v1/brain/process → BrainInspiredCoordinator.process_user_input()
      → RoutingManager routes to relevant brain regions
      → CapabilityOrchestrator executes cognitive capabilities
      → Result includes response, agents involved, activation trace

    When BMAM is unavailable, falls back to simple heuristic reasoning.
    """

    def __init__(self, client: Optional[BMAMClient] = None) -> None:
        self._client = client

    def plan(
        self,
        scene: SceneState,
        memories: Sequence[MemoryItem],
        user_query: Optional[str] = None,
    ) -> ReasoningResult:
        # If BMAM available and we have a query, delegate to BMAM
        if self._client and user_query:
            try:
                result = self._client.process_input(
                    user_input=user_query,
                    context={
                        "scene_summary": scene.summary,
                        "entities": scene.entities,
                        "memory_count": len(memories),
                        "source": "aura",
                    },
                )
                if result.get("success"):
                    return ReasoningResult(
                        intent=f"answer_query:{user_query}",
                        rationale=result.get("response", "BMAM processed successfully."),
                        actions=["respond", "include_context"],
                        metadata={
                            "source": "bmam",
                            "agents_involved": result.get("agents_involved", []),
                            "memories_retrieved": result.get("memories_retrieved", []),
                            "processing_time": result.get("processing_time", 0),
                            "bmam_response": result.get("response", ""),
                            "insights": result.get("insights", {}),
                        },
                    )
            except ConnectionError:
                logger.warning("BMAM reasoning unavailable, using fallback")
            except Exception as e:
                logger.error("BMAM reasoning failed: %s", e)

        # Fallback
        if user_query:
            return ReasoningResult(
                intent=f"answer_query:{user_query}",
                rationale="BMAM unavailable; using fallback reasoning.",
                actions=["respond", "include_context"],
                metadata={"source": "fallback", "memory_count": len(memories)},
            )
        return ReasoningResult(
            intent="summarize_environment",
            rationale="No query; summarizing.",
            actions=["respond"],
            metadata={"source": "fallback"},
        )


def build_bmam_reasoner(**kwargs: Any) -> BMAMReasoner:
    config = BMAMConfig.from_env()
    if kwargs.get("bmam_base_url"):
        config.base_url = kwargs["bmam_base_url"]
    return BMAMReasoner(client=BMAMClient(config))
