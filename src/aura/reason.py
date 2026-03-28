from __future__ import annotations

from typing import List, Optional, Sequence

from .types import MemoryItem, ReasoningResult, SceneState


class Reasoner:
    def plan(
        self,
        scene: SceneState,
        memories: Sequence[MemoryItem],
        user_query: Optional[str] = None,
    ) -> ReasoningResult:
        raise NotImplementedError


class SimpleReasoner(Reasoner):
    def plan(
        self,
        scene: SceneState,
        memories: Sequence[MemoryItem],
        user_query: Optional[str] = None,
    ) -> ReasoningResult:
        if user_query:
            intent = f"answer_query:{user_query}"
            rationale = "User query provided; prioritize answering with current scene and memory."
            actions: List[str] = ["respond", "include_context"]
        else:
            intent = "summarize_environment"
            rationale = "No explicit query; provide a concise environment summary."
            actions = ["respond"]

        metadata = {
            "scene_summary": scene.summary,
            "memory_count": len(memories),
        }
        return ReasoningResult(
            intent=intent,
            rationale=rationale,
            actions=actions,
            metadata=metadata,
        )
