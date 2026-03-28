"""LLM-driven reasoner — uses language model for intent planning and action selection."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

from ..llm import LLMEngine
from ..reason import Reasoner
from ..types import MemoryItem, ReasoningResult, SceneState

logger = logging.getLogger(__name__)

_REASON_SYSTEM = """\
You are a reasoning engine for an autonomous agent. Given the current scene and relevant memories,
decide what the agent should do next.

Output a JSON object with:
- "intent": a short description of the goal (e.g., "answer_query:What is X?", "explore_environment", "recall_and_respond")
- "rationale": 1-2 sentences explaining WHY this is the right action
- "actions": list of action strings to take (e.g., ["respond", "include_context", "use_memory"])
- "confidence": float 0.0-1.0 indicating how confident you are

Respond ONLY with valid JSON, no markdown."""


class LLMReasoner(Reasoner):
    """Uses LLM to plan agent intent based on scene + memories."""

    def __init__(self, llm: Optional[LLMEngine] = None) -> None:
        self._llm = llm

    def plan(
        self,
        scene: SceneState,
        memories: Sequence[MemoryItem],
        user_query: Optional[str] = None,
    ) -> ReasoningResult:
        if not self._llm:
            return self._fallback_plan(scene, memories, user_query)

        try:
            # Build context
            memory_text = ""
            if memories:
                lines = [f"- [{m.metadata.get('kind', 'obs')}] {m.content}" for m in memories[:10]]
                memory_text = "\n".join(lines)

            user_msg_parts = [f"Scene: {scene.summary}"]
            if scene.entities:
                user_msg_parts.append(f"Entities: {', '.join(scene.entities[:15])}")
            if memory_text:
                user_msg_parts.append(f"Relevant memories:\n{memory_text}")
            if user_query:
                user_msg_parts.append(f"User query: {user_query}")
            else:
                user_msg_parts.append("No explicit query. Decide the best proactive action.")

            result = self._llm.chat_json(
                [
                    {"role": "system", "content": _REASON_SYSTEM},
                    {"role": "user", "content": "\n\n".join(user_msg_parts)},
                ],
                temperature=0.3,
                max_tokens=512,
            )

            return ReasoningResult(
                intent=result.get("intent", "respond"),
                rationale=result.get("rationale", "LLM-driven reasoning."),
                actions=result.get("actions", ["respond"]),
                metadata={
                    "confidence": result.get("confidence", 0.5),
                    "scene_summary": scene.summary,
                    "memory_count": len(memories),
                    "source": "llm",
                },
            )
        except Exception as e:
            logger.warning("LLM reasoning failed, using fallback: %s", e)
            return self._fallback_plan(scene, memories, user_query)

    def _fallback_plan(
        self,
        scene: SceneState,
        memories: Sequence[MemoryItem],
        user_query: Optional[str] = None,
    ) -> ReasoningResult:
        if user_query:
            return ReasoningResult(
                intent=f"answer_query:{user_query}",
                rationale="User query provided; answering with scene and memory context.",
                actions=["respond", "include_context"],
                metadata={"source": "fallback", "memory_count": len(memories)},
            )
        return ReasoningResult(
            intent="summarize_environment",
            rationale="No query; summarizing current environment state.",
            actions=["respond"],
            metadata={"source": "fallback"},
        )


def build_llm_reasoner(**kwargs: Any) -> LLMReasoner:
    return LLMReasoner(llm=kwargs.get("llm"))
