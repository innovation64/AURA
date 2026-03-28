"""Smart actor — routes reasoning intents to structured actions."""

from __future__ import annotations

from typing import Any

from ..act import Actor
from ..types import Action, ReasoningResult, SceneState


class SmartActor(Actor):
    """Maps reasoning intents to concrete action types with structured payloads."""

    def act(self, reasoning: ReasoningResult, scene: SceneState) -> Action:
        intent = reasoning.intent.lower()

        # Determine action type from intent
        if intent.startswith("answer_query"):
            action_type = "answer"
        elif "explore" in intent:
            action_type = "explore"
        elif "recall" in intent or "remember" in intent:
            action_type = "recall"
        elif "store" in intent or "learn" in intent:
            action_type = "store"
        else:
            action_type = "respond"

        # Build structured payload
        payload = {
            "intent": reasoning.intent,
            "rationale": reasoning.rationale,
            "actions": reasoning.actions,
            "scene_summary": scene.summary,
            "entities": scene.entities,
        }

        # Include confidence if available
        if "confidence" in reasoning.metadata:
            payload["confidence"] = reasoning.metadata["confidence"]

        return Action(type=action_type, payload=payload)


def build_llm_actor(**kwargs: Any) -> SmartActor:
    return SmartActor()
