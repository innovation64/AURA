from __future__ import annotations

from .types import Action, ReasoningResult, SceneState


class Actor:
    def act(self, reasoning: ReasoningResult, scene: SceneState) -> Action:
        raise NotImplementedError


class StubActor(Actor):
    def act(self, reasoning: ReasoningResult, scene: SceneState) -> Action:
        if reasoning.intent.startswith("answer_query"):
            action_type = "answer"
        else:
            action_type = "summary"

        payload = {
            "intent": reasoning.intent,
            "rationale": reasoning.rationale,
            "scene": scene.summary,
        }
        return Action(type=action_type, payload=payload)
