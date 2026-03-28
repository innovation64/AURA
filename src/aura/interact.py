from __future__ import annotations

from .types import Action, Interaction, SceneState


class Interactor:
    def respond(self, action: Action, scene: SceneState) -> Interaction:
        raise NotImplementedError


class BasicInteractor(Interactor):
    def respond(self, action: Action, scene: SceneState) -> Interaction:
        if action.type == "answer":
            message = f"Based on the environment, {scene.summary}."
        elif action.type == "summary":
            message = f"Environment summary: {scene.summary}."
        else:
            message = f"Action '{action.type}' prepared."

        payload = {
            "action": action.payload,
            "scene": scene.context,
        }
        return Interaction(message=message, payload=payload)
