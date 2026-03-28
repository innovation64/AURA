"""LLM-driven interactor — generates natural language responses."""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..interact import Interactor
from ..llm import LLMEngine
from ..types import Action, Interaction, SceneState

logger = logging.getLogger(__name__)

_INTERACT_SYSTEM = """\
You are the response generator for an autonomous agent. Given the agent's action decision and scene context,
produce a factually grounded response.

RULES:
- Only state facts that are directly supported by the scene context and entities provided
- Reference specific entities, locations, times, and actions from the scene
- If the action includes a query, answer it by citing concrete details from the context
- Do not speculate or add information not present in the scene
- Be concise but precise — prefer specific details over vague descriptions"""


class LLMInteractor(Interactor):
    """Uses LLM to generate natural language responses grounded in scene context."""

    def __init__(self, llm: Optional[LLMEngine] = None) -> None:
        self._llm = llm

    def respond(self, action: Action, scene: SceneState) -> Interaction:
        if not self._llm:
            return self._fallback_respond(action, scene)

        try:
            user_parts = [
                f"Action type: {action.type}",
                f"Intent: {action.payload.get('intent', 'unknown')}",
                f"Scene summary: {scene.summary}",
            ]
            if scene.entities:
                user_parts.append(f"Key entities: {', '.join(scene.entities[:15])}")
            if scene.context:
                # Include concrete scene details for factual grounding
                ctx_items = []
                for k, v in scene.context.items():
                    if k not in ("signals",) and v:
                        ctx_items.append(f"{k}: {str(v)[:200]}")
                if ctx_items:
                    user_parts.append(f"Scene details:\n" + "\n".join(ctx_items[:10]))
            user_parts.append("Ground your response in the above facts. Do not add unsupported claims.")

            message = self._llm.chat(
                [
                    {"role": "system", "content": _INTERACT_SYSTEM},
                    {"role": "user", "content": "\n".join(user_parts)},
                ],
                temperature=0.5,
                max_tokens=512,
            )

            return Interaction(
                message=message,
                payload={
                    "action_type": action.type,
                    "scene": scene.context,
                    "source": "llm",
                },
            )
        except Exception as e:
            logger.warning("LLM interaction failed, using fallback: %s", e)
            return self._fallback_respond(action, scene)

    def _fallback_respond(self, action: Action, scene: SceneState) -> Interaction:
        if action.type == "answer":
            message = f"Based on the current context: {scene.summary}"
        elif action.type == "explore":
            message = f"Exploring the environment. Current state: {scene.summary}"
        else:
            message = f"Environment: {scene.summary}"

        return Interaction(
            message=message,
            payload={"action": action.payload, "scene": scene.context, "source": "fallback"},
        )


def build_llm_interactor(**kwargs: Any) -> LLMInteractor:
    return LLMInteractor(llm=kwargs.get("llm"))
