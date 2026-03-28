"""LLM-driven scene builder — extracts entities, relations, and builds structured scene."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from ..llm import LLMEngine
from ..scene import SceneModel
from ..types import EnvironmentSignal, SceneState

logger = logging.getLogger(__name__)

_SCENE_SYSTEM = """\
You are a scene analysis engine. Given a set of environment signals, produce a structured scene summary.

Output a JSON object with:
- "summary": one-sentence natural language description of the current scene
- "entities": list of entity names mentioned
- "relations": list of {"subject": str, "predicate": str, "object": str}
- "context_type": one of ["conversation", "observation", "task", "query", "unknown"]

Respond ONLY with valid JSON, no markdown."""


class LLMScene(SceneModel):
    """Uses LLM to build rich scene representations from signals."""

    def __init__(self, llm: Optional[LLMEngine] = None) -> None:
        self._llm = llm

    def build(self, signals: Sequence[EnvironmentSignal]) -> SceneState:
        # Filter out low-confidence signals (e.g., irrelevant system probes)
        signals = [s for s in signals if s.confidence >= 0.3]

        # Collect all signal data
        signal_texts = []
        all_entities: List[str] = []

        for sig in signals:
            payload = sig.payload
            if isinstance(payload, dict):
                # Extract pre-identified entities
                if "entities" in payload and isinstance(payload["entities"], list):
                    all_entities.extend(str(e) for e in payload["entities"])
                # Build text representation
                if "text" in payload:
                    signal_texts.append(f"[{sig.source}/{sig.modality}] {payload['text']}")
                elif "value" in payload:
                    signal_texts.append(f"[{sig.source}/{sig.modality}] {payload['value']}")
                elif "output" in payload:
                    signal_texts.append(f"[{sig.source}/tool] {payload['output']}")
                else:
                    signal_texts.append(f"[{sig.source}/{sig.modality}] {payload}")

        # If LLM available, use it for rich scene understanding
        if self._llm and signal_texts:
            try:
                user_msg = "Signals:\n" + "\n".join(signal_texts[:20])
                result = self._llm.chat_json(
                    [
                        {"role": "system", "content": _SCENE_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.1,
                    max_tokens=512,
                )
                summary = result.get("summary", f"{len(signals)} signals observed")
                entities = result.get("entities", all_entities)
                relations = result.get("relations", [])
                context_type = result.get("context_type", "unknown")

                unique_entities = sorted(set(str(e) for e in entities))
                return SceneState(
                    summary=summary,
                    entities=unique_entities,
                    context={
                        "signals": [{"source": s.source, "modality": s.modality} for s in signals],
                        "relations": relations,
                        "context_type": context_type,
                    },
                )
            except Exception as e:
                logger.warning("LLM scene building failed, using fallback: %s", e)

        # Fallback: heuristic scene building
        unique_entities = sorted(set(all_entities))
        summary = f"{len(signals)} signals observed"
        if unique_entities:
            summary += f", entities: {', '.join(unique_entities[:10])}"

        return SceneState(
            summary=summary,
            entities=unique_entities,
            context={"signals": [{"source": s.source, "modality": s.modality} for s in signals]},
        )


def build_llm_scene(**kwargs: Any) -> LLMScene:
    return LLMScene(llm=kwargs.get("llm"))
