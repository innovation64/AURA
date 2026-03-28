"""Smart sense adapter with entity extraction and confidence scoring."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..sense import SenseAdapter
from ..types import EnvironmentSignal


class SmartSense(SenseAdapter):
    """Enhanced sense adapter: extracts entities, classifies modality, scores confidence."""

    def __init__(self, source: str = "smart_sense", llm: Any = None) -> None:
        self.source = source
        self._llm = llm

    def ingest(self, raw: Any) -> List[EnvironmentSignal]:
        if raw is None:
            return []
        if isinstance(raw, list):
            return [sig for item in raw for sig in self._process_item(item)]
        return self._process_item(raw)

    def _process_item(self, item: Any) -> List[EnvironmentSignal]:
        if isinstance(item, EnvironmentSignal):
            return [item]

        if isinstance(item, dict):
            payload = dict(item)
            modality = payload.pop("modality", self._detect_modality(payload))
            confidence = payload.pop("confidence", 1.0)
            entities = payload.get("entities") or self._extract_entities_simple(payload)
            if entities:
                payload["entities"] = entities
            return [EnvironmentSignal(
                source=payload.pop("source", self.source),
                payload=payload,
                modality=modality,
                confidence=confidence,
            )]

        # String input
        text = str(item)
        entities = self._extract_entities_simple({"text": text})
        return [EnvironmentSignal(
            source=self.source,
            payload={"text": text, "entities": entities},
            modality="text",
            confidence=0.9,
        )]

    def _detect_modality(self, payload: Dict[str, Any]) -> str:
        if "image" in payload or "image_url" in payload:
            return "vision"
        if "audio" in payload:
            return "audio"
        if "sensor" in payload or "temperature" in payload:
            return "sensor"
        return "text"

    def _extract_entities_simple(self, payload: Dict[str, Any]) -> List[str]:
        """Extract capitalized words as candidate entities (heuristic)."""
        text = " ".join(str(v) for v in payload.values() if isinstance(v, str))
        if not text:
            return []
        # Find capitalized multi-word phrases or single capitalized words
        candidates = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        # Deduplicate
        seen = set()
        result = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                result.append(c)
        return result


def build_llm_sense(**kwargs: Any) -> SmartSense:
    return SmartSense(llm=kwargs.get("llm"))
