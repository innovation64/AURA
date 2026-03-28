from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

from .types import EnvironmentSignal, SceneState


class SceneModel:
    def build(self, signals: Sequence[EnvironmentSignal]) -> SceneState:
        raise NotImplementedError


# Modalities that carry task-relevant content (keep in scene summary)
_TASK_MODALITIES = frozenset({"generic", "agent", "social", "environment", "probe", "benchmark"})
# Modalities that are system-level (exclude from scene summary unless query-relevant)
_SYSTEM_MODALITIES = frozenset({"system", "tool", "docker", "network", "filesystem"})

# Keys commonly carrying location/time/social info in signal payloads
_LOCATION_KEYS = ("location", "current_location", "loc", "place", "position")
_TIME_KEYS = ("time", "hour", "timestamp", "current_time", "time_of_day")
_ACTION_KEYS = ("action", "current_action", "activity", "doing")
_AGENT_KEYS = ("agent", "agent_name", "name", "agents", "nearby_agents", "agents_here")


class BasicScene(SceneModel):
    def build(self, signals: Sequence[EnvironmentSignal]) -> SceneState:
        entities: List[str] = []
        context_signals: List[Dict[str, Any]] = []

        # Grounding slots extracted from signals
        locations: List[str] = []
        times: List[str] = []
        actions: List[str] = []
        agents_seen: List[str] = []
        content_parts: List[str] = []

        for signal in signals:
            payload = signal.payload or {}

            # Skip low-confidence signals
            if signal.confidence < 0.3:
                continue

            if isinstance(payload, dict):
                # Extract entities
                found = payload.get("entities")
                if isinstance(found, list):
                    entities.extend(str(item) for item in found)

                # Extract grounding information from payload
                _extract_into(payload, _LOCATION_KEYS, locations)
                _extract_into(payload, _TIME_KEYS, times)
                _extract_into(payload, _ACTION_KEYS, actions)
                _extract_into(payload, _AGENT_KEYS, agents_seen)

                # For task-relevant signals, extract meaningful content
                if signal.modality in _TASK_MODALITIES:
                    content = _extract_content(payload)
                    if content:
                        content_parts.append(content)
                elif signal.modality == "tool":
                    # Tool outputs: only include if they contain grounding info
                    output = payload.get("output")
                    if isinstance(output, dict) and _has_grounding_info(output):
                        content = _extract_content(output)
                        if content:
                            content_parts.append(f"[{signal.source}] {content}")

            context_signals.append(
                {
                    "source": signal.source,
                    "modality": signal.modality,
                    "payload": payload,
                }
            )

        # Build a grounded summary instead of just counting signals
        unique_entities = sorted(set(entities))
        summary = _build_summary(
            locations=locations,
            times=times,
            actions=actions,
            agents_seen=agents_seen,
            entities=unique_entities,
            content_parts=content_parts,
            total_signals=len(signals),
        )

        context: Dict[str, Any] = {"signals": context_signals}
        if locations:
            context["location"] = locations[-1]
        if times:
            context["time"] = times[-1]
        if actions:
            context["current_action"] = actions[-1]
        if agents_seen:
            context["nearby_agents"] = list(set(agents_seen))

        return SceneState(summary=summary, entities=unique_entities, context=context)


def _extract_into(payload: Dict[str, Any], keys: tuple, target: List[str]) -> None:
    """Extract string values from payload for given keys into target list."""
    for key in keys:
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            target.append(val.strip())
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    target.append(item)
                elif isinstance(item, dict):
                    name = item.get("name", "")
                    if name:
                        target.append(str(name))


def _extract_content(payload: Dict[str, Any]) -> str:
    """Extract human-readable content from a payload dict."""
    # Try common content fields
    for key in ("summary", "description", "content", "text", "message", "value"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()[:300]

    # For structured data with a few informative fields, build a brief summary
    informative = {}
    for k, v in payload.items():
        if k in ("entities", "signals", "error"):
            continue
        if isinstance(v, (str, int, float, bool)):
            informative[k] = v
        elif isinstance(v, list) and len(v) <= 5:
            informative[k] = v
    if informative and len(informative) <= 8:
        parts = []
        for k, v in informative.items():
            parts.append(f"{k}: {v}")
        return "; ".join(parts)[:300]

    return ""


def _has_grounding_info(output: Dict[str, Any]) -> bool:
    """Check if a tool output contains location/time/agent grounding info."""
    all_keys = set(output.keys())
    grounding_keys = set(_LOCATION_KEYS + _TIME_KEYS + _ACTION_KEYS + _AGENT_KEYS)
    return bool(all_keys & grounding_keys)


def _build_summary(
    locations: List[str],
    times: List[str],
    actions: List[str],
    agents_seen: List[str],
    entities: List[str],
    content_parts: List[str],
    total_signals: int,
) -> str:
    """Build a grounded natural-language summary from extracted info."""
    parts = []

    if times:
        parts.append(f"Time: {times[-1]}")
    if locations:
        parts.append(f"Location: {locations[-1]}")
    if actions:
        parts.append(f"Activity: {actions[-1]}")

    unique_agents = sorted(set(agents_seen))
    if unique_agents:
        parts.append(f"Agents: {', '.join(unique_agents[:10])}")

    if entities:
        # Don't repeat agents already listed
        extra_entities = [e for e in entities if e not in set(unique_agents)]
        if extra_entities:
            parts.append(f"Entities: {', '.join(extra_entities[:10])}")

    # Include meaningful content (deduplicated, truncated)
    if content_parts:
        seen = set()
        unique_content = []
        for c in content_parts:
            key = c[:50]
            if key not in seen:
                seen.add(key)
                unique_content.append(c)
        if unique_content:
            parts.append("Context: " + " | ".join(unique_content[:5]))

    if not parts:
        return f"{total_signals} signals observed (no grounding info extracted)"

    return "; ".join(parts)
