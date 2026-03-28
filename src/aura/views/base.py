"""Base classes for agent-type environment views."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from aura.types import EnvironmentSignal

logger = logging.getLogger(__name__)


@dataclass
class ViewConfig:
    """Configuration for an agent-type view."""

    agent_type: str
    focus_paths: List[str] = field(default_factory=list)
    focus_services: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    max_context_items: int = 20


class EnvironmentView(ABC):
    """Abstract base class for agent-type environment views.

    Different agent types need different environment perspectives.
    Subclasses implement filtering, prioritisation, and rendering
    logic specific to a particular agent archetype.
    """

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return the agent type identifier for this view."""

    def filter_signals(
        self, signals: List[EnvironmentSignal], config: ViewConfig
    ) -> List[EnvironmentSignal]:
        """Filter signals to those relevant for this agent type.

        The base implementation keeps all signals.  Subclasses should
        override to drop irrelevant noise.
        """
        return list(signals)

    def prioritize(
        self, signals: List[EnvironmentSignal]
    ) -> List[EnvironmentSignal]:
        """Sort signals by relevance (most relevant first).

        The base implementation sorts by confidence descending, then
        by timestamp descending (newest first).
        """
        return sorted(
            signals,
            key=lambda s: (s.confidence, s.timestamp),
            reverse=True,
        )

    def summarize(self, signals: List[EnvironmentSignal]) -> str:
        """Produce a human-readable text summary of the signals."""
        if not signals:
            return "No relevant environment signals."
        lines = [f"## Environment Summary ({self.agent_type})"]
        for sig in signals:
            lines.append(f"- [{sig.source}] {sig.modality}: {sig.payload}")
        return "\n".join(lines)

    def render(
        self, signals: List[EnvironmentSignal], config: ViewConfig
    ) -> dict:
        """Produce a structured view dictionary.

        The base implementation returns a generic structure.
        Subclasses should override for agent-specific schemas.
        """
        filtered = self.filter_signals(signals, config)
        prioritized = self.prioritize(filtered)
        capped = prioritized[: config.max_context_items]
        return {
            "agent_type": self.agent_type,
            "signal_count": len(capped),
            "signals": [
                {
                    "source": s.source,
                    "modality": s.modality,
                    "payload": s.payload,
                    "confidence": s.confidence,
                    "timestamp": s.timestamp.isoformat(),
                }
                for s in capped
            ],
            "summary": self.summarize(capped),
        }


class ViewRegistry:
    """Registry that maps agent types to their environment views."""

    def __init__(self) -> None:
        self._views: Dict[str, EnvironmentView] = {}

    def register(self, view: EnvironmentView) -> None:
        """Register a view instance for its agent type."""
        agent_type = view.agent_type
        logger.info("Registering view for agent type: %s", agent_type)
        self._views[agent_type] = view

    def get_view(self, agent_type: str) -> Optional[EnvironmentView]:
        """Look up the view for *agent_type*, or return ``None``."""
        return self._views.get(agent_type)

    def render_for_agent(
        self,
        signals: List[EnvironmentSignal],
        agent_type: str,
        config: ViewConfig,
    ) -> dict:
        """Render signals through the view registered for *agent_type*.

        Falls back to the base ``EnvironmentView.render`` behaviour when
        no specific view has been registered.
        """
        view = self.get_view(agent_type)
        if view is None:
            logger.warning(
                "No view registered for agent type '%s'; "
                "returning raw signal list.",
                agent_type,
            )
            # Provide a minimal fallback
            return {
                "agent_type": agent_type,
                "signal_count": len(signals),
                "signals": [
                    {
                        "source": s.source,
                        "modality": s.modality,
                        "payload": s.payload,
                        "confidence": s.confidence,
                        "timestamp": s.timestamp.isoformat(),
                    }
                    for s in signals[: config.max_context_items]
                ],
            }
        return view.render(signals, config)
