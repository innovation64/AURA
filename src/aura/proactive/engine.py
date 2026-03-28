"""ProactiveEngine — orchestrates the full proactive context loop.

This is AURA's core innovation: instead of agents passively querying,
the engine continuously monitors the environment and proactively
provides relevant context.

Architecture:
    Probes → ChangeDetector → RelevanceScorer → ContextAssembler → PushController
       ↑                                                              ↓
       └──────────── AttentionTracker (feedback) ←────────────────────┘
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from ..probes.base import ProbeRegistry, ProbeResult
from ..types import EnvironmentSignal
from .change_detector import ChangeDetector, ChangeEvent
from .context_assembler import ContextAssembler, EnvironmentContext
from .relevance_scorer import RelevanceScorer, TaskContext

logger = logging.getLogger(__name__)


class PushDecision:
    """Result of the push controller's evaluation."""

    def __init__(
        self,
        should_push: bool,
        reason: str = "",
        context: Optional[EnvironmentContext] = None,
        priority: str = "normal",
    ):
        self.should_push = should_push
        self.reason = reason
        self.context = context
        self.priority = priority


class ProactiveEngine:
    """Full proactive context engine with all components.

    Orchestrates: Probes → ChangeDetector → RelevanceScorer → ContextAssembler.
    """

    def __init__(
        self,
        probes: ProbeRegistry,
        detector: Optional[ChangeDetector] = None,
        scorer: Optional[RelevanceScorer] = None,
        assembler: Optional[ContextAssembler] = None,
    ):
        self.probes = probes
        self.detector = detector or ChangeDetector()
        self.scorer = scorer or RelevanceScorer()
        self.assembler = assembler or ContextAssembler()

        self._task_context: Optional[TaskContext] = None
        self._current_context: Optional[EnvironmentContext] = None
        self._cached_signals: List[EnvironmentSignal] = []
        self._last_poll: float = 0
        self._min_poll_interval: float = 5.0
        self._running = False
        self._probe_snapshots: Dict[str, Any] = {}

    async def tick(self, task_context: Optional[TaskContext] = None) -> Optional[PushDecision]:
        """One cycle of the proactive loop."""
        if task_context:
            self._task_context = task_context

        # 1. Poll all probes
        results = await self.probes.poll_all()
        all_signals: List[EnvironmentSignal] = []
        for r in results:
            all_signals.extend(r.signals)
            self._probe_snapshots[r.source] = r

        if not all_signals:
            return None

        # 2. Detect changes
        events = self.detector.detect(all_signals)
        if not events:
            return None

        # 3. Score relevance
        relevance_scores: Dict[str, float] = {}
        tc = self._task_context or TaskContext()
        for i, event in enumerate(events):
            eid = f"evt_{i}"
            relevance_scores[eid] = self.scorer.score(event, tc)

        # 4. Assemble context
        self._current_context = self.assembler.assemble(
            change_events=events,
            relevance_scores=relevance_scores,
            probe_snapshots={k: v.metadata for k, v in self._probe_snapshots.items()},
            task_context=self._task_context,
        )

        # 5. Cache signals
        self._cached_signals = all_signals
        self._last_poll = time.time()

        # 6. Decide whether to push
        has_critical = any(e.severity > 0.8 for e in events)
        has_relevant = any(v > 0.4 for v in relevance_scores.values())

        if has_critical:
            return PushDecision(
                should_push=True,
                reason="Critical environment change detected",
                context=self._current_context,
                priority="critical",
            )
        elif has_relevant:
            return PushDecision(
                should_push=True,
                reason="Relevant environment change detected",
                context=self._current_context,
                priority="normal",
            )

        return None

    async def poll_signals(self) -> List[EnvironmentSignal]:
        """Poll probes and return raw environment signals."""
        now = time.time()
        if now - self._last_poll < self._min_poll_interval and self._cached_signals:
            return list(self._cached_signals)

        results = await self.probes.poll_all()
        signals: List[EnvironmentSignal] = []
        for r in results:
            signals.extend(r.signals)
            self._probe_snapshots[r.source] = r

        self._cached_signals = signals
        self._last_poll = now
        return signals

    def get_cached_signals(self) -> List[EnvironmentSignal]:
        """Return cached signals without polling (non-blocking)."""
        return list(self._cached_signals)

    def get_current_context(self) -> Optional[EnvironmentContext]:
        """Return the latest assembled context."""
        return self._current_context

    def update_task(self, task_context: TaskContext) -> None:
        """Update what the agent is currently working on."""
        self._task_context = task_context

    def list_probes(self) -> List[str]:
        """Return names of registered probes."""
        return self.probes.list_probes()

    async def run_loop(
        self,
        interval: float = 10.0,
        callback: Optional[Callable[[PushDecision], Any]] = None,
    ) -> None:
        """Background loop that continuously monitors and pushes."""
        self._running = True
        logger.info("Proactive engine loop started (interval=%.1fs)", interval)

        while self._running:
            try:
                decision = await self.tick()
                if decision and decision.should_push and callback:
                    callback(decision)
            except Exception as e:
                logger.error("Proactive tick failed: %s", e)

            await asyncio.sleep(interval)

    def stop(self) -> None:
        """Stop the background loop."""
        self._running = False


class SimpleProactiveEngine:
    """Lightweight proactive engine for synchronous use — polls probes
    and returns signals without the full change detection pipeline.

    Used by AURAAgent.run() when full async engine is overkill.
    """

    def __init__(self, probes: ProbeRegistry):
        self.probes = probes
        self._cached_signals: List[EnvironmentSignal] = []
        self._last_poll: float = 0
        self._min_interval: float = 10.0

    async def poll_signals(self) -> List[EnvironmentSignal]:
        """Poll all probes and return signals."""
        now = time.time()
        if now - self._last_poll < self._min_interval and self._cached_signals:
            return list(self._cached_signals)

        try:
            results = await self.probes.poll_all()
            signals: List[EnvironmentSignal] = []
            for r in results:
                signals.extend(r.signals)
            self._cached_signals = signals
            self._last_poll = now
            return signals
        except Exception as e:
            logger.debug("Probe poll failed: %s", e)
            return list(self._cached_signals)

    def get_cached_signals(self) -> List[EnvironmentSignal]:
        """Return cached signals without polling."""
        return list(self._cached_signals)

    def get_current_context(self) -> Optional[Dict[str, Any]]:
        """Return simple context dict from cached signals."""
        if not self._cached_signals:
            return None
        return {
            "signals": len(self._cached_signals),
            "sources": list({s.source for s in self._cached_signals}),
            "last_poll": self._last_poll,
        }

    def list_probes(self) -> List[str]:
        return self.probes.list_probes()
