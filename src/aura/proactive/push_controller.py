"""Push controller — decides when and what to push to agents."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

from .context_assembler import EnvironmentContext

logger = logging.getLogger(__name__)


@dataclass
class PushRecord:
    timestamp: float
    priority: str
    num_alerts: int
    was_acknowledged: bool = False


class PushController:
    """Controls when and what to push to agents — the proactive decision maker.

    Prevents alert fatigue through:
    - Minimum interval between pushes (adaptive based on ack rate)
    - Rate limiting (max pushes per window, adaptive)
    - Critical override (bypasses throttling)
    - Acknowledgement-driven adaptation: pushes more when agent uses context,
      less when agent ignores it

    Supports simulation mode via logical_time for non-realtime experiments.
    """

    def __init__(
        self,
        min_push_interval: float = 10.0,
        critical_override: bool = True,
        relevance_threshold: float = 0.4,
        batch_window: float = 3.0,
        max_pushes_per_minute: int = 6,
        use_logical_time: bool = False,
    ):
        self.min_push_interval = min_push_interval
        self.critical_override = critical_override
        self.relevance_threshold = relevance_threshold
        self.batch_window = batch_window
        self.max_pushes_per_minute = max_pushes_per_minute
        self.use_logical_time = use_logical_time

        self._push_history: deque[PushRecord] = deque(maxlen=200)
        self._last_push_time: float = 0
        self._pending_batch: List[EnvironmentContext] = []
        self._batch_start: float = 0
        self._logical_clock: int = 0
        # Adaptive rate: starts at configured value, adjusts based on ack rate
        self._adaptive_max_rate: float = float(max_pushes_per_minute)
        self._adaptive_interval: float = min_push_interval

    def _now(self) -> float:
        """Return current time — logical clock for simulation, wall clock otherwise."""
        if self.use_logical_time:
            return float(self._logical_clock)
        return time.time()

    def tick(self) -> None:
        """Advance logical clock by one step (simulation mode)."""
        self._logical_clock += 1

    def should_push(self, context: EnvironmentContext) -> bool:
        """Evaluate whether to push this context to agents."""
        now = self._now()

        # Critical alerts always bypass (if enabled)
        if self.critical_override and context.critical_alerts:
            return True

        # Check rate limit using adaptive rate
        if self.use_logical_time:
            # In simulation: limit pushes per logical window (last 10 ticks)
            window = 10
            recent = sum(
                1 for r in self._push_history
                if now - r.timestamp < window
            )
            if recent >= self._adaptive_max_rate:
                logger.debug("Push rate limited (%d/%.0f per window)", recent, self._adaptive_max_rate)
                return False
        else:
            recent = sum(
                1 for r in self._push_history
                if now - r.timestamp < 60
            )
            if recent >= self._adaptive_max_rate:
                logger.debug("Push rate limited (%d/%.0f per minute)", recent, self._adaptive_max_rate)
                return False

        # Check minimum interval (adaptive)
        if now - self._last_push_time < self._adaptive_interval:
            return False

        # Check if there are relevant changes
        if not context.relevant_changes and not context.critical_alerts:
            return False

        return True

    def record_push(self, priority: str, num_alerts: int) -> None:
        """Record that a push was made."""
        self._last_push_time = self._now()
        self._push_history.append(PushRecord(
            timestamp=self._last_push_time,
            priority=priority,
            num_alerts=num_alerts,
        ))

    def record_acknowledgement(self) -> None:
        """Record that the agent acknowledged the last push and adapt rates."""
        if self._push_history:
            self._push_history[-1].was_acknowledged = True
        self._adapt_rates()

    def record_ignore(self) -> None:
        """Record that the agent ignored the last push and adapt rates."""
        self._adapt_rates()

    def _adapt_rates(self) -> None:
        """Adapt push frequency based on recent acknowledgement rate.

        If the agent is using pushed context (high ack rate), allow more
        frequent pushes. If ignoring, slow down to avoid fatigue.
        """
        # Look at the last 10 pushes
        recent = list(self._push_history)[-10:]
        if len(recent) < 3:
            return
        ack_rate = sum(1 for r in recent if r.was_acknowledged) / len(recent)

        # High ack rate (>0.6): agent values pushes, increase rate
        if ack_rate > 0.6:
            self._adaptive_max_rate = min(
                self.max_pushes_per_minute * 2.0,
                self._adaptive_max_rate + 1.0,
            )
            self._adaptive_interval = max(0.0, self._adaptive_interval - 1.0)
        # Low ack rate (<0.3): agent ignoring pushes, decrease rate
        elif ack_rate < 0.3:
            self._adaptive_max_rate = max(2.0, self._adaptive_max_rate - 1.0)
            self._adaptive_interval = min(
                self.min_push_interval * 3.0,
                self._adaptive_interval + 2.0,
            )
        # Medium ack rate: slowly converge back to defaults
        else:
            self._adaptive_max_rate += (self.max_pushes_per_minute - self._adaptive_max_rate) * 0.1
            self._adaptive_interval += (self.min_push_interval - self._adaptive_interval) * 0.1

    def classify_priority(self, context: EnvironmentContext) -> str:
        """Classify the push priority based on context content."""
        if context.critical_alerts:
            max_severity = max(e.severity for e in context.critical_alerts)
            if max_severity > 0.9:
                return "critical"
            return "high"
        if context.relevant_changes:
            max_severity = max(e.severity for e in context.relevant_changes)
            if max_severity > 0.6:
                return "normal"
        return "low"

    def get_stats(self) -> dict:
        """Return push statistics."""
        now = self._now()
        if self.use_logical_time:
            recent = list(self._push_history)
        else:
            recent = [r for r in self._push_history if now - r.timestamp < 3600]
        acknowledged = sum(1 for r in recent if r.was_acknowledged)
        return {
            "total_pushes": len(self._push_history),
            "pushes_recent": len(recent),
            "acknowledged_recent": acknowledged,
            "ack_rate": acknowledged / max(len(recent), 1),
            "adaptive_max_rate": self._adaptive_max_rate,
            "adaptive_interval": self._adaptive_interval,
            "last_push_ago": now - self._last_push_time if self._last_push_time else None,
        }
