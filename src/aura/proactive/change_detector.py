"""Detects significant environment changes from probe signals.

Implements multiple detection strategies (threshold, statistical, state
transition, rate-of-change, pattern) to convert raw
:class:`~aura.types.EnvironmentSignal` streams into actionable
:class:`ChangeEvent` objects.
"""

from __future__ import annotations

import logging
import statistics
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from aura.types import EnvironmentSignal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ChangeEvent:
    """Represents a detected environment change worth reporting."""

    event_type: str  # "anomaly" | "state_change" | "drift" | "spike" | "new_entity" | "pattern"
    source: str
    severity: float  # 0.0 .. 1.0
    description: str
    signals: List[EnvironmentSignal]
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS: Dict[str, float] = {
    "cpu_percent": 90.0,
    "memory_percent": 85.0,
    "disk_percent": 95.0,
    "load_1m": 8.0,
    "error_count": 5.0,
    "response_time_ms": 5000.0,
}

# Mapping from event_type helpers to severity floors so that each strategy
# can assign meaningful default severities.
_SEVERITY_FLOORS: Dict[str, float] = {
    "threshold": 0.7,
    "anomaly": 0.5,
    "state_change": 0.6,
    "spike": 0.6,
    "pattern": 0.5,
    "new_entity": 0.5,
}


# ---------------------------------------------------------------------------
# ChangeDetector
# ---------------------------------------------------------------------------

class ChangeDetector:
    """Maintains rolling windows of signal values and applies multiple
    detection strategies to identify significant changes.

    Parameters
    ----------
    window_size:
        Maximum number of historical values kept per source.
    thresholds:
        Static thresholds keyed by payload field name.  Merged on top of
        built-in defaults.
    z_score_limit:
        Number of standard deviations from the rolling mean that counts as
        an anomaly.  Defaults to ``2.0``.
    spike_pct:
        Minimum percentage change between consecutive polls to be flagged
        as a spike.  Defaults to ``0.50`` (50 %).
    error_repeat_count:
        Number of same-source errors within the window to trigger a
        pattern alert.  Defaults to ``3``.
    """

    def __init__(
        self,
        window_size: int = 100,
        thresholds: Optional[Dict[str, float]] = None,
        z_score_limit: float = 2.0,
        spike_pct: float = 0.50,
        error_repeat_count: int = 3,
    ) -> None:
        self.window_size = window_size
        self.thresholds: Dict[str, float] = {**_DEFAULT_THRESHOLDS}
        if thresholds:
            self.thresholds.update(thresholds)
        self.z_score_limit = z_score_limit
        self.spike_pct = spike_pct
        self.error_repeat_count = error_repeat_count

        # Rolling windows: source -> field -> deque of float values
        self._windows: Dict[str, Dict[str, Deque[float]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.window_size))
        )
        # Previous discrete states: source -> field -> last known value
        self._states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        # Error counters per source within window
        self._error_windows: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_discrete_states(self) -> None:
        """Clear cached discrete states so first-occurrence detection fires again.

        Call this between episodes so that the detector correctly detects
        file conflicts, service status changes, etc. each time the
        environment resets.  Numeric rolling windows are preserved to
        maintain cross-episode anomaly detection history.
        """
        self._states.clear()

    def detect(self, signals: List[EnvironmentSignal]) -> List[ChangeEvent]:
        """Run all detection strategies on a batch of signals.

        Returns a list of :class:`ChangeEvent` instances (may be empty).
        """
        events: List[ChangeEvent] = []
        for signal in signals:
            try:
                events.extend(self._detect_single(signal))
            except Exception:
                logger.exception("Detection failed for signal from %s", signal.source)
        return events

    # ------------------------------------------------------------------
    # Internal: per-signal detection
    # ------------------------------------------------------------------

    def _detect_single(self, signal: EnvironmentSignal) -> List[ChangeEvent]:
        events: List[ChangeEvent] = []
        source = signal.source
        ts = signal.timestamp.timestamp() if hasattr(signal.timestamp, "timestamp") else float(signal.timestamp)
        payload = signal.payload

        # Track errors for pattern detection
        if signal.modality == "error" or payload.get("error"):
            self._error_windows[source].append(ts)

        numeric_fields, discrete_fields = self._classify_fields(payload)

        # --- (a) Threshold-based ---
        for fld, val in numeric_fields:
            if fld in self.thresholds and val > self.thresholds[fld]:
                severity = min(1.0, _SEVERITY_FLOORS["threshold"] + 0.3 * (val - self.thresholds[fld]) / max(self.thresholds[fld], 1.0))
                events.append(ChangeEvent(
                    event_type="anomaly",
                    source=source,
                    severity=round(severity, 3),
                    description=f"{fld} exceeded threshold: {val:.2f} > {self.thresholds[fld]:.2f}",
                    signals=[signal],
                    timestamp=ts,
                    context={"field": fld, "value": val, "threshold": self.thresholds[fld]},
                ))

        # --- (b) Statistical z-score anomaly ---
        for fld, val in numeric_fields:
            window = self._windows[source][fld]
            if len(window) >= 5:
                mean = statistics.mean(window)
                stdev = statistics.pstdev(window)
                if stdev > 0:
                    z = abs(val - mean) / stdev
                    if z > self.z_score_limit:
                        severity = min(1.0, _SEVERITY_FLOORS["anomaly"] + 0.1 * (z - self.z_score_limit))
                        events.append(ChangeEvent(
                            event_type="anomaly",
                            source=source,
                            severity=round(severity, 3),
                            description=f"{fld} deviates from rolling mean (z={z:.2f}, mean={mean:.2f}, val={val:.2f})",
                            signals=[signal],
                            timestamp=ts,
                            context={"field": fld, "value": val, "z_score": round(z, 3), "mean": round(mean, 3), "stdev": round(stdev, 3)},
                        ))

        # --- (d) Rate-of-change / spike ---
        for fld, val in numeric_fields:
            window = self._windows[source][fld]
            if window:
                prev = window[-1]
                if prev != 0:
                    change_ratio = abs(val - prev) / abs(prev)
                    if change_ratio > self.spike_pct:
                        severity = min(1.0, _SEVERITY_FLOORS["spike"] + 0.4 * (change_ratio - self.spike_pct))
                        events.append(ChangeEvent(
                            event_type="spike",
                            source=source,
                            severity=round(severity, 3),
                            description=f"{fld} spiked {change_ratio*100:.1f}% (prev={prev:.2f}, now={val:.2f})",
                            signals=[signal],
                            timestamp=ts,
                            context={"field": fld, "previous": prev, "current": val, "change_pct": round(change_ratio * 100, 1)},
                        ))

        # Update rolling windows *after* comparisons so current value is
        # compared against historical data.
        for fld, val in numeric_fields:
            self._windows[source][fld].append(val)

        # --- (c) State transition + first occurrence ---
        for fld, val in discrete_fields:
            prev = self._states[source].get(fld)
            self._states[source][fld] = val
            if prev is None:
                # First time seeing this field — emit new_entity event
                # Critical for detecting file conflicts, new alerts, etc.
                severity = self._first_occurrence_severity(fld, val)
                if severity > 0:
                    events.append(ChangeEvent(
                        event_type="new_entity",
                        source=source,
                        severity=round(severity, 3),
                        description=f"New {fld} detected: '{val}'",
                        signals=[signal],
                        timestamp=ts,
                        context={"field": fld, "value": val},
                    ))
            elif prev != val:
                severity = self._state_transition_severity(fld, prev, val)
                events.append(ChangeEvent(
                    event_type="state_change",
                    source=source,
                    severity=round(severity, 3),
                    description=f"{fld} changed from '{prev}' to '{val}'",
                    signals=[signal],
                    timestamp=ts,
                    context={"field": fld, "old": prev, "new": val},
                ))

        # --- (e) Pattern: repeated errors ---
        err_window = self._error_windows[source]
        if len(err_window) >= self.error_repeat_count:
            # Check if the last N errors occurred within the rolling window
            recent = list(err_window)[-self.error_repeat_count:]
            span = recent[-1] - recent[0]
            if span < 300:  # within 5 minutes
                severity = min(1.0, _SEVERITY_FLOORS["pattern"] + 0.1 * (len(err_window) - self.error_repeat_count))
                events.append(ChangeEvent(
                    event_type="pattern",
                    source=source,
                    severity=round(severity, 3),
                    description=f"Repeated errors from {source}: {len(err_window)} in {span:.0f}s",
                    signals=[signal],
                    timestamp=ts,
                    context={"error_count": len(err_window), "span_seconds": round(span, 1)},
                ))
                # Clear so we don't re-alert every signal
                self._error_windows[source].clear()

        return events

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_fields(payload: Dict[str, Any]) -> Tuple[List[Tuple[str, float]], List[Tuple[str, Any]]]:
        """Separate payload fields into numeric and discrete categories."""
        numeric: List[Tuple[str, float]] = []
        discrete: List[Tuple[str, Any]] = []
        for key, val in payload.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                numeric.append((key, float(val)))
            elif isinstance(val, str):
                discrete.append((key, val))
            elif isinstance(val, bool):
                discrete.append((key, val))
        return numeric, discrete

    @staticmethod
    def _first_occurrence_severity(field: str, value: Any) -> float:
        """Heuristic severity for first-occurrence discrete fields.

        Returns >0 for fields that are inherently noteworthy (conflicts,
        errors, status fields with bad values). Returns 0 for mundane
        fields like 'name' to avoid noise.
        """
        fld_lower = field.lower()
        val_lower = str(value).lower()

        # Conflict-related fields are always noteworthy
        if "conflict" in fld_lower or "conflict" in val_lower:
            return 0.75
        # Error/failure fields
        if "error" in fld_lower or "fail" in fld_lower:
            return 0.8
        # Status fields with bad values
        if fld_lower in ("status", "state"):
            negative = {"down", "stopped", "error", "failed", "crashed", "unhealthy"}
            if val_lower in negative:
                return 0.9
            return 0.0  # normal status first-seen is not noteworthy
        # Type fields that indicate something happened
        if fld_lower == "type" and val_lower in ("conflict", "error", "alert", "security", "spike"):
            return 0.7
        # Generic 'name' or mundane fields — not noteworthy
        if fld_lower in ("name", "pid", "modality"):
            return 0.0
        # Default: mildly noteworthy
        return 0.3

    @staticmethod
    def _state_transition_severity(field: str, old: Any, new: Any) -> float:
        """Heuristic severity for state transitions."""
        # Service going down is worse than coming up
        negative_states = {"down", "stopped", "error", "failed", "unreachable", "crashed", "unhealthy"}
        positive_states = {"up", "running", "healthy", "ok", "started", "ready"}
        new_lower = str(new).lower()
        old_lower = str(old).lower()
        if new_lower in negative_states:
            return 0.9
        if old_lower in negative_states and new_lower in positive_states:
            return 0.4  # recovery is less severe
        return _SEVERITY_FLOORS["state_change"]
