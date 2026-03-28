"""Filesystem probe -- detects file creation, modification, and deletion."""

from __future__ import annotations

import fnmatch
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from aura.types import EnvironmentSignal

from .base import Probe, ProbeResult

import asyncio

# Default patterns to ignore when scanning.
DEFAULT_IGNORE_PATTERNS: List[str] = [
    ".git",
    "__pycache__",
    "node_modules",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "*.pyc",
    "*.pyo",
    "*.swp",
    "*.swo",
    ".DS_Store",
    "*.egg-info",
]


@dataclass(frozen=True)
class _FileSnapshot:
    path: str
    mtime: float
    size: int


def _should_ignore(name: str, ignore_patterns: List[str]) -> bool:
    """Return True if *name* matches any ignore pattern."""
    for pat in ignore_patterns:
        if fnmatch.fnmatch(name, pat):
            return True
    return False


def _scan_tree(
    root: str,
    max_depth: int,
    ignore_patterns: List[str],
    _depth: int = 0,
) -> Dict[str, _FileSnapshot]:
    """Walk directory tree up to *max_depth* levels, returning file snapshots."""
    snapshots: Dict[str, _FileSnapshot] = {}
    if _depth > max_depth:
        return snapshots
    try:
        entries = os.scandir(root)
    except (PermissionError, OSError):
        return snapshots

    for entry in entries:
        if _should_ignore(entry.name, ignore_patterns):
            continue
        try:
            if entry.is_file(follow_symlinks=False):
                stat = entry.stat(follow_symlinks=False)
                snapshots[entry.path] = _FileSnapshot(
                    path=entry.path,
                    mtime=stat.st_mtime,
                    size=stat.st_size,
                )
            elif entry.is_dir(follow_symlinks=False):
                snapshots.update(
                    _scan_tree(entry.path, max_depth, ignore_patterns, _depth + 1)
                )
        except (PermissionError, OSError):
            continue
    return snapshots


class FileSystemProbe(Probe):
    """Detects file creation, modification, and deletion via polling."""

    def __init__(
        self,
        watch_paths: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        max_depth: int = 3,
        batch_window_seconds: float = 2.0,
    ) -> None:
        super().__init__()
        self._watch_paths = watch_paths or [os.getcwd()]
        self._ignore_patterns = ignore_patterns or list(DEFAULT_IGNORE_PATTERNS)
        self._max_depth = max_depth
        self._batch_window = batch_window_seconds
        self._prev_state: Optional[Dict[str, _FileSnapshot]] = None

    @property
    def name(self) -> str:
        return "filesystem"

    @property
    def interval_seconds(self) -> float:
        return 10.0

    def _scan_all(self) -> Dict[str, _FileSnapshot]:
        combined: Dict[str, _FileSnapshot] = {}
        for root in self._watch_paths:
            combined.update(_scan_tree(root, self._max_depth, self._ignore_patterns))
        return combined

    async def poll(self) -> ProbeResult:
        t0 = time.time()
        signals: List[EnvironmentSignal] = []

        try:
            current = await asyncio.to_thread(self._scan_all)

            if self._prev_state is not None:
                prev_paths: Set[str] = set(self._prev_state.keys())
                curr_paths: Set[str] = set(current.keys())

                created = curr_paths - prev_paths
                deleted = prev_paths - curr_paths
                common = prev_paths & curr_paths

                modified: List[str] = []
                for p in common:
                    old, new = self._prev_state[p], current[p]
                    if old.mtime != new.mtime or old.size != new.size:
                        modified.append(p)

                # Batch rapid changes: group all into one signal if within window.
                changes: List[Dict[str, Any]] = []
                for p in sorted(created):
                    changes.append({"event": "created", "path": p})
                for p in sorted(modified):
                    snap = current[p]
                    changes.append(
                        {
                            "event": "modified",
                            "path": p,
                            "size": snap.size,
                            "mtime": snap.mtime,
                        }
                    )
                for p in sorted(deleted):
                    changes.append({"event": "deleted", "path": p})

                if changes:
                    signals.append(
                        EnvironmentSignal(
                            source="probe.filesystem",
                            modality="filesystem",
                            payload={
                                "changes": changes,
                                "summary": {
                                    "created": len(created),
                                    "modified": len(modified),
                                    "deleted": len(deleted),
                                },
                            },
                            confidence=1.0,
                        )
                    )

            self._prev_state = current

        except Exception as exc:
            signals.append(
                EnvironmentSignal(
                    source="probe.filesystem",
                    modality="filesystem",
                    payload={"error": str(exc)},
                    confidence=0.0,
                )
            )

        latency = (time.time() - t0) * 1000
        return ProbeResult(
            source=self.name,
            timestamp=t0,
            signals=signals,
            latency_ms=round(latency, 2),
            metadata={"watched_paths": self._watch_paths, "max_depth": self._max_depth},
        )
