"""Git repository probe -- tracks uncommitted changes, new commits, branch switches, remote divergence."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

from aura.types import EnvironmentSignal

from .base import Probe, ProbeResult


async def _run_git(
    *args: str, cwd: Optional[str] = None, timeout: float = 10.0
) -> Optional[str]:
    """Run a git subprocess and return stdout, or None on failure."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        if proc.returncode != 0:
            return None
        return stdout.decode(errors="replace").strip()
    except (FileNotFoundError, asyncio.TimeoutError, OSError):
        return None


class GitProbe(Probe):
    """Tracks git repository state: uncommitted changes, commits, branches, remote divergence."""

    def __init__(self, repo_path: Optional[str] = None) -> None:
        super().__init__()
        self._repo_path = repo_path or os.getcwd()
        self._prev_branch: Optional[str] = None
        self._prev_head: Optional[str] = None
        self._prev_dirty: Optional[bool] = None
        self._prev_diff_summary: Optional[Dict[str, int]] = None

    @property
    def name(self) -> str:
        return "git"

    @property
    def interval_seconds(self) -> float:
        return 15.0

    async def _is_git_repo(self) -> bool:
        result = await _run_git("rev-parse", "--is-inside-work-tree", cwd=self._repo_path)
        return result == "true"

    async def _current_branch(self) -> Optional[str]:
        return await _run_git("rev-parse", "--abbrev-ref", "HEAD", cwd=self._repo_path)

    async def _head_commit(self) -> Optional[str]:
        return await _run_git("rev-parse", "HEAD", cwd=self._repo_path)

    async def _is_dirty(self) -> bool:
        result = await _run_git("status", "--porcelain", cwd=self._repo_path)
        return result is not None and len(result) > 0

    async def _diff_summary(self) -> Dict[str, int]:
        """Return files changed, insertions, deletions for uncommitted work."""
        result = await _run_git("diff", "--shortstat", cwd=self._repo_path)
        summary: Dict[str, int] = {"files_changed": 0, "insertions": 0, "deletions": 0}
        if not result:
            return summary
        # Example: " 3 files changed, 12 insertions(+), 4 deletions(-)"
        for part in result.split(","):
            part = part.strip()
            if "file" in part:
                summary["files_changed"] = int(part.split()[0])
            elif "insertion" in part:
                summary["insertions"] = int(part.split()[0])
            elif "deletion" in part:
                summary["deletions"] = int(part.split()[0])
        return summary

    async def _ahead_behind(self) -> Optional[Dict[str, int]]:
        """Return how many commits ahead/behind the tracking remote."""
        result = await _run_git(
            "rev-list", "--left-right", "--count", "@{upstream}...HEAD", cwd=self._repo_path
        )
        if result is None:
            return None
        parts = result.split()
        if len(parts) != 2:
            return None
        return {"behind": int(parts[0]), "ahead": int(parts[1])}

    async def poll(self) -> ProbeResult:
        t0 = time.time()
        signals: List[EnvironmentSignal] = []

        try:
            if not await self._is_git_repo():
                return ProbeResult(
                    source=self.name,
                    timestamp=t0,
                    signals=[],
                    latency_ms=round((time.time() - t0) * 1000, 2),
                    metadata={"git_repo": False},
                )

            branch, head, dirty, diff_summary, ahead_behind = await asyncio.gather(
                self._current_branch(),
                self._head_commit(),
                self._is_dirty(),
                self._diff_summary(),
                self._ahead_behind(),
            )

            # --- Branch switch ---
            if self._prev_branch is not None and branch != self._prev_branch:
                signals.append(
                    EnvironmentSignal(
                        source="probe.git",
                        modality="git",
                        payload={
                            "event": "git.branch_switch",
                            "from": self._prev_branch,
                            "to": branch,
                        },
                        confidence=1.0,
                    )
                )

            # --- New commits ---
            if self._prev_head is not None and head != self._prev_head:
                signals.append(
                    EnvironmentSignal(
                        source="probe.git",
                        modality="git",
                        payload={
                            "event": "git.new_commits",
                            "prev_head": self._prev_head,
                            "current_head": head,
                            "branch": branch,
                        },
                        confidence=1.0,
                    )
                )

            # --- Uncommitted changes ---
            if dirty:
                signals.append(
                    EnvironmentSignal(
                        source="probe.git",
                        modality="git",
                        payload={
                            "event": "git.uncommitted_changes",
                            "branch": branch,
                            "diff_summary": diff_summary,
                        },
                        confidence=1.0,
                    )
                )

            # --- Remote divergence ---
            if ahead_behind is not None and (
                ahead_behind["ahead"] > 0 or ahead_behind["behind"] > 0
            ):
                signals.append(
                    EnvironmentSignal(
                        source="probe.git",
                        modality="git",
                        payload={
                            "event": "git.remote_diverged",
                            "branch": branch,
                            "ahead": ahead_behind["ahead"],
                            "behind": ahead_behind["behind"],
                        },
                        confidence=1.0,
                    )
                )

            # Update tracked state.
            self._prev_branch = branch
            self._prev_head = head
            self._prev_dirty = dirty
            self._prev_diff_summary = diff_summary

        except Exception as exc:
            signals.append(
                EnvironmentSignal(
                    source="probe.git",
                    modality="git",
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
            metadata={"repo_path": self._repo_path},
        )
