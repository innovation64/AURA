"""Trajectory collection for recording (state, action, result) tuples."""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryStep:
    """A single (environment_state, agent_action, result) record."""

    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    environment_state: dict = field(default_factory=dict)
    agent_action: str = ""
    action_metadata: dict = field(default_factory=dict)
    result: Optional[str] = None
    reward: float = 0.5
    context_was_used: bool = False
    episode_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrajectoryStep":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class TrajectoryCollector:
    """Collects trajectory steps, grouping them into episodes.

    Steps are buffered in memory and can be flushed to disk as JSONL.
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        max_buffer: int = 10000,
    ) -> None:
        self._storage_path = storage_path
        self._max_buffer = max_buffer

        self._buffer: List[TrajectoryStep] = []
        self._episodes: Dict[str, List[TrajectoryStep]] = {}
        self._current_episode_id: Optional[str] = None
        self._current_task: str = ""

        # Aggregate counters
        self._total_episodes: int = 0
        self._total_steps: int = 0
        self._total_reward: float = 0.0

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def start_episode(self, task_description: str = "") -> str:
        """Begin a new episode and return its id."""
        if self._current_episode_id is not None:
            logger.warning(
                "Starting new episode while episode %s is still active; "
                "ending the previous episode first.",
                self._current_episode_id,
            )
            self.end_episode()

        episode_id = str(uuid.uuid4())
        self._current_episode_id = episode_id
        self._current_task = task_description
        self._episodes[episode_id] = []
        logger.info("Started episode %s: %s", episode_id, task_description)
        return episode_id

    def record_step(
        self,
        environment_state: dict,
        agent_action: str,
        result: Optional[str] = None,
        reward: float = 0.5,
        context_was_used: bool = False,
        metadata: Optional[dict] = None,
    ) -> TrajectoryStep:
        """Record a single trajectory step in the current episode."""
        if self._current_episode_id is None:
            # Auto-start an episode for convenience
            self.start_episode()

        step = TrajectoryStep(
            environment_state=environment_state,
            agent_action=agent_action,
            result=result,
            reward=reward,
            context_was_used=context_was_used,
            action_metadata=metadata if metadata is not None else {},
            episode_id=self._current_episode_id,  # type: ignore[arg-type]
        )

        self._buffer.append(step)
        self._episodes[self._current_episode_id].append(step)  # type: ignore[index]
        self._total_steps += 1
        self._total_reward += reward

        # Auto-flush when buffer is full
        if len(self._buffer) >= self._max_buffer:
            logger.info("Buffer full (%d steps); auto-saving.", len(self._buffer))
            self.save()

        return step

    def end_episode(self) -> List[TrajectoryStep]:
        """End the current episode and return its steps."""
        if self._current_episode_id is None:
            logger.warning("end_episode called with no active episode.")
            return []

        episode_id = self._current_episode_id
        steps = self._episodes.get(episode_id, [])
        self._total_episodes += 1
        self._current_episode_id = None
        self._current_task = ""
        logger.info(
            "Ended episode %s with %d steps.", episode_id, len(steps)
        )
        return list(steps)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_episode(self, episode_id: str) -> List[TrajectoryStep]:
        """Return all steps for *episode_id*."""
        return list(self._episodes.get(episode_id, []))

    def get_statistics(self) -> dict:
        """Return aggregate statistics about collected trajectories."""
        total_episodes = self._total_episodes
        # Count in-progress episode too
        if self._current_episode_id is not None:
            total_episodes += 1

        total_steps = self._total_steps
        avg_reward = self._total_reward / max(total_steps, 1)
        avg_steps_per_episode = total_steps / max(total_episodes, 1)

        return {
            "total_episodes": total_episodes,
            "total_steps": total_steps,
            "avg_reward": round(avg_reward, 4),
            "avg_steps_per_episode": round(avg_steps_per_episode, 2),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> None:
        """Write buffered steps to a JSONL file and clear the buffer."""
        target = path or self._storage_path
        if target is None:
            target = Path("trajectory_data.jsonl")

        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)

        with open(target, "a", encoding="utf-8") as fh:
            for step in self._buffer:
                fh.write(json.dumps(step.to_dict(), default=str) + "\n")

        logger.info("Saved %d steps to %s.", len(self._buffer), target)
        self._buffer.clear()

    @staticmethod
    def load(path: Path) -> List[TrajectoryStep]:
        """Load trajectory steps from a JSONL file."""
        path = Path(path)
        if not path.exists():
            logger.warning("Trajectory file %s does not exist.", path)
            return []

        steps: List[TrajectoryStep] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    steps.append(TrajectoryStep.from_dict(data))
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning(
                        "Skipping malformed line %d in %s: %s",
                        line_no,
                        path,
                        exc,
                    )
        logger.info("Loaded %d steps from %s.", len(steps), path)
        return steps
