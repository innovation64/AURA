"""Room — a shared perception space where multiple agents observe the same environment.

Architecture:
    Shared (per-room):   Sense → Explore → Scene   (executed once, broadcast to all)
    Per-agent (external): Memory → Reason → Act     (each agent handles its own)

A Room is the spatial primitive for multi-agent environment awareness.
Like a physical room: everyone inside sees the same scene,
but each person thinks and acts independently.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from aura.explore import Explorer, ExplorationOutcome, HeuristicPlanner
from aura.scene import BasicScene, SceneModel
from aura.sense import BasicSense, SenseAdapter
from aura.tools import ToolRegistry
from aura.builtin_tools import default_tools
from aura.types import EnvironmentSignal, SceneState

logger = logging.getLogger("aura.server.room")


@dataclass
class AgentSubscription:
    """A connected agent subscribing to room updates."""
    agent_id: str
    agent_name: str = ""
    agent_type: str = "generic"  # "nanobot" | "openclaw" | "nexus" | "generic"
    connected_at: float = field(default_factory=time.time)
    websocket: Any = None  # starlette WebSocket, set at runtime


@dataclass
class RoomInfo:
    """Serializable room metadata."""
    room_id: str
    name: str
    scene_summary: str
    agent_count: int
    signal_count: int
    created_at: float
    last_update: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Room:
    """Shared perception space for multiple agents.

    The room owns the Sense → Explore → Scene pipeline.
    When new signals arrive, the room rebuilds the shared SceneState
    and pushes it to all subscribed agents via WebSocket.
    """

    def __init__(
        self,
        room_id: str,
        name: str = "",
        sense: Optional[SenseAdapter] = None,
        scene_model: Optional[SceneModel] = None,
        explorer: Optional[Explorer] = None,
        evolver: Optional[Any] = None,
    ) -> None:
        self.room_id = room_id
        self.name = name or room_id
        self.created_at = time.time()
        self.last_update = self.created_at

        # Shared perception components
        self.sense = sense or BasicSense(source="room")
        self.scene_model = scene_model or BasicScene()
        self.explorer = explorer or Explorer(
            planner=HeuristicPlanner(),
            registry=ToolRegistry(default_tools()),
            max_steps=3,
        )

        # Evolver (optional world-level mutation engine)
        self.evolver = evolver

        # State
        self._signals: List[EnvironmentSignal] = []
        self._current_scene: Optional[SceneState] = None
        self._subscribers: Dict[str, AgentSubscription] = {}
        self._lock = asyncio.Lock()

    @property
    def current_scene(self) -> Optional[SceneState]:
        return self._current_scene

    @property
    def agents(self) -> List[AgentSubscription]:
        return list(self._subscribers.values())

    @property
    def agent_count(self) -> int:
        return len(self._subscribers)

    @property
    def signal_count(self) -> int:
        return len(self._signals)

    def info(self) -> RoomInfo:
        return RoomInfo(
            room_id=self.room_id,
            name=self.name,
            scene_summary=self._current_scene.summary if self._current_scene else "(no scene)",
            agent_count=self.agent_count,
            signal_count=self.signal_count,
            created_at=self.created_at,
            last_update=self.last_update,
        )

    # ── Agent subscription ──────────────────────────────────

    def subscribe(self, sub: AgentSubscription) -> None:
        self._subscribers[sub.agent_id] = sub
        logger.info(
            "Agent %s (%s) joined room %s [%d agents]",
            sub.agent_id, sub.agent_type, self.room_id, self.agent_count,
        )

    def unsubscribe(self, agent_id: str) -> None:
        removed = self._subscribers.pop(agent_id, None)
        if removed:
            logger.info(
                "Agent %s left room %s [%d agents]",
                agent_id, self.room_id, self.agent_count,
            )

    def has_agent(self, agent_id: str) -> bool:
        return agent_id in self._subscribers

    # ── Signal ingestion + scene rebuild ────────────────────

    async def ingest(
        self,
        raw_input: Any,
        explore: bool = False,
        user_query: Optional[str] = None,
    ) -> SceneState:
        """Ingest environment signals, rebuild scene, broadcast to all agents."""
        async with self._lock:
            # 1) Sense: raw input → signals
            new_signals = self.sense.ingest(raw_input)
            self._signals.extend(new_signals)

            # 2) Explore: optional tool probing
            exploration: Optional[ExplorationOutcome] = None
            if explore and self.explorer:
                exploration = self.explorer.explore(
                    self._signals, user_query=user_query, raw_input=raw_input,
                )
                self._signals.extend(exploration.extra_signals)

            # 3) Scene: rebuild from all signals
            self._current_scene = self.scene_model.build(self._signals)
            self.last_update = time.time()

        # 4) Broadcast to all subscribers
        await self._broadcast_scene(exploration)

        logger.info(
            "Room %s updated: %d signals → %s",
            self.room_id, len(self._signals), self._current_scene.summary,
        )
        return self._current_scene

    async def explore_environment(
        self, user_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Trigger active exploration and update the shared scene."""
        if not self.explorer:
            return {"error": "exploration not enabled"}

        async with self._lock:
            outcome = self.explorer.explore(
                self._signals, user_query=user_query,
            )
            self._signals.extend(outcome.extra_signals)
            self._current_scene = self.scene_model.build(self._signals)
            self.last_update = time.time()

        await self._broadcast_scene(outcome)
        return outcome.summary()

    # ── Broadcasting ────────────────────────────────────────

    async def _broadcast_scene(
        self, exploration: Optional[ExplorationOutcome] = None,
    ) -> None:
        """Push current scene to all connected agents via WebSocket."""
        if not self._current_scene:
            return

        message = {
            "type": "scene.update",
            "room_id": self.room_id,
            "timestamp": self.last_update,
            "scene": {
                "summary": self._current_scene.summary,
                "entities": self._current_scene.entities,
                "context": self._current_scene.context,
            },
            "agents_in_room": [
                {"agent_id": s.agent_id, "agent_name": s.agent_name, "agent_type": s.agent_type}
                for s in self._subscribers.values()
            ],
        }
        if exploration:
            message["exploration"] = exploration.summary()

        payload = json.dumps(message, ensure_ascii=False, default=str)

        disconnected: List[str] = []
        for agent_id, sub in self._subscribers.items():
            if sub.websocket is None:
                continue
            try:
                await sub.websocket.send_text(payload)
            except Exception:
                disconnected.append(agent_id)

        for agent_id in disconnected:
            self.unsubscribe(agent_id)

    async def _broadcast_evolution(self, result: Any) -> None:
        """Push evolution results to all connected agents via WebSocket."""
        message = {
            "type": "evolution.update",
            "room_id": self.room_id,
            "timestamp": time.time(),
            "summary": result.summary() if hasattr(result, "summary") else str(result),
            "mutations": [
                {
                    "type": m.type.value if hasattr(m.type, "value") else str(m.type),
                    "target": m.target,
                    "reason": m.reason,
                }
                for m in (result.mutations if hasattr(result, "mutations") else [])
            ],
        }

        payload = json.dumps(message, ensure_ascii=False, default=str)

        disconnected: List[str] = []
        for agent_id, sub in self._subscribers.items():
            if sub.websocket is None:
                continue
            try:
                await sub.websocket.send_text(payload)
            except Exception:
                disconnected.append(agent_id)

        for agent_id in disconnected:
            self.unsubscribe(agent_id)

    def clear_signals(self) -> None:
        """Clear accumulated signals (reset room state)."""
        self._signals.clear()
        self._current_scene = None


class RoomManager:
    """Manages the lifecycle of all rooms."""

    def __init__(self) -> None:
        self._rooms: Dict[str, Room] = {}

    def create(self, name: str = "", room_id: Optional[str] = None) -> Room:
        rid = room_id or uuid.uuid4().hex[:8]
        if rid in self._rooms:
            return self._rooms[rid]
        room = Room(room_id=rid, name=name)
        self._rooms[rid] = room
        logger.info("Room created: %s (%s)", rid, name)
        return room

    def get(self, room_id: str) -> Optional[Room]:
        return self._rooms.get(room_id)

    def delete(self, room_id: str) -> bool:
        room = self._rooms.pop(room_id, None)
        if room:
            logger.info("Room deleted: %s", room_id)
            return True
        return False

    def list_all(self) -> List[RoomInfo]:
        return [room.info() for room in self._rooms.values()]
