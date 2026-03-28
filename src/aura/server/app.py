"""AURA Server — shared environment perception for multi-agent systems.

Exposes AURA's Sense → Explore → Scene pipeline as a service.
Multiple agents (nanobot, OpenClaw, NEXUS, etc.) connect to "rooms"
and receive shared SceneState updates via WebSocket.

Endpoints:
    POST   /v1/rooms                  Create a room
    GET    /v1/rooms                  List all rooms
    GET    /v1/rooms/{id}             Get room info + current scene
    DELETE /v1/rooms/{id}             Delete a room
    POST   /v1/rooms/{id}/signals     Inject environment signals
    POST   /v1/rooms/{id}/explore     Trigger active exploration
    POST   /v1/rooms/{id}/clear       Clear room signals
    WS     /v1/rooms/{id}/subscribe   Agent subscribes to scene updates
    GET    /v1/health                 Health check
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field

from .room import AgentSubscription, RoomManager

logger = logging.getLogger("aura.server")

# ── Global state ────────────────────────────────────────

rooms = RoomManager()

# ── Request / Response models ───────────────────────────


class CreateRoomRequest(BaseModel):
    name: str = ""
    room_id: Optional[str] = None


class CreateRoomResponse(BaseModel):
    room_id: str
    name: str


class SignalInput(BaseModel):
    source: str = "external"
    payload: Dict[str, Any] = Field(default_factory=dict)
    modality: str = "generic"
    confidence: float = 1.0


class IngestRequest(BaseModel):
    signals: List[SignalInput] = Field(default_factory=list)
    raw: Optional[Any] = None
    explore: bool = False
    query: Optional[str] = None


class IngestResponse(BaseModel):
    scene_summary: str
    entities: List[str]
    signal_count: int
    agent_count: int


class ExploreRequest(BaseModel):
    query: Optional[str] = None


class EvolveRequest(BaseModel):
    tick_index: int = 0


class RoomDetail(BaseModel):
    room_id: str
    name: str
    scene_summary: str
    entities: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    agents: List[Dict[str, Any]] = Field(default_factory=list)
    signal_count: int
    created_at: float
    last_update: float


# ── Lifespan ────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AURA Server starting — multi-agent shared perception service")
    yield
    logger.info("AURA Server shutting down")


# ── App ─────────────────────────────────────────────────

app = FastAPI(
    title="AURA Server",
    description="Multi-agent shared environment perception service",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Room endpoints ──────────────────────────────────────


@app.post("/v1/rooms", status_code=201)
async def create_room(req: CreateRoomRequest) -> CreateRoomResponse:
    room = rooms.create(name=req.name, room_id=req.room_id)
    return CreateRoomResponse(room_id=room.room_id, name=room.name)


@app.get("/v1/rooms")
async def list_rooms():
    return {"rooms": [r.to_dict() for r in rooms.list_all()]}


@app.get("/v1/rooms/{room_id}")
async def get_room(room_id: str) -> RoomDetail:
    room = rooms.get(room_id)
    if not room:
        raise HTTPException(404, f"Room '{room_id}' not found")

    scene = room.current_scene
    return RoomDetail(
        room_id=room.room_id,
        name=room.name,
        scene_summary=scene.summary if scene else "(no scene)",
        entities=scene.entities if scene else [],
        context=scene.context if scene else {},
        agents=[
            {
                "agent_id": a.agent_id,
                "agent_name": a.agent_name,
                "agent_type": a.agent_type,
                "connected_at": a.connected_at,
            }
            for a in room.agents
        ],
        signal_count=room.signal_count,
        created_at=room.created_at,
        last_update=room.last_update,
    )


@app.delete("/v1/rooms/{room_id}")
async def delete_room(room_id: str):
    if not rooms.delete(room_id):
        raise HTTPException(404, f"Room '{room_id}' not found")
    return {"deleted": True}


# ── Signal ingestion ────────────────────────────────────


@app.post("/v1/rooms/{room_id}/signals")
async def ingest_signals(room_id: str, req: IngestRequest) -> IngestResponse:
    room = rooms.get(room_id)
    if not room:
        raise HTTPException(404, f"Room '{room_id}' not found")

    # Build raw input for AURA's Sense adapter
    if req.signals:
        from aura.types import EnvironmentSignal

        raw_input = [
            EnvironmentSignal(
                source=s.source,
                payload=s.payload,
                modality=s.modality,
                confidence=s.confidence,
            )
            for s in req.signals
        ]
    elif req.raw is not None:
        raw_input = req.raw
    else:
        raise HTTPException(400, "Provide 'signals' or 'raw'")

    scene = await room.ingest(
        raw_input=raw_input,
        explore=req.explore,
        user_query=req.query,
    )

    return IngestResponse(
        scene_summary=scene.summary,
        entities=scene.entities,
        signal_count=room.signal_count,
        agent_count=room.agent_count,
    )


# ── Exploration ─────────────────────────────────────────


@app.post("/v1/rooms/{room_id}/explore")
async def explore_room(room_id: str, req: ExploreRequest):
    room = rooms.get(room_id)
    if not room:
        raise HTTPException(404, f"Room '{room_id}' not found")

    result = await room.explore_environment(user_query=req.query)
    scene = room.current_scene
    return {
        "exploration": result,
        "scene_summary": scene.summary if scene else "(no scene)",
        "entities": scene.entities if scene else [],
    }


# ── Evolution ──────────────────────────────────────────


@app.post("/v1/rooms/{room_id}/evolve")
async def evolve_room(room_id: str, req: EvolveRequest):
    room = rooms.get(room_id)
    if not room:
        raise HTTPException(404, f"Room '{room_id}' not found")

    if not room.evolver:
        return {"error": "No evolver configured for this room", "mutations": []}

    # Evolve requires a WorldState-compatible object; return empty if not available
    result = room.evolver.evolve(
        world_state=None,  # Rooms don't have a built-in world; caller provides context
        activity_signals=[],
        tick_index=req.tick_index,
    )

    if result.mutations:
        await room._broadcast_evolution(result)

    return {
        "summary": result.summary(),
        "mutations": [
            {
                "type": m.type.value if hasattr(m.type, "value") else str(m.type),
                "target": m.target,
                "payload": m.payload,
                "reason": m.reason,
            }
            for m in result.mutations
        ],
    }


# ── Clear ───────────────────────────────────────────────


@app.post("/v1/rooms/{room_id}/clear")
async def clear_room(room_id: str):
    room = rooms.get(room_id)
    if not room:
        raise HTTPException(404, f"Room '{room_id}' not found")
    room.clear_signals()
    return {"cleared": True}


# ── WebSocket subscription ──────────────────────────────


@app.websocket("/v1/rooms/{room_id}/subscribe")
async def subscribe_to_room(ws: WebSocket, room_id: str):
    """Agent connects via WebSocket to receive real-time scene updates.

    Handshake protocol:
        1. Agent connects to /v1/rooms/{room_id}/subscribe
        2. Agent sends JSON: {"agent_id": "...", "agent_name": "...", "agent_type": "nanobot"}
        3. Server sends back: {"type": "subscribed", "room_id": "...", "scene": {...}}
        4. Server pushes {"type": "scene.update", ...} on every scene change
        5. Agent can send {"type": "signal", "payload": {...}} to inject signals
    """
    room = rooms.get(room_id)
    if not room:
        await ws.close(code=4004, reason=f"Room '{room_id}' not found")
        return

    await ws.accept()

    # Wait for agent identification
    try:
        raw = await ws.receive_text()
        data = json.loads(raw)
        agent_id = data.get("agent_id", uuid.uuid4().hex[:8])
        agent_name = data.get("agent_name", agent_id)
        agent_type = data.get("agent_type", "generic")
    except Exception:
        await ws.close(code=4000, reason="Expected JSON with agent_id")
        return

    # Subscribe
    sub = AgentSubscription(
        agent_id=agent_id,
        agent_name=agent_name,
        agent_type=agent_type,
        websocket=ws,
    )
    room.subscribe(sub)

    # Send current scene as welcome
    scene = room.current_scene
    welcome = {
        "type": "subscribed",
        "room_id": room_id,
        "agent_id": agent_id,
        "scene": {
            "summary": scene.summary if scene else "(no scene yet)",
            "entities": scene.entities if scene else [],
            "context": scene.context if scene else {},
        },
        "agents_in_room": [
            {"agent_id": a.agent_id, "agent_name": a.agent_name, "agent_type": a.agent_type}
            for a in room.agents
        ],
    }
    await ws.send_text(json.dumps(welcome, ensure_ascii=False, default=str))

    # Listen for incoming messages from agent
    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type", "")

            if msg_type == "signal":
                # Agent can inject signals into the room
                from aura.types import EnvironmentSignal

                signal = EnvironmentSignal(
                    source=f"agent:{agent_id}",
                    payload=data.get("payload", {}),
                    modality=data.get("modality", "agent"),
                    confidence=data.get("confidence", 1.0),
                )
                await room.ingest(raw_input=[signal])

            elif msg_type == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        pass
    finally:
        room.unsubscribe(agent_id)


# ── Health ──────────────────────────────────────────────


@app.get("/v1/health")
async def health():
    total_agents = sum(r.agent_count for r in rooms._rooms.values())
    return {
        "status": "ok",
        "service": "aura-server",
        "version": "0.1.0",
        "rooms": len(rooms._rooms),
        "agents_connected": total_agents,
    }
