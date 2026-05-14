"""Lightweight HTTP API for the AURA Town frontend."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from .simulation import TownSimulation
from .asset_manager import AssetManager
from .ai_art import AIArtGenerator


_LOCK = threading.Lock()
_STEP_BUSY = threading.Lock()   # non-blocking: prevents concurrent sim.step()
_CHAT_BUSY = threading.Lock()   # non-blocking: prevents concurrent sim.chat()
_SIM: Optional[TownSimulation] = None
_ASSET_MGR: Optional[AssetManager] = None
_AI_ART: Optional[AIArtGenerator] = None


def _get_sim() -> TownSimulation:
    global _SIM
    if _SIM is None:
        _SIM = TownSimulation()
        _SIM.initialize()
    return _SIM


def _get_asset_mgr() -> AssetManager:
    global _ASSET_MGR
    if _ASSET_MGR is None:
        _ASSET_MGR = AssetManager()
    return _ASSET_MGR


def _get_ai_art() -> AIArtGenerator:
    global _AI_ART
    if _AI_ART is None:
        from .config import DEFAULT_CONFIG
        _AI_ART = AIArtGenerator(
            api_url=DEFAULT_CONFIG.image_gen_api_url or None,
            api_key=DEFAULT_CONFIG.image_gen_api_key or None,
        )
    return _AI_ART


def _cors(handler: BaseHTTPRequestHandler) -> None:
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


def _json_response(handler: BaseHTTPRequestHandler, payload: Dict[str, Any], status: int = 200) -> None:
    data = json.dumps(payload).encode("utf-8")
    try:
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(data)))
        _cors(handler)
        handler.end_headers()
        handler.wfile.write(data)
    except BrokenPipeError:
        pass  # Client disconnected before response was sent


def _read_json(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return {}


class TownRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _handle_state(self) -> None:
        with _LOCK:
            state = _get_sim().get_state()
        # Inject asset overrides for frontend rendering
        state["asset_overrides"] = _get_asset_mgr().get_asset_overrides()
        _json_response(self, {"ok": True, "state": state})

    def _handle_step(self) -> None:
        # Reject if a step is already in progress (non-blocking acquire).
        if not _STEP_BUSY.acquire(blocking=False):
            with _LOCK:
                state = _get_sim().get_state()
            _json_response(self, {"ok": True, "busy": True, "state": state})
            return
        try:
            with _LOCK:
                sim = _get_sim()
            sim.step()
            with _LOCK:
                state = sim.get_state()
            _json_response(self, {"ok": True, "state": state})
        finally:
            _STEP_BUSY.release()

    def _handle_reset(self) -> None:
        # Accept an optional `seed` to propagate to TownConfig.world_seed and
        # TownConfig.llm_seed so multi-seed experiment runners get genuinely
        # different (but reproducible) draws across reset boundaries. Without
        # this, "3 seeds" only differ in Python-level randomness, not in the
        # backbone draw or the procedural-world seed.
        payload = _read_json(self) if self.headers.get("Content-Length") else {}
        seed = payload.get("seed")
        with _LOCK:
            from .config import DEFAULT_CONFIG, TownConfig
            from dataclasses import replace
            if seed is not None:
                cfg = replace(DEFAULT_CONFIG, world_seed=int(seed), llm_seed=int(seed))
                sim = TownSimulation(cfg)
            else:
                sim = TownSimulation()
            sim.initialize()
            global _SIM
            _SIM = sim
            state = sim.get_state()
        _json_response(self, {"ok": True, "state": state, "seed": seed})

    def _handle_probe(self) -> None:
        payload = _read_json(self)
        enabled = bool(payload.get("enabled", True))
        max_steps = int(payload.get("max_steps", 2))
        with _LOCK:
            sim = _get_sim()
            sim.update_probe_settings(enabled, max_steps)
            state = sim.get_state()
        _json_response(self, {"ok": True, "state": state})

    def _handle_ablation(self) -> None:
        """POST /api/ablation — toggle memory/reflection for ablation experiments."""
        payload = _read_json(self)
        memory_enabled = bool(payload.get("memory_enabled", True))
        reflection_enabled = bool(payload.get("reflection_enabled", True))
        with _LOCK:
            sim = _get_sim()
            sim.update_ablation_settings(memory_enabled, reflection_enabled)
            state = sim.get_state()
        _json_response(self, {"ok": True, "state": state})

    def _handle_action_mode(self) -> None:
        """POST /api/action_mode - switch between AURA proactive and ReAct reactive."""
        payload = _read_json(self)
        react_mode = bool(payload.get("react_mode", False))
        with _LOCK:
            sim = _get_sim()
            sim.update_action_mode(react_mode)
            state = sim.get_state()
        _json_response(self, {"ok": True, "react_mode": react_mode, "state": state})

    def _handle_chat(self) -> None:
        payload = _read_json(self)
        user = payload.get("user", "")
        message = payload.get("message", "")
        # When read_only=true the chat does NOT write event log or memory —
        # used by the RQ2 paired-snapshot runner so conditions can replay the
        # same simulation state without cross-condition pollution.
        read_only = bool(payload.get("read_only", False))
        if not user or not message:
            _json_response(
                self,
                {"ok": False, "error": "user and message required"},
                status=400,
            )
            return
        # Reject if a chat is already in progress.
        if not _CHAT_BUSY.acquire(blocking=False):
            _json_response(self, {"ok": False, "error": "Another chat is in progress"}, status=429)
            return
        try:
            with _LOCK:
                sim = _get_sim()
            result = sim.chat(user, message, read_only=read_only)
            with _LOCK:
                state = sim.get_state()
            _json_response(self, {"ok": True, "chat": result, "state": state})
        finally:
            _CHAT_BUSY.release()

    def _handle_agent(self, name: str) -> None:
        with _LOCK:
            sim = _get_sim()
            detail = sim.get_agent_detail(name)
        if detail is None:
            _json_response(self, {"ok": False, "error": "agent not found"}, status=404)
            return
        _json_response(self, {"ok": True, "agent": detail})

    def _handle_chunks(self, params: dict) -> None:
        """GET /api/chunks?x=&y=&w=&h= — return chunk biome data for a viewport."""
        x = int(params.get("x", ["0"])[0])
        y = int(params.get("y", ["0"])[0])
        w = int(params.get("w", ["320"])[0])
        h = int(params.get("h", ["256"])[0])
        with _LOCK:
            sim = _get_sim()
            biomes = sim.world.chunk_manager.get_biome_map(x, y, w, h)
        _json_response(self, {"ok": True, "chunk_biomes": biomes})

    def _handle_evolution(self) -> None:
        """GET /api/evolution — return evolution history and world properties."""
        with _LOCK:
            sim = _get_sim()
            data = {
                "world_properties": sim.world.get_world_properties(),
                "evolution_log": sim.world.evolution_log[-50:],
                "evolve_enabled": sim.config.evolve_enabled,
                "evolve_interval": sim.config.evolve_interval,
                "evolve_max_mutations": sim.config.evolve_max_mutations,
            }
        _json_response(self, {"ok": True, "evolution": data})

    def _handle_force_evolve(self) -> None:
        """POST /api/evolve — force an evolution check."""
        with _LOCK:
            sim = _get_sim()
            events = sim._run_evolution()
            state = sim.get_state()
        _json_response(self, {
            "ok": True,
            "events": [
                {"time": e.time, "agent": e.agent, "type": e.event_type, "description": e.description}
                for e in events
            ],
            "state": state,
        })

    def _handle_agent_control(self) -> None:
        """POST /api/agent/control — toggle player control mode."""
        payload = _read_json(self)
        name = payload.get("agent", "")
        controlled = bool(payload.get("controlled", False))
        if not name:
            _json_response(self, {"ok": False, "error": "agent name required"}, status=400)
            return
        with _LOCK:
            sim = _get_sim()
            result = sim.set_agent_control(name, controlled)
        if result is None:
            _json_response(self, {"ok": False, "error": "agent not found"}, status=404)
            return
        _json_response(self, {"ok": True, "state": result})

    def _handle_agent_action(self) -> None:
        """POST /api/agent/action — queue a manual action for a controlled agent."""
        payload = _read_json(self)
        name = payload.get("agent", "")
        if not name:
            _json_response(self, {"ok": False, "error": "agent name required"}, status=400)
            return
        action_data = {
            "action": payload.get("action", ""),
            "location": payload.get("location", ""),
            "emoji": payload.get("emoji", ""),
        }
        with _LOCK:
            sim = _get_sim()
            result = sim.set_agent_action(name, action_data)
        if result is None:
            _json_response(self, {"ok": False, "error": "agent not found or not player-controlled"}, status=400)
            return
        _json_response(self, {"ok": True, "state": result})

    def _handle_agent_interact(self) -> None:
        """POST /api/agent/interact — initiate a conversation with a nearby agent."""
        payload = _read_json(self)
        agent_name = payload.get("agent", "")
        target_name = payload.get("target", "")
        if not agent_name or not target_name:
            _json_response(self, {"ok": False, "error": "agent and target required"}, status=400)
            return
        with _LOCK:
            sim = _get_sim()
            result = sim.initiate_conversation(agent_name, target_name)
        if result is None:
            _json_response(self, {"ok": False, "error": "agents not found or not at same location"}, status=400)
            return
        _json_response(self, {"ok": True, "conversation": result["conversation"], "state": result["state"]})

    def _handle_agent_move(self) -> None:
        """POST /api/agent/move — D-pad directional movement."""
        payload = _read_json(self)
        name = payload.get("agent", "")
        direction = payload.get("direction", "")
        steps = int(payload.get("steps", 2))
        if not name or not direction:
            _json_response(self, {"ok": False, "error": "agent and direction required"}, status=400)
            return
        with _LOCK:
            sim = _get_sim()
            result = sim.move_agent_direction(name, direction, steps)
        if result is None:
            _json_response(self, {"ok": False, "error": "agent not found or invalid direction"}, status=400)
            return
        _json_response(self, {"ok": True, "state": result})

    def _handle_agent_explore(self) -> None:
        """POST /api/agent/explore — send agent exploring in a direction."""
        payload = _read_json(self)
        name = payload.get("agent", "")
        direction = payload.get("direction", "")
        if not name or not direction:
            _json_response(self, {"ok": False, "error": "agent and direction required"}, status=400)
            return
        with _LOCK:
            sim = _get_sim()
            result = sim.explore_direction(name, direction)
        if result is None:
            _json_response(self, {"ok": False, "error": "agent not found or invalid direction"}, status=400)
            return
        _json_response(self, {"ok": True, "state": result})

    def _handle_location_detail(self, params: dict) -> None:
        """GET /api/location?name=... — location detail + occupants."""
        name = params.get("name", [""])[0]
        if not name:
            _json_response(self, {"ok": False, "error": "name required"}, status=400)
            return
        with _LOCK:
            sim = _get_sim()
            detail = sim.world.get_location_detail(name)
            if detail is None:
                _json_response(self, {"ok": False, "error": "location not found"}, status=404)
                return
            # Add current occupants
            occupants = []
            for agent in sim.agents:
                if agent.state.current_location_name == detail["name"]:
                    occupants.append({
                        "name": agent.name,
                        "emoji": agent.profile.emoji,
                        "action": agent.state.current_action,
                    })
            detail["occupants"] = occupants
        _json_response(self, {"ok": True, "location": detail})

    def _handle_evolve_settings(self) -> None:
        """POST /api/evolve/settings — update evolve settings."""
        payload = _read_json(self)
        with _LOCK:
            sim = _get_sim()
            if "enabled" in payload:
                sim.config.evolve_enabled = bool(payload["enabled"])
            if "interval" in payload:
                sim.config.evolve_interval = max(1, int(payload["interval"]))
            if "max_mutations" in payload:
                sim.config.evolve_max_mutations = max(1, int(payload["max_mutations"]))
            state = sim.get_state()
        _json_response(self, {"ok": True, "state": state})

    # ── Map generation endpoints (Phase 1B) ─────────────────────────

    def _handle_map_generate(self) -> None:
        """POST /api/map/generate — generate map from natural language."""
        payload = _read_json(self)
        prompt = payload.get("prompt", "")
        origin_x = int(payload.get("origin_x", 80))
        origin_y = int(payload.get("origin_y", 0))
        max_locations = int(payload.get("max_locations", 8))
        if not prompt:
            _json_response(self, {"ok": False, "error": "prompt required"}, status=400)
            return
        with _LOCK:
            sim = _get_sim()
            spec = sim._map_generator.from_natural_language(
                prompt, (origin_x, origin_y), max_locations
            )
            if spec is None:
                _json_response(self, {"ok": False, "error": "Map generation failed"}, status=500)
                return
            placed = sim._map_generator.apply_to_world(sim.world, spec)
            # Register as a region
            from .regions import RegionInfo
            region = RegionInfo(
                id=f"gen_{spec.name.lower().replace(' ', '_')}",
                name=spec.name,
                biome=spec.biome,
                world_x=spec.origin_x,
                world_y=spec.origin_y,
                width=spec.width,
                height=spec.height,
                description=spec.description,
                location_count=len(placed),
            )
            sim._region_manager.register_region(region)
            state = sim.get_state()
        _json_response(self, {
            "ok": True,
            "spec": {
                "name": spec.name,
                "biome": spec.biome,
                "description": spec.description,
                "origin": [spec.origin_x, spec.origin_y],
                "locations_placed": len(placed),
                "locations": [{"name": l.name, "type": l.type} for l in placed],
            },
            "state": state,
        })

    def _handle_map_template(self) -> None:
        """POST /api/map/template — generate map from template."""
        payload = _read_json(self)
        template_key = payload.get("template", "")
        origin_x = int(payload.get("origin_x", 80))
        origin_y = int(payload.get("origin_y", 0))
        customizations = payload.get("customizations")
        if not template_key:
            _json_response(self, {"ok": False, "error": "template key required"}, status=400)
            return
        with _LOCK:
            sim = _get_sim()
            spec = sim._map_generator.from_template(
                template_key, (origin_x, origin_y), customizations
            )
            if spec is None:
                _json_response(self, {"ok": False, "error": f"Template '{template_key}' not found"}, status=404)
                return
            placed = sim._map_generator.apply_to_world(sim.world, spec)
            # Register as a region
            from .regions import RegionInfo
            region = RegionInfo(
                id=f"tmpl_{template_key}",
                name=spec.name,
                biome=spec.biome,
                world_x=spec.origin_x,
                world_y=spec.origin_y,
                width=spec.width,
                height=spec.height,
                description=spec.description,
                location_count=len(placed),
            )
            sim._region_manager.register_region(region)
            state = sim.get_state()
        _json_response(self, {
            "ok": True,
            "spec": {
                "name": spec.name,
                "biome": spec.biome,
                "description": spec.description,
                "origin": [spec.origin_x, spec.origin_y],
                "locations_placed": len(placed),
                "locations": [{"name": l.name, "type": l.type} for l in placed],
            },
            "state": state,
        })

    def _handle_map_templates(self) -> None:
        """GET /api/map/templates — list available templates."""
        from .map_generator import MapGenerator
        templates = MapGenerator.list_templates()
        _json_response(self, {"ok": True, "templates": templates})

    # ── World map endpoints (Phase 2A) ────────────────────────────

    def _handle_worldmap(self) -> None:
        """GET /api/worldmap — world map data."""
        with _LOCK:
            sim = _get_sim()
            wm = sim._region_manager.get_world_map_data(sim.agents)
        _json_response(self, {
            "ok": True,
            "world_map": {
                "regions": wm.regions,
                "agent_positions": wm.agent_positions,
                "connections": wm.connections,
                "world_bounds": wm.world_bounds,
            },
        })

    def _handle_worldmap_teleport(self) -> None:
        """POST /api/worldmap/teleport — teleport agent to region center."""
        payload = _read_json(self)
        agent_name = payload.get("agent", "")
        region_id = payload.get("region_id", "")
        if not agent_name or not region_id:
            _json_response(self, {"ok": False, "error": "agent and region_id required"}, status=400)
            return
        with _LOCK:
            sim = _get_sim()
            region = sim._region_manager.get_region_by_id(region_id)
            if region is None:
                _json_response(self, {"ok": False, "error": "region not found"}, status=404)
                return
            agent = None
            for a in sim.agents:
                if a.name == agent_name:
                    agent = a
                    break
            if agent is None:
                _json_response(self, {"ok": False, "error": "agent not found"}, status=404)
                return
            # Teleport to a location inside this region, preferring one
            # that is not also inside another (smaller) region.
            target_x = region.world_x + region.width // 2
            target_y = region.world_y + region.height // 2
            # Try to find a location exclusively inside this region
            best_loc = None
            for loc in sim.world.locations:
                if (region.world_x <= loc.x < region.world_x + region.width and
                        region.world_y <= loc.y < region.world_y + region.height):
                    actual_region = sim._region_manager.get_region_at(loc.x, loc.y)
                    if actual_region and actual_region.id == region_id:
                        best_loc = loc
                        break
                    elif best_loc is None:
                        best_loc = loc
            if best_loc:
                target_x = best_loc.x + best_loc.width // 2
                target_y = best_loc.y + best_loc.height // 2
            agent.state.x = target_x
            agent.state.y = target_y
            loc = sim.world.get_location_at(agent.state.x, agent.state.y)
            agent.state.current_location_name = loc.name if loc else ""
            # Ensure chunks are generated around new position
            from .chunks import CHUNK_SIZE
            cx = agent.state.x // CHUNK_SIZE
            cy = agent.state.y // CHUNK_SIZE
            for ddx in range(-1, 2):
                for ddy in range(-1, 2):
                    sim.world.ensure_chunk_locations(cx + ddx, cy + ddy)
                    agent.state.explored_chunks.add((cx + ddx, cy + ddy))
            state = sim.get_state()
        _json_response(self, {"ok": True, "state": state})

    # ── Asset management endpoints (Phase 3B) ─────────────────────

    def _handle_assets_upload(self) -> None:
        """POST /api/assets/upload — upload a custom asset (multipart or JSON base64)."""
        payload = _read_json(self)
        name = payload.get("name", "")
        asset_type = payload.get("asset_type", "")
        target = payload.get("target", "")
        filename = payload.get("filename", "upload.png")

        # Support base64-encoded file data in JSON
        import base64
        file_b64 = payload.get("file_data", "")
        if not name or not asset_type or not file_b64:
            _json_response(self, {"ok": False, "error": "name, asset_type, and file_data required"}, status=400)
            return
        try:
            file_data = base64.b64decode(file_b64)
        except Exception:
            _json_response(self, {"ok": False, "error": "invalid base64 file_data"}, status=400)
            return

        mgr = _get_asset_mgr()
        asset = mgr.upload_asset(name, asset_type, file_data, filename, target)
        if asset is None:
            _json_response(self, {"ok": False, "error": "Upload failed"}, status=500)
            return
        _json_response(self, {"ok": True, "asset": asset.to_dict()})

    def _handle_assets_delete(self, asset_id: str) -> None:
        """DELETE /api/assets/{id} — delete a custom asset."""
        mgr = _get_asset_mgr()
        if mgr.delete_asset(asset_id):
            _json_response(self, {"ok": True})
        else:
            _json_response(self, {"ok": False, "error": "asset not found"}, status=404)

    def _handle_assets_list(self) -> None:
        """GET /api/assets — list all custom assets."""
        mgr = _get_asset_mgr()
        assets = mgr.list_assets()
        overrides = mgr.get_asset_overrides()
        _json_response(self, {"ok": True, "assets": assets, "overrides": overrides})

    # ── AI Art generation endpoints (Phase 3C) ────────────────────

    def _handle_ai_art_building(self) -> None:
        """POST /api/ai-art/building — generate building sprite."""
        payload = _read_json(self)
        building_type = payload.get("building_type", "home")
        biome = payload.get("biome", "town_center")
        name = payload.get("name", "")
        width = int(payload.get("width", 64))
        height = int(payload.get("height", 64))

        gen = _get_ai_art()
        if not gen.available:
            _json_response(self, {"ok": False, "error": "Image generation API not configured"}, status=503)
            return

        img_data = gen.generate_building_sprite(building_type, biome, name, width, height)
        if img_data is None:
            _json_response(self, {"ok": False, "error": "Generation failed"}, status=500)
            return

        # Auto-save to custom assets
        mgr = _get_asset_mgr()
        asset = mgr.upload_asset(
            name=f"AI {building_type} sprite",
            asset_type="building_sprite",
            file_data=img_data,
            filename=f"ai_{building_type}.png",
            target=f"building:{building_type}",
        )
        result = {"ok": True}
        if asset:
            result["asset"] = asset.to_dict()
        _json_response(self, result)

    def _handle_ai_art_background(self) -> None:
        """POST /api/ai-art/background — generate region background."""
        payload = _read_json(self)
        biome = payload.get("biome", "town_center")
        season = payload.get("season", "spring")
        description = payload.get("description", "")
        width = int(payload.get("width", 640))
        height = int(payload.get("height", 512))

        gen = _get_ai_art()
        if not gen.available:
            _json_response(self, {"ok": False, "error": "Image generation API not configured"}, status=503)
            return

        img_data = gen.generate_background(biome, season, description, width, height)
        if img_data is None:
            _json_response(self, {"ok": False, "error": "Generation failed"}, status=500)
            return

        mgr = _get_asset_mgr()
        asset = mgr.upload_asset(
            name=f"AI {biome} background",
            asset_type="background",
            file_data=img_data,
            filename=f"ai_bg_{biome}.png",
            target=f"background:{biome}",
        )
        result = {"ok": True}
        if asset:
            result["asset"] = asset.to_dict()
        _json_response(self, result)

    def _serve_static(self, path: str) -> bool:
        base = Path(__file__).resolve().parent.parent / "visualization-ui" / "dist"
        if not base.exists():
            return False

        rel = path.lstrip("/") or "index.html"
        target = (base / rel).resolve()
        if not str(target).startswith(str(base)):
            return False
        if target.is_dir():
            target = target / "index.html"
        if not target.exists():
            target = base / "index.html"
        try:
            data = target.read_bytes()
        except OSError:
            return False

        if target.suffix == ".html":
            ctype = "text/html; charset=utf-8"
        elif target.suffix == ".js":
            ctype = "application/javascript; charset=utf-8"
        elif target.suffix == ".css":
            ctype = "text/css; charset=utf-8"
        elif target.suffix in {".svg", ".png"}:
            ctype = "image/svg+xml" if target.suffix == ".svg" else "image/png"
        else:
            ctype = "application/octet-stream"

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=3600")
        _cors(self)
        self.end_headers()
        self.wfile.write(data)
        return True

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        _cors(self)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            _json_response(self, {"ok": True})
            return
        if parsed.path == "/api/state":
            self._handle_state()
            return
        if parsed.path == "/api/evolution":
            self._handle_evolution()
            return
        if parsed.path == "/api/chunks":
            params = parse_qs(parsed.query)
            self._handle_chunks(params)
            return
        if parsed.path == "/api/agent":
            params = parse_qs(parsed.query)
            name = params.get("name", [""])[0]
            if not name:
                _json_response(self, {"ok": False, "error": "name required"}, status=400)
                return
            self._handle_agent(name)
            return
        if parsed.path == "/api/location":
            params = parse_qs(parsed.query)
            self._handle_location_detail(params)
            return
        if parsed.path == "/api/map/templates":
            self._handle_map_templates()
            return
        if parsed.path == "/api/worldmap":
            self._handle_worldmap()
            return
        if parsed.path == "/api/assets":
            self._handle_assets_list()
            return

        if self._serve_static(parsed.path):
            return

        _json_response(self, {"ok": False, "error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/step":
            self._handle_step()
            return
        if self.path == "/api/reset":
            self._handle_reset()
            return
        if self.path == "/api/probe":
            self._handle_probe()
            return
        if self.path == "/api/ablation":
            self._handle_ablation()
            return
        if self.path == "/api/action_mode":
            self._handle_action_mode()
            return
        if self.path == "/api/chat":
            self._handle_chat()
            return
        if self.path == "/api/agent/control":
            self._handle_agent_control()
            return
        if self.path == "/api/agent/action":
            self._handle_agent_action()
            return
        if self.path == "/api/agent/interact":
            self._handle_agent_interact()
            return
        if self.path == "/api/agent/move":
            self._handle_agent_move()
            return
        if self.path == "/api/agent/explore":
            self._handle_agent_explore()
            return
        if self.path == "/api/evolve":
            self._handle_force_evolve()
            return
        if self.path == "/api/evolve/settings":
            self._handle_evolve_settings()
            return
        if self.path == "/api/map/generate":
            self._handle_map_generate()
            return
        if self.path == "/api/map/template":
            self._handle_map_template()
            return
        if self.path == "/api/worldmap/teleport":
            self._handle_worldmap_teleport()
            return
        if self.path == "/api/assets/upload":
            self._handle_assets_upload()
            return
        if self.path.startswith("/api/assets/") and len(self.path.split("/")) == 4:
            # POST /api/assets/{id} for delete (since some clients can't DELETE)
            asset_id = self.path.split("/")[-1]
            if asset_id != "upload":
                self._handle_assets_delete(asset_id)
                return
        if self.path == "/api/ai-art/building":
            self._handle_ai_art_building()
            return
        if self.path == "/api/ai-art/background":
            self._handle_ai_art_background()
            return
        _json_response(self, {"ok": False, "error": "not found"}, status=404)

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/assets/"):
            asset_id = parsed.path.split("/")[-1]
            self._handle_assets_delete(asset_id)
            return
        _json_response(self, {"ok": False, "error": "not found"}, status=404)


def main(host: str = "127.0.0.1", port: int = 7861) -> None:
    # Ensure .env is loaded before the first TownConfig is created
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    server = ThreadingHTTPServer((host, port), TownRequestHandler)
    print(f"AURA Town API listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
