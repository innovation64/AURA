import React, { useCallback, useEffect, useRef, useState } from "react";
import useTileLoader from "../hooks/useTileLoader.js";
import {
  CANVAS_W, CANVAS_H, DISPLAY_SIZE, VIEWPORT_W, VIEWPORT_H,
  GRID_W, GRID_H, setGridSize,
} from "../lib/tileConfig.js";
import { renderMap, hitTestAgent, hitTestCell, renderFogOfWar } from "../lib/mapRenderer.js";
import { renderDayNightOverlay, WeatherSystem, AmbientSystem } from "../lib/effects.js";

/**
 * Pixel-art town map rendered on a <canvas> element with camera support.
 * Now features continuous 30fps animation loop with weather, day/night, and ambient effects.
 *
 * Props:
 *   state       — environment state { locations, agents, grid_w, grid_h, world_properties, ... }
 *   activeUser  — currently selected agent name
 *   onSelectAgent(name) — callback when an agent is clicked
 */
export default function TownMap({ state, activeUser, onSelectAgent, controlMode, onControlMove, onDpadMove, onLocationClick }) {
  const canvasRef = useRef(null);
  const { loaded, image } = useTileLoader();
  const [tooltip, setTooltip] = useState(null);
  const prevAgents = useRef(null);
  const animRef = useRef(null);   // animation frame positions
  const rafRef = useRef(null);    // requestAnimationFrame id
  const animStartRef = useRef(0);
  const stateRef = useRef(state);
  stateRef.current = state;

  // Effect systems
  const weatherSystemRef = useRef(new WeatherSystem(200));
  const ambientSystemRef = useRef(new AmbientSystem(30));

  // Camera state: top-left corner of viewport in world grid coords
  const [camera, setCamera] = useState({ cx: 14, cy: 16 });
  const cameraRef = useRef(camera);
  cameraRef.current = camera;

  // Walk animation frame (toggles 0/1 during movement)
  const walkFrameRef = useRef(0);
  // Water animation frame (cycles 0-3)
  const waterFrameRef = useRef(0);

  // Sync grid size from server state
  useEffect(() => {
    if (state?.grid_width && state?.grid_height) {
      setGridSize(state.grid_width, state.grid_height);
    }
  }, [state?.grid_width, state?.grid_height]);

  // Sync weather/season from state to effect systems
  useEffect(() => {
    if (!state?.world_properties) return;
    const wp = state.world_properties;
    weatherSystemRef.current.setWeather(wp.weather || "clear");
    ambientSystemRef.current.update(state.hour || 12, wp.season || "spring");
  }, [state?.world_properties?.weather, state?.world_properties?.season, state?.hour]);

  // Clamp camera to world bounds (now uses dynamic GRID_W/H)
  const clampCamera = useCallback((cx, cy) => {
    const maxCx = Math.max(0, GRID_W - VIEWPORT_W);
    const maxCy = Math.max(0, GRID_H - VIEWPORT_H);
    return {
      cx: Math.max(0, Math.min(cx, maxCx)),
      cy: Math.max(0, Math.min(cy, maxCy)),
    };
  }, []);

  // Smoothly scroll camera to center on a world position
  const scrollTo = useCallback((worldX, worldY) => {
    const target = clampCamera(
      worldX - VIEWPORT_W / 2,
      worldY - VIEWPORT_H / 2,
    );
    // Animate camera over 400ms
    const start = { ...cameraRef.current };
    const t0 = performance.now();
    const duration = 400;

    const tick = (now) => {
      const t = Math.min((now - t0) / duration, 1);
      const ease = t * (2 - t);
      const cx = start.cx + (target.cx - start.cx) * ease;
      const cy = start.cy + (target.cy - start.cy) * ease;
      setCamera({ cx, cy });
      if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [clampCamera]);

  // When activeUser changes, scroll camera to that agent
  useEffect(() => {
    if (!activeUser || !state?.agents) return;
    const agent = state.agents.find((a) => a.name === activeUser);
    if (agent) scrollTo(agent.x, agent.y);
  }, [activeUser, state?.agents, scrollTo]);

  // Keyboard: WASD / arrow keys — in control mode moves agent, otherwise pans camera
  useEffect(() => {
    const handleKey = (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      const dirMap = {
        w: "north", ArrowUp: "north",
        s: "south", ArrowDown: "south",
        a: "west", ArrowLeft: "west",
        d: "east", ArrowRight: "east",
      };
      const dir = dirMap[e.key];
      if (!dir) return;
      e.preventDefault();

      if (controlMode && onDpadMove) {
        onDpadMove(dir);
      } else {
        const speed = 2;
        let { cx, cy } = cameraRef.current;
        switch (dir) {
          case "north": cy -= speed; break;
          case "south": cy += speed; break;
          case "west":  cx -= speed; break;
          case "east":  cx += speed; break;
        }
        setCamera(clampCamera(cx, cy));
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [clampCamera, controlMode, onDpadMove]);

  // Mouse drag to pan camera
  const dragRef = useRef(null);

  const handleMouseDown = useCallback((e) => {
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      camCx: cameraRef.current.cx,
      camCy: cameraRef.current.cy,
    };
  }, []);

  const handleDragMove = useCallback((e) => {
    if (!dragRef.current) return;
    const dx = e.clientX - dragRef.current.startX;
    const dy = e.clientY - dragRef.current.startY;
    dragDistRef.current = Math.max(dragDistRef.current, Math.abs(dx) + Math.abs(dy));
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const pixelsPerCell = rect.width / VIEWPORT_W;
    const newCx = dragRef.current.camCx - dx / pixelsPerCell;
    const newCy = dragRef.current.camCy - dy / pixelsPerCell;
    setCamera(clampCamera(newCx, newCy));
  }, [clampCamera]);

  const handleMouseUp = useCallback(() => {
    dragRef.current = null;
  }, []);

  // Attach drag listeners to window so dragging outside canvas still works
  useEffect(() => {
    window.addEventListener("mousemove", handleDragMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleDragMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [handleDragMove, handleMouseUp]);

  // Build animation positions map for agent movement lerp
  const getAnimPositions = useCallback(() => {
    return animRef.current || {};
  }, []);

  // Kick off 300ms movement animation when agents change position
  useEffect(() => {
    if (!state?.agents) return;
    const prev = prevAgents.current;
    const curr = state.agents;

    if (prev && prev.length === curr.length) {
      const hasMovement = curr.some((a) => {
        const p = prev.find((pa) => pa.name === a.name);
        return p && (p.x !== a.x || p.y !== a.y);
      });

      if (hasMovement) {
        const startPositions = {};
        for (const a of prev) {
          startPositions[a.name] = { x: a.x, y: a.y };
        }
        const endPositions = {};
        for (const a of curr) {
          endPositions[a.name] = { x: a.x, y: a.y };
        }

        animStartRef.current = performance.now();
        const duration = 300;

        const animate = (now) => {
          const t = Math.min((now - animStartRef.current) / duration, 1);
          const ease = t * (2 - t); // ease-out quad
          const positions = {};
          for (const a of curr) {
            const s = startPositions[a.name] || endPositions[a.name];
            const e = endPositions[a.name];
            positions[a.name] = {
              x: s.x + (e.x - s.x) * ease,
              y: s.y + (e.y - s.y) * ease,
            };
          }
          animRef.current = positions;

          // Toggle walk frame every ~150ms
          walkFrameRef.current = Math.floor((now - animStartRef.current) / 150) % 2;

          if (t >= 1) {
            animRef.current = null;
            walkFrameRef.current = 0;
          }
        };

        // Use a short interval to update animation positions
        const id = setInterval(() => animate(performance.now()), 16);
        setTimeout(() => clearInterval(id), duration + 20);
        prevAgents.current = curr;
        return;
      }
    }

    prevAgents.current = curr;
    animRef.current = null;
  }, [state?.agents]);

  // Continuous 30fps animation loop for weather, day/night, ambient effects
  useEffect(() => {
    if (!loaded) return;

    let lastWaterFrame = 0;
    let running = true;

    const loop = (now) => {
      if (!running) return;

      const ctx = canvasRef.current?.getContext("2d");
      if (!ctx) { rafRef.current = requestAnimationFrame(loop); return; }

      const currentState = stateRef.current;
      if (!currentState) { rafRef.current = requestAnimationFrame(loop); return; }

      // Update water animation frame every 500ms
      const newWaterFrame = Math.floor(now / 500) % 4;
      waterFrameRef.current = newWaterFrame;

      // Render map layers
      renderMap(
        ctx, image, currentState, activeUser,
        animRef.current, cameraRef.current,
        walkFrameRef.current, waterFrameRef.current, now
      );

      // Fog of war (after map, before effects)
      const cam = cameraRef.current;
      if (currentState.explored_chunks) {
        renderFogOfWar(ctx, cam.cx, cam.cy, currentState.explored_chunks);
      }

      // Weather particles
      weatherSystemRef.current.render(ctx, CANVAS_W, CANVAS_H);

      // Ambient entities (butterflies, birds, petals, etc.)
      ambientSystemRef.current.render(ctx, CANVAS_W, CANVAS_H);

      // Day/night overlay
      const hour = currentState.hour || 12;
      // Collect building rects for night window glow
      let buildingRects = null;
      if (hour >= 20 || hour < 6) {
        const cam = cameraRef.current;
        buildingRects = (currentState.locations || [])
          .filter(l => l.type !== "park" && l.type !== "square")
          .map(l => ({
            x: (l.x - cam.cx) * DISPLAY_SIZE,
            y: (l.y - cam.cy) * DISPLAY_SIZE,
            w: l.width * DISPLAY_SIZE,
            h: l.height * DISPLAY_SIZE,
          }))
          .filter(r => r.x + r.w > 0 && r.x < CANVAS_W && r.y + r.h > 0 && r.y < CANVAS_H);
      }
      renderDayNightOverlay(ctx, hour, CANVAS_W, CANVAS_H, buildingRects);

      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);

    return () => {
      running = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [loaded, image, activeUser]);

  // Track drag distance to distinguish click vs drag
  const dragDistRef = useRef(0);

  const handleCanvasMouseDown = useCallback((e) => {
    dragDistRef.current = 0;
    handleMouseDown(e);
  }, [handleMouseDown]);

  // Canvas click handler — only fire if not dragging
  const handleClick = useCallback(
    (e) => {
      // If user dragged more than 4px, treat as pan not click
      if (dragDistRef.current > 4) return;
      if (!state) return;
      const rect = e.target.getBoundingClientRect();
      const scaleX = CANVAS_W / rect.width;
      const scaleY = CANVAS_H / rect.height;
      const cx = (e.clientX - rect.left) * scaleX;
      const cy = (e.clientY - rect.top) * scaleY;

      // In control mode, clicking a location sends a move command
      if (controlMode) {
        const hit = hitTestCell(cx, cy, state, animRef.current, cameraRef.current);
        if (hit?.agent) {
          onSelectAgent?.(hit.agent.name);
        } else if (hit?.location && onControlMove) {
          onControlMove(hit.location.name);
        }
        return;
      }

      const hit = hitTestCell(cx, cy, state, animRef.current, cameraRef.current);
      if (hit?.agent) {
        onSelectAgent?.(hit.agent.name);
      } else if (hit?.location && onLocationClick) {
        onLocationClick(hit.location.name);
      }
    },
    [state, onSelectAgent, controlMode, onControlMove, onLocationClick]
  );

  // Mouse move for tooltip
  const handleMouseMove = useCallback(
    (e) => {
      if (!state) return;
      const rect = e.target.getBoundingClientRect();
      const scaleX = CANVAS_W / rect.width;
      const scaleY = CANVAS_H / rect.height;
      const cx = (e.clientX - rect.left) * scaleX;
      const cy = (e.clientY - rect.top) * scaleY;
      const hit = hitTestCell(cx, cy, state, animRef.current, cameraRef.current);
      if (hit?.agent) {
        setTooltip({
          x: e.clientX - rect.left,
          y: e.clientY - rect.top,
          text: `${hit.agent.emoji} ${hit.agent.name}: ${hit.agent.action}`,
        });
      } else if (hit?.location) {
        const locText = controlMode
          ? `${hit.location.emoji} ${hit.location.name} \u2014 Click to move here`
          : `${hit.location.emoji} ${hit.location.name} (${hit.location.type})`;
        setTooltip({
          x: e.clientX - rect.left,
          y: e.clientY - rect.top,
          text: locText,
        });
      } else {
        setTooltip(null);
      }
    },
    [state]
  );

  const handleMouseLeave = useCallback(() => setTooltip(null), []);

  if (!loaded) {
    return (
      <div className="map-view">
        <div className="map-loading">Loading pixel art tileset...</div>
      </div>
    );
  }

  return (
    <div className="map-view">
      <div className="map-container">
        <div style={{ position: "relative", display: "inline-block" }}>
          <canvas
            ref={canvasRef}
            width={CANVAS_W}
            height={CANVAS_H}
            className={`town-canvas${controlMode ? " control-cursor" : ""}`}
            onMouseDown={handleCanvasMouseDown}
            onClick={handleClick}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
          />
          {tooltip && (
            <div
              className="canvas-tooltip"
              style={{ left: tooltip.x + 12, top: tooltip.y - 8 }}
            >
              {tooltip.text}
            </div>
          )}
          {controlMode && onDpadMove && (
            <div className="dpad-overlay">
              <div className="dpad-cell" />
              <button className="dpad-btn" onClick={() => onDpadMove("north")} title="North">{"\u25B2"}</button>
              <div className="dpad-cell" />
              <button className="dpad-btn" onClick={() => onDpadMove("west")} title="West">{"\u25C4"}</button>
              <div className="dpad-center" />
              <button className="dpad-btn" onClick={() => onDpadMove("east")} title="East">{"\u25BA"}</button>
              <div className="dpad-cell" />
              <button className="dpad-btn" onClick={() => onDpadMove("south")} title="South">{"\u25BC"}</button>
              <div className="dpad-cell" />
            </div>
          )}
        </div>
        <div className="map-legend">
          {state?.locations?.map((loc) => (
            <span className="legend-chip" key={loc.name}>
              {loc.emoji} {loc.name}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
