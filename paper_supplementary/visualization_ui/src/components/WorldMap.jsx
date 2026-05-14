import React, { useCallback, useEffect, useRef, useState } from "react";
import { renderWorldMap, hitTestRegion } from "../lib/worldMapRenderer.js";
import { CANVAS_W, CANVAS_H } from "../lib/tileConfig.js";

/**
 * RPG-style world map component showing regions as clickable nodes.
 *
 * Props:
 *   worldMapData  — { regions, agent_positions, connections, world_bounds }
 *   onSelectRegion(region) — callback when a region is clicked
 *   onBack() — callback for "Back to Region" button
 *   activeRegion — currently active region ID
 */
export default function WorldMap({ worldMapData, onSelectRegion, onBack, activeRegion }) {
  const canvasRef = useRef(null);
  const rafRef = useRef(null);
  const [hoveredRegion, setHoveredRegion] = useState(null);
  const [tooltip, setTooltip] = useState(null);

  // Animation loop for pulsing effects
  useEffect(() => {
    let running = true;

    const loop = () => {
      if (!running) return;
      const ctx = canvasRef.current?.getContext("2d");
      if (!ctx || !worldMapData) {
        rafRef.current = requestAnimationFrame(loop);
        return;
      }

      renderWorldMap(ctx, null, worldMapData, CANVAS_W, CANVAS_H, {
        hoveredRegion,
        selectedRegion: activeRegion,
      });

      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);
    return () => {
      running = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [worldMapData, hoveredRegion, activeRegion]);

  const handleClick = useCallback((e) => {
    if (!worldMapData) return;
    const rect = e.target.getBoundingClientRect();
    const scaleX = CANVAS_W / rect.width;
    const scaleY = CANVAS_H / rect.height;
    const cx = (e.clientX - rect.left) * scaleX;
    const cy = (e.clientY - rect.top) * scaleY;

    const region = hitTestRegion(cx, cy, worldMapData, CANVAS_W, CANVAS_H);
    if (region && region.discovered) {
      onSelectRegion?.(region);
    }
  }, [worldMapData, onSelectRegion]);

  const handleMouseMove = useCallback((e) => {
    if (!worldMapData) return;
    const rect = e.target.getBoundingClientRect();
    const scaleX = CANVAS_W / rect.width;
    const scaleY = CANVAS_H / rect.height;
    const cx = (e.clientX - rect.left) * scaleX;
    const cy = (e.clientY - rect.top) * scaleY;

    const region = hitTestRegion(cx, cy, worldMapData, CANVAS_W, CANVAS_H);
    if (region) {
      setHoveredRegion(region.id);
      setTooltip({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        text: `${region.emoji} ${region.name} (${region.biome}) — ${region.location_count} locations, ${region.population} agents`,
      });
    } else {
      setHoveredRegion(null);
      setTooltip(null);
    }
  }, [worldMapData]);

  const handleMouseLeave = useCallback(() => {
    setHoveredRegion(null);
    setTooltip(null);
  }, []);

  return (
    <div className="world-map-view">
      <div className="world-map-header">
        <h3>{"\uD83D\uDDFA\uFE0F"} World Map</h3>
        {onBack && (
          <button className="btn btn-sm" onClick={onBack}>
            {"\u2190"} Back to Region
          </button>
        )}
      </div>
      <div className="world-map-container" style={{ position: "relative", display: "inline-block" }}>
        <canvas
          ref={canvasRef}
          width={CANVAS_W}
          height={CANVAS_H}
          className="town-canvas world-canvas"
          onClick={handleClick}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          style={{ cursor: hoveredRegion ? "pointer" : "default" }}
        />
        {tooltip && (
          <div
            className="canvas-tooltip"
            style={{ left: tooltip.x + 12, top: tooltip.y - 8 }}
          >
            {tooltip.text}
          </div>
        )}
      </div>
      {worldMapData?.regions && (
        <div className="world-map-legend">
          {worldMapData.regions.map(r => (
            <span
              className={`legend-chip ${activeRegion === r.id ? "active" : ""}`}
              key={r.id}
              onClick={() => r.discovered && onSelectRegion?.(r)}
              style={{ cursor: r.discovered ? "pointer" : "default" }}
            >
              {r.emoji} {r.name}
              <span className="legend-count">{r.location_count}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
