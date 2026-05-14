/**
 * World map renderer for AURA Town — RPG-style overview map.
 * Renders regions as clickable nodes, agent positions, and connections.
 */

const REGION_COLORS = {
  town_center: "#e8d44d",
  farmland: "#7cb342",
  riverside: "#42a5f5",
  forest: "#2e7d32",
  mountain: "#78909c",
};

const REGION_BG = {
  town_center: "rgba(232,212,77,0.15)",
  farmland: "rgba(124,179,66,0.15)",
  riverside: "rgba(66,165,245,0.15)",
  forest: "rgba(46,125,50,0.15)",
  mountain: "rgba(120,144,156,0.15)",
};

/**
 * Render the world map overview.
 * @param {CanvasRenderingContext2D} ctx
 * @param {Image|null} img - tilemap image (unused for world map, kept for API compat)
 * @param {object} worldMapData - { regions, agent_positions, connections, world_bounds }
 * @param {number} canvasW
 * @param {number} canvasH
 * @param {object} [options] - { hoveredRegion, selectedRegion }
 */
export function renderWorldMap(ctx, img, worldMapData, canvasW, canvasH, options = {}) {
  if (!worldMapData) return;

  const { regions = [], agent_positions = [], connections = [], world_bounds = {} } = worldMapData;
  const { hoveredRegion, selectedRegion } = options;

  ctx.clearRect(0, 0, canvasW, canvasH);

  // Background
  ctx.fillStyle = "#0a0e1a";
  ctx.fillRect(0, 0, canvasW, canvasH);

  // Grid pattern
  ctx.strokeStyle = "rgba(255,255,255,0.04)";
  ctx.lineWidth = 1;
  for (let x = 0; x < canvasW; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvasH);
    ctx.stroke();
  }
  for (let y = 0; y < canvasH; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(canvasW, y);
    ctx.stroke();
  }

  if (regions.length === 0) {
    ctx.fillStyle = "rgba(255,255,255,0.3)";
    ctx.font = "16px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("No regions discovered yet", canvasW / 2, canvasH / 2);
    return;
  }

  // Compute scale: map world coords to canvas
  const bounds = computeBounds(regions, world_bounds);
  const padding = 60;
  const scaleX = (canvasW - padding * 2) / Math.max(bounds.width, 1);
  const scaleY = (canvasH - padding * 2) / Math.max(bounds.height, 1);
  const scale = Math.min(scaleX, scaleY, 4);

  const toScreen = (wx, wy) => ({
    x: padding + (wx - bounds.minX) * scale,
    y: padding + (wy - bounds.minY) * scale,
  });

  // Draw connections
  ctx.lineWidth = 2;
  for (const conn of connections) {
    const regionA = regions.find(r => r.id === conn.from);
    const regionB = regions.find(r => r.id === conn.to);
    if (!regionA || !regionB) continue;

    const a = toScreen(regionA.world_x + regionA.width / 2, regionA.world_y + regionA.height / 2);
    const b = toScreen(regionB.world_x + regionB.width / 2, regionB.world_y + regionB.height / 2);

    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }

  // Draw region areas
  for (const region of regions) {
    const pos = toScreen(region.world_x, region.world_y);
    const w = region.width * scale;
    const h = region.height * scale;
    const biome = region.biome || "town_center";
    const isHovered = hoveredRegion === region.id;
    const isSelected = selectedRegion === region.id;

    // Region area fill
    ctx.fillStyle = REGION_BG[biome] || "rgba(128,128,128,0.1)";
    ctx.beginPath();
    ctx.roundRect(pos.x, pos.y, w, h, 8);
    ctx.fill();

    // Border
    const borderColor = REGION_COLORS[biome] || "#888";
    ctx.strokeStyle = isSelected
      ? borderColor
      : isHovered
        ? borderColor
        : `${borderColor}66`;
    ctx.lineWidth = isSelected ? 3 : isHovered ? 2.5 : 1.5;
    ctx.stroke();

    // Hover/selected glow
    if (isHovered || isSelected) {
      ctx.shadowColor = borderColor;
      ctx.shadowBlur = 12;
      ctx.strokeStyle = borderColor;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // Region center
    const cx = pos.x + w / 2;
    const cy = pos.y + h / 2;

    // Emoji icon
    ctx.font = `${Math.max(16, Math.min(w * 0.3, 32))}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(region.emoji || "\uD83D\uDDFA", cx, cy - 8);

    // Region name
    ctx.font = `bold ${Math.max(9, Math.min(w * 0.08, 14))}px Inter, sans-serif`;
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(region.name, cx, cy + 8);

    // Stats
    const stats = `${region.location_count || 0} loc | ${region.population || 0} pop`;
    ctx.font = `${Math.max(7, Math.min(w * 0.06, 10))}px Inter, sans-serif`;
    ctx.fillStyle = "rgba(255,255,255,0.5)";
    ctx.fillText(stats, cx, cy + 22);

    // Fog for undiscovered
    if (!region.discovered) {
      ctx.fillStyle = "rgba(5,5,15,0.7)";
      ctx.beginPath();
      ctx.roundRect(pos.x, pos.y, w, h, 8);
      ctx.fill();
      ctx.fillStyle = "rgba(255,255,255,0.2)";
      ctx.font = "20px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("?", cx, cy);
    }
  }

  // Draw agent position dots
  for (const agent of agent_positions) {
    const pos = toScreen(agent.x, agent.y);
    const time = Date.now() / 1000;
    const pulse = 0.5 + 0.5 * Math.sin(time * 3);

    // Outer pulse ring
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, 6 + pulse * 3, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(111, 227, 255, ${0.2 + pulse * 0.15})`;
    ctx.fill();

    // Inner dot
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = "#6fe3ff";
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1;
    ctx.stroke();

    // Agent name
    ctx.font = "bold 9px Inter, sans-serif";
    ctx.fillStyle = "#6fe3ff";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(agent.name, pos.x, pos.y + 8);
  }

  // Title
  ctx.font = "bold 14px Inter, sans-serif";
  ctx.fillStyle = "rgba(255,255,255,0.6)";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText("World Map", 12, 12);
}

/**
 * Hit-test: find which region was clicked.
 * @returns {object|null} region data if clicked, null otherwise
 */
export function hitTestRegion(canvasX, canvasY, worldMapData, canvasW, canvasH) {
  if (!worldMapData?.regions?.length) return null;

  const { regions, world_bounds } = worldMapData;
  const bounds = computeBounds(regions, world_bounds);
  const padding = 60;
  const scaleX = (canvasW - padding * 2) / Math.max(bounds.width, 1);
  const scaleY = (canvasH - padding * 2) / Math.max(bounds.height, 1);
  const scale = Math.min(scaleX, scaleY, 4);

  for (const region of regions) {
    const x = padding + (region.world_x - bounds.minX) * scale;
    const y = padding + (region.world_y - bounds.minY) * scale;
    const w = region.width * scale;
    const h = region.height * scale;

    if (canvasX >= x && canvasX <= x + w && canvasY >= y && canvasY <= y + h) {
      return region;
    }
  }
  return null;
}

function computeBounds(regions, world_bounds) {
  let minX = world_bounds?.min_x ?? 0;
  let minY = world_bounds?.min_y ?? 0;
  let maxX = world_bounds?.max_x ?? 60;
  let maxY = world_bounds?.max_y ?? 60;

  for (const r of regions) {
    minX = Math.min(minX, r.world_x);
    minY = Math.min(minY, r.world_y);
    maxX = Math.max(maxX, r.world_x + r.width);
    maxY = Math.max(maxY, r.world_y + r.height);
  }

  return {
    minX: minX - 10,
    minY: minY - 10,
    width: maxX - minX + 20,
    height: maxY - minY + 20,
  };
}
