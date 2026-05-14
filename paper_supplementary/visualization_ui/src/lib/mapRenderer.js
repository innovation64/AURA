import {
  TILE_SIZE, SCALE, DISPLAY_SIZE,
  GRID_W, GRID_H, VIEWPORT_W, VIEWPORT_H,
  CANVAS_W, CANVAS_H, tileCoords,
  TILES, ROAD_VARIANTS, BUILDING_PATTERNS,
  PARK_FILL, SQUARE_FILL, DECORATIONS, DECORATION_DENSITY,
  CHARACTER_SPRITES,
  WATER_TILES, SEASON_DECORATIONS, BIOME_TERRAIN, CHUNK_SIZE,
} from "./tileConfig.js";
import { generateRoads, roadNeighborMask } from "./pathfinder.js";

/* -- Custom asset override cache ---------------------- */
const _assetOverrideCache = new Map(); // target -> { img, loading }

function getOverrideImage(url) {
  if (!url) return null;
  const cached = _assetOverrideCache.get(url);
  if (cached) return cached.loaded ? cached.img : null;
  const entry = { img: new Image(), loaded: false };
  entry.img.onload = () => { entry.loaded = true; };
  entry.img.onerror = () => { entry.loaded = true; entry.img = null; };
  entry.img.src = url;
  _assetOverrideCache.set(url, entry);
  return null;
}

// Per-type accent color + entrance decoration tile
const BUILDING_ACCENTS = {
  home:     { color: "#e67e22", sign: TILES.item_sign },
  cafe:     { color: "#e74c3c", sign: TILES.item_food },
  bakery:   { color: "#f39c12", sign: TILES.item_food },
  library:  { color: "#3498db", sign: TILES.item_book },
  shop:     { color: "#e74c3c", sign: TILES.item_chest },
  townhall: { color: "#9b59b6", sign: TILES.item_sign },
  pharmacy: { color: "#1abc9c", sign: TILES.item_potion },
  school:   { color: "#2980b9", sign: TILES.item_scroll },
  gallery:  { color: "#8e44ad", sign: TILES.item_gem },
  teahouse: { color: "#27ae60", sign: TILES.item_food },
  temple:   { color: "#c0392b", sign: TILES.item_ring },
};

/* -- helpers ----------------------------------------- */

/** Draw a single tile from the spritesheet onto the canvas. */
function drawTile(ctx, img, tileIndex, dx, dy) {
  if (!img) {
    ctx.fillStyle = "#3a7d44";
    ctx.fillRect(dx, dy, DISPLAY_SIZE, DISPLAY_SIZE);
    return;
  }
  const { sx, sy } = tileCoords(tileIndex);
  ctx.drawImage(img, sx, sy, TILE_SIZE, TILE_SIZE, dx, dy, DISPLAY_SIZE, DISPLAY_SIZE);
}

/** Seeded pseudo-random for deterministic decoration scatter. */
function seededRand(x, y) {
  let h = (x * 374761393 + y * 668265263 + 1013904223) | 0;
  h = ((h ^ (h >> 13)) * 1274126177) | 0;
  return ((h ^ (h >> 16)) >>> 0) / 4294967296;
}

/** Check if a world grid cell (gx,gy) is within the viewport. */
function inView(gx, gy, cx, cy) {
  return gx >= cx - 1 && gx <= cx + VIEWPORT_W && gy >= cy - 1 && gy <= cy + VIEWPORT_H;
}

/* -- location lookup --------------------------------- */

/** Build a Map of "x,y" -> loc for only the visible viewport area. */
function buildLocationMap(locations, cx, cy) {
  const map = new Map();
  const x0 = Math.floor(cx) - 1;
  const y0 = Math.floor(cy) - 1;
  const x1 = Math.ceil(cx + VIEWPORT_W) + 1;
  const y1 = Math.ceil(cy + VIEWPORT_H) + 1;
  for (const loc of locations) {
    // Skip locations completely outside viewport
    if (loc.x + loc.width < x0 || loc.x > x1) continue;
    if (loc.y + loc.height < y0 || loc.y > y1) continue;
    for (let dy = 0; dy < loc.height; dy++) {
      for (let dx = 0; dx < loc.width; dx++) {
        const gx = loc.x + dx, gy = loc.y + dy;
        if (gx >= x0 && gx <= x1 && gy >= y0 && gy <= y1) {
          map.set(`${gx},${gy}`, loc);
        }
      }
    }
  }
  return map;
}

/** Get the biome for a world coordinate using chunk biomes from state. */
function getBiomeAt(x, y, chunkBiomes) {
  if (!chunkBiomes) return "town_center";
  const cx = Math.floor(x / CHUNK_SIZE);
  const cy = Math.floor(y / CHUNK_SIZE);
  return chunkBiomes[`${cx},${cy}`] || "town_center";
}

/* -- Layer 1: Terrain (grass/biome) -- viewport culled  */

function renderTerrain(ctx, img, cx, cy, chunkBiomes, season) {
  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const x1 = Math.ceil(cx + VIEWPORT_W + 1);
  const y1 = Math.ceil(cy + VIEWPORT_H + 1);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const biome = getBiomeAt(x, y, chunkBiomes);
      const palette = BIOME_TERRAIN[biome] || BIOME_TERRAIN.town_center;
      const variant = palette[Math.floor(seededRand(x, y) * palette.length)];
      const dx = (x - cx) * DISPLAY_SIZE;
      const dy = (y - cy) * DISPLAY_SIZE;
      drawTile(ctx, img, variant, dx, dy);

      // Season tinting per-tile overlay
      if (season === "autumn") {
        ctx.fillStyle = "rgba(139,90,43,0.15)";
        ctx.fillRect(dx, dy, DISPLAY_SIZE, DISPLAY_SIZE);
      } else if (season === "winter") {
        ctx.fillStyle = "rgba(200,220,240,0.2)";
        ctx.fillRect(dx, dy, DISPLAY_SIZE, DISPLAY_SIZE);
      }
    }
  }
}

/* -- Layer 1b: Water (riverside biome) -- animated --- */

function renderWater(ctx, img, cx, cy, chunkBiomes, animFrame) {
  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const x1 = Math.ceil(cx + VIEWPORT_W + 1);
  const y1 = Math.ceil(cy + VIEWPORT_H + 1);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const biome = getBiomeAt(x, y, chunkBiomes);
      if (biome !== "riverside") continue;
      // Only draw water on some cells (river pattern)
      const r = seededRand(x + 300, y + 300);
      if (r > 0.3) continue; // ~30% of riverside cells are actual water
      const tileIdx = WATER_TILES[(animFrame + Math.floor(r * 4)) % WATER_TILES.length];
      drawTile(ctx, img, tileIdx, (x - cx) * DISPLAY_SIZE, (y - cy) * DISPLAY_SIZE);
    }
  }
}

/* -- Layer 2: Roads -- viewport culled --------------- */

function renderRoads(ctx, img, roadSet, cx, cy) {
  for (const key of roadSet) {
    const [x, y] = key.split(",").map(Number);
    if (!inView(x, y, cx, cy)) continue;
    const mask = roadNeighborMask(x, y, roadSet);
    const tileIdx = ROAD_VARIANTS[mask] ?? TILES.road_cross;
    drawTile(ctx, img, tileIdx, (x - cx) * DISPLAY_SIZE, (y - cy) * DISPLAY_SIZE);
  }
}

/* -- Layer 3: Buildings -- viewport culled ----------- */

function renderBuildings(ctx, img, locations, cx, cy, overrides) {
  for (const loc of locations) {
    // Quick AABB check: skip if entire building is offscreen
    if (loc.x + loc.width < cx - 1 || loc.x > cx + VIEWPORT_W + 1) continue;
    if (loc.y + loc.height < cy - 1 || loc.y > cy + VIEWPORT_H + 1) continue;

    // Check for custom asset override
    if (overrides) {
      const overrideUrl = overrides[`building_sprite:${loc.type}`] || overrides[`building_sprite:${loc.name}`];
      if (overrideUrl) {
        const overrideImg = getOverrideImage(overrideUrl);
        if (overrideImg) {
          const dx = (loc.x - cx) * DISPLAY_SIZE;
          const dy = (loc.y - cy) * DISPLAY_SIZE;
          const dw = loc.width * DISPLAY_SIZE;
          const dh = loc.height * DISPLAY_SIZE;
          ctx.imageSmoothingEnabled = false;
          ctx.drawImage(overrideImg, dx, dy, dw, dh);
          continue;
        }
      }
    }

    const pattern = BUILDING_PATTERNS[loc.type];
    if (pattern) {
      for (let r = 0; r < pattern.h; r++) {
        for (let c = 0; c < pattern.w; c++) {
          const gx = loc.x + c, gy = loc.y + r;
          drawTile(ctx, img, pattern.tiles[r][c], (gx - cx) * DISPLAY_SIZE, (gy - cy) * DISPLAY_SIZE);
        }
      }
      // Fill remaining cells of larger buildings with walls
      for (let dy = 0; dy < loc.height; dy++) {
        for (let dx = 0; dx < loc.width; dx++) {
          if (dy < pattern.h && dx < pattern.w) continue;
          const gx = loc.x + dx, gy = loc.y + dy;
          const wallTile = dy === 0 ? TILES.roof_or_tl : TILES.wall_brown_1;
          drawTile(ctx, img, wallTile, (gx - cx) * DISPLAY_SIZE, (gy - cy) * DISPLAY_SIZE);
        }
      }
    } else if (loc.type === "park") {
      for (let dy = 0; dy < loc.height; dy++) {
        for (let dx = 0; dx < loc.width; dx++) {
          const gx = loc.x + dx, gy = loc.y + dy;
          const r = seededRand(gx + 100, gy + 100);
          const tile = PARK_FILL[Math.floor(r * PARK_FILL.length)];
          drawTile(ctx, img, tile, (gx - cx) * DISPLAY_SIZE, (gy - cy) * DISPLAY_SIZE);
        }
      }
    } else if (loc.type === "square") {
      for (let dy = 0; dy < loc.height; dy++) {
        for (let dx = 0; dx < loc.width; dx++) {
          const gx = loc.x + dx, gy = loc.y + dy;
          const r = seededRand(gx + 200, gy + 200);
          const tile = SQUARE_FILL[Math.floor(r * SQUARE_FILL.length)];
          drawTile(ctx, img, tile, (gx - cx) * DISPLAY_SIZE, (gy - cy) * DISPLAY_SIZE);
        }
      }
    } else {
      const fallback = BUILDING_PATTERNS.home;
      for (let r = 0; r < Math.min(fallback.h, loc.height); r++) {
        for (let c = 0; c < Math.min(fallback.w, loc.width); c++) {
          const gx = loc.x + c, gy = loc.y + r;
          drawTile(ctx, img, fallback.tiles[r][c], (gx - cx) * DISPLAY_SIZE, (gy - cy) * DISPLAY_SIZE);
        }
      }
    }
  }
}

/* -- Layer 3b: Building labels + accents ------------- */

function renderBuildingLabels(ctx, img, locations, cx, cy) {
  for (const loc of locations) {
    if (loc.x + loc.width < cx - 1 || loc.x > cx + VIEWPORT_W + 1) continue;
    if (loc.y + loc.height < cy - 2 || loc.y > cy + VIEWPORT_H + 1) continue;

    const accent = BUILDING_ACCENTS[loc.type];
    const bx = (loc.x - cx) * DISPLAY_SIZE;
    const by = (loc.y - cy) * DISPLAY_SIZE;
    const bw = loc.width * DISPLAY_SIZE;
    const bh = loc.height * DISPLAY_SIZE;

    // Colored border around building
    if (accent && loc.type !== "park" && loc.type !== "square") {
      ctx.strokeStyle = accent.color;
      ctx.lineWidth = 2;
      ctx.strokeRect(bx + 1, by + 1, bw - 2, bh - 2);

      // Entrance decoration tile — placed one cell below the door (center-bottom)
      if (accent.sign != null) {
        const doorX = loc.x + Math.floor(loc.width / 2);
        const doorY = loc.y + loc.height; // cell below building
        drawTile(ctx, img, accent.sign, (doorX - cx) * DISPLAY_SIZE, (doorY - cy) * DISPLAY_SIZE);
      }
    }

    // Name label above building
    const labelX = bx + bw / 2;
    const labelY = by - 6;
    const name = loc.name.replace("'s Home", "").replace("'s ", " ");
    ctx.font = `bold ${6 * SCALE}px Inter, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";

    // Background pill
    const textW = ctx.measureText(name).width;
    const pillColor = accent?.color || "#666";
    ctx.fillStyle = "rgba(0,0,0,0.7)";
    ctx.beginPath();
    const r = 4;
    const pw = textW + 12, ph = 7 * SCALE + 4;
    const px = labelX - pw / 2, ppy = labelY - ph + 2;
    ctx.roundRect(px, ppy, pw, ph, r);
    ctx.fill();

    // Colored left accent bar
    ctx.fillStyle = pillColor;
    ctx.fillRect(px, ppy, 3, ph);

    // Text
    ctx.fillStyle = "#fff";
    ctx.fillText(name, labelX, labelY);
  }
}

/* -- Layer 4: Decorations -- viewport culled, seasonal */

function renderDecorations(ctx, img, locMap, roadSet, cx, cy, season) {
  const decoPool = SEASON_DECORATIONS[season] || DECORATIONS;
  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const x1 = Math.ceil(cx + VIEWPORT_W + 1);
  const y1 = Math.ceil(cy + VIEWPORT_H + 1);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      if (locMap.has(`${x},${y}`)) continue;
      if (roadSet.has(`${x},${y}`)) continue;
      const r = seededRand(x + 500, y + 500);
      if (r < DECORATION_DENSITY) {
        const tile = decoPool[Math.floor(seededRand(x + 600, y + 600) * decoPool.length)];
        drawTile(ctx, img, tile, (x - cx) * DISPLAY_SIZE, (y - cy) * DISPLAY_SIZE);
      }
    }
  }
}

/* -- Layer 5: Pixel-art character sprites ------------ */

/**
 * Draw a 16x16-scaled pixel-art character.
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} px  - center X on canvas
 * @param {number} py  - center Y on canvas
 * @param {object} colors - { hair, shirt, skin, pants }
 * @param {number} dir - 0=down, 1=left, 2=right, 3=up
 * @param {number} frame - 0 or 1 for walk animation
 * @param {boolean} isActive
 */
function drawPixelCharacter(ctx, px, py, colors, dir, frame, isActive) {
  const s = SCALE; // pixel scale factor
  // Character is drawn at 16x16 native pixels, scaled by s
  // Origin: center of the cell
  const ox = px - 8 * s;
  const oy = py - 8 * s;

  // Active highlight glow
  if (isActive) {
    ctx.beginPath();
    ctx.arc(px, py, 14 * s, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(111, 227, 255, 0.18)";
    ctx.fill();
    ctx.strokeStyle = "rgba(111, 227, 255, 0.7)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // Shadow
  ctx.fillStyle = "rgba(0,0,0,0.25)";
  ctx.fillRect(ox + 3 * s, oy + 14 * s, 10 * s, 2 * s);

  const { hair, shirt, skin, pants } = colors;

  // --- Hair / Head (rows 1-4) ---
  ctx.fillStyle = hair;
  if (dir === 3) {
    // facing up: hair covers face
    ctx.fillRect(ox + 4 * s, oy + 0 * s, 8 * s, 5 * s);
  } else {
    ctx.fillRect(ox + 4 * s, oy + 0 * s, 8 * s, 2 * s);
    // Face
    ctx.fillStyle = skin;
    ctx.fillRect(ox + 4 * s, oy + 2 * s, 8 * s, 3 * s);
    // Eyes
    ctx.fillStyle = "#1a1a2e";
    if (dir === 0) {
      // facing down
      ctx.fillRect(ox + 5 * s, oy + 3 * s, 2 * s, 1 * s);
      ctx.fillRect(ox + 9 * s, oy + 3 * s, 2 * s, 1 * s);
    } else if (dir === 1) {
      // facing left
      ctx.fillRect(ox + 4 * s, oy + 3 * s, 2 * s, 1 * s);
      ctx.fillRect(ox + 8 * s, oy + 3 * s, 2 * s, 1 * s);
    } else if (dir === 2) {
      // facing right
      ctx.fillRect(ox + 6 * s, oy + 3 * s, 2 * s, 1 * s);
      ctx.fillRect(ox + 10 * s, oy + 3 * s, 2 * s, 1 * s);
    }
  }

  // --- Body / Shirt (rows 5-9) ---
  ctx.fillStyle = shirt;
  ctx.fillRect(ox + 3 * s, oy + 5 * s, 10 * s, 5 * s);

  // Arms
  if (frame === 0) {
    ctx.fillRect(ox + 1 * s, oy + 5 * s, 2 * s, 4 * s);
    ctx.fillRect(ox + 13 * s, oy + 5 * s, 2 * s, 4 * s);
  } else {
    // Walk frame: arms swing
    ctx.fillRect(ox + 1 * s, oy + 6 * s, 2 * s, 3 * s);
    ctx.fillRect(ox + 13 * s, oy + 4 * s, 2 * s, 3 * s);
  }

  // Hands
  ctx.fillStyle = skin;
  if (frame === 0) {
    ctx.fillRect(ox + 1 * s, oy + 9 * s, 2 * s, 1 * s);
    ctx.fillRect(ox + 13 * s, oy + 9 * s, 2 * s, 1 * s);
  } else {
    ctx.fillRect(ox + 1 * s, oy + 9 * s, 2 * s, 1 * s);
    ctx.fillRect(ox + 13 * s, oy + 7 * s, 2 * s, 1 * s);
  }

  // --- Legs / Pants (rows 10-13) ---
  ctx.fillStyle = pants;
  if (frame === 0) {
    // Standing
    ctx.fillRect(ox + 4 * s, oy + 10 * s, 3 * s, 4 * s);
    ctx.fillRect(ox + 9 * s, oy + 10 * s, 3 * s, 4 * s);
  } else {
    // Walk: legs split
    ctx.fillRect(ox + 3 * s, oy + 10 * s, 3 * s, 4 * s);
    ctx.fillRect(ox + 10 * s, oy + 10 * s, 3 * s, 4 * s);
  }

  // Shoes
  ctx.fillStyle = "#2c2c2c";
  if (frame === 0) {
    ctx.fillRect(ox + 4 * s, oy + 14 * s, 3 * s, 1 * s);
    ctx.fillRect(ox + 9 * s, oy + 14 * s, 3 * s, 1 * s);
  } else {
    ctx.fillRect(ox + 2 * s, oy + 14 * s, 4 * s, 1 * s);
    ctx.fillRect(ox + 10 * s, oy + 14 * s, 4 * s, 1 * s);
  }
}

function renderAgents(ctx, agents, activeUser, animPositions, cx, cy, walkFrame, overrides) {
  for (const agent of agents) {
    const pos = animPositions?.[agent.name] || { x: agent.x, y: agent.y };
    // Skip if offscreen
    if (!inView(Math.floor(pos.x), Math.floor(pos.y), cx, cy)) continue;

    const px = (pos.x - cx) * DISPLAY_SIZE + DISPLAY_SIZE / 2;
    const py = (pos.y - cy) * DISPLAY_SIZE + DISPLAY_SIZE / 2;
    const isActive = agent.name === activeUser;

    // Determine facing direction from movement delta
    let dir = 0; // default: down
    if (animPositions?.[agent.name]) {
      const target = { x: agent.x, y: agent.y };
      const cur = animPositions[agent.name];
      const ddx = target.x - cur.x;
      const ddy = target.y - cur.y;
      if (Math.abs(ddx) > Math.abs(ddy)) {
        dir = ddx < 0 ? 1 : 2; // left or right
      } else if (ddy < -0.01) {
        dir = 3; // up
      } else if (ddy > 0.01) {
        dir = 0; // down
      }
    }

    // Walk frame: alternate during animation
    const frame = animPositions?.[agent.name] ? walkFrame : 0;

    // Check for custom character sprite override
    const spriteUrl = overrides?.[`character_sprite:${agent.name}`];
    const spriteImg = spriteUrl ? getOverrideImage(spriteUrl) : null;
    if (spriteImg) {
      // Active highlight
      if (isActive) {
        ctx.beginPath();
        ctx.arc(px, py, 14 * SCALE, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(111, 227, 255, 0.18)";
        ctx.fill();
        ctx.strokeStyle = "rgba(111, 227, 255, 0.7)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(spriteImg, px - DISPLAY_SIZE / 2, py - DISPLAY_SIZE / 2, DISPLAY_SIZE, DISPLAY_SIZE);
    } else {
      const colors = CHARACTER_SPRITES[agent.name] || {
        hair: "#333", shirt: "#888", skin: "#f5c6a0", pants: "#444",
      };
      drawPixelCharacter(ctx, px, py, colors, dir, frame, isActive);
    }

    // Name label below character
    const labelY = py + 12 * SCALE;
    ctx.font = `bold ${7 * SCALE}px Inter, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = isActive ? "#6fe3ff" : "#e2ebf8";
    ctx.strokeStyle = "rgba(0,0,0,0.85)";
    ctx.lineWidth = 3;
    ctx.strokeText(agent.name, px, labelY);
    ctx.fillText(agent.name, px, labelY);
  }
}

/* -- Layer 6: Speech & Thought Bubbles --------------- */

const MOOD_EMOJI = {
  happy: "\u{1F60A}",
  thinking: "\u{1F914}",
  tired: "\u{1F634}",
  excited: "\u2728",
  neutral: "",
};

function renderBubbles(ctx, agents, activeUser, animPositions, cx, cy) {
  for (const agent of agents) {
    const pos = animPositions?.[agent.name] || { x: agent.x, y: agent.y };
    if (!inView(Math.floor(pos.x), Math.floor(pos.y), cx, cy)) continue;

    const px = (pos.x - cx) * DISPLAY_SIZE + DISPLAY_SIZE / 2;
    const py = (pos.y - cy) * DISPLAY_SIZE + DISPLAY_SIZE / 2;

    // Mood emoji (top-right of agent head)
    const moodEmoji = MOOD_EMOJI[agent.mood] || "";
    if (moodEmoji) {
      ctx.font = `${6 * SCALE}px sans-serif`;
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(moodEmoji, px + 8 * SCALE, py - 10 * SCALE);
    }

    // Speech bubble (white rounded rect with triangle tail)
    if (agent.speech_bubble) {
      const text = agent.speech_bubble.length > 40
        ? agent.speech_bubble.slice(0, 40) + "\u2026"
        : agent.speech_bubble;
      const bubbleY = py - 24 * SCALE;
      ctx.font = `${5 * SCALE}px Inter, sans-serif`;
      const tw = ctx.measureText(text).width;
      const bw = tw + 12;
      const bh = 8 * SCALE + 4;
      const bx = px - bw / 2;
      const by = bubbleY - bh;

      // Rounded rect
      ctx.fillStyle = "rgba(255,255,255,0.95)";
      ctx.beginPath();
      ctx.roundRect(bx, by, bw, bh, 4);
      ctx.fill();
      ctx.strokeStyle = "rgba(0,0,0,0.2)";
      ctx.lineWidth = 1;
      ctx.stroke();

      // Triangle tail
      ctx.fillStyle = "rgba(255,255,255,0.95)";
      ctx.beginPath();
      ctx.moveTo(px - 3, by + bh);
      ctx.lineTo(px, by + bh + 4 * SCALE);
      ctx.lineTo(px + 3, by + bh);
      ctx.fill();

      // Text
      ctx.fillStyle = "#1a1a2e";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, px, by + bh / 2);
    }
    // Thought bubble (purple rounded rect with cloud tail)
    else if (agent.thought_bubble) {
      const text = agent.thought_bubble.length > 40
        ? agent.thought_bubble.slice(0, 40) + "\u2026"
        : agent.thought_bubble;
      const bubbleY = py - 24 * SCALE;
      ctx.font = `italic ${5 * SCALE}px Inter, sans-serif`;
      const tw = ctx.measureText(text).width;
      const bw = tw + 12;
      const bh = 8 * SCALE + 4;
      const bx = px - bw / 2;
      const by = bubbleY - bh;

      // Rounded rect (purple)
      ctx.fillStyle = "rgba(167,139,250,0.9)";
      ctx.beginPath();
      ctx.roundRect(bx, by, bw, bh, 4);
      ctx.fill();

      // Cloud tail (3 small circles)
      ctx.fillStyle = "rgba(167,139,250,0.9)";
      for (let i = 0; i < 3; i++) {
        const r = (3 - i) * SCALE * 0.8;
        const dotX = px - 2 + i * 3;
        const dotY = by + bh + 2 + i * 3;
        ctx.beginPath();
        ctx.arc(dotX, dotY, r, 0, Math.PI * 2);
        ctx.fill();
      }

      // Text
      ctx.fillStyle = "#fff";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, px, by + bh / 2);
    }
  }
}

/* -- Layer 7: Interaction Visualization -------------- */

function renderInteractions(ctx, agents, animPositions, cx, cy, time) {
  const drawnPairs = new Set();
  for (const agent of agents) {
    if (!agent.interaction_partner) continue;
    const pairKey = [agent.name, agent.interaction_partner].sort().join(":");
    if (drawnPairs.has(pairKey)) continue;
    drawnPairs.add(pairKey);

    const partner = agents.find(a => a.name === agent.interaction_partner);
    if (!partner || partner.interaction_partner !== agent.name) continue;

    const posA = animPositions?.[agent.name] || { x: agent.x, y: agent.y };
    const posB = animPositions?.[partner.name] || { x: partner.x, y: partner.y };

    if (!inView(Math.floor(posA.x), Math.floor(posA.y), cx, cy) &&
        !inView(Math.floor(posB.x), Math.floor(posB.y), cx, cy)) continue;

    const ax = (posA.x - cx) * DISPLAY_SIZE + DISPLAY_SIZE / 2;
    const ay = (posA.y - cy) * DISPLAY_SIZE + DISPLAY_SIZE / 2;
    const bx = (posB.x - cx) * DISPLAY_SIZE + DISPLAY_SIZE / 2;
    const by = (posB.y - cy) * DISPLAY_SIZE + DISPLAY_SIZE / 2;

    const midX = (ax + bx) / 2;
    const midY = (ay + by) / 2;

    // Pulsing chat emoji
    const alpha = 0.4 + 0.6 * Math.abs(Math.sin(time * 0.003));
    ctx.globalAlpha = alpha;
    ctx.font = `${10 * SCALE}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("\u{1F4AC}", midX, midY - 6 * SCALE);
    ctx.globalAlpha = 1.0;
  }
}

/* -- Layer 8: Destination Markers -------------------- */

function renderDestinationMarkers(ctx, agents, locations, animPositions, cx, cy, time) {
  for (const agent of agents) {
    if (!agent.destination) continue;

    const pos = animPositions?.[agent.name] || { x: agent.x, y: agent.y };
    if (!inView(Math.floor(pos.x), Math.floor(pos.y), cx, cy)) continue;

    // Find destination location
    const destLoc = locations.find(l =>
      l.name.toLowerCase().includes(agent.destination.toLowerCase()) ||
      agent.destination.toLowerCase().includes(l.name.toLowerCase())
    );
    if (!destLoc) continue;

    const ax = (pos.x - cx) * DISPLAY_SIZE + DISPLAY_SIZE / 2;
    const ay = (pos.y - cy) * DISPLAY_SIZE + DISPLAY_SIZE / 2;
    const destCx = destLoc.x + destLoc.width / 2;
    const destCy = destLoc.y + destLoc.height / 2;
    const dx = (destCx - cx) * DISPLAY_SIZE;
    const dy = (destCy - cy) * DISPLAY_SIZE;

    // Dotted line from agent to destination
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "rgba(111, 227, 255, 0.3)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(dx, dy);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // Pulsing target circle at destination
    const pulse = 0.5 + 0.5 * Math.sin(time * 0.005);
    const radius = (6 + pulse * 4) * SCALE;
    ctx.strokeStyle = `rgba(111, 227, 255, ${0.3 + pulse * 0.3})`;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(dx, dy, radius, 0, Math.PI * 2);
    ctx.stroke();
  }
}

/* -- Layer 9: Fog of War ----------------------------- */

export function renderFogOfWar(ctx, cx, cy, exploredChunks) {
  if (!exploredChunks || exploredChunks.length === 0) return;

  const exploredSet = new Set(exploredChunks);
  const x0 = Math.floor(cx / CHUNK_SIZE) - 1;
  const y0 = Math.floor(cy / CHUNK_SIZE) - 1;
  const x1 = Math.ceil((cx + VIEWPORT_W) / CHUNK_SIZE) + 1;
  const y1 = Math.ceil((cy + VIEWPORT_H) / CHUNK_SIZE) + 1;

  for (let chunkY = y0; chunkY <= y1; chunkY++) {
    for (let chunkX = x0; chunkX <= x1; chunkX++) {
      const key = `${chunkX},${chunkY}`;
      if (exploredSet.has(key)) continue;

      // Fill with dark overlay
      const worldX = chunkX * CHUNK_SIZE;
      const worldY = chunkY * CHUNK_SIZE;
      const screenX = (worldX - cx) * DISPLAY_SIZE;
      const screenY = (worldY - cy) * DISPLAY_SIZE;
      const size = CHUNK_SIZE * DISPLAY_SIZE;

      ctx.fillStyle = "rgba(5, 5, 15, 0.85)";
      ctx.fillRect(screenX, screenY, size, size);

      // Centered "?" text
      ctx.fillStyle = "rgba(255, 255, 255, 0.15)";
      ctx.font = `bold ${12 * SCALE}px Inter, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("?", screenX + size / 2, screenY + size / 2);
    }
  }
}

/* -- Layer 10: Exploration Paths & Frontier Markers --- */

function renderExplorationPaths(ctx, agents, cx, cy, time) {
  for (const agent of agents) {
    const goal = agent.exploration_goal;
    if (!goal || !goal.path_preview || goal.path_preview.length === 0) continue;

    const pos = { x: agent.x, y: agent.y };
    const agentScreenX = (pos.x - cx) * DISPLAY_SIZE + DISPLAY_SIZE / 2;
    const agentScreenY = (pos.y - cy) * DISPLAY_SIZE + DISPLAY_SIZE / 2;

    // Draw dotted path line
    ctx.save();
    const dashOffset = (time * 0.02) % 12;
    ctx.setLineDash([6, 6]);
    ctx.lineDashOffset = -dashOffset;
    ctx.strokeStyle = "rgba(255, 200, 50, 0.4)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(agentScreenX, agentScreenY);

    for (const waypoint of goal.path_preview) {
      const wx = (waypoint[0] - cx) * DISPLAY_SIZE + DISPLAY_SIZE / 2;
      const wy = (waypoint[1] - cy) * DISPLAY_SIZE + DISPLAY_SIZE / 2;
      ctx.lineTo(wx, wy);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // Pulsing target circle at goal
    if (goal.target_world) {
      const tx = (goal.target_world[0] - cx) * DISPLAY_SIZE + DISPLAY_SIZE / 2;
      const ty = (goal.target_world[1] - cy) * DISPLAY_SIZE + DISPLAY_SIZE / 2;
      const pulse = 0.5 + 0.5 * Math.sin(time * 0.004);
      const radius = (8 + pulse * 6) * SCALE;

      ctx.beginPath();
      ctx.arc(tx, ty, radius, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255, 200, 50, ${0.3 + pulse * 0.4})`;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Goal type label
      ctx.font = `bold ${5 * SCALE}px Inter, sans-serif`;
      ctx.fillStyle = `rgba(255, 200, 50, ${0.5 + pulse * 0.3})`;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(goal.goal_type || "explore", tx, ty - radius - 4);
    }
  }
}

function renderFrontierMarkers(ctx, frontierChunks, cx, cy, time) {
  if (!frontierChunks || frontierChunks.length === 0) return;

  const frontierSet = new Set(frontierChunks);
  const x0 = Math.floor(cx / CHUNK_SIZE) - 1;
  const y0 = Math.floor(cy / CHUNK_SIZE) - 1;
  const x1 = Math.ceil((cx + VIEWPORT_W) / CHUNK_SIZE) + 1;
  const y1 = Math.ceil((cy + VIEWPORT_H) / CHUNK_SIZE) + 1;

  const pulse = 0.3 + 0.7 * Math.abs(Math.sin(time * 0.002));

  for (let chunkY = y0; chunkY <= y1; chunkY++) {
    for (let chunkX = x0; chunkX <= x1; chunkX++) {
      const key = `${chunkX},${chunkY}`;
      if (!frontierSet.has(key)) continue;

      const worldX = chunkX * CHUNK_SIZE;
      const worldY = chunkY * CHUNK_SIZE;
      const screenX = (worldX - cx) * DISPLAY_SIZE;
      const screenY = (worldY - cy) * DISPLAY_SIZE;
      const size = CHUNK_SIZE * DISPLAY_SIZE;

      // Golden pulsing border
      ctx.strokeStyle = `rgba(255, 215, 0, ${0.2 + pulse * 0.3})`;
      ctx.lineWidth = 2;
      ctx.strokeRect(screenX + 2, screenY + 2, size - 4, size - 4);

      // "?" marker
      ctx.fillStyle = `rgba(255, 215, 0, ${0.15 + pulse * 0.2})`;
      ctx.font = `bold ${10 * SCALE}px Inter, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("?", screenX + size / 2, screenY + size / 2);
    }
  }
}

/* -- Layer 11: Status Indicators ---------------------- */

const STATUS_ICONS = {
  // Action keyword -> tile index for overhead icon
  writing: TILES.item_book,
  reading: TILES.item_book,
  studying: TILES.item_scroll,
  cooking: TILES.item_food,
  eating: TILES.item_food,
  drinking: TILES.item_food,
  shopping: TILES.item_coin,
  buying: TILES.item_coin,
  selling: TILES.item_coin,
  praying: TILES.item_ring,
  meditating: TILES.item_ring,
  healing: TILES.item_potion,
  sleeping: TILES.item_key, // placeholder
  exploring: TILES.item_target,
  chatting: TILES.item_sign,
  working: TILES.item_chest,
  painting: TILES.item_gem,
};

function getStatusIcon(action) {
  if (!action) return null;
  const lower = action.toLowerCase();
  for (const [keyword, tile] of Object.entries(STATUS_ICONS)) {
    if (lower.includes(keyword)) return tile;
  }
  return null;
}

function renderStatusIndicators(ctx, img, agents, animPositions, cx, cy) {
  for (const agent of agents) {
    const tileIdx = getStatusIcon(agent.action);
    if (tileIdx == null) continue;

    const pos = animPositions?.[agent.name] || { x: agent.x, y: agent.y };
    if (!inView(Math.floor(pos.x), Math.floor(pos.y), cx, cy)) continue;

    const px = (pos.x - cx) * DISPLAY_SIZE + DISPLAY_SIZE / 2;
    const py = (pos.y - cy) * DISPLAY_SIZE + DISPLAY_SIZE / 2;

    // Draw status icon above and to the left of the agent
    const iconSize = DISPLAY_SIZE * 0.6;
    const iconX = px - DISPLAY_SIZE * 0.7;
    const iconY = py - DISPLAY_SIZE * 0.9;

    // Background circle
    ctx.beginPath();
    ctx.arc(iconX + iconSize / 2, iconY + iconSize / 2, iconSize * 0.55, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(0,0,0,0.5)";
    ctx.fill();

    // Draw the tile icon
    drawTile(ctx, img, tileIdx, iconX, iconY);
  }
}

/* -- Public API -------------------------------------- */

/**
 * Full render of the town map with camera support.
 * @param {CanvasRenderingContext2D} ctx
 * @param {Image|null} tilemapImg
 * @param {object} state - { locations, agents, grid_w, grid_h, chunk_biomes, world_properties, ... }
 * @param {string|null} activeUser
 * @param {object|null} animPositions - { [agentName]: {x, y} }
 * @param {{cx: number, cy: number}} camera - viewport top-left in world coords
 * @param {number} [walkFrame=0] - 0 or 1 for walk animation
 * @param {number} [waterFrame=0] - water animation frame (cycles 0-3)
 */
export function renderMap(ctx, tilemapImg, state, activeUser, animPositions, camera, walkFrame = 0, waterFrame = 0, time = 0) {
  if (!state) return;

  const cx = camera?.cx ?? 0;
  const cy = camera?.cy ?? 0;

  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

  const locations = state.locations || [];
  const agents = state.agents || [];
  const chunkBiomes = state.chunk_biomes || null;
  const season = state.world_properties?.season || "spring";
  const locMap = buildLocationMap(locations, cx, cy);
  const roadSet = generateRoads(locations);

  renderTerrain(ctx, tilemapImg, cx, cy, chunkBiomes, season);
  renderWater(ctx, tilemapImg, cx, cy, chunkBiomes, waterFrame);
  renderRoads(ctx, tilemapImg, roadSet, cx, cy);
  renderBuildings(ctx, tilemapImg, locations, cx, cy, state.asset_overrides);
  renderBuildingLabels(ctx, tilemapImg, locations, cx, cy);
  renderDecorations(ctx, tilemapImg, locMap, roadSet, cx, cy, season);
  renderFrontierMarkers(ctx, state.frontier_chunks, cx, cy, time);
  renderExplorationPaths(ctx, agents, cx, cy, time);
  renderDestinationMarkers(ctx, agents, locations, animPositions, cx, cy, time);
  renderAgents(ctx, agents, activeUser, animPositions, cx, cy, walkFrame, state.asset_overrides);
  renderStatusIndicators(ctx, tilemapImg, agents, animPositions, cx, cy);
  renderInteractions(ctx, agents, animPositions, cx, cy, time);
  renderBubbles(ctx, agents, activeUser, animPositions, cx, cy);
}

/**
 * Hit-test: find agent near canvas pixel coordinates.
 * @returns {object|null} the clicked agent, or null
 */
export function hitTestAgent(canvasX, canvasY, agents, animPositions, camera) {
  const cx = camera?.cx ?? 0;
  const cy = camera?.cy ?? 0;
  const hitRadius = DISPLAY_SIZE * 0.7;
  for (const agent of agents) {
    const pos = animPositions?.[agent.name] || { x: agent.x, y: agent.y };
    const px = (pos.x - cx) * DISPLAY_SIZE + DISPLAY_SIZE / 2;
    const py = (pos.y - cy) * DISPLAY_SIZE + DISPLAY_SIZE / 2;
    const dx = canvasX - px, dy = canvasY - py;
    if (dx * dx + dy * dy < hitRadius * hitRadius) return agent;
  }
  return null;
}

/**
 * Find what's at a canvas pixel (for tooltip).
 * @returns {{ agent?: object, location?: object } | null}
 */
export function hitTestCell(canvasX, canvasY, state, animPositions, camera) {
  const cx = camera?.cx ?? 0;
  const cy = camera?.cy ?? 0;

  const agent = hitTestAgent(canvasX, canvasY, state.agents || [], animPositions, camera);
  if (agent) return { agent };

  // Convert canvas pixel to world grid coords
  const gx = Math.floor(canvasX / DISPLAY_SIZE + cx);
  const gy = Math.floor(canvasY / DISPLAY_SIZE + cy);
  const locMap = buildLocationMap(state.locations || [], cx, cy);
  const loc = locMap.get(`${gx},${gy}`);
  if (loc) return { location: loc };
  return null;
}
