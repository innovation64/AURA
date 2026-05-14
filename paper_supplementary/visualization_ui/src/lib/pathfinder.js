import { GRID_W, GRID_H, CHUNK_SIZE } from "./tileConfig.js";

/**
 * Generate a chunk-grid road network that extends naturally with the world.
 * - Horizontal roads every CHUNK_SIZE cells
 * - Vertical roads every CHUNK_SIZE cells
 * - Each building connects to nearest grid road via short spur paths
 *
 * @param {Array} locations - [{x, y, width, height, type, ...}]
 * @returns {Set<string>} set of "x,y" keys for road cells
 */
export function generateRoads(locations) {
  const roads = new Set();
  if (!locations?.length) return roads;

  // Find world bounds from locations
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const loc of locations) {
    minX = Math.min(minX, loc.x);
    maxX = Math.max(maxX, loc.x + loc.width);
    minY = Math.min(minY, loc.y);
    maxY = Math.max(maxY, loc.y + loc.height);
  }

  // Add margin
  minX = Math.max(0, minX - 2);
  minY = Math.max(0, minY - 2);
  maxX += 2;
  maxY += 2;

  // Grid roads: horizontal every CHUNK_SIZE cells, vertical every CHUNK_SIZE cells
  // Align to chunk boundaries
  const gridSpacing = CHUNK_SIZE;

  // Find nearest grid lines
  const firstGridY = Math.floor(minY / gridSpacing) * gridSpacing + Math.floor(gridSpacing / 2);
  const firstGridX = Math.floor(minX / gridSpacing) * gridSpacing + Math.floor(gridSpacing / 2);

  // Horizontal grid roads
  for (let gy = firstGridY; gy <= maxY; gy += gridSpacing) {
    if (gy < minY) continue;
    for (let x = minX; x < maxX; x++) {
      roads.add(`${x},${gy}`);
    }
  }

  // Vertical grid roads
  for (let gx = firstGridX; gx <= maxX; gx += gridSpacing) {
    if (gx < minX) continue;
    for (let y = minY; y < maxY; y++) {
      roads.add(`${gx},${y}`);
    }
  }

  // Connect each building entrance to nearest grid road via spur
  for (const loc of locations) {
    if (loc.type === "park" || loc.type === "square") continue;

    const entranceX = loc.x + Math.floor(loc.width / 2);
    const entranceY = loc.y + loc.height; // one cell below building

    // Find nearest horizontal grid road
    const nearestGridY = Math.round((entranceY - firstGridY) / gridSpacing) * gridSpacing + firstGridY;

    // Vertical spur from entrance to nearest grid road
    const startY = Math.min(entranceY, nearestGridY);
    const endY = Math.max(entranceY, nearestGridY);
    for (let y = startY; y <= endY; y++) {
      roads.add(`${entranceX},${y}`);
    }

    // Find nearest vertical grid road
    const nearestGridX = Math.round((entranceX - firstGridX) / gridSpacing) * gridSpacing + firstGridX;

    // Horizontal spur from entrance to nearest vertical grid road
    const startX = Math.min(entranceX, nearestGridX);
    const endX = Math.max(entranceX, nearestGridX);
    for (let x = startX; x <= endX; x++) {
      roads.add(`${x},${entranceY}`);
    }
  }

  return roads;
}

/**
 * Compute the 4-neighbor bitmask for a road cell.
 * bit 0 = North, bit 1 = East, bit 2 = South, bit 3 = West
 */
export function roadNeighborMask(x, y, roadSet) {
  let mask = 0;
  if (roadSet.has(`${x},${y - 1}`)) mask |= 0b0001; // N
  if (roadSet.has(`${x + 1},${y}`)) mask |= 0b0010; // E
  if (roadSet.has(`${x},${y + 1}`)) mask |= 0b0100; // S
  if (roadSet.has(`${x - 1},${y}`)) mask |= 0b1000; // W
  return mask;
}
