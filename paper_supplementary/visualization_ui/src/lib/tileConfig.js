/**
 * Tile configuration for the Kenney Tiny Town tileset.
 * Uses tilemap_packed.png (192x176, 12 cols x 11 rows, 16x16 per tile).
 */

export const TILE_SIZE = 16;
export const SCALE = 1.4;
export const DISPLAY_SIZE = TILE_SIZE * SCALE; // 22.4

// Viewport: full 60x60 grid visible at once (whole-town screenshot mode)
export const VIEWPORT_W = 60;
export const VIEWPORT_H = 60;
export const CANVAS_W = VIEWPORT_W * DISPLAY_SIZE; // 1344
export const CANVAS_H = VIEWPORT_H * DISPLAY_SIZE; // 1344

// World grid size — read dynamically from state, these are fallback defaults
export let GRID_W = 60;
export let GRID_H = 60;

/** Call once when state arrives to sync grid dimensions. */
export function setGridSize(w, h) {
  GRID_W = w || 60;
  GRID_H = h || 60;
}

export const TILEMAP_COLS = 12;

// Helper: convert tile index (0-131) -> { sx, sy } source coords in packed tilemap
export function tileCoords(index) {
  const col = index % TILEMAP_COLS;
  const row = Math.floor(index / TILEMAP_COLS);
  return { sx: col * TILE_SIZE, sy: row * TILE_SIZE };
}

// Semantic tile names -> tile index in packed tilemap (col + row*12)
export const TILES = {
  // Row 0 -- terrain & trees
  grass_1:       0,   // plain green grass
  grass_2:       1,   // grass variant
  dirt:          2,   // dirt/sand
  tree_small:    3,   // small orange tree
  bush:          4,   // green bush
  tree_round:    5,   // round green tree
  tree_large:    6,   // larger round tree
  tree_pine:     7,   // tall pine
  tree_tall:     8,   // tall tree
  tree_autumn_1: 9,   // autumn tree 1
  tree_autumn_2: 10,  // autumn tree 2
  tree_autumn_3: 11,  // autumn tree 3

  // Row 1 -- more terrain & vegetation
  grass_edge_1:  12,  // grass edge
  grass_edge_2:  13,  // grass-dirt edge
  grass_edge_3:  14,  // hedge edge
  flower:        15,  // flowers
  hedge_1:       16,  // hedge
  hedge_2:       17,  // hedge variant
  tree_big_1:    18,  // big tree
  tree_big_2:    19,  // big tree variant
  tree_big_3:    20,  // tree variant
  roof_or_tl:    21,  // orange roof top-left
  roof_or_tr:    22,  // orange roof top-right
  roof_or_peak:  23,  // orange roof peak

  // Row 2 -- roads & blue roofs
  road_cross:    24,  // cobblestone cross
  road_h:        25,  // horizontal road
  road_v:        26,  // vertical road
  road_tl:       27,  // corner top-left
  road_tr:       28,  // corner top-right
  road_bl:       29,  // corner bottom-left
  road_br:       30,  // corner bottom-right (or t-junction)
  roof_bl_tl:    31,  // blue roof top-left
  roof_bl_tr:    32,  // blue roof top-right
  roof_bl_ml:    33,  // blue roof mid-left
  pipe_1:        34,  // pipe piece
  pipe_2:        35,  // pipe piece

  // Row 3 -- building walls & orange roofs
  wall_stone_1:  36,  // stone wall top
  wall_stone_2:  37,  // stone wall
  wall_stone_3:  38,  // stone wall variant
  wall_wood_1:   39,  // wood wall top
  wall_wood_2:   40,  // wood wall
  wall_wood_3:   41,  // wood wall variant
  roof_or_ml:    42,  // orange roof mid-left
  roof_or_mr:    43,  // orange roof mid-right
  roof_or_bl:    44,  // orange roof bottom-left
  fence_1:       45,  // fence piece
  fence_2:       46,  // fence piece
  fence_3:       47,  // fence piece

  // Row 4 -- building fronts & items
  wall_brown_1:  48,  // brown wall
  wall_brown_2:  49,  // brown wall with window
  door_wood:     50,  // wooden door
  wall_gray_1:   51,  // gray/stone wall
  wall_gray_2:   52,  // gray wall with window
  door_gray:     53,  // gray door
  item_sign:     54,  // sign
  item_coin:     55,  // coin
  item_chest:    56,  // chest
  item_key:      57,  // key
  item_ring:     58,  // ring
  item_target:   59,  // target

  // Row 5 -- more building parts & items
  wall_dark_1:   60,  // dark wall
  wall_dark_2:   61,  // dark wall window
  door_dark:     62,  // dark door
  wall_brick_1:  63,  // brick wall
  wall_brick_2:  64,  // brick wall window
  door_brick:    65,  // brick door
  item_potion:   66,
  item_food:     67,
  item_skull:    68,
  item_gem:      69,
  item_book:     70,
  item_scroll:   71,

  // Row 6 -- stone floor & castle
  floor_white_1: 72,  // white stone floor
  floor_white_2: 73,
  floor_white_3: 74,
  floor_white_4: 75,
  castle_wall_1: 76,
  castle_wall_2: 77,
  castle_top_1:  78,
  castle_top_2:  79,
  tool_1:        80,
  tool_2:        81,
  fountain:      82,  // fountain/well
  tool_3:        83,

  // Row 7 -- dungeon & archway
  floor_dark_1:  84,
  floor_dark_2:  85,
  floor_dark_3:  86,
  floor_dark_4:  87,
  arch_left:     88,
  arch_center:   89,
  arch_right:    90,
  bench:         91,  // bench
  weapon_1:      92,
  weapon_2:      93,
  weapon_3:      94,
  weapon_4:      95,
};

// Road tile selection based on 4-neighbor connectivity (N,E,S,W bitmask)
// bit 0 = North, bit 1 = East, bit 2 = South, bit 3 = West
export const ROAD_VARIANTS = {
  0b0000: TILES.road_cross,  // isolated -> cross
  0b1111: TILES.road_cross,  // all connected -> cross
  0b0101: TILES.road_v,      // N+S -> vertical
  0b1010: TILES.road_h,      // E+W -> horizontal
  0b0110: TILES.road_tl,     // S+E -> corner
  0b1100: TILES.road_tr,     // W+S -> corner (flipped)
  0b0011: TILES.road_bl,     // N+E -> corner
  0b1001: TILES.road_br,     // N+W -> corner
  // T-junctions and other combos -> cross
  0b0111: TILES.road_cross,
  0b1011: TILES.road_cross,
  0b1101: TILES.road_cross,
  0b1110: TILES.road_cross,
  // Dead ends -> straight
  0b0001: TILES.road_v,
  0b0100: TILES.road_v,
  0b0010: TILES.road_h,
  0b1000: TILES.road_h,
};

// Building patterns: location type -> 2D array of tile indices
// Each pattern is [row][col] of the building footprint
export const BUILDING_PATTERNS = {
  // --- Orange roof, brown walls ---
  home: {
    tiles: [
      [TILES.roof_or_tl,  TILES.roof_or_tr,  TILES.roof_or_tl],
      [TILES.wall_brown_2, TILES.door_wood,   TILES.wall_brown_2],
      [TILES.wall_brown_1, TILES.wall_brown_1, TILES.wall_brown_1],
    ],
    w: 3, h: 3,
  },
  // --- Blue roof, brown walls ---
  cafe: {
    tiles: [
      [TILES.roof_bl_tl,  TILES.roof_bl_tr,  TILES.roof_bl_tl],
      [TILES.wall_brown_2, TILES.door_wood,   TILES.wall_brown_2],
      [TILES.wall_brown_1, TILES.wall_brown_1, TILES.wall_brown_1],
    ],
    w: 3, h: 3,
  },
  // --- Blue roof, gray stone ---
  library: {
    tiles: [
      [TILES.roof_bl_tl, TILES.roof_bl_tr, TILES.roof_bl_tl, TILES.roof_bl_tr],
      [TILES.wall_gray_2, TILES.wall_gray_2, TILES.door_gray,  TILES.wall_gray_2],
      [TILES.wall_gray_1, TILES.wall_gray_1, TILES.wall_gray_1, TILES.wall_gray_1],
    ],
    w: 4, h: 3,
  },
  // --- Orange roof, brick walls ---
  shop: {
    tiles: [
      [TILES.roof_or_tl,   TILES.roof_or_tr,  TILES.roof_or_tl],
      [TILES.wall_brick_2, TILES.door_brick,  TILES.wall_brick_2],
    ],
    w: 3, h: 2,
  },
  // --- Castle-style: arches + stone --- (Town Hall)
  townhall: {
    tiles: [
      [TILES.castle_top_1, TILES.castle_top_2, TILES.castle_top_1, TILES.castle_top_2, TILES.castle_top_1],
      [TILES.castle_wall_1,TILES.castle_wall_2,TILES.castle_wall_1,TILES.castle_wall_2,TILES.castle_wall_1],
      [TILES.wall_gray_2,  TILES.wall_gray_2,  TILES.door_gray,    TILES.wall_gray_2,  TILES.wall_gray_2],
      [TILES.wall_gray_1,  TILES.wall_gray_1,  TILES.wall_gray_1,  TILES.wall_gray_1,  TILES.wall_gray_1],
    ],
    w: 5, h: 4,
  },
  // --- Orange roof, wood walls --- (Bakery)
  bakery: {
    tiles: [
      [TILES.roof_or_tl,  TILES.roof_or_peak, TILES.roof_or_tr],
      [TILES.wall_wood_2,  TILES.door_wood,    TILES.wall_wood_2],
    ],
    w: 3, h: 2,
  },
  // --- Orange roof, brick --- (Pharmacy)
  pharmacy: {
    tiles: [
      [TILES.roof_or_ml,  TILES.roof_or_mr,  TILES.roof_or_ml],
      [TILES.wall_brick_2, TILES.door_brick,  TILES.wall_brick_1],
    ],
    w: 3, h: 2,
  },
  // --- Blue roof, gray stone --- (School)
  school: {
    tiles: [
      [TILES.roof_bl_tl,  TILES.roof_bl_tr,  TILES.roof_bl_tl,  TILES.roof_bl_tr],
      [TILES.wall_gray_2,  TILES.door_gray,   TILES.wall_gray_2,  TILES.wall_gray_2],
      [TILES.wall_gray_1,  TILES.wall_gray_1, TILES.wall_gray_1,  TILES.wall_gray_1],
    ],
    w: 4, h: 3,
  },
  // --- Dark walls, dark door --- (Art Gallery)
  gallery: {
    tiles: [
      [TILES.roof_bl_tl,  TILES.roof_bl_tr,  TILES.roof_bl_tl],
      [TILES.wall_dark_2,  TILES.door_dark,   TILES.wall_dark_2],
      [TILES.wall_dark_1,  TILES.wall_dark_1, TILES.wall_dark_1],
    ],
    w: 3, h: 3,
  },
  // --- Wood walls, blue roof --- (Tea House)
  teahouse: {
    tiles: [
      [TILES.roof_bl_tl,  TILES.roof_bl_tr,  TILES.roof_bl_tl],
      [TILES.wall_wood_2,  TILES.door_wood,   TILES.wall_wood_2],
      [TILES.wall_wood_1,  TILES.wall_wood_1, TILES.wall_wood_1],
    ],
    w: 3, h: 3,
  },
  // --- Stone + arch --- (Temple)
  temple: {
    tiles: [
      [TILES.castle_top_1, TILES.castle_top_2, TILES.castle_top_1],
      [TILES.arch_left,    TILES.arch_center,  TILES.arch_right],
      [TILES.wall_stone_1, TILES.door_gray,    TILES.wall_stone_1],
    ],
    w: 3, h: 3,
  },
};

// Park fills with trees, flowers, fountain
export const PARK_FILL = [TILES.tree_round, TILES.tree_pine, TILES.flower, TILES.bush];

// Town square fills
export const SQUARE_FILL = [TILES.floor_white_1, TILES.floor_white_2, TILES.fountain, TILES.bench];

// Decoration candidates for scattering on empty grass
export const DECORATIONS = [
  TILES.flower, TILES.bush, TILES.tree_small, TILES.hedge_1,
];
export const DECORATION_DENSITY = 0.08;

// Water tiles (reuse dark floor tiles as water)
export const WATER_TILES = [TILES.floor_dark_1, TILES.floor_dark_2, TILES.floor_dark_3, TILES.floor_dark_4];

// Season-specific decoration palettes
export const SEASON_DECORATIONS = {
  spring: [TILES.flower, TILES.bush, TILES.tree_round, TILES.tree_large],
  summer: [TILES.tree_round, TILES.tree_large, TILES.tree_big_1, TILES.bush],
  autumn: [TILES.tree_autumn_1, TILES.tree_autumn_2, TILES.tree_autumn_3, TILES.bush],
  winter: [TILES.tree_pine, TILES.tree_tall, TILES.hedge_1],
};

// Biome terrain palettes
export const BIOME_TERRAIN = {
  town_center: [TILES.grass_1, TILES.grass_1, TILES.grass_1, TILES.grass_2],
  farmland: [TILES.dirt, TILES.grass_1, TILES.grass_2, TILES.dirt],
  riverside: [TILES.grass_1, TILES.grass_edge_1, TILES.grass_edge_2, TILES.grass_1],
  forest: [TILES.grass_1, TILES.grass_1, TILES.grass_2, TILES.grass_2],
  mountain: [TILES.dirt, TILES.dirt, TILES.grass_edge_3, TILES.dirt],
};

// Time phase definitions for day/night cycle
export const TIME_PHASES = [
  { name: "dawn",     start: 5,  end: 7,  overlay: "rgba(255,200,100,0.15)" },
  { name: "day",      start: 7,  end: 17, overlay: null },
  { name: "dusk",     start: 17, end: 19, overlay: "rgba(255,130,50,0.2)" },
  { name: "twilight", start: 19, end: 21, overlay: "rgba(30,30,80,0.3)" },
  { name: "night",    start: 21, end: 5,  overlay: "rgba(10,10,40,0.55)" },
];

// Chunk size constant (must match backend)
export const CHUNK_SIZE = 16;

// Character sprite color schemes for pixel-art characters
export const CHARACTER_SPRITES = {
  "Lin Wei":   { hair: "#3b2314", shirt: "#e74c3c", skin: "#f5c6a0", pants: "#2c3e50" },
  "Zhang Hao": { hair: "#1a1a2e", shirt: "#3498db", skin: "#f0d0a0", pants: "#34495e" },
  "Chen Mei":  { hair: "#4a2810", shirt: "#2ecc71", skin: "#f5c6a0", pants: "#7f8c8d" },
  "Liu Yang":  { hair: "#2c1810", shirt: "#f39c12", skin: "#f0d0a0", pants: "#2c3e50" },
  "Wang Jun":  { hair: "#555555", shirt: "#9b59b6", skin: "#f5c6a0", pants: "#2c3e50" },
};
