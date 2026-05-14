"""Render an AURATown simulation snapshot for the main paper.

A single tick (18:00, Scene B) is rendered showing:
- 5 agents at their current positions
- Public state (location, action) visible next to each agent
- Private state in dashed callouts marked PROBE-ONLY
- A user query banner + AURA's IntentFrame inference + the routed probe arrow

This makes the public/private split + proactive-probing mechanism legible at a glance.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOWN_MAP = ROOT / "demo" / "town" / "assets" / "town_map.json"
OUT = ROOT / "paper" / "figures" / "auratown-snapshot.png"

# Hard-coded snapshot for tick = 18:00, Scene B.
# Public state matches what the scene snapshot would expose.
# Private state matches what is hidden behind probe tools.
AGENT_STATE = {
    "Lin Wei": {
        "loc": "Lin Wei's Home",
        "action": "cooking dinner",
        "private": {
            "availability": "busy",
            "unspoken_goal": "dinner before evening yoga",
            "emotional_state": "focused",
        },
        "color": "#c0392b",
        "marker_text": "L",
    },
    "Wang Jun": {
        "loc": "Sunrise Cafe",
        "action": "drinking tea",
        "private": {
            "availability": "free",
            "unspoken_goal": "wind down after work",
            "emotional_state": "relaxed",
        },
        "color": "#e67e22",
        "marker_text": "W",
    },
    "Zhang Hao": {
        "loc": "Art Gallery",
        "action": "viewing exhibition",
        "private": {
            "availability": "free",
            "unspoken_goal": "find inspiration",
            "emotional_state": "curious",
        },
        "color": "#2980b9",
        "marker_text": "Z",
    },
    "Chen Mei": {
        "loc": "Riverside Walk",
        "action": "walking",
        "private": {
            "availability": "free",
            "unspoken_goal": "decompress",
            "emotional_state": "calm",
        },
        "color": "#27ae60",
        "marker_text": "C",
    },
    "Liu Yang": {
        "loc": "Town Park",
        "action": "meditating",
        "private": {
            "availability": "busy",
            "unspoken_goal": "evening practice",
            "emotional_state": "centered",
        },
        "color": "#8e44ad",
        "marker_text": "Y",
    },
}

# Loc-type → fill colour
TYPE_COLOR = {
    "home":     "#a8d8a8",
    "teahouse": "#f5c272",
    "cafe":     "#f5c272",
    "bakery":   "#f5c272",
    "shop":     "#f5c272",
    "pharmacy": "#f5c272",
    "gallery":  "#f5c272",
    "townhall": "#a0c4d6",
    "temple":   "#a0c4d6",
    "library":  "#a0c4d6",
    "school":   "#a0c4d6",
    "square":   "#a0c4d6",
    "park":     "#d4b3e0",
}


def loc_lookup(town_map: dict) -> dict:
    return {loc["name"]: loc for loc in town_map["locations"]}


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    town = json.loads(TOWN_MAP.read_text())
    locs = loc_lookup(town)
    W, H = town["width"], town["height"]

    fig, ax = plt.subplots(figsize=(14, 9.5))
    ax.set_xlim(-22, W + 30)   # left margin for private callout, right for panel
    ax.set_ylim(-3, H + 6)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    # 1. Map background
    ax.add_patch(patches.Rectangle((0, 0), W, H, facecolor="#f7f2e7", edgecolor="black", lw=1.5))
    # light grid
    for i in range(0, W + 1, 10):
        ax.plot([i, i], [0, H], color="#cfc8b6", lw=0.4)
    for j in range(0, H + 1, 10):
        ax.plot([0, W], [j, j], color="#cfc8b6", lw=0.4)

    # 2. Location footprints
    for loc in town["locations"]:
        color = TYPE_COLOR.get(loc["type"], "#cccccc")
        ax.add_patch(patches.Rectangle(
            (loc["x"], loc["y"]), loc["width"], loc["height"],
            facecolor=color, edgecolor="#666", lw=0.6, alpha=0.85,
        ))
        # short label inside footprint (only if footprint is big enough)
        if loc["width"] * loc["height"] >= 9:
            short = loc["name"].replace("'s Home", "").replace("Town ", "")
            cx = loc["x"] + loc["width"] / 2
            cy = loc["y"] + loc["height"] / 2
            ax.text(cx, cy, short, fontsize=6, ha="center", va="center", color="#333")

    # 3. Agents
    for name, st in AGENT_STATE.items():
        loc = locs[st["loc"]]
        ax_x = loc["x"] + loc["width"] / 2
        ax_y = loc["y"] + loc["height"] / 2
        # ring marker
        ax.add_patch(patches.Circle((ax_x, ax_y), 1.6,
                                    facecolor=st["color"], edgecolor="black", lw=1.0, zorder=5))
        ax.text(ax_x, ax_y, st["marker_text"], fontsize=8, ha="center", va="center",
                color="white", fontweight="bold", zorder=6)
        # name + public action label (right of agent)
        label_x, label_y = ax_x + 2.2, ax_y - 0.5
        ax.text(label_x, label_y,
                f"{name}\n{st['loc']}\n{st['action']}",
                fontsize=6.5, ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=st["color"], lw=0.8),
                zorder=7)

    # 4. Private-state callout for Lin Wei (the query target) — dashed border = HIDDEN
    # Placed to the LEFT of the map so it doesn't overlap public-state labels.
    lin = locs["Lin Wei's Home"]
    callout_cx = -10.5   # centre x (left of map)
    callout_cy = 8       # centre y (aligned to upper-left region)
    callout_w  = 19
    callout_h  = 8
    private_text = (
        "PRIVATE  (probe-only)\n\n"
        "availability:       busy\n"
        "unspoken_goal:  dinner before yoga\n"
        "emotional_state: focused"
    )
    ax.add_patch(FancyBboxPatch(
        (callout_cx - callout_w / 2, callout_cy - callout_h / 2),
        callout_w, callout_h,
        boxstyle="round,pad=0.3",
        facecolor="#fff8e1", edgecolor="#c0392b", lw=1.2, linestyle="--", zorder=4,
    ))
    ax.text(callout_cx, callout_cy, private_text,
            fontsize=6, ha="center", va="center", color="#7c2f1d", zorder=8)
    # connector from Lin Wei to private callout
    ax.add_patch(FancyArrowPatch(
        (lin["x"] + lin["width"] / 2, lin["y"] + lin["height"] / 2),
        (callout_cx + callout_w / 2 - 0.2, callout_cy),
        arrowstyle="-", color="#c0392b", lw=1.0, linestyle="--", zorder=3,
    ))

    # 5. User query + IntentFrame panel on the right
    panel_x = W + 3
    panel_y = 2
    panel_w = 26
    panel_h = 50

    ax.add_patch(FancyBboxPatch(
        (panel_x, panel_y), panel_w, panel_h,
        boxstyle="round,pad=0.5", facecolor="#eef4fb", edgecolor="#34495e", lw=1.0, zorder=2,
    ))

    # Header: user query
    ax.text(panel_x + panel_w / 2, panel_y + 3.5,
            "User query  @  tick 18:00",
            fontsize=9, ha="center", va="center", fontweight="bold", color="#34495e")
    ax.text(panel_x + panel_w / 2, panel_y + 7,
            "\"Where is Lin Wei?\"",
            fontsize=11, ha="center", va="center", style="italic", color="#1f2d3a")

    # Divider
    ax.plot([panel_x + 1.5, panel_x + panel_w - 1.5], [panel_y + 10, panel_y + 10],
            color="#34495e", lw=0.6)

    # IntentFrame
    ax.text(panel_x + panel_w / 2, panel_y + 12.5,
            "AURA  IntentFrame",
            fontsize=8.5, ha="center", va="center", fontweight="bold", color="#0b5394")
    intent_lines = [
        ("literal",  "find Lin Wei's location"),
        ("implicit", "is she free to chat now?"),
        ("gap",      "0.6"),
        ("probe",    "get_agent_private_state(Lin Wei)"),
        ("alert",    "1"),
    ]
    for i, (k, v) in enumerate(intent_lines):
        y = panel_y + 16 + i * 2.6
        ax.text(panel_x + 2, y, f"{k}:",      fontsize=7, ha="left", va="center",
                fontweight="bold", color="#0b5394")
        ax.text(panel_x + 8.5, y, v,           fontsize=7, ha="left", va="center", color="#1f2d3a")

    # Divider
    ax.plot([panel_x + 1.5, panel_x + panel_w - 1.5], [panel_y + 31, panel_y + 31],
            color="#34495e", lw=0.6)

    # Answer
    ax.text(panel_x + panel_w / 2, panel_y + 33.5,
            "Routed probe → answer",
            fontsize=8.5, ha="center", va="center", fontweight="bold", color="#0b5394")
    answer_text = (
        "\"Lin Wei is cooking dinner\n"
        "at home; she is busy preparing\n"
        "for evening yoga, so this is\n"
        "probably not the best moment\n"
        "to interrupt.\""
    )
    ax.text(panel_x + panel_w / 2, panel_y + 41.5, answer_text,
            fontsize=7, ha="center", va="center", style="italic", color="#1f2d3a")

    # Arrow from panel → Lin Wei's private callout
    ax.add_patch(FancyArrowPatch(
        (panel_x, panel_y + 23.5),
        (callout_cx + callout_w / 2 - 1, callout_cy + 1.5),
        arrowstyle="->", color="#0b5394", lw=1.0, mutation_scale=10, zorder=5,
        connectionstyle="arc3,rad=-0.35",
    ))

    # 6. Legend strip at bottom-left
    legend_y = H + 1.2
    legend_items = [
        ("home",   "#a8d8a8"),
        ("commerce", "#f5c272"),
        ("civic",  "#a0c4d6"),
        ("park",   "#d4b3e0"),
    ]
    for i, (k, c) in enumerate(legend_items):
        lx = 1 + i * 11
        ax.add_patch(patches.Rectangle((lx, legend_y), 2, 1.5, facecolor=c, edgecolor="#666"))
        ax.text(lx + 2.5, legend_y + 0.7, k, fontsize=7, ha="left", va="center", color="#333")
    # private-state caption
    ax.text(45, legend_y + 0.7,
            "─ ─ ─  hidden state, retrievable only via probe tool",
            fontsize=7, ha="left", va="center", color="#7c2f1d")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"saved → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
