"""Gradio Web UI for AURA Town simulation."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from .config import DEFAULT_CONFIG, TownConfig
from .simulation import TownSimulation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global simulation state ─────────────────────────────────────────

sim: Optional[TownSimulation] = None
auto_running = False
auto_thread: Optional[threading.Thread] = None


def get_sim() -> TownSimulation:
    global sim
    if sim is None:
        sim = TownSimulation()
        sim.initialize()
    return sim


# ── Map rendering ───────────────────────────────────────────────────

def render_map(state: Dict[str, Any]) -> str:
    """Render the town map as an HTML grid with emoji."""
    locations = state["locations"]
    agents = state["agents"]
    grid_w, grid_h = 20, 20
    cell_size = 36

    # Build a grid lookup: (x, y) -> (emoji, name, type)
    grid: Dict[Tuple[int, int], Tuple[str, str, str]] = {}
    for loc in locations:
        for dx in range(loc["width"]):
            for dy in range(loc["height"]):
                gx, gy = loc["x"] + dx, loc["y"] + dy
                grid[(gx, gy)] = (loc["emoji"], loc["name"], loc["type"])

    # Agent positions
    agent_pos: Dict[Tuple[int, int], List[Dict]] = {}
    for a in agents:
        pos = (a["x"], a["y"])
        agent_pos.setdefault(pos, []).append(a)

    # Color map for location types
    bg_colors = {
        "home": "#FFF3E0",
        "cafe": "#EFEBE9",
        "library": "#E8EAF6",
        "shop": "#FFF9C4",
        "park": "#E8F5E9",
        "square": "#F3E5F5",
    }

    html = f"""
    <div style="display:inline-grid; grid-template-columns: repeat({grid_w}, {cell_size}px);
                grid-template-rows: repeat({grid_h}, {cell_size}px);
                gap: 1px; background: #ccc; border: 2px solid #999; border-radius: 8px;
                padding: 2px; font-size: 14px;">
    """

    for y in range(grid_h):
        for x in range(grid_w):
            loc_info = grid.get((x, y))
            agents_here = agent_pos.get((x, y), [])

            bg = "#f5f5f0"
            title = ""
            content = ""

            if loc_info:
                emoji, name, loc_type = loc_info
                bg = bg_colors.get(loc_type, "#f0f0f0")
                title = name

            if agents_here:
                # Show agent emojis
                content = "".join(a["emoji"] for a in agents_here)
                agent_names = ", ".join(a["name"] for a in agents_here)
                title = agent_names + (f" @ {title}" if title else "")
            elif loc_info:
                # Show location emoji only at center
                for loc in locations:
                    if loc["name"] == loc_info[1]:
                        cx = loc["x"] + loc["width"] // 2
                        cy = loc["y"] + loc["height"] // 2
                        if x == cx and y == cy:
                            content = loc_info[0]
                        break

            html += (
                f'<div style="background:{bg}; display:flex; align-items:center; '
                f'justify-content:center; width:{cell_size}px; height:{cell_size}px; '
                f'font-size:16px; cursor:default; border-radius:3px;" '
                f'title="{title}">{content}</div>'
            )

    html += "</div>"

    # Legend
    html += '<div style="margin-top:12px; font-size:13px; color:#555;">'
    html += "<b>Locations:</b> "
    for loc in locations:
        bg = bg_colors.get(loc["type"], "#f0f0f0")
        html += (
            f'<span style="background:{bg}; padding:2px 6px; margin:2px; '
            f'border-radius:4px; border:1px solid #ddd;">'
            f'{loc["emoji"]} {loc["name"]}</span> '
        )
    html += "<br><b>Agents:</b> "
    for a in agents:
        html += f'{a["emoji"]} {a["name"]}  '
    html += "</div>"

    return html


# ── Event log formatting ────────────────────────────────────────────

def format_events(state: Dict[str, Any]) -> str:
    """Format events as readable log text."""
    events = state.get("events", [])
    if not events:
        return "No events yet. Click **Step** or **Auto-run** to start the simulation."

    lines = []
    type_icons = {
        "action": "🎬",
        "conversation": "💬",
        "reflection": "💭",
        "movement": "🚶",
        "plan": "📋",
        "system": "⚙️",
        "probe": "🛰️",
    }
    for evt in reversed(events[-30:]):
        icon = type_icons.get(evt["type"], "•")
        line = f"**[{evt['time']}]** {icon} {evt['description']}"
        # Show conversation dialogue
        if evt["type"] == "conversation" and "dialogue" in evt.get("details", {}):
            for dl in evt["details"]["dialogue"][:4]:
                line += f"\n> {dl}"
        if evt["type"] == "probe" and "steps" in evt.get("details", {}):
            for step in evt["details"]["steps"][:3]:
                status = "ok" if step.get("ok") else f"error: {step.get('error')}"
                line += f"\n> {step.get('tool')}({step.get('arguments')}) → {status}"
        lines.append(line)

    return "\n\n".join(lines)


# ── Agent detail panel ──────────────────────────────────────────────

def format_agent_detail(agent_name: str) -> str:
    """Get formatted detail for the selected agent."""
    s = get_sim()
    detail = s.get_agent_detail(agent_name)
    if not detail:
        return "Select an agent to see details."

    lines = [
        f"# {detail['emoji']} {detail['name']}",
        f"**Age:** {detail['age']} | **Occupation:** {detail['occupation']}",
        f"**Location:** {detail['location']}",
        f"**Current Activity:** {detail['action']}",
        "",
        f"**Personality:** {detail['personality']}",
        "",
        "---",
        "### Today's Plan",
    ]
    if detail["plan"]:
        for item in detail["plan"]:
            lines.append(f"- {item}")
    else:
        lines.append("- No plan yet")

    lines.extend(["", "---", "### Recent Memories"])
    if detail["recent_memories"]:
        for m in detail["recent_memories"][:10]:
            lines.append(f"- {m}")
    else:
        lines.append("- No memories yet")

    if detail["reflections"]:
        lines.extend(["", "---", "### Reflections"])
        for r in detail["reflections"]:
            lines.append(f"- 💭 {r}")

    lines.extend(["", "---", "### Active Probe"])
    probe_summary = detail.get("probe_summary") or "No probe data."
    lines.append(f"- {probe_summary}")
    probe_steps = detail.get("probe_steps") or []
    for step in probe_steps[:6]:
        status = "ok" if step.get("ok") else f"error: {step.get('error')}"
        lines.append(f"- {step.get('tool')}({step.get('arguments')}) → {status}")

    lines.extend(["", "---", "### Relationships"])
    for name, desc in detail["relationships"].items():
        lines.append(f"- **{name}:** {desc}")

    lines.append(f"\n*Total memories: {detail['memory_count']}*")

    return "\n".join(lines)


# ── Control handlers ────────────────────────────────────────────────

def do_step() -> Tuple[str, str, str, str]:
    """Execute one simulation step."""
    s = get_sim()
    s.step()
    state = s.get_state()
    time_display = f"**{state['time']}**"
    return render_map(state), format_events(state), time_display, ""


def do_reset() -> Tuple[str, str, str, str]:
    """Reset the simulation."""
    global sim, auto_running
    auto_running = False
    sim = TownSimulation()
    sim.initialize()
    state = sim.get_state()
    time_display = f"**{state['time']}**"
    return (
        render_map(state),
        format_events(state),
        time_display,
        "Simulation reset.",
    )


def do_auto_toggle(current_label: str) -> Tuple[str, str, str, str, str]:
    """Toggle auto-run mode."""
    global auto_running
    auto_running = not auto_running

    s = get_sim()
    state = s.get_state()
    new_label = "Stop ⏹" if auto_running else "Auto-run ▶▶"
    status = "Auto-run started..." if auto_running else "Auto-run stopped."
    return (
        render_map(state),
        format_events(state),
        f"**{state['time']}**",
        status,
        new_label,
    )


def auto_step_tick(selected_agent: str) -> Tuple[str, str, str, str]:
    """Called periodically during auto-run."""
    global auto_running
    if not auto_running:
        s = get_sim()
        state = s.get_state()
        return (
            render_map(state),
            format_events(state),
            f"**{state['time']}**",
            format_agent_detail(selected_agent) if selected_agent else "",
        )

    s = get_sim()
    s.step()
    state = s.get_state()
    return (
        render_map(state),
        format_events(state),
        f"**{state['time']}**",
        format_agent_detail(selected_agent) if selected_agent else "",
    )


def on_agent_select(agent_name: str) -> str:
    """Handle agent selection."""
    if not agent_name:
        return "Select an agent to see details."
    return format_agent_detail(agent_name)


def update_probe_settings(enabled: bool, steps: int) -> str:
    s = get_sim()
    s.update_probe_settings(enabled, steps)
    status = "enabled" if enabled else "disabled"
    return f"Active probe {status} (max steps: {steps})."


# ── Build Gradio UI ─────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    """Create the Gradio Blocks interface."""
    agent_names = [p.name for p in get_sim().agents]

    with gr.Blocks(
        title="AURA Town",
        theme=gr.themes.Soft(),
        css="""
        .event-log { max-height: 400px; overflow-y: auto; }
        .map-container { display: flex; justify-content: center; }
        """,
    ) as app:
        gr.Markdown(
            "# 🏘️ AURA Town — Multi-Agent Simulation\n"
            "*Powered by AURA pipeline + GPT-4o-mini*"
        )

        with gr.Row():
            # ── Left: Town Map ──────────────────────────────────
            with gr.Column(scale=3):
                map_display = gr.HTML(
                    value=render_map(get_sim().get_state()),
                    label="Town Map",
                    elem_classes=["map-container"],
                )

            # ── Right: Agent Panel ──────────────────────────────
            with gr.Column(scale=2):
                agent_selector = gr.Dropdown(
                    choices=agent_names,
                    value=agent_names[0],
                    label="Select Agent",
                    interactive=True,
                )
                agent_detail = gr.Markdown(
                    value=format_agent_detail(agent_names[0]),
                    label="Agent Details",
                )

        # ── Event Log ───────────────────────────────────────────
        gr.Markdown("### Event Log")
        event_log = gr.Markdown(
            value=format_events(get_sim().get_state()),
            elem_classes=["event-log"],
        )

        # ── Controls ────────────────────────────────────────────
        with gr.Row():
            step_btn = gr.Button("Step ▶", variant="primary")
            auto_btn = gr.Button("Auto-run ▶▶", variant="secondary")
            reset_btn = gr.Button("Reset 🔄", variant="stop")
            time_display = gr.Markdown(
                value=f"**{get_sim().get_state()['time']}**"
            )
            status_display = gr.Markdown(value="Ready.")

        with gr.Row():
            probe_toggle = gr.Checkbox(
                value=get_sim().config.probe_enabled,
                label="Active Probe",
            )
            probe_steps = gr.Slider(
                minimum=0,
                maximum=4,
                step=1,
                value=get_sim().config.probe_max_steps,
                label="Probe Steps",
            )
        gr.Markdown(
            "_Active Probe triggers extra tool-calling LLM requests per agent._",
        )

        # Auto-run timer
        auto_timer = gr.Timer(value=3, active=False)

        # ── Event wiring ────────────────────────────────────────

        step_btn.click(
            fn=do_step,
            outputs=[map_display, event_log, time_display, status_display],
        )

        reset_btn.click(
            fn=do_reset,
            outputs=[map_display, event_log, time_display, status_display],
        )

        auto_btn.click(
            fn=do_auto_toggle,
            inputs=[auto_btn],
            outputs=[map_display, event_log, time_display, status_display, auto_btn],
        ).then(
            fn=lambda: gr.Timer(active=auto_running),
            outputs=[auto_timer],
        )

        auto_timer.tick(
            fn=auto_step_tick,
            inputs=[agent_selector],
            outputs=[map_display, event_log, time_display, agent_detail],
        )

        agent_selector.change(
            fn=on_agent_select,
            inputs=[agent_selector],
            outputs=[agent_detail],
        )

        probe_toggle.change(
            fn=update_probe_settings,
            inputs=[probe_toggle, probe_steps],
            outputs=[status_display],
        )
        probe_steps.change(
            fn=update_probe_settings,
            inputs=[probe_toggle, probe_steps],
            outputs=[status_display],
        )

        # Also refresh agent detail on step
        step_btn.click(
            fn=lambda sel: format_agent_detail(sel) if sel else "",
            inputs=[agent_selector],
            outputs=[agent_detail],
        )

    return app


# ── Entry point ─────────────────────────────────────────────────────

def main() -> None:
    app = create_app()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
