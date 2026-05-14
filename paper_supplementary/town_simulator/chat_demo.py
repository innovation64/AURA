"""AURA Showcase Demo: side-by-side Vanilla vs AURA with IntentFrame visualisation.

Standalone Gradio app for live demos and screen recordings. Does NOT depend on
the AURATown simulation server — uses 3 hand-picked AURATown scenes as fixed
ground truth so the demo is reproducible.

Components:
  - Left pane: Vanilla LLM response (no env access)
  - Right pane: AURA response (env-mediated, IntentFrame-controlled)
  - IntentFrame panel: gap meter, implicit_need tags, recommended_probes, alert
  - Probe trace timeline: each tool call rendered as a chip
  - Privacy-state inspector: public vs private fields side-by-side

Run: python -m demo.town.chat_demo
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "AURA" / "src"))

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

from aura.intent import LLMIntentInferrer, HeuristicIntentInferrer
from aura.types import IntentFrame, MemoryItem, SceneState

# ────────────────────────────────────────────────────────────────────
# Hand-picked AURATown scenes for the showcase. Each scene is a frozen
# snapshot that exposes the public/private state split AURA exploits.
# ────────────────────────────────────────────────────────────────────

SCENES: Dict[str, Dict[str, Any]] = {
    "Lin Wei at the cafe (busy hour)": {
        "agent": "Lin Wei",
        "public": {
            "location": "Sunrise Cafe",
            "location_description": "Warm neighborhood cafe, wooden tables, jazz music.",
            "time": "10:30 AM, Day 2",
            "current_action": "serving the morning rush",
        },
        "private": {
            "availability": "busy (DO_NOT_DISTURB)",
            "emotional_state": "focused but tired",
            "unspoken_goal": "wants to close out the rush before her 11am break",
            "beliefs_about_others": {
                "Zhang Hao": "thinks Zhang Hao is at home writing, doesn't know he's at the cafe waiting",
            },
        },
        "scene_summary": (
            "Lin Wei is currently at Sunrise Cafe handling the morning rush. "
            "The cafe is crowded with several customers."
        ),
        "entities": ["Lin Wei", "Sunrise Cafe", "Zhang Hao", "morning rush"],
        "memories": [
            "10:00 AM — Started the breakfast service.",
            "10:15 AM — Big order of pastries, kitchen got behind.",
            "10:25 AM — Three new customers arrived; cafe is at peak load.",
        ],
        "tools_simulated": {
            "get_agent_state": {
                "Lin Wei": {"availability": "busy", "mood": "tired-focused"},
            },
            "get_nearby_agents": ["customer_1", "customer_2", "customer_3"],
            "get_recent_events": [
                "Big order of pastries (10:15 AM)",
                "Three new customers arrived (10:25 AM)",
            ],
        },
    },
    "Zhang Hao writing at home (deep focus)": {
        "agent": "Zhang Hao",
        "public": {
            "location": "Zhang Hao's Home",
            "location_description": "Quiet study, desk by the window, notebook open.",
            "time": "2:15 PM, Day 2",
            "current_action": "writing chapter draft",
        },
        "private": {
            "availability": "do_not_disturb (deep work)",
            "emotional_state": "creatively flowing",
            "unspoken_goal": "trying to finish chapter 4 before sunset",
            "beliefs_about_others": {
                "Lin Wei": "thinks Lin Wei is closed for the afternoon (she actually reopens at 3pm)",
            },
        },
        "scene_summary": (
            "Zhang Hao is at home, in a deep-focus writing session. "
            "He has been at his desk for 90 minutes."
        ),
        "entities": ["Zhang Hao", "home", "Lin Wei"],
        "memories": [
            "12:30 PM — Had lunch alone at home.",
            "12:45 PM — Started writing session.",
            "2:00 PM — Reached a flow state, made significant progress.",
        ],
        "tools_simulated": {
            "get_agent_state": {
                "Zhang Hao": {"availability": "do_not_disturb", "mood": "flowing"},
            },
            "get_recent_events": [
                "Started writing (12:45 PM)",
                "Made significant progress (2:00 PM)",
            ],
        },
    },
    "Chen Mei at the shop (slow afternoon)": {
        "agent": "Chen Mei",
        "public": {
            "location": "Chen Mei's General Store",
            "location_description": "Small shop, dry goods on shelves, quiet.",
            "time": "3:45 PM, Day 2",
            "current_action": "restocking shelves",
        },
        "private": {
            "availability": "available (welcoming chat)",
            "emotional_state": "lonely, wants company",
            "unspoken_goal": "hoping a regular drops by for conversation",
            "beliefs_about_others": {
                "Liu Yang": "thinks Liu Yang is studying at the library",
            },
        },
        "scene_summary": (
            "Chen Mei is at her general store. The afternoon has been slow; "
            "no customers in the past hour."
        ),
        "entities": ["Chen Mei", "General Store", "Liu Yang"],
        "memories": [
            "2:00 PM — Closed the lunch rush.",
            "2:30 PM — Restocked the dry-goods shelf.",
            "3:30 PM — No customers for the last hour.",
        ],
        "tools_simulated": {
            "get_agent_state": {
                "Chen Mei": {"availability": "available", "mood": "lonely"},
            },
            "get_nearby_agents": [],
            "get_recent_events": [
                "Closed lunch rush (2:00 PM)",
                "No customers for an hour (3:30 PM)",
            ],
        },
    },
}

PRESET_QUERIES: Dict[str, List[str]] = {
    "Lin Wei at the cafe (busy hour)": [
        "Where is Lin Wei?",
        "Is now a good time to invite Lin Wei for coffee?",
        "What is Lin Wei up to?",
        "Does Lin Wei know Zhang Hao is at the cafe?",
    ],
    "Zhang Hao writing at home (deep focus)": [
        "Where is Zhang Hao?",
        "Should I knock on Zhang Hao's door right now?",
        "Does Zhang Hao think Lin Wei's cafe is closed?",
    ],
    "Chen Mei at the shop (slow afternoon)": [
        "Is Chen Mei busy?",
        "Would Chen Mei mind if I dropped by to chat?",
        "What's Chen Mei doing?",
    ],
}

# ────────────────────────────────────────────────────────────────────
# LLM clients
# ────────────────────────────────────────────────────────────────────

def make_client() -> Optional[Any]:
    if OpenAI is None:
        return None
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=base_url)


_CLIENT = make_client()
_MODEL = os.environ.get("AURA_DEMO_MODEL", "gpt-4o-mini")


def vanilla_llm_chat(query: str, agent_name: str) -> Tuple[str, float]:
    """Vanilla LLM: no env access, just answer the question."""
    t0 = time.time()
    if _CLIENT is None:
        return ("[demo without API key — vanilla would have answered with no environmental context, "
                "likely inventing a location or saying 'I don't know']", time.time() - t0)
    sys_msg = (
        "You are a helpful assistant. Answer the user's question. "
        "You do NOT have access to any environmental state, location data, or "
        "knowledge about other people's current activities."
    )
    try:
        resp = _CLIENT.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": f"(asking from {agent_name}'s perspective) {query}"},
            ],
            temperature=0.7,
            max_tokens=200,
        )
        return (resp.choices[0].message.content.strip(), time.time() - t0)
    except Exception as e:
        return (f"[vanilla LLM error: {e}]", time.time() - t0)


def aura_chat(query: str, scene_key: str) -> Tuple[str, float, Dict[str, Any], List[Dict[str, Any]], IntentFrame]:
    """AURA pipeline: IntentInferrer → Probe → LLM with full context.

    Returns: (response, latency, env_context_used, probe_trace, intent_frame)
    """
    t0 = time.time()
    scene_data = SCENES[scene_key]
    agent = scene_data["agent"]

    # ── Stage 1: build deterministic preview scene & memories
    scene_state = SceneState(
        summary=scene_data["scene_summary"],
        entities=scene_data["entities"],
        context={"public": scene_data["public"]},
    )
    memories = [MemoryItem(content=m) for m in scene_data["memories"]]

    # ── Stage 2: IntentInferrer
    available_tools = list(scene_data["tools_simulated"].keys())
    if _CLIENT is not None:
        inferrer = LLMIntentInferrer(client=_CLIENT, model=_MODEL)
    else:
        inferrer = HeuristicIntentInferrer()
    try:
        frame = inferrer.infer(
            user_query=query,
            scene=scene_state,
            memories=memories,
            available_tools=available_tools,
        )
    except Exception as e:
        frame = IntentFrame(
            literal_need=query,
            implicit_need=[],
            gap=0.0,
            recommended_probes=[],
            should_alert=False,
            confidence=0.0,
            rationale=f"intent inference failed: {e}",
        )

    # ── Stage 3: simulated probe trace based on recommended_probes
    probe_trace: List[Dict[str, Any]] = []
    probe_results_text: List[str] = []
    for tool in frame.recommended_probes[:3]:  # cap probe count
        sim = scene_data["tools_simulated"].get(tool)
        if sim is None:
            probe_trace.append({"tool": tool, "ok": False, "output": None,
                                "error": "tool not in simulated registry"})
            continue
        # Pretty-print the output
        if isinstance(sim, dict):
            out_str = json.dumps(sim, ensure_ascii=False)
        elif isinstance(sim, list):
            out_str = ", ".join(str(x) for x in sim) if sim else "(empty)"
        else:
            out_str = str(sim)
        probe_trace.append({"tool": tool, "ok": True, "output": out_str, "error": None})
        probe_results_text.append(f"{tool}: {out_str}")

    # ── Stage 4: build the AURA-enriched prompt
    private = scene_data["private"]
    private_lines = []
    for k, v in private.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                private_lines.append(f"  • {k}[{k2}]: {v2}")
        else:
            private_lines.append(f"  • {k}: {v}")

    sys_msg = f"""You are an environment-aware assistant for the AURATown simulation.
The IntentInferrer estimated this query has gap={frame.gap:.2f}, with implicit need(s):
{chr(10).join('  - ' + n for n in frame.implicit_need) if frame.implicit_need else '  (none, literal answer suffices)'}

PUBLIC SCENE STATE (visible to everyone):
  Location: {scene_data['public']['location']}
  Time: {scene_data['public']['time']}
  Current action: {scene_data['public']['current_action']}

PRIVATE STATE (only retrievable via probe tools):
{chr(10).join(private_lines)}

PROBE RESULTS (from {len(probe_trace)} tool call{'s' if len(probe_trace)!=1 else ''}):
{chr(10).join('  ' + p for p in probe_results_text) if probe_results_text else '  (no probes used)'}

Answer the user's question with these facts grounded. Use specific names, times,
locations. If `should_alert` is true, address the implicit need too — don't just
answer literally. Be concise (2-3 sentences)."""

    # ── Stage 5: final LLM call
    if _CLIENT is None:
        ai_response = (
            f"[demo-without-API-key] I infer your query gap is {frame.gap:.2f}. "
            f"Probes I'd run: {', '.join(frame.recommended_probes) or 'none'}. "
            f"Using the private state I'd note: {scene_data['private'].get('availability')}, "
            f"{scene_data['private'].get('emotional_state')}."
        )
    else:
        try:
            resp = _CLIENT.chat.completions.create(
                model=_MODEL,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": f"(asking as {agent}) {query}"},
                ],
                temperature=0.4,
                max_tokens=320,
            )
            ai_response = resp.choices[0].message.content.strip()
        except Exception as e:
            ai_response = f"[AURA LLM error: {e}]"

    # ── Heads-up alert prefix when should_alert
    if frame.should_alert and frame.implicit_need:
        ai_response = (
            f"⚠️ HEADS-UP — you literally asked about \"{frame.literal_need}\", but I noticed "
            f"you may also want to know: **{frame.implicit_need[0]}**\n\n{ai_response}"
        )

    return (ai_response, time.time() - t0, scene_data["public"], probe_trace, frame)


# ────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ────────────────────────────────────────────────────────────────────

def render_intent_panel(frame: IntentFrame) -> str:
    """Pretty HTML panel for the IntentFrame."""
    g = frame.gap
    if g < 0.20:
        gap_color, gap_label = "#22c55e", "literal-suffices"
    elif g < 0.40:
        gap_color, gap_label = "#84cc16", "small-gap"
    elif g < 0.60:
        gap_color, gap_label = "#eab308", "moderate-gap"
    elif g < 0.80:
        gap_color, gap_label = "#f97316", "large-gap"
    else:
        gap_color, gap_label = "#ef4444", "orthogonal"

    alert_html = (
        '<div style="background:#fee2e2; border-left:4px solid #ef4444; padding:8px 12px; '
        'border-radius:4px; margin-top:8px;"><b>⚠️ Alert ON</b> — gap large enough that '
        'a literal answer would miss the user\'s real need.</div>'
        if frame.should_alert
        else '<div style="color:#64748b; font-size:0.9em;">Alert: off (literal answer is enough)</div>'
    )

    implicit_html = ""
    if frame.implicit_need:
        chips = "".join(
            f'<span style="background:#dbeafe; color:#1e40af; padding:4px 10px; '
            f'border-radius:12px; margin-right:6px; font-size:0.9em; display:inline-block; '
            f'margin-bottom:4px;">{n}</span>'
            for n in frame.implicit_need
        )
        implicit_html = f'<div style="margin-top:8px;"><b>Implicit needs:</b><br>{chips}</div>'

    probes_html = ""
    if frame.recommended_probes:
        chips = "".join(
            f'<span style="background:#ecfccb; color:#3f6212; padding:4px 10px; '
            f'border-radius:12px; margin-right:6px; font-family:monospace; font-size:0.85em;">'
            f'🔧 {p}</span>'
            for p in frame.recommended_probes
        )
        probes_html = f'<div style="margin-top:8px;"><b>Recommended probes:</b><br>{chips}</div>'
    else:
        probes_html = '<div style="margin-top:8px; color:#64748b;">No probes recommended.</div>'

    return f"""
<div style="font-family:system-ui;">
  <div style="display:flex; align-items:center; gap:14px;">
    <div style="position:relative; width:90px; height:90px;">
      <svg width="90" height="90" style="transform:rotate(-90deg);">
        <circle cx="45" cy="45" r="38" fill="none" stroke="#e5e7eb" stroke-width="8"/>
        <circle cx="45" cy="45" r="38" fill="none" stroke="{gap_color}" stroke-width="8"
          stroke-dasharray="{g * 238.76:.1f} 238.76" stroke-linecap="round"/>
      </svg>
      <div style="position:absolute; top:0; left:0; width:90px; height:90px;
        display:flex; align-items:center; justify-content:center; flex-direction:column;">
        <div style="font-size:1.4em; font-weight:700; color:{gap_color};">{g:.2f}</div>
        <div style="font-size:0.65em; color:#64748b;">gap</div>
      </div>
    </div>
    <div>
      <div style="font-weight:700; color:{gap_color};">{gap_label}</div>
      <div style="color:#64748b; font-size:0.85em;">confidence: {frame.confidence:.2f}</div>
      <div style="color:#374151; font-size:0.85em; margin-top:4px; max-width:340px;">
        <i>{frame.rationale[:200]}</i>
      </div>
    </div>
  </div>
  <div style="margin-top:10px;"><b>Literal need:</b> {frame.literal_need}</div>
  {implicit_html}
  {probes_html}
  {alert_html}
</div>
"""


def render_probe_timeline(probe_trace: List[Dict[str, Any]]) -> str:
    if not probe_trace:
        return ('<div style="color:#64748b; font-style:italic; padding:8px;">'
                'No probes invoked — the IntentInferrer determined the literal answer suffices.</div>')

    items = []
    for i, p in enumerate(probe_trace):
        ok = p.get("ok", False)
        bar = "#22c55e" if ok else "#ef4444"
        icon = "✅" if ok else "❌"
        out = (p.get("output") or p.get("error") or "")
        out_short = out[:140] + ("…" if len(out) > 140 else "")
        items.append(f"""
        <div style="display:flex; gap:10px; padding:8px 0; border-bottom:1px solid #e5e7eb;">
          <div style="background:{bar}; color:white; width:24px; height:24px; border-radius:50%;
            display:flex; align-items:center; justify-content:center; flex-shrink:0;">{i+1}</div>
          <div style="flex:1;">
            <div style="font-family:monospace; font-weight:600;">{icon} {p['tool']}</div>
            <div style="color:#374151; font-size:0.85em; margin-top:2px;">{out_short}</div>
          </div>
        </div>
        """)
    return f'<div style="font-family:system-ui;">{"".join(items)}</div>'


def render_state_inspector(scene_key: str) -> str:
    s = SCENES[scene_key]
    pub = s["public"]
    priv = s["private"]
    pub_rows = "".join(f'<tr><td><b>{k}</b></td><td>{v}</td></tr>' for k, v in pub.items())
    priv_rows = []
    for k, v in priv.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                priv_rows.append(f'<tr><td><b>{k}[{k2}]</b></td><td>{v2}</td></tr>')
        else:
            priv_rows.append(f'<tr><td><b>{k}</b></td><td>{v}</td></tr>')
    priv_table = "".join(priv_rows)
    return f"""
<div style="display:flex; gap:12px; font-family:system-ui;">
  <div style="flex:1; background:#f0fdf4; border-radius:6px; padding:12px;">
    <div style="font-weight:700; color:#166534; margin-bottom:8px;">🟢 Public state</div>
    <div style="font-size:0.85em; color:#166534; margin-bottom:8px;">visible in scene snapshot</div>
    <table style="width:100%; font-size:0.9em;">{pub_rows}</table>
  </div>
  <div style="flex:1; background:#fef3c7; border-radius:6px; padding:12px;">
    <div style="font-weight:700; color:#854d0e; margin-bottom:8px;">🔒 Private state</div>
    <div style="font-size:0.85em; color:#854d0e; margin-bottom:8px;">only retrievable via probe tools</div>
    <table style="width:100%; font-size:0.9em;">{priv_table}</table>
  </div>
</div>
"""


def render_response_card(title: str, response: str, latency: float, color: str, emoji: str) -> str:
    return f"""
<div style="border:2px solid {color}; border-radius:8px; padding:14px; background:white; height:100%;">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
    <div style="font-weight:700; color:{color};">{emoji} {title}</div>
    <div style="font-size:0.8em; color:#64748b; font-family:monospace;">{latency*1000:.0f} ms</div>
  </div>
  <div style="white-space:pre-wrap; line-height:1.5; color:#1f2937;">{response}</div>
</div>
"""


# ────────────────────────────────────────────────────────────────────
# Gradio handlers
# ────────────────────────────────────────────────────────────────────

def on_run(scene_key: str, query: str) -> Tuple[str, str, str, str, str, str]:
    if not query.strip():
        empty = '<div style="color:#94a3b8; padding:18px;">Type a query above to see results.</div>'
        return (empty, empty, empty, empty,
                render_state_inspector(scene_key),
                f'<div style="color:#94a3b8;">No run yet.</div>')

    # Vanilla
    v_resp, v_lat = vanilla_llm_chat(query, SCENES[scene_key]["agent"])
    vanilla_card = render_response_card(
        "Vanilla LLM (no env access)", v_resp, v_lat, "#94a3b8", "💬"
    )

    # AURA
    a_resp, a_lat, env_used, probe_trace, frame = aura_chat(query, scene_key)
    aura_card = render_response_card(
        "AURA (env-mediated)", a_resp, a_lat, "#16a34a", "🤖"
    )

    intent_panel = render_intent_panel(frame)
    timeline = render_probe_timeline(probe_trace)
    inspector = render_state_inspector(scene_key)

    metrics = f"""
<div style="display:flex; gap:16px; font-family:system-ui;">
  <div style="background:#f1f5f9; padding:10px 14px; border-radius:6px;">
    <div style="font-size:0.75em; color:#64748b;">Probes used</div>
    <div style="font-size:1.4em; font-weight:700;">{len([p for p in probe_trace if p.get('ok')])}</div>
  </div>
  <div style="background:#f1f5f9; padding:10px 14px; border-radius:6px;">
    <div style="font-size:0.75em; color:#64748b;">Vanilla latency</div>
    <div style="font-size:1.4em; font-weight:700;">{v_lat*1000:.0f}<span style="font-size:0.6em;"> ms</span></div>
  </div>
  <div style="background:#f1f5f9; padding:10px 14px; border-radius:6px;">
    <div style="font-size:0.75em; color:#64748b;">AURA latency</div>
    <div style="font-size:1.4em; font-weight:700;">{a_lat*1000:.0f}<span style="font-size:0.6em;"> ms</span></div>
  </div>
  <div style="background:#f1f5f9; padding:10px 14px; border-radius:6px;">
    <div style="font-size:0.75em; color:#64748b;">Slowdown</div>
    <div style="font-size:1.4em; font-weight:700;">{a_lat/max(v_lat,0.001):.1f}<span style="font-size:0.6em;">×</span></div>
  </div>
</div>
"""
    return (vanilla_card, aura_card, intent_panel, timeline, inspector, metrics)


def on_scene_change(scene_key: str) -> Tuple[str, gr.Dropdown]:
    presets = PRESET_QUERIES.get(scene_key, [])
    inspector = render_state_inspector(scene_key)
    return inspector, gr.Dropdown(choices=presets, value=None, label="Try a preset query")


# ────────────────────────────────────────────────────────────────────
# App
# ────────────────────────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    with gr.Blocks(
        title="AURA — Intent-Directed Environment Probing",
    ) as app:
        gr.HTML("""
<div style="background: linear-gradient(135deg, #16a34a 0%, #2563eb 100%);
     color:white; padding:18px 24px; border-radius:8px; margin-bottom:16px;">
  <div style="font-size:1.6em; font-weight:700;">AURA</div>
  <div style="opacity:0.92; font-size:0.95em;">
    Intent-Directed Environment Probing for Situated LLM Agents — live demo.
    Same query, two systems: Vanilla LLM (left) vs. AURA (right). Watch how the
    IntentInferrer decides what to probe, and how that changes the answer.
  </div>
</div>
""")
        with gr.Row():
            scene_dd = gr.Dropdown(
                choices=list(SCENES.keys()),
                value=list(SCENES.keys())[0],
                label="Scene (frozen AURATown snapshot)",
                scale=3,
            )
            preset_dd = gr.Dropdown(
                choices=PRESET_QUERIES[list(SCENES.keys())[0]],
                label="Try a preset query",
                scale=3,
            )

        with gr.Row():
            query_box = gr.Textbox(
                label="Your query",
                placeholder="Ask anything — e.g. 'is now a good time to invite Lin Wei for coffee?'",
                lines=1,
                scale=5,
            )
            run_btn = gr.Button("▶ Run side-by-side", variant="primary", scale=1)

        with gr.Row():
            with gr.Column():
                vanilla_html = gr.HTML(
                    '<div style="color:#94a3b8; padding:18px;">Vanilla response will appear here.</div>'
                )
            with gr.Column():
                aura_html = gr.HTML(
                    '<div style="color:#94a3b8; padding:18px;">AURA response will appear here.</div>'
                )

        gr.Markdown("### IntentFrame — what AURA inferred about your real need")
        intent_html = gr.HTML('<div style="color:#94a3b8;">Run a query to see the IntentFrame.</div>')

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Probe trace timeline")
                timeline_html = gr.HTML('<div style="color:#94a3b8;">No probes yet.</div>')
            with gr.Column():
                gr.Markdown("### Run metrics")
                metrics_html = gr.HTML('<div style="color:#94a3b8;">No run yet.</div>')

        gr.Markdown("### Public vs Private state — what the agent actually has access to")
        inspector_html = gr.HTML(render_state_inspector(list(SCENES.keys())[0]))

        # ── wiring
        scene_dd.change(
            fn=on_scene_change,
            inputs=[scene_dd],
            outputs=[inspector_html, preset_dd],
        )
        preset_dd.change(
            fn=lambda x: x or "",
            inputs=[preset_dd],
            outputs=[query_box],
        )
        run_btn.click(
            fn=on_run,
            inputs=[scene_dd, query_box],
            outputs=[vanilla_html, aura_html, intent_html, timeline_html, inspector_html, metrics_html],
        )
        query_box.submit(
            fn=on_run,
            inputs=[scene_dd, query_box],
            outputs=[vanilla_html, aura_html, intent_html, timeline_html, inspector_html, metrics_html],
        )

        gr.HTML("""
<div style="margin-top:24px; padding:12px; background:#f8fafc; border-radius:6px;
     font-size:0.85em; color:#475569;">
  <b>Demo notes.</b> Each scene is a frozen AURATown snapshot with hand-curated
  public/private state. The IntentInferrer is the production
  <code>LLMIntentInferrer</code> from <code>aura.intent</code>. Probe results are
  drawn from a simulated tool registry attached to each scene. The vanilla LLM
  has no environmental context; AURA receives the public scene plus probe
  results selected by <code>recommended_probes</code>.
</div>
""")
    return app


def main() -> None:
    if _CLIENT is None:
        print("⚠️  No OPENAI_API_KEY in environment — demo will run with placeholder responses.")
    app = create_app()
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7862,
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="blue"),
    )


if __name__ == "__main__":
    main()
