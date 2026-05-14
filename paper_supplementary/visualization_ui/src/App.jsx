import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import TownMap from "./components/TownMap.jsx";
import WorldMap from "./components/WorldMap.jsx";
import MapGeneratorPanel from "./components/MapGeneratorPanel.jsx";
import AssetManager from "./components/AssetManager.jsx";

/* ── API ─────────────────────────────────────── */
const API = {
  state: () => fetch("/api/state").then((r) => r.json()),
  step: () => fetch("/api/step", { method: "POST" }).then((r) => r.json()),
  reset: () => fetch("/api/reset", { method: "POST" }).then((r) => r.json()),
  probe: (payload) =>
    fetch("/api/probe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then((r) => r.json()),
  chat: (user, message) =>
    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user, message }),
    }).then((r) => r.json()),
  agentControl: (agent, controlled) =>
    fetch("/api/agent/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agent, controlled }),
    }).then((r) => r.json()),
  agentAction: (agent, action, location, emoji) =>
    fetch("/api/agent/action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agent, action, location, emoji }),
    }).then((r) => r.json()),
  agentInteract: (agent, target) =>
    fetch("/api/agent/interact", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agent, target }),
    }).then((r) => r.json()),
  agentMove: (agent, direction, steps = 2) =>
    fetch("/api/agent/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agent, direction, steps }),
    }).then((r) => r.json()),
  agentExplore: (agent, direction) =>
    fetch("/api/agent/explore", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agent, direction }),
    }).then((r) => r.json()),
  locationDetail: (name) =>
    fetch(`/api/location?name=${encodeURIComponent(name)}`).then((r) => r.json()),
  worldmap: () => fetch("/api/worldmap").then((r) => r.json()),
  teleport: (agent, region_id) =>
    fetch("/api/worldmap/teleport", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agent, region_id }),
    }).then((r) => r.json()),
};

const EVT_ICONS = {
  action: "\u{1F3AC}",
  conversation: "\u{1F4AC}",
  reflection: "\u{1F4AD}",
  movement: "\u{1F6B6}",
  plan: "\u{1F4CB}",
  system: "\u2699\uFE0F",
  probe: "\u{1F6F0}\uFE0F",
  chat: "\u{1F916}",
  weather_change: "\u{1F326}\uFE0F",
  season_change: "\u{1F343}",
  micro_event: "\u2728",
  evolution: "\u{1F331}",
  popularity_upgrade: "\u{2B50}",
  exploration: "\u{1F9ED}",
};

const WEATHER_ICONS = {
  clear: "\u2600\uFE0F",
  partly_cloudy: "\u26C5",
  cloudy: "\u2601\uFE0F",
  rain: "\u{1F327}\uFE0F",
  storm: "\u26C8\uFE0F",
  windy: "\u{1F32C}\uFE0F",
  fog: "\u{1F32B}\uFE0F",
  snow: "\u{1F328}\uFE0F",
  blizzard: "\u{1F328}\uFE0F",
};

const SEASON_ICONS = {
  spring: "\u{1F338}",
  summer: "\u{1F33B}",
  autumn: "\u{1F341}",
  winter: "\u{2744}\uFE0F",
};

function truncate(s, n) {
  return s && s.length > n ? s.slice(0, n) + "\u2026" : s;
}

/* ── App ─────────────────────────────────────── */
export default function App() {
  const [state, setState] = useState(null);
  const [activeUser, setActiveUser] = useState(null);
  const [autoRun, setAutoRun] = useState(false);
  const [stepSpeed, setStepSpeed] = useState(3); // seconds between steps
  const [probeOn, setProbeOn] = useState(true);
  const [probeMax, setProbeMax] = useState(2);

  // Chat state
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

  // Player control state
  const [controlActionInput, setControlActionInput] = useState("");
  const [controlLocationSelect, setControlLocationSelect] = useState("");
  const [interactLoading, setInteractLoading] = useState(false);

  // Explore & location detail state
  const [exploreDir, setExploreDir] = useState("");
  const [locationDetail, setLocationDetail] = useState(null);

  // View state
  const [centerView, setCenterView] = useState("chat");
  const [expandedEnv, setExpandedEnv] = useState({});

  // World map state (Phase 2B)
  const [mapLevel, setMapLevel] = useState("region"); // "world" | "region"
  const [activeRegion, setActiveRegion] = useState("town_center");

  const chatEndRef = useRef(null);
  const eventEndRef = useRef(null);

  // Initial fetch
  useEffect(() => {
    API.state()
      .then((r) => {
        if (r.ok) {
          setState(r.state);
          if (r.state.agents.length) setActiveUser(r.state.agents[0].name);
        }
      })
      .catch(() => {});
  }, []);

  // Sync probe settings
  useEffect(() => {
    if (!state) return;
    setProbeOn(state.probe_enabled ?? true);
    setProbeMax(state.probe_max_steps ?? 2);
  }, [state]);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages.length]);

  // Auto-run with configurable speed (skips when step is busy)
  useEffect(() => {
    if (!autoRun) return;
    const ms = stepSpeed * 1000;
    const id = setInterval(() => {
      API.step()
        .then((r) => {
          if (r.ok && !r.busy) setState(r.state);
          // busy → skip silently, next interval will retry
        })
        .catch(() => setAutoRun(false));
    }, ms);
    return () => clearInterval(id);
  }, [autoRun, stepSpeed]);

  const doStep = useCallback(() => {
    API.step().then((r) => {
      if (r.ok) setState(r.state);
    });
  }, []);

  const doReset = useCallback(() => {
    setAutoRun(false);
    setChatMessages([]);
    API.reset().then((r) => {
      if (r.ok) {
        setState(r.state);
        setActiveUser(r.state.agents[0]?.name || null);
      }
    });
  }, []);

  const updateProbe = useCallback((on, steps) => {
    API.probe({ enabled: on, max_steps: steps }).then((r) => {
      if (r.ok) setState(r.state);
    });
  }, []);

  // Track whether auto-run was on before chat started
  const autoRunBeforeChatRef = useRef(false);

  // Send chat message — pauses auto-run to free up API bandwidth
  const sendChat = useCallback(() => {
    if (!chatInput.trim() || !activeUser || chatLoading) return;
    const msg = chatInput.trim();
    setChatInput("");
    setChatLoading(true);

    // Pause auto-run while chatting so LLM calls don't compete
    autoRunBeforeChatRef.current = autoRun;
    if (autoRun) setAutoRun(false);

    const userMsg = {
      id: Date.now(),
      type: "user",
      user: activeUser,
      content: msg,
      timestamp: new Date().toLocaleTimeString(),
    };
    setChatMessages((prev) => [...prev, userMsg]);

    const thinkingId = Date.now() + 1;
    setChatMessages((prev) => [
      ...prev,
      {
        id: thinkingId,
        type: "env-thinking",
        content: "Environment Agent is gathering context\u2026",
      },
    ]);

    API.chat(activeUser, msg)
      .then((r) => {
        if (r.ok) {
          setState(r.state);
          const chat = r.chat;
          setChatMessages((prev) => {
            const filtered = prev.filter((m) => m.id !== thinkingId);
            return [
              ...filtered,
              {
                id: Date.now() + 2,
                type: "env-enrichment",
                context: chat.env_context,
                probe: chat.probe,
              },
              {
                id: Date.now() + 3,
                type: "ai",
                content: chat.ai_response,
                timestamp: new Date().toLocaleTimeString(),
              },
            ];
          });
        }
      })
      .catch((err) => {
        setChatMessages((prev) => {
          const filtered = prev.filter((m) => m.id !== thinkingId);
          return [
            ...filtered,
            {
              id: Date.now() + 4,
              type: "error",
              content: `Failed to get response: ${err.message}`,
            },
          ];
        });
      })
      .finally(() => {
        setChatLoading(false);
        // Resume auto-run if it was on before
        if (autoRunBeforeChatRef.current) setAutoRun(true);
      });
  }, [chatInput, activeUser, chatLoading, autoRun]);

  const activeAgent = useMemo(
    () => state?.agents?.find((a) => a.name === activeUser) || null,
    [state, activeUser]
  );
  const controlMode = activeAgent?.player_controlled || false;

  const nearbyAgents = useMemo(() => {
    if (!activeAgent || !state?.agents) return [];
    return state.agents.filter(
      (a) => a.name !== activeAgent.name && a.location === activeAgent.location && a.location !== "on the road"
    );
  }, [activeAgent, state?.agents]);

  const toggleControl = useCallback(() => {
    if (!activeUser) return;
    const newVal = !controlMode;
    API.agentControl(activeUser, newVal).then((r) => {
      if (r.ok) setState(r.state);
    });
  }, [activeUser, controlMode]);

  const sendControlAction = useCallback(() => {
    if (!controlActionInput.trim() || !activeUser) return;
    API.agentAction(activeUser, controlActionInput.trim(), "", "").then((r) => {
      if (r.ok) setState(r.state);
    });
    setControlActionInput("");
  }, [controlActionInput, activeUser]);

  const sendControlMove = useCallback(() => {
    if (!controlLocationSelect || !activeUser) return;
    API.agentAction(activeUser, "walking", controlLocationSelect, "").then((r) => {
      if (r.ok) setState(r.state);
    });
    setControlLocationSelect("");
  }, [controlLocationSelect, activeUser]);

  const handleControlMove = useCallback((locationName) => {
    if (!activeUser) return;
    API.agentAction(activeUser, "walking", locationName, "").then((r) => {
      if (r.ok) setState(r.state);
    });
  }, [activeUser]);

  const handleInteract = useCallback((targetName) => {
    if (!activeUser || interactLoading) return;
    setInteractLoading(true);
    API.agentInteract(activeUser, targetName)
      .then((r) => {
        if (r.ok) setState(r.state);
      })
      .finally(() => setInteractLoading(false));
  }, [activeUser, interactLoading]);

  const handleDpadMove = useCallback((direction) => {
    if (!activeUser) return;
    API.agentMove(activeUser, direction, 2).then((r) => {
      if (r.ok) setState(r.state);
    });
  }, [activeUser]);

  const sendExplore = useCallback(() => {
    if (!exploreDir || !activeUser) return;
    API.agentExplore(activeUser, exploreDir).then((r) => {
      if (r.ok) setState(r.state);
    });
    setExploreDir("");
  }, [exploreDir, activeUser]);

  const handleLocationClick = useCallback((locationName) => {
    API.locationDetail(locationName).then((r) => {
      if (r.ok) setLocationDetail(r.location);
    });
  }, []);

  // World map handlers
  const handleSelectRegion = useCallback((region) => {
    setActiveRegion(region.id);
    // Teleport active agent to region and switch to town map view
    if (activeUser) {
      API.teleport(activeUser, region.id).then((r) => {
        if (r.ok) {
          setState(r.state);
          setMapLevel("region");
          setCenterView("map");
        }
      });
    } else {
      setMapLevel("region");
      setCenterView("map");
    }
  }, [activeUser]);

  const handleMapGenerate = useCallback((result) => {
    if (result?.state) setState(result.state);
  }, []);

  const events = useMemo(
    () => (state?.events ? [...state.events].reverse() : []),
    [state?.events]
  );

  if (!state) {
    return (
      <div className="app loading-screen">
        <div className="loading-content">
          <div className="loading-icon">
            <span className="pulse-ring" />
            <span className="pulse-core">{"\u{1F310}"}</span>
          </div>
          <div className="loading-text">Connecting to AURA Environment\u2026</div>
          <div className="loading-sub">
            Make sure the API server is running on port 7861
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      {/* ── Header ──────────────────────────── */}
      <header className="header">
        <div className="header-left">
          <div className="header-logo">{"\u{1F310}"}</div>
          <div className="header-titles">
            <h1>AURA Environment Intelligence</h1>
            <span className="subtitle">
              Human &times; AI &times; Environment Agent
            </span>
          </div>
        </div>
        <div className="header-center">
          <span className="time-badge">{state.time}</span>
        </div>
        <div className="header-right">
          <div className="user-selector">
            <span className="user-selector-label">Playing as</span>
            <select
              value={activeUser || ""}
              onChange={(e) => setActiveUser(e.target.value)}
              className="user-select"
            >
              {state.agents.map((a) => (
                <option key={a.name} value={a.name}>
                  {a.emoji} {a.name}
                </option>
              ))}
            </select>
            <button
              className={`btn ${controlMode ? "btn-warn" : "btn-ctrl"}`}
              onClick={toggleControl}
              title={controlMode ? "Release agent to autonomous mode" : "Take direct control of this agent"}
            >
              {controlMode ? "\uD83C\uDFAE Release" : "\uD83C\uDFAE Control"}
            </button>
            {controlMode && <span className="ctrl-active-badge">CONTROLLED</span>}
          </div>
          <button className="btn btn-primary" onClick={doStep}>
            {"\u25B6"} Step
          </button>
          <button
            className={`btn ${autoRun ? "btn-warn" : ""}`}
            onClick={() => setAutoRun((v) => !v)}
          >
            {autoRun ? "\u23F9 Stop" : "\u23E9 Auto"}
          </button>
          <div className="speed-control">
            <input
              type="range"
              min="0.5"
              max="10"
              step="0.5"
              value={stepSpeed}
              onChange={(e) => setStepSpeed(Number(e.target.value))}
              className="speed-slider"
            />
            <span className="speed-label">{stepSpeed}s</span>
          </div>
          <button className="btn" onClick={doReset}>
            {"\u21BB"} Reset
          </button>
        </div>
      </header>

      {/* ── 3-Column Layout ─────────────────── */}
      <div className="main">
        {/* ── LEFT: Group Members ────────────── */}
        <aside className="sidebar">
          <div className="sidebar-header">
            <span>{"\u{1F465}"} Group Members</span>
            <span className="badge">{state.agents.length}</span>
          </div>
          <div className="agent-list">
            {state.agents.map((a) => (
              <div
                key={a.name}
                className={`agent-card ${activeUser === a.name ? "active" : ""}`}
                onClick={() => setActiveUser(a.name)}
              >
                <div className="agent-card-top">
                  <span className="agent-emoji">{a.emoji}</span>
                  <div className="agent-info">
                    <div className="agent-name">
                      {a.name}
                      {activeUser === a.name && (
                        <span className="you-badge">YOU</span>
                      )}
                      {a.player_controlled && (
                        <span className="ctrl-badge">{"\uD83C\uDFAE"} CTRL</span>
                      )}
                    </div>
                    <div className="agent-occupation">{a.occupation}</div>
                  </div>
                  <div className="agent-status-dot" />
                </div>
                <div className="agent-action">{a.action}</div>
                <div className="agent-location">
                  {"\u{1F4CD}"} {a.location}
                </div>
                {a.exploration_goal && (
                  <div className="agent-exploring">
                    {"\u{1F9ED}"} {a.exploration_goal.reason}
                  </div>
                )}
                {a.curiosity > 0.3 && (
                  <div className="agent-curiosity">
                    {"\u2728"} Curiosity: {Math.round(a.curiosity * 100)}%
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Activity Feed */}
          <div className="activity-feed">
            <div className="activity-header">Activity Feed</div>
            <div className="activity-list">
              {events.slice(0, 12).map((evt, i) => (
                <div className="activity-item" key={`${evt.time}-${i}`}>
                  <span className="activity-icon">
                    {EVT_ICONS[evt.type] || "\u2022"}
                  </span>
                  <span className="activity-text">
                    {truncate(evt.description, 55)}
                  </span>
                </div>
              ))}
              {events.length === 0 && (
                <div className="activity-empty">
                  No events yet. Click Step to begin.
                </div>
              )}
            </div>
          </div>
        </aside>

        {/* ── CENTER: Collaborative Space ────── */}
        <div className="center">
          <div className="center-tabs">
            <button
              className={`tab ${centerView === "chat" ? "active" : ""}`}
              onClick={() => setCenterView("chat")}
            >
              {"\u{1F4AC}"} Chat
            </button>
            <button
              className={`tab ${centerView === "map" ? "active" : ""}`}
              onClick={() => { setCenterView("map"); setMapLevel("region"); }}
            >
              {"\u{1F5FA}\uFE0F"} Map
            </button>
            <button
              className={`tab ${centerView === "worldmap" ? "active" : ""}`}
              onClick={() => { setCenterView("worldmap"); }}
            >
              {"\u{1F30D}"} World
            </button>
            <button
              className={`tab ${centerView === "generate" ? "active" : ""}`}
              onClick={() => setCenterView("generate")}
            >
              {"\u2728"} Generate
            </button>
            <button
              className={`tab ${centerView === "assets" ? "active" : ""}`}
              onClick={() => setCenterView("assets")}
            >
              {"\u{1F3A8}"} Assets
            </button>
            <button
              className={`tab ${centerView === "events" ? "active" : ""}`}
              onClick={() => setCenterView("events")}
            >
              {"\u{1F4CB}"} Events
            </button>
          </div>

          {/* ── Control Panel (visible when in control mode) ── */}
          {controlMode && (
            <div className="control-panel">
              <div className="control-panel-header">
                {"\uD83C\uDFAE"} Agent Control Panel
              </div>
              <div className="control-panel-body">
                <div className="control-row">
                  <input
                    type="text"
                    className="control-input"
                    placeholder="Set action (e.g., reading a book)"
                    value={controlActionInput}
                    onChange={(e) => setControlActionInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        sendControlAction();
                      }
                    }}
                  />
                  <button
                    className="btn btn-primary btn-sm"
                    onClick={sendControlAction}
                    disabled={!controlActionInput.trim()}
                  >
                    Set Action
                  </button>
                </div>
                <div className="control-row">
                  <select
                    className="control-select"
                    value={controlLocationSelect}
                    onChange={(e) => setControlLocationSelect(e.target.value)}
                  >
                    <option value="">Move to location...</option>
                    {state.locations?.map((loc) => (
                      <option key={loc.name} value={loc.name}>
                        {loc.emoji} {loc.name}
                      </option>
                    ))}
                  </select>
                  <button
                    className="btn btn-primary btn-sm"
                    onClick={sendControlMove}
                    disabled={!controlLocationSelect}
                  >
                    Go
                  </button>
                </div>
                <div className="control-row explore-row">
                  <select
                    className="control-select"
                    value={exploreDir}
                    onChange={(e) => setExploreDir(e.target.value)}
                  >
                    <option value="">Explore direction...</option>
                    <option value="north">{"\u2B06"} North</option>
                    <option value="south">{"\u2B07"} South</option>
                    <option value="east">{"\u27A1"} East</option>
                    <option value="west">{"\u2B05"} West</option>
                    <option value="northeast">{"\u2197"} Northeast</option>
                    <option value="northwest">{"\u2196"} Northwest</option>
                    <option value="southeast">{"\u2198"} Southeast</option>
                    <option value="southwest">{"\u2199"} Southwest</option>
                  </select>
                  <button
                    className="btn btn-primary btn-sm"
                    onClick={sendExplore}
                    disabled={!exploreDir}
                  >
                    Explore
                  </button>
                </div>
                {nearbyAgents.length > 0 && (
                  <div className="control-nearby">
                    <div className="control-nearby-label">Nearby agents:</div>
                    <div className="control-nearby-list">
                      {nearbyAgents.map((a) => (
                        <div className="control-nearby-item" key={a.name}>
                          <span>{a.emoji} {a.name} &middot; {a.action}</span>
                          <button
                            className="btn btn-sm btn-talk"
                            onClick={() => handleInteract(a.name)}
                            disabled={interactLoading}
                          >
                            {interactLoading ? "\u22EF" : "\uD83D\uDCAC Talk"}
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── Chat View ── */}
          {centerView === "chat" && (
            <div className="chat-container">
              <div className="chat-messages">
                {chatMessages.length === 0 && (
                  <div className="chat-welcome">
                    <div className="chat-welcome-icon">{"\u{1F310}"}</div>
                    <h2>AURA Collaborative Space</h2>
                    <p>
                      You are <strong>{activeUser}</strong>. Ask the AI anything
                      &mdash; the <em>Environment Agent</em> will enrich your
                      query with real-time context from the town before the AI
                      responds.
                    </p>
                    <div className="chat-flow-preview">
                      <div className="flow-chip human">{"\u{1F464}"} You ask</div>
                      <span className="flow-arrow">{"\u2192"}</span>
                      <div className="flow-chip env">
                        {"\u{1F310}"} Env enriches
                      </div>
                      <span className="flow-arrow">{"\u2192"}</span>
                      <div className="flow-chip ai">
                        {"\u{1F916}"} AI responds
                      </div>
                    </div>
                    <div className="chat-welcome-hints">
                      <div
                        className="hint"
                        onClick={() =>
                          setChatInput("What's happening around me right now?")
                        }
                      >
                        What's happening around me?
                      </div>
                      <div
                        className="hint"
                        onClick={() =>
                          setChatInput("Who is nearby and what are they doing?")
                        }
                      >
                        Who is nearby?
                      </div>
                      <div
                        className="hint"
                        onClick={() =>
                          setChatInput(
                            "What should I do next based on my daily plan?"
                          )
                        }
                      >
                        What should I do next?
                      </div>
                      <div
                        className="hint"
                        onClick={() =>
                          setChatInput(
                            "Tell me about the recent conversations in town"
                          )
                        }
                      >
                        Recent town conversations?
                      </div>
                    </div>
                  </div>
                )}

                {chatMessages.map((msg) => {
                  if (msg.type === "user") {
                    return (
                      <div className="chat-msg chat-msg-user" key={msg.id}>
                        <div className="chat-msg-avatar user-avatar">
                          {"\u{1F464}"}
                        </div>
                        <div className="chat-msg-body">
                          <div className="chat-msg-label">
                            <span className="chat-msg-role">{msg.user}</span>
                            <span className="chat-msg-time">
                              {msg.timestamp}
                            </span>
                          </div>
                          <div className="chat-msg-bubble user-bubble">
                            {msg.content}
                          </div>
                        </div>
                      </div>
                    );
                  }

                  if (msg.type === "env-thinking") {
                    return (
                      <div className="chat-msg chat-msg-env" key={msg.id}>
                        <div className="chat-msg-avatar env-avatar">
                          {"\u{1F310}"}
                        </div>
                        <div className="chat-msg-body">
                          <div className="chat-msg-label">
                            <span className="chat-msg-role env-role">
                              Environment Agent
                            </span>
                          </div>
                          <div className="chat-msg-bubble env-bubble thinking">
                            <div className="thinking-dots">
                              <span />
                              <span />
                              <span />
                            </div>
                            <span>{msg.content}</span>
                          </div>
                        </div>
                      </div>
                    );
                  }

                  if (msg.type === "env-enrichment") {
                    const ctx = msg.context || {};
                    const probe = msg.probe;
                    const isExpanded = expandedEnv[msg.id] !== false;
                    return (
                      <div className="chat-msg chat-msg-env" key={msg.id}>
                        <div className="chat-msg-avatar env-avatar">
                          {"\u{1F310}"}
                        </div>
                        <div className="chat-msg-body">
                          <div className="chat-msg-label">
                            <span className="chat-msg-role env-role">
                              Environment Agent
                            </span>
                            <span className="env-badge">CONTEXT ENRICHMENT</span>
                          </div>
                          <div className="chat-msg-bubble env-bubble enrichment">
                            <div
                              className="env-toggle"
                              onClick={() =>
                                setExpandedEnv((prev) => ({
                                  ...prev,
                                  [msg.id]: !isExpanded,
                                }))
                              }
                            >
                              <span>
                                {isExpanded ? "\u25BC" : "\u25B6"} Context
                                gathered for your query
                              </span>
                            </div>
                            {isExpanded && (
                              <div className="env-details">
                                <div className="env-item">
                                  <span className="env-icon">
                                    {"\u{1F4CD}"}
                                  </span>
                                  <span className="env-key">Location:</span>
                                  <span className="env-val">
                                    {ctx.location}
                                  </span>
                                </div>
                                <div className="env-item">
                                  <span className="env-icon">
                                    {"\u23F0"}
                                  </span>
                                  <span className="env-key">Time:</span>
                                  <span className="env-val">{ctx.time}</span>
                                </div>
                                <div className="env-item">
                                  <span className="env-icon">
                                    {"\u{1F3AC}"}
                                  </span>
                                  <span className="env-key">Activity:</span>
                                  <span className="env-val">
                                    {ctx.current_action}
                                  </span>
                                </div>
                                {ctx.nearby_agents?.length > 0 && (
                                  <div className="env-item env-item-list">
                                    <span className="env-icon">
                                      {"\u{1F465}"}
                                    </span>
                                    <span className="env-key">Nearby:</span>
                                    <div className="env-list">
                                      {ctx.nearby_agents.map((a, i) => (
                                        <span className="env-chip" key={i}>
                                          {a.name} &middot; {a.action}
                                        </span>
                                      ))}
                                    </div>
                                  </div>
                                )}
                                {ctx.recent_memories?.length > 0 && (
                                  <div className="env-item env-item-list">
                                    <span className="env-icon">
                                      {"\u{1F9E0}"}
                                    </span>
                                    <span className="env-key">Memories:</span>
                                    <div className="env-list">
                                      {ctx.recent_memories
                                        .slice(0, 3)
                                        .map((m, i) => (
                                          <div className="env-memory" key={i}>
                                            {truncate(m, 80)}
                                          </div>
                                        ))}
                                    </div>
                                  </div>
                                )}
                                {ctx.recent_events?.length > 0 && (
                                  <div className="env-item env-item-list">
                                    <span className="env-icon">
                                      {"\u{1F4CB}"}
                                    </span>
                                    <span className="env-key">Events:</span>
                                    <div className="env-list">
                                      {ctx.recent_events
                                        .slice(0, 3)
                                        .map((e, i) => (
                                          <div className="env-memory" key={i}>
                                            [{e.time}] {truncate(e.description, 60)}
                                          </div>
                                        ))}
                                    </div>
                                  </div>
                                )}
                                {probe?.steps?.length > 0 && (
                                  <div className="env-item env-item-list">
                                    <span className="env-icon">
                                      {"\u{1F6F0}\uFE0F"}
                                    </span>
                                    <span className="env-key">
                                      Probe ({probe.steps.length} calls):
                                    </span>
                                    <div className="env-list">
                                      {probe.steps.map((s, i) => (
                                        <div
                                          className="env-probe-step"
                                          key={i}
                                        >
                                          <span
                                            className={`probe-dot ${s.ok ? "ok" : "err"}`}
                                          >
                                            {s.ok ? "\u2713" : "\u2717"}
                                          </span>
                                          <span className="probe-tool">
                                            {s.tool}
                                          </span>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  }

                  if (msg.type === "ai") {
                    return (
                      <div className="chat-msg chat-msg-ai" key={msg.id}>
                        <div className="chat-msg-avatar ai-avatar">
                          {"\u{1F916}"}
                        </div>
                        <div className="chat-msg-body">
                          <div className="chat-msg-label">
                            <span className="chat-msg-role ai-role">
                              AI Assistant
                            </span>
                            <span className="chat-msg-time">
                              {msg.timestamp}
                            </span>
                          </div>
                          <div className="chat-msg-bubble ai-bubble">
                            {msg.content}
                          </div>
                        </div>
                      </div>
                    );
                  }

                  if (msg.type === "error") {
                    return (
                      <div className="chat-msg chat-msg-error" key={msg.id}>
                        <div className="chat-msg-bubble error-bubble">
                          {"\u26A0\uFE0F"} {msg.content}
                        </div>
                      </div>
                    );
                  }
                  return null;
                })}
                <div ref={chatEndRef} />
              </div>

              <div className="chat-input-area">
                <input
                  type="text"
                  className="chat-input"
                  placeholder={`Ask as ${activeUser}\u2026 (Environment Agent will enrich your query)`}
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      sendChat();
                    }
                  }}
                  disabled={chatLoading}
                />
                <button
                  className="btn btn-send"
                  onClick={sendChat}
                  disabled={chatLoading || !chatInput.trim()}
                >
                  {chatLoading ? "\u22EF" : "Send \u2192"}
                </button>
              </div>
            </div>
          )}

          {/* ── Map View ── */}
          {centerView === "map" && (
            <>
              <TownMap
                state={state}
                activeUser={activeUser}
                onSelectAgent={setActiveUser}
                controlMode={controlMode}
                onControlMove={handleControlMove}
                onDpadMove={handleDpadMove}
                onLocationClick={handleLocationClick}
              />
              {/* Location Detail Panel */}
              {locationDetail && (
                <div className="location-detail-panel">
                  <div className="location-detail-header">
                    <span className="location-detail-title">
                      {locationDetail.emoji} {locationDetail.name}
                    </span>
                    <button
                      className="btn btn-sm"
                      onClick={() => setLocationDetail(null)}
                    >
                      {"\u2715"} Close
                    </button>
                  </div>
                  <div className="location-detail-body">
                    <div className="location-detail-meta">
                      <span className="location-type-badge">{locationDetail.type}</span>
                      {locationDetail.owner && (
                        <span className="location-owner">Owner: {locationDetail.owner}</span>
                      )}
                      <span className="location-visits">
                        Visits: {locationDetail.visit_count || 0}
                      </span>
                    </div>
                    <p className="location-description">{locationDetail.description}</p>
                    {locationDetail.atmosphere && (
                      <div className="location-atmosphere">
                        {"\u2728"} <em>{locationDetail.atmosphere}</em>
                      </div>
                    )}
                    {locationDetail.interior_objects?.length > 0 && (
                      <div className="location-section">
                        <div className="location-section-title">Interior</div>
                        {locationDetail.interior_objects.map((obj, i) => (
                          <div className="interior-object" key={i}>
                            <strong>{obj.name}</strong>
                            <span> &mdash; {obj.description}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    {locationDetail.items?.length > 0 && (
                      <div className="location-section">
                        <div className="location-section-title">Items</div>
                        <div className="item-chips">
                          {locationDetail.items.map((item, i) => (
                            <span className="item-chip" key={i}>{item}</span>
                          ))}
                        </div>
                      </div>
                    )}
                    {locationDetail.occupants?.length > 0 && (
                      <div className="location-section">
                        <div className="location-section-title">Current Occupants</div>
                        {locationDetail.occupants.map((occ, i) => (
                          <div className="location-occupant" key={i}>
                            {occ.emoji} {occ.name} &middot; {occ.action}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </>
          )}

          {/* ── World Map View ── */}
          {centerView === "worldmap" && (
            <WorldMap
              worldMapData={state?.world_map}
              activeRegion={activeRegion}
              onSelectRegion={handleSelectRegion}
              onBack={() => { setCenterView("map"); setMapLevel("region"); }}
            />
          )}

          {/* ── Generate View ── */}
          {centerView === "generate" && (
            <MapGeneratorPanel onGenerate={handleMapGenerate} />
          )}

          {/* ── Assets View ── */}
          {centerView === "assets" && (
            <AssetManager onAssetsChange={() => {
              API.state().then((r) => { if (r.ok) setState(r.state); });
            }} />
          )}

          {/* ── Events View ── */}
          {centerView === "events" && (
            <div className="events-view">
              <div className="event-list">
                {events.length === 0 && (
                  <div className="events-empty">
                    No events yet. Click <strong>Step</strong> to begin the
                    simulation.
                  </div>
                )}
                {events.map((evt, i) => (
                  <div
                    className={`event-item type-${evt.type}`}
                    key={`${evt.time}-${i}`}
                  >
                    <div className="event-meta">
                      {EVT_ICONS[evt.type] || "\u2022"} {evt.time} &middot;{" "}
                      {evt.agent}
                    </div>
                    <div className="event-desc">{evt.description}</div>
                    {evt.type === "conversation" &&
                      evt.details?.dialogue?.length > 0 && (
                        <div className="event-dialogue">
                          {evt.details.dialogue.map((line, j) => (
                            <div className="event-dialogue-line" key={j}>
                              {line}
                            </div>
                          ))}
                        </div>
                      )}
                    {evt.type === "probe" &&
                      evt.details?.steps?.length > 0 && (
                        <div className="event-probe-trace">
                          {evt.details.steps.map((s, j) => (
                            <div key={j}>
                              {"\u2192"} {s.tool}({JSON.stringify(s.arguments)}){" "}
                              {s.ok ? "\u2713" : `\u2717 ${s.error}`}
                            </div>
                          ))}
                        </div>
                      )}
                  </div>
                ))}
                <div ref={eventEndRef} />
              </div>
            </div>
          )}
        </div>

        {/* ── RIGHT: Environment Intelligence ── */}
        <aside className="env-panel">
          <div className="env-panel-header">
            <span>{"\u{1F310}"} Environment Intelligence</span>
            <span className="badge active-badge">ACTIVE</span>
          </div>

          {/* Scene Context */}
          <div className="env-section">
            <div className="env-section-title">Scene Context</div>
            {activeAgent && (
              <div className="env-scene">
                <div className="scene-row">
                  <span className="scene-icon">{"\u{1F4CD}"}</span>
                  <span className="scene-label">Location</span>
                  <span className="scene-value">{activeAgent.location}</span>
                </div>
                <div className="scene-row">
                  <span className="scene-icon">{"\u23F0"}</span>
                  <span className="scene-label">Time</span>
                  <span className="scene-value">{state.time}</span>
                </div>
                <div className="scene-row">
                  <span className="scene-icon">{"\u{1F3AC}"}</span>
                  <span className="scene-label">Action</span>
                  <span className="scene-value">{activeAgent.action}</span>
                </div>
                <div className="scene-row">
                  <span className="scene-icon">{"\u{1F464}"}</span>
                  <span className="scene-label">Role</span>
                  <span className="scene-value">{activeAgent.occupation}</span>
                </div>
                <div className="scene-row">
                  <span className="scene-icon">{"\uD83C\uDFAE"}</span>
                  <span className="scene-label">Mode</span>
                  <span className={`scene-value ${controlMode ? "ctrl-mode-active" : ""}`}>
                    {controlMode ? "Player Controlled" : "Autonomous"}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Weather / Season / Biome */}
          {state.world_properties && (
            <div className="env-section">
              <div className="env-section-title">World Environment</div>
              <div className="env-scene">
                <div className="scene-row">
                  <span className="scene-icon">
                    {SEASON_ICONS[state.world_properties.season] || "\u{1F343}"}
                  </span>
                  <span className="scene-label">Season</span>
                  <span className="scene-value" style={{ textTransform: "capitalize" }}>
                    {state.world_properties.season || "spring"}
                  </span>
                </div>
                <div className="scene-row">
                  <span className="scene-icon">
                    {WEATHER_ICONS[state.world_properties.weather] || "\u2600\uFE0F"}
                  </span>
                  <span className="scene-label">Weather</span>
                  <span className="scene-value" style={{ textTransform: "capitalize" }}>
                    {(state.world_properties.weather || "clear").replace("_", " ")}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Collaboration Flow */}
          <div className="env-section">
            <div className="env-section-title">Collaboration Flow</div>
            <div className="collab-flow">
              <div className="flow-node flow-human">
                <div className="flow-node-icon">{"\u{1F464}"}</div>
                <div className="flow-node-label">Human</div>
              </div>
              <div className="flow-connector">
                <div className="flow-line" />
                <div className="flow-arrow-icon">{"\u25B6"}</div>
              </div>
              <div className="flow-node flow-env-node">
                <div className="flow-node-icon">{"\u{1F310}"}</div>
                <div className="flow-node-label">Env Agent</div>
              </div>
              <div className="flow-connector">
                <div className="flow-line" />
                <div className="flow-arrow-icon">{"\u25B6"}</div>
              </div>
              <div className="flow-node flow-ai-node">
                <div className="flow-node-icon">{"\u{1F916}"}</div>
                <div className="flow-node-label">AI</div>
              </div>
            </div>
          </div>

          {/* Probe Settings */}
          <div className="env-section">
            <div className="env-section-title">Probe Settings</div>
            <div className="probe-controls">
              <label className="probe-toggle">
                <input
                  type="checkbox"
                  checked={probeOn}
                  onChange={(e) => {
                    const v = e.target.checked;
                    setProbeOn(v);
                    updateProbe(v, probeMax);
                  }}
                />
                <span>Active Probe</span>
                <span className={`status-dot ${probeOn ? "on" : "off"}`} />
              </label>
              <label className="probe-slider">
                <span>Max Steps: {probeMax}</span>
                <input
                  type="range"
                  min="0"
                  max="4"
                  value={probeMax}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    setProbeMax(v);
                    updateProbe(probeOn, v);
                  }}
                />
              </label>
            </div>
          </div>

          {/* Recent Probe Activity */}
          {activeAgent?.probe_steps?.length > 0 && (
            <div className="env-section">
              <div className="env-section-title">Recent Probe</div>
              <div className="probe-activity">
                <div className="probe-summary">
                  {activeAgent.probe_summary}
                </div>
                {activeAgent.probe_steps.map((step, i) => (
                  <div className="probe-trace-item" key={i}>
                    <span className={`trace-dot ${step.ok ? "ok" : "err"}`}>
                      {step.ok ? "\u2713" : "\u2717"}
                    </span>
                    <span className="trace-tool">{step.tool}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Group Insights */}
          <div className="env-section">
            <div className="env-section-title">Group Insights</div>
            <div className="group-insights">
              <div className="insight-row">
                <span className="insight-label">Active Agents</span>
                <span className="insight-value">
                  {state.agents.filter((a) => a.action !== "sleeping").length} /{" "}
                  {state.agents.length}
                </span>
              </div>
              <div className="insight-row">
                <span className="insight-label">Conversations</span>
                <span className="insight-value">
                  {state.events?.filter((e) => e.type === "conversation")
                    .length || 0}
                </span>
              </div>
              <div className="insight-row">
                <span className="insight-label">Probes Run</span>
                <span className="insight-value">
                  {state.events?.filter((e) => e.type === "probe").length || 0}
                </span>
              </div>
              <div className="insight-row">
                <span className="insight-label">Total Events</span>
                <span className="insight-value">
                  {state.events?.length || 0}
                </span>
              </div>
            </div>
          </div>

          {/* Agent Memories */}
          {activeAgent?.memories?.length > 0 && (
            <div className="env-section">
              <div className="env-section-title">
                {activeAgent.emoji} {activeAgent.name}'s Memory
              </div>
              <div className="memory-list">
                {activeAgent.memories.slice(0, 6).map((m, i) => (
                  <div className="memory-item" key={i}>
                    <span className="memory-kind">[{m.kind}]</span>
                    <span className="memory-text">
                      {truncate(m.content, 70)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
