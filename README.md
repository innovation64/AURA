# AURA: Intent-Directed Probing for Implicit-Need Surfacing in Situated LLM Agents

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center">
  <img src="docs/assets/auratown-illustrated.png" alt="AURATown — illustrated bird's-eye view of the 5-agent social simulation testbed" width="860"/>
</p>

<p align="center">
  <em>AURATown — the 5-agent social simulation testbed where AURA's mechanisms are evaluated.<br/>
  Technical 60×60 grid map with named coordinates: see <a href="docs/assets/auratown-map.svg">auratown-map.svg</a>.</em>
</p>

**AURA** is an *environment agent* that bridges the alignment gap between
users and LLM-driven AI agents. Instead of passively answering literal
queries, AURA infers the user's **implicit information need**, decides
how much environmental probing that need warrants, and pre-loads grounded
context *before* reasoning begins.

The defining mechanism is an **`IntentFrame`** — a learned theory-of-mind
structure that models the gap between what the user *asked* and what
the user *wants* — on top of a deterministic `Perceive → Scene → Memory →
Reason` pipeline.

---

## The Problem

Asking *"where is Lin Wei?"* often means *"is she free to chat?"* A
reactive LLM agent answers the surface question; the user's real need
never surfaces. AURA addresses this alignment gap at the architectural
level.

<p align="center">
  <img src="docs/assets/story-comic.png" alt="4-panel comic: user asks 'where is Lin Wei?', vanilla LLM gives a literal answer, AURA infers the implicit need (is she free?) by probing the cafe scene, and replies with a grounded answer plus a heads-up alert" width="860"/>
</p>

<p align="center">
  <em>What AURA does in 30 seconds: it doesn't just answer the literal question — it infers the user's implicit need, probes the environment, and surfaces what actually matters with an explicit heads-up.</em>
</p>

```mermaid
flowchart LR
    U[Human User<br/>literal query q]:::human
    A[LLM Agents<br/>a_1 … a_N]:::agent
    E[(Environment<br/>scene + hidden private state)]:::env
    EA[Environment Agent π_E<br/>IntentFrame + 7-stage pipeline]:::aura

    U -->|query| EA
    EA -->|grounded response + alert| U
    A -->|actions| EA
    EA -->|enriched context| A
    EA <-->|observe / probe| E
    U -.->|interaction| A

    classDef human fill:#ffe7cc,stroke:#d97706,color:#111
    classDef agent fill:#dcfce7,stroke:#16a34a,color:#111
    classDef env   fill:#e5e7eb,stroke:#475569,color:#111
    classDef aura  fill:#dbeafe,stroke:#2563eb,color:#111,stroke-width:2px
```

AURA sits in the middle of the Human–Agent–Environment triangle,
mediating context flow in both directions.

<p align="center">
  <img src="docs/assets/triangle-mediator.png" alt="Triangle: Human user (top-left) and AI agents (top-right) connected through a translucent mediator layer (the Environment Agent π_E) over a small town environment (bottom)" width="720"/>
</p>

---

## The Architecture

<p align="center">
  <img src="docs/assets/aura-architecture.png" alt="AURA's eight-stage pipeline: Sense → Scene → Memory → IntentInferrer → Explore → Reason → Act → Interact, embedded in the human user / AI agents / environment context, with the IntentInferrer highlighted as the single agentic stage" width="900"/>
</p>

An **eight-stage pipeline**. The first three stages (Sense, Scene,
Memory) build grounded context deterministically; the
**`IntentInferrer`** (★ in the figure) is the point at which the LLM
takes over control flow (per-query probe budget, tool selection,
alerting). Explore, Reason, Act, Interact then run on the enriched
context.

```mermaid
flowchart LR
    Q[User Query] --> S1[1 Sense<br/>raw → EnvSignal]
    S1 --> S2[2 Scene<br/>aggregate state]
    S2 --> S3[3 Memory<br/>weighted retrieval]
    S3 --> II{{IntentInferrer<br/>literal / implicit / gap}}:::intent
    II -->|gap, recommended probes| S4[4 Explore<br/>bounded probe loop]
    S4 -.tools.-> T[(Tool Registry<br/>world.time · nearby<br/>agents · events · memory)]
    S4 --> S5[5 Reason<br/>plan over context]
    S5 --> S6[6 Act<br/>executables]
    S6 --> S7[7 Interact<br/>render + alert]
    S7 --> R[Response]

    classDef intent fill:#fef3c7,stroke:#d97706,color:#111,stroke-width:2px
```

**Deterministic shape, agentic content.** Stages 1–3 are code-determined
so context assembly is predictable. Stages 4, 7 are LLM-directed via
the `IntentFrame` so per-query judgment is local where it matters.

---

## IntentFrame: Modeling the Implicit Need

<p align="center">
  <img src="docs/assets/intentframe-thinking.png" alt="IntentFrame visualisation: a robot character producing a thought-bubble that contains six fields (literal_need, implicit_need, gap, recommended_probes, alert, confidence) plus an input-to-output workflow strip below" width="640"/>
</p>

Each query produces an `IntentFrame`:

```python
IntentFrame(
    literal_need       = "Lin Wei's current location",
    implicit_need      = ["is Lin Wei available to chat?", "mood"],
    gap                = 0.5,                               # [0, 1]
    recommended_probes = ["get_nearby_agents", "get_agent_plan"],
    should_alert       = True,
    confidence         = 0.8,
    rationale          = "location query; real need is social availability",
)
```

The `gap` drives the probe budget; `recommended_probes` is whitelisted
against the live tool registry; `should_alert` opts in to a
`[heads-up]` prefix on the response.

```mermaid
flowchart TD
    Q[User query] --> II[IntentInferrer]
    SC[Scene snapshot] --> II
    M[Recent memories] --> II
    II --> F[IntentFrame]
    F -->|gap g| B{g → budget B}
    B -->|g &lt; 0.2| B0[B=0 · literal answer only]:::b0
    B -->|0.2 ≤ g &lt; 0.6| B1[B=1-2 · single targeted probe]:::b1
    B -->|g ≥ 0.6| B3[B=3 · full probe loop]:::b3
    B0 --> ANS[LLM generate]
    B1 --> ANS
    B3 --> ANS
    F -->|should_alert| AL["[heads-up] prefix"] --> ANS
    ANS --> OUT[Response]

    classDef b0 fill:#e0f2fe,stroke:#0284c7,color:#111
    classDef b1 fill:#fde68a,stroke:#d97706,color:#111
    classDef b3 fill:#fecaca,stroke:#dc2626,color:#111
```

The budget map is a **ceiling**, not a target — the LLM often stops
short of it when a single well-targeted probe has already returned
actionable information. See `aura.intent.LLMIntentInferrer` for the
prompt calibration and `aura.intent.HeuristicIntentInferrer` for the
deterministic fallback.

---

## Interaction Paradigms

AURA cleanly separates three information-flow regimes, parameterised by
a single `ParadigmConfig`:

```mermaid
flowchart TB
    subgraph Reactive[Reactive · ReAct-style]
        direction LR
        QA[query] --> LA[LLM]
        LA -->|tool call| TA[env]
        TA --> LA
        LA --> RA[answer]
    end

    subgraph Proactive[Proactive · AURA default]
        direction LR
        QB[query] --> IB[IntentInferrer]
        IB --> PB[probe loop]
        PB --> LB[LLM with pre-loaded context]
        LB --> RB[answer + alert]
    end

    subgraph Collab[Collaborative · Proactive + feedback]
        direction LR
        QC[query] --> IC[IntentInferrer]
        IC --> PC[probe]
        PC --> LC[LLM]
        LC --> RC[answer]
        RC --> FB[human feedback]
        FB -.update relevance.-> IC
    end
```

Reactive spends the tool budget *during* reasoning. Proactive spends it
*before* reasoning, with intent inference directing where to look.
Collaborative closes the loop with human attention-budget signals.

---

## Highlights

- **IntentFrame** — learned literal/implicit gap estimation that routes
  per-query probe budget, tool preference, and proactive alerting.
- **7-stage pluggable pipeline** — Sense · Scene · Memory · Reason ·
  Explore · Act · Interact, every stage swappable via the backend
  registry (`default`, `llm`, `bmam`, `model`).
- **Bounded probe loop** — pre-reasoning tool-call loop with a hard
  ceiling (`explore_max_steps`) and domain-contextualised tool set.
- **Proactive context engine** — environment probes (system, git,
  docker, filesystem, network, process) push relevant signals to the
  agent before it asks.
- **Runtime safety guard** — 5-level intervention (Observe → Hint →
  Suggest → Constrain → Redirect) detects loops, stagnation, goal
  drift.
- **Workflow optimisation** — `WorkflowEngine` reuses successful tool
  sequences; `StrategyAuditor` retires ineffective strategies.
- **Second-order ToM probe** — `get_agent_belief_about` tool lets one
  agent query another agent's beliefs about a third, for
  false-belief-style reasoning.

---

## Installation

```bash
# Core (zero external dependencies, runs on stubs)
pip install -e .

# With LLM backend + server support
pip install -e ".[server]"

# Development (tests, linting, type checking)
pip install -e ".[dev]"
```

## Quick Start

### CLI

```bash
# Direct environment understanding
aura "office, 2 people, projector on" --query "summarize environment"

# With pre-reasoning probing
aura "meeting room" --query "any project files?" --probe --max-steps 2

# Restrict which tools the explorer may call
aura "lab" --query "check GPU status" --probe --allow-tool "system.*"
```

### Python API

```python
from aura import AURAAgent, AURAConfig
from aura.intent import LLMIntentInferrer

config = AURAConfig(
    backend="llm",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    explore_enabled=True,
    explore_max_steps=3,
    intent_backend="llm",     # or "heuristic" for deterministic
    guard_enabled=True,
)

agent = AURAAgent(config)
result = agent.step(
    raw_input="Sunrise Cafe, 10:15 AM. Lin Wei at counter.",
    query="where is Lin Wei?",
)
print(result.text)
# → Lin Wei is at the Sunrise Cafe counter.
#   [heads-up] she is currently busy serving customers — you may want
#   to wait until she has a free moment before starting a long chat.
```

### Server

```bash
aura-server --host 0.0.0.0 --port 8000
# WebSocket + HTTP, streaming responses, transparency metadata
```

---

## Backends

| Backend   | What it is                                   | Deps        |
|-----------|----------------------------------------------|-------------|
| `default` | Stub implementations, no external calls      | None        |
| `llm`     | OpenAI-compatible LLM for all stages         | API key     |
| `bmam`    | Bridge to [BMAM](https://github.com/) five-brain-region system | BMAM server |
| `model`   | Neural plasticity memory layer               | API key     |

Select with `AURAConfig(backend=...)` or the `--backend` CLI flag.

## Configuration (excerpt)

```python
AURAConfig(
    # Backend
    backend="llm",                    # default | llm | bmam | model
    llm_model="gpt-4o-mini",

    # Intent (new — the ToM stage)
    intent_enabled=True,
    intent_backend="llm",             # llm | heuristic
    intent_gap_to_budget=(0.2, 0.4, 0.6, 0.8),

    # Exploration
    explore_enabled=True,
    explore_max_steps=3,
    smart_planner=True,

    # Proactive engine (separate from Explore)
    proactive_enabled=True,
    proactive_poll_interval=10.0,
    proactive_relevance_threshold=0.4,

    # Safety
    guard_enabled=True,
    guard_window=8,
    guard_threshold=0.7,

    # Workflow optimisation
    workflow_enabled=True,
    workflow_reuse_rate=0.6,
)
```

---

## Project Structure

```
src/aura/
├── core.py              # AURAAgent — main orchestrator
├── types.py             # IntentFrame, SceneState, MemoryItem
├── intent.py            # IntentInferrer (Heuristic + LLM)  ← the ToM stage
├── sense.py             # Environment input adapter
├── scene.py             # Scene state building
├── memory.py            # Semantic memory (TF-IDF)
├── reason.py            # Reasoning interface
├── act.py               # Action interface
├── interact.py          # Interaction interface (+ alert rendering)
├── explore.py           # Bounded probe loop
├── tools.py             # Tool registry (incl. get_agent_belief_about)
├── smart_planner.py     # Context-aware probe planner
├── guard.py             # Runtime safety (5-level intervention)
├── workflow.py          # WorkflowEngine (reuse + synthesis)
├── auditor.py           # StrategyAuditor (retire ineffective ones)
├── feedback.py          # Conditional feedback store
├── llm.py               # OpenAI-compatible engine
├── backend.py           # Plugin backend registry
├── defaults/            # LLM-based default implementations
├── adapters/            # External bridges (BMAM, model)
├── paradigm/            # Reactive / Proactive / Collaborative
├── proactive/           # Proactive context engine + probes
├── probes/              # Environment sensors (system, git, docker, …)
├── views/               # Specialized agent personas
├── eval/                # Benchmarking & metrics
├── trajectory/          # Training-data collection
└── server/              # FastAPI server mode
```

## Testing

```bash
pytest tests/                       # full suite
pytest tests/test_core_integration.py
pytest tests/test_intent.py         # IntentInferrer
pytest tests/ -m "not slow"
```

## Experiments

Research validation scripts in `experiments/`:

```bash
python -m experiments.run_paradigm_comparison   # reactive vs proactive vs collab
python -m experiments.run_ablation              # per-component ablation
python -m experiments.run_scalability           # N-agent scaling
python -m experiments.run_feedback_convergence  # collaborative loop
```

A companion multi-agent testbed (**AURATown**, 5 agents, social
simulation with hidden private state) and an implicit-intent benchmark
(25 queries × 5 subcategories) live in the sibling evaluation project.

## License

[MIT](LICENSE)
