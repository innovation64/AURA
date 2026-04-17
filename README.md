# AURA — Environment-Aware Agent Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AURA** is an environment-aware agent framework that converts raw, multimodal context into structured signals for LLM reasoning and interaction. It features a modular 7-stage pipeline, proactive context pushing, runtime safety guards, and pluggable backends.

---

## Highlights

- **Proactive Context Engine** — Environment probes (system, git, docker, filesystem, network, process) detect changes and push relevant signals to the agent *before* it asks.
- **Pluggable Pipeline** — Every stage (Sense → Scene → Memory → Reason → Act → Interact) is an interface you can swap with a custom implementation.
- **Multiple Backends** — `default` (zero-dependency stubs), `llm` (OpenAI-compatible), `bmam` (five-brain-region bridge), `model` (neural plasticity memory).
- **Runtime Safety** — `ExecutionGuard` with 5-level intervention (Observe → Hint → Suggest → Constrain → Redirect) detects loops, stagnation, and goal drift.
- **Workflow Optimization** — Learns from past tool sequences, reuses known workflows, and synthesizes composite tools at runtime via `ToolForge`.
- **Three Interaction Paradigms** — Reactive (pull), Proactive (push), and Collaborative (push + online feedback loop).
- **Specialized Views** — Pre-built agent personas for coding, research, and sysadmin tasks.

## Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────────────┐
│              AURAAgent (core)                │
├─────────────────────────────────────────────┤
│ 1. Sense      → SenseAdapter               │
│ 2. Proactive  → ProactiveEngine             │
│    ├─ Probes (system, git, docker, …)       │
│    ├─ ChangeDetector                        │
│    ├─ RelevanceScorer                       │
│    └─ ContextAssembler                      │
│ 3. Explore    → Explorer + Planner          │
│ 4. Scene      → SceneModel                  │
│ 5. Memory     → MemoryStore (TF-IDF)       │
│ 6. Reason     → Reasoner                    │
│ 7. Guard      → ExecutionGuard              │
│ 8. Act        → Actor                       │
│ 9. Interact   → Interactor                  │
│10. Workflow   → WorkflowEngine              │
└─────────────────────────────────────────────┘
    │
    ▼
User Response
```

## Installation

```bash
# Core (zero external dependencies)
pip install -e .

# With LLM server support
pip install -e ".[server]"

# Development (tests, linting, type checking)
pip install -e ".[dev]"
```

## Quick Start

### CLI

```bash
# Basic environment understanding
aura "office, 2 people, projector on" --query "summarize environment"

# With active exploration (probing the local environment)
aura "meeting room" --query "any project files?" --probe --max-steps 2

# Restrict which tools the explorer may call
aura "lab" --query "check GPU status" --probe --allow-tool "system.*"
```

### Python API

```python
from aura import AURAAgent, AURAConfig

config = AURAConfig(
    backend="llm",
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",
    explore_enabled=True,
    guard_enabled=True,
)

agent = AURAAgent(config)
result = agent.step(
    raw_input="office, 2 people, projector on",
    query="summarize the environment",
)
print(result.text)
```

### Server Mode

```bash
# Start the AURA WebSocket/HTTP server
aura-server --host 0.0.0.0 --port 8000
```

## Backends

| Backend   | Description                                  | Dependencies |
|-----------|----------------------------------------------|-------------|
| `default` | Stub implementations, no external calls      | None        |
| `llm`     | OpenAI-compatible LLM for all stages         | API key     |
| `bmam`    | Bridge to BMAM five-brain-region system      | BMAM server |
| `model`   | Neural plasticity memory layer               | API key     |

Select a backend via `AURAConfig(backend="llm")` or the `--backend` CLI flag.

## Configuration

All configuration is done through `AURAConfig`. Key options:

```python
AURAConfig(
    # Backend
    backend="llm",                    # default | llm | bmam | model
    llm_api_key="sk-...",
    llm_model="gpt-4o-mini",

    # Exploration
    explore_enabled=True,
    explore_max_steps=5,
    smart_planner=True,               # Context-aware planner (vs heuristic)

    # Proactive engine
    proactive_enabled=True,
    proactive_poll_interval=10.0,     # seconds
    proactive_relevance_threshold=0.4,

    # Safety
    guard_enabled=True,
    guard_window=8,
    guard_threshold=0.7,

    # Workflow optimization
    workflow_enabled=True,
    workflow_reuse_rate=0.6,

    # World evolution
    evolve_enabled=False,
    evolve_interval=5,
)
```

## Project Structure

```
src/aura/
├── core.py              # AURAAgent — main orchestrator
├── types.py             # Core data types
├── sense.py             # Environment input adapter
├── scene.py             # Scene state building
├── memory.py            # Semantic memory (TF-IDF)
├── reason.py            # Reasoning interface
├── act.py               # Action interface
├── interact.py          # Interaction interface
├── explore.py           # Exploration/probing engine
├── tools.py             # Tool registry & execution
├── guard.py             # Runtime safety monitoring
├── feedback.py          # Conditional feedback store
├── workflow.py          # Workflow memory & optimization
├── auditor.py           # Strategy effectiveness auditing
├── evolve.py            # World-level evolution
├── llm.py               # OpenAI-compatible LLM engine
├── backend.py           # Plugin backend registry
├── builtin_tools.py     # System/workspace tools
├── smart_planner.py     # Context-aware exploration planner
├── defaults/            # LLM-based default implementations
├── adapters/            # External system bridges (BMAM, model)
├── paradigm/            # Interaction paradigms (reactive/proactive/collaborative)
├── proactive/           # Proactive context engine
├── probes/              # Environment sensors
├── views/               # Specialized agent personas
├── eval/                # Benchmarking & metrics
├── trajectory/          # Training data collection
└── server/              # FastAPI server mode
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test suites
pytest tests/test_core_integration.py
pytest tests/test_proactive.py
pytest tests/test_guard.py

# With markers
pytest tests/ -m "not slow"
```

## Experiments

Research validation scripts are in `experiments/`:

```bash
python -m experiments.run_paradigm_comparison
python -m experiments.run_ablation
python -m experiments.run_scalability
python -m experiments.run_feedback_convergence
```

## License

[MIT](LICENSE)
