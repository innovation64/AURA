"""Backend registry — switch between default / bmam / model implementations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Type

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BACKENDS: Dict[str, "BackendFactory"] = {}


@dataclass
class BackendFactory:
    """Holds callables that produce AURA component implementations."""

    name: str
    description: str = ""
    build_sense: Optional[Callable[..., Any]] = None
    build_scene: Optional[Callable[..., Any]] = None
    build_memory: Optional[Callable[..., Any]] = None
    build_reasoner: Optional[Callable[..., Any]] = None
    build_actor: Optional[Callable[..., Any]] = None
    build_interactor: Optional[Callable[..., Any]] = None
    build_explorer: Optional[Callable[..., Any]] = None
    # Plasticity is an optional cross-cutting concern
    build_plasticity: Optional[Callable[..., Any]] = None
    # Evolve is an optional world-level cross-cutting concern
    build_evolver: Optional[Callable[..., Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def register_backend(name: str, factory: BackendFactory) -> None:
    _BACKENDS[name] = factory
    logger.info("Registered AURA backend: %s", name)


def get_backend(name: str) -> BackendFactory:
    if name not in _BACKENDS:
        available = ", ".join(_BACKENDS.keys()) or "(none)"
        raise KeyError(f"Unknown backend '{name}'. Available: {available}")
    return _BACKENDS[name]


def list_backends() -> Dict[str, str]:
    return {name: f.description for name, f in _BACKENDS.items()}


# ---------------------------------------------------------------------------
# Built-in backend registration helpers
# ---------------------------------------------------------------------------

def _register_default_backend() -> None:
    """Register the 'default' backend (stub implementations, always available)."""
    from .sense import BasicSense
    from .scene import BasicScene
    from .memory import EphemeralMemory
    from .reason import SimpleReasoner
    from .act import StubActor
    from .interact import BasicInteractor

    from .evolve import NoOpEvolver

    register_backend("default", BackendFactory(
        name="default",
        description="Built-in stub implementations (no LLM required)",
        build_sense=lambda **kw: BasicSense(),
        build_scene=lambda **kw: BasicScene(),
        build_memory=lambda **kw: EphemeralMemory(max_items=kw.get("memory_limit", 100)),
        build_reasoner=lambda **kw: SimpleReasoner(),
        build_actor=lambda **kw: StubActor(),
        build_interactor=lambda **kw: BasicInteractor(),
        build_evolver=lambda **kw: NoOpEvolver(),
    ))


def _register_llm_backend() -> None:
    """Register the 'llm' backend (LLM-driven defaults, standalone)."""
    from .defaults import build_llm_sense, build_llm_scene, build_llm_memory
    from .defaults import build_llm_reasoner, build_llm_actor, build_llm_interactor
    from .defaults.llm_evolver import build_llm_evolver

    register_backend("llm", BackendFactory(
        name="llm",
        description="LLM-driven implementations (standalone, no BMAM required)",
        build_sense=build_llm_sense,
        build_scene=build_llm_scene,
        build_memory=build_llm_memory,
        build_reasoner=build_llm_reasoner,
        build_actor=build_llm_actor,
        build_interactor=build_llm_interactor,
        build_evolver=build_llm_evolver,
    ))


def _register_bmam_backend() -> None:
    """Register the 'bmam' backend (bridging to BMAM brain system)."""
    from .adapters.bmam import build_bmam_memory, build_bmam_reasoner, build_bmam_scene
    from .defaults import build_llm_sense, build_llm_actor, build_llm_interactor
    from .defaults.llm_evolver import build_llm_evolver

    register_backend("bmam", BackendFactory(
        name="bmam",
        description="BMAM five-brain-region system adapter",
        build_sense=build_llm_sense,        # reuse LLM sense
        build_scene=build_bmam_scene,        # BMAM knowledge graph
        build_memory=build_bmam_memory,      # BMAM distributed memory
        build_reasoner=build_bmam_reasoner,  # BMAM cognitive reasoning
        build_actor=build_llm_actor,         # reuse LLM actor
        build_interactor=build_llm_interactor,
        build_evolver=build_llm_evolver,
    ))


def _register_model_backend() -> None:
    """Register the 'model' backend (neural plasticity memory model)."""
    from .adapters.model import build_model_memory, build_model_plasticity
    from .defaults import build_llm_sense, build_llm_scene
    from .defaults import build_llm_reasoner, build_llm_actor, build_llm_interactor
    from .defaults.llm_evolver import build_llm_evolver

    register_backend("model", BackendFactory(
        name="model",
        description="Neural plasticity memory model adapter",
        build_sense=build_llm_sense,
        build_scene=build_llm_scene,
        build_memory=build_model_memory,       # neural memory
        build_reasoner=build_llm_reasoner,
        build_actor=build_llm_actor,
        build_interactor=build_llm_interactor,
        build_plasticity=build_model_plasticity,  # plasticity hooks
        build_evolver=build_llm_evolver,
    ))


def ensure_backends_registered() -> None:
    """Lazily register all built-in backends on first use."""
    if _BACKENDS:
        return
    _register_default_backend()
    try:
        _register_llm_backend()
    except Exception as e:
        logger.debug("LLM backend not available: %s", e)
    try:
        _register_bmam_backend()
    except Exception as e:
        logger.debug("BMAM backend not available: %s", e)
    try:
        _register_model_backend()
    except Exception as e:
        logger.debug("Model backend not available: %s", e)
