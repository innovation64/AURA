"""Configuration for AURA Town simulation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Load .env from project root so API keys are available in os.environ
try:
    from dotenv import load_dotenv

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass  # python-dotenv not installed; rely on shell environment


@dataclass
class TownConfig:
    # OpenAI settings
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    openai_base_url: Optional[str] = field(
        default_factory=lambda: os.environ.get("OPENAI_BASE_URL")
    )
    model: str = field(
        default_factory=lambda: os.environ.get("AURA_TOWN_MODEL", "gpt-4o-mini")
    )
    temperature: float = 0.7
    max_tokens: int = 512

    # LLM call settings
    max_retries: int = 2
    retry_delay: float = 0.5
    request_timeout: float = 10.0

    # Simulation settings
    grid_width: int = 60
    grid_height: int = 60
    tick_minutes: int = 30
    start_hour: int = 6
    end_hour: int = 23

    # Memory settings
    max_memories: int = 200
    reflection_threshold: int = 10  # reflect every N observations
    importance_weight: float = 0.4
    recency_weight: float = 0.3
    relevance_weight: float = 0.3

    # Agent settings
    agent_count: int = 5

    # Active probe settings
    probe_enabled: bool = True
    probe_max_steps: int = 3
    probe_cooldown_ticks: int = 2

    # Evolution settings
    evolve_enabled: bool = True
    evolve_interval: int = 5
    evolve_max_mutations: int = 3

    # Procedural evolution settings
    procedural_evolve_enabled: bool = True
    weather_transition_interval: int = 3
    season_length: int = 40
    micro_event_probability: float = 0.3

    # Chunk / infinite world settings
    chunk_size: int = 16
    world_seed: int = 42

    # LLM seed: when set, every chat-completion call passes seed=<value>
    # to the OpenAI Chat Completions API so backbone draws are
    # deterministic-when-supported. Together with `world_seed` this is
    # what `_run_multi_seed` callers should override per seed; otherwise
    # the "3 seeds" only differ in Python-level random sampling, not
    # backbone output.
    llm_seed: Optional[int] = None

    # Image generation API settings (Phase 3C)
    image_gen_api_url: str = field(
        default_factory=lambda: os.environ.get("AURA_IMAGE_GEN_URL", "")
    )
    image_gen_api_key: str = field(
        default_factory=lambda: os.environ.get("AURA_IMAGE_GEN_KEY", "")
    )

    @property
    def llm_available(self) -> bool:
        """True when an API key is configured (non-empty)."""
        return bool(self.openai_api_key and self.openai_api_key.strip())

    def __post_init__(self) -> None:
        if not self.llm_available:
            # No key → only try once so fallback kicks in instantly
            self.max_retries = 1
            self.request_timeout = 5.0


DEFAULT_CONFIG = TownConfig()
