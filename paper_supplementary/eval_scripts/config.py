"""Experiment configuration for AURA paper evaluation."""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class EvalConfig:
    # LLM settings — backbone model for agent reasoning
    model: str = "gpt-4o-mini"
    # Judge model — ideally stronger than backbone; using same model to control cost
    judge_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    judge_temperature: float = 0.1  # Low temp for consistent judging

    # API
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))

    # AURA server
    aura_server: str = "http://127.0.0.1:7861"

    # Experiment parameters
    num_simulation_steps: int = 100     # RQ1: steps to run
    num_chat_queries: int = 50          # RQ2: number of test queries
    num_social_episodes: int = 30       # RQ3: social interaction episodes
    num_memory_tests: int = 100         # RQ4: memory retrieval test cases
    probe_budgets: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])  # RQ6

    # Output
    results_dir: str = "evaluation/results"
    seed: int = 42

    # Multi-seed for statistical significance
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    current_seed: int = 42  # set during iteration


# SOTOPIA-style 7-dimension evaluation rubric
SOTOPIA_DIMENSIONS = {
    "believability": {
        "description": "How natural, authentic, and consistent with the character's personality is the behavior?",
        "scale": (0, 10),
    },
    "relationship": {
        "description": "Did the interaction improve or damage the relationship between agents?",
        "scale": (-5, 5),
    },
    "knowledge": {
        "description": "Did the agent demonstrate relevant knowledge and curiosity about its environment?",
        "scale": (0, 10),
    },
    "secret": {
        "description": "Did the agent properly protect confidential or private information?",
        "scale": (-10, 0),
    },
    "social_rules": {
        "description": "Did the agent follow social norms, etiquette, and conventions?",
        "scale": (-10, 0),
    },
    "financial": {
        "description": "Were there material gains or losses from the interaction?",
        "scale": (-5, 5),
    },
    "goal": {
        "description": "Did the agent make progress toward or achieve its stated objectives?",
        "scale": (0, 10),
    },
}

# Grounding categories for action evaluation
GROUNDING_CATEGORIES = [
    "location_consistency",   # Is the action consistent with the agent's current location?
    "time_appropriateness",   # Is the action appropriate for the current time of day?
    "social_awareness",       # Does the action account for nearby agents?
    "memory_utilization",     # Does the action reflect past experiences?
    "plan_adherence",         # Does the action align with the daily plan?
]
