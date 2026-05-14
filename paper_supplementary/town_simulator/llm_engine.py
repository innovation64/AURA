"""LLM engine for AURA Town using OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from .config import TownConfig

logger = logging.getLogger(__name__)


class LLMEngine:
    """Wrapper around OpenAI-compatible API with retry logic."""

    def __init__(self, config: TownConfig, seed: Optional[int] = None) -> None:
        self.config = config
        self._client: Any = None
        self._last_call_time: float = 0.0
        self._min_interval: float = 0.1  # min seconds between calls
        # When set, every chat completion call passes seed=<value> to the
        # OpenAI Chat Completions API for deterministic-when-supported
        # output. Providers that don't support the kwarg will raise; we
        # catch and fall back to no-seed inside chat().
        self._seed: Optional[int] = seed
        self._seed_supported: bool = True  # flips False after first rejection

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package required: pip install openai"
                )
            kwargs: Dict[str, Any] = {
                "api_key": self.config.openai_api_key,
                "timeout": self.config.request_timeout,
            }
            if self.config.openai_base_url:
                kwargs["base_url"] = self.config.openai_base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_time = time.time()

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a chat completion request with retry logic."""
        # Fast-fail when no API key is configured
        if not self.config.openai_api_key or not self.config.openai_api_key.strip():
            raise RuntimeError("No OpenAI API key configured — using fallback.")

        client = self._get_client()
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                self._rate_limit()
                kwargs: Dict[str, Any] = dict(
                    model=self.config.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                )
                if self._seed is not None and self._seed_supported:
                    kwargs["seed"] = self._seed
                try:
                    response = client.chat.completions.create(**kwargs)
                except TypeError:
                    # SDK rejects seed kwarg — disable for the rest of this run
                    self._seed_supported = False
                    kwargs.pop("seed", None)
                    response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                return content.strip() if content else ""
            except Exception as e:
                last_error = e
                # 401 = invalid key, retrying won't help
                err_str = str(e)
                if "401" in err_str or "authentication" in err_str.lower():
                    logger.warning("LLM auth error (no retry): %s", e)
                    break
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.config.max_retries,
                    e,
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))

        raise RuntimeError(f"LLM call failed after {self.config.max_retries} retries: {last_error}")

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send a chat request and parse the response as JSON."""
        raw = self.chat(messages, temperature, max_tokens)
        # Try to extract JSON from response
        text = raw.strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines[1:] if not l.strip().startswith("```")]
            text = "\n".join(lines)
        # Attempt 1: direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Attempt 2: find first { ... } or [ ... ] in text
        for open_ch, close_ch in [("{", "}"), ("[", "]")]:
            start = text.find(open_ch)
            if start == -1:
                continue
            depth = 0
            for i in range(start, len(text)):
                if text[i] == open_ch:
                    depth += 1
                elif text[i] == close_ch:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break
        logger.warning("Failed to parse LLM JSON response: %s", raw[:200])
        return {"raw": raw}

    def score_importance(self, description: str) -> int:
        """Rate the importance of a memory on a scale of 1-10."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You rate the importance of events/observations for a person's memory "
                    "on a scale of 1 (mundane, routine) to 10 (life-changing, critical). "
                    "Respond with ONLY a single integer."
                ),
            },
            {
                "role": "user",
                "content": f"Rate the importance (1-10): {description}",
            },
        ]
        try:
            result = self.chat(messages, temperature=0.0, max_tokens=8)
            score = int(result.strip().split()[0])
            return max(1, min(10, score))
        except (ValueError, IndexError, RuntimeError):
            return 5  # default moderate importance
