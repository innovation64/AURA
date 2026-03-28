"""Shared LLM engine for AURA — supports any OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM backend."""

    api_key: str = ""
    base_url: Optional[str] = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1024
    max_retries: int = 1
    retry_delay: float = 0.2

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Load from environment (compatible with BMAM's .env pattern)."""
        import os
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"),
            model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS", "1024")),
        )


class LLMEngine:
    """Lightweight wrapper around OpenAI-compatible chat API."""

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig()
        self._client: Any = None
        self._last_call: float = 0.0

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package required: pip install openai")
            kwargs: Dict[str, Any] = {"api_key": self.config.api_key}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        client = self._get_client()
        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            try:
                # Simple rate limiting
                elapsed = time.time() - self._last_call
                if elapsed < 0.1:
                    time.sleep(0.1 - elapsed)
                self._last_call = time.time()

                response = client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=tokens,
                )
                content = response.choices[0].message.content
                return content.strip() if content else ""
            except Exception as e:
                last_error = e
                logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, self.config.max_retries, e)
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))

        raise RuntimeError(f"LLM call failed after {self.config.max_retries} retries: {last_error}")

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        raw = self.chat(messages, temperature, max_tokens)
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines[1:] if not line.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON: %s", raw[:200])
            return {"raw": raw}
