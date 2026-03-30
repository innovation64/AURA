"""BMAM HTTP client — connects to BMAM's FastAPI REST API.

BMAM runs as a service (default: localhost:8100) with endpoints:
  POST /v1/memories           — store memory
  POST /v1/memories/search    — semantic search
  POST /v1/brain/retrieve     — 5-brain-region distributed retrieval
  POST /v1/brain/process      — full pipeline (store + retrieve + reason)
  POST /v1/brain/consolidate  — trigger consolidation
  POST /v1/brain/forget       — trigger forgetting
  POST /v1/brain/feedback     — submit learning feedback
  GET  /v1/system/health      — health check

AURA calls BMAM through this client, never importing BMAM internals directly.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)


@dataclass
class BMAMConfig:
    """Connection settings for BMAM API service."""

    base_url: str = "http://localhost:8100"
    api_key: str = ""
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "BMAMConfig":
        return cls(
            base_url=os.getenv("BMAM_API_URL", os.getenv("BMAM_BASE_URL", "http://localhost:8100")),
            api_key=os.getenv("BMAM_API_KEY", ""),
            timeout=float(os.getenv("BMAM_TIMEOUT", "30")),
        )


class BMAMClient:
    """HTTP client for BMAM REST API.

    All BMAM operations go through this client.
    No direct import of BMAM Python code.
    """

    def __init__(self, config: Optional[BMAMConfig] = None) -> None:
        self.config = config or BMAMConfig.from_env()
        self._base = self.config.base_url.rstrip("/")

    def _request(self, method: str, path: str, body: Optional[Dict] = None) -> Dict[str, Any]:
        url = f"{self._base}{path}"
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key

        data = json.dumps(body).encode("utf-8") if body else None

        # Use requests if available (follows redirects for POST/307)
        try:
            import requests as _req
            if method == "GET":
                resp = _req.get(url, headers=headers, timeout=self.config.timeout)
            else:
                resp = _req.post(url, data=data, headers=headers, timeout=self.config.timeout)
            resp.raise_for_status()
            return resp.json()
        except ImportError:
            pass
        except Exception as e:
            logger.error("BMAM API request failed: %s %s → %s", method, url, e)
            raise ConnectionError(f"BMAM API unavailable at {self._base}: {e}") from e

        # Fallback: urllib (does not follow 307 redirects for POST)
        req = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=self.config.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except URLError as e:
            logger.error("BMAM API request failed: %s %s → %s", method, url, e)
            raise ConnectionError(f"BMAM API unavailable at {self._base}: {e}") from e

    def _get(self, path: str) -> Dict[str, Any]:
        return self._request("GET", path)

    def _post(self, path: str, body: Dict) -> Dict[str, Any]:
        return self._request("POST", path, body)

    # ── Health ────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        return self._get("/v1/system/health")

    def is_available(self) -> bool:
        try:
            self.health()
            return True
        except (ConnectionError, Exception):
            return False

    # ── Memory CRUD ───────────────────────────────────────────

    def store_memory(
        self,
        content: str,
        user_id: str = "default",
        importance: float = 0.5,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """POST /v1/memories — store a new memory."""
        return self._post("/v1/memories", {
            "content": content,
            "user_id": user_id,
            "importance": importance,
            "metadata": metadata or {},
        })

    # ── Search / Retrieval ────────────────────────────────────

    def search(
        self,
        query: str,
        limit: int = 5,
        use_brain_retrieval: bool = False,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """POST /v1/memories/search — semantic or brain-distributed search."""
        return self._post("/v1/memories/search", {
            "query": query,
            "limit": limit,
            "use_brain_retrieval": use_brain_retrieval,
            "context": context or {},
        })

    def brain_retrieve(
        self,
        query: str,
        k: int = 5,
        context: Optional[Dict] = None,
        force_slow_path: bool = False,
    ) -> Dict[str, Any]:
        """POST /v1/brain/retrieve — 5-brain-region distributed retrieval."""
        return self._post("/v1/brain/retrieve", {
            "query": query,
            "k": k,
            "context": context or {},
            "force_slow_path": force_slow_path,
        })

    def process_input(
        self,
        user_input: str,
        user_id: str = "default",
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """POST /v1/brain/process — full pipeline."""
        return self._post("/v1/brain/process", {
            "input": user_input,
            "user_id": user_id,
            "context": context or {},
        })

    # ── Brain Operations ──────────────────────────────────────

    def consolidate(self, evaluation_mode: bool = False) -> Dict[str, Any]:
        """POST /v1/brain/consolidate."""
        return self._post("/v1/brain/consolidate", {
            "evaluation_mode": evaluation_mode,
        })

    def forget(self, capacity_threshold: float = 0.8) -> Dict[str, Any]:
        """POST /v1/brain/forget."""
        return self._post("/v1/brain/forget", {
            "capacity_threshold": capacity_threshold,
        })

    def feedback(
        self,
        query: str,
        response: str,
        reward_signal: float,
        query_type: str = "general",
    ) -> Dict[str, Any]:
        """POST /v1/brain/feedback — submit learning feedback."""
        return self._post("/v1/brain/feedback", {
            "query": query,
            "response": response,
            "reward_signal": reward_signal,
            "query_type": query_type,
        })

    def get_preferences(
        self, query: str, user_id: str = "default", k: int = 5,
    ) -> Dict[str, Any]:
        """POST /v1/brain/preferences/ — retrieve preferences relevant to query."""
        return self._post("/v1/brain/preferences/", {
            "query": query,
            "user_id": user_id,
            "k": k,
        })

    def get_persona_portrait(
        self, user_id: str = "default",
    ) -> Dict[str, Any]:
        """POST /v1/brain/persona/portrait/ — synthesize user persona portrait."""
        return self._post("/v1/brain/persona/portrait/", {
            "user_id": user_id,
        })

    # ── Archives (Soul Transfer) ─────────────────────────────

    def export_archive(self, archive_name: str = "aura_export") -> Dict[str, Any]:
        """POST /v1/archives/export/ — export memory archive (.bma)."""
        return self._post("/v1/archives/export/", {
            "archive_name": archive_name,
        })

    def import_archive(self, archive_path: str) -> Dict[str, Any]:
        """POST /v1/archives/import/ — import memory archive."""
        return self._post("/v1/archives/import/", {
            "archive_path": archive_path,
        })

    # ── Soul State ───────────────────────────────────────────

    def get_component_health(self) -> Dict[str, Any]:
        """GET /v1/brain/health/components — get brain region health status."""
        return self._get("/v1/brain/health/components")
