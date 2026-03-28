"""Network probe -- monitors service availability via TCP and HTTP checks."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from aura.types import EnvironmentSignal

from .base import Probe, ProbeResult


async def _tcp_check(host: str, port: int, timeout: float = 5.0) -> Tuple[bool, float]:
    """Attempt a TCP connection. Returns (success, latency_ms)."""
    t0 = time.time()
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        latency = (time.time() - t0) * 1000
        writer.close()
        await writer.wait_closed()
        return True, round(latency, 2)
    except (OSError, asyncio.TimeoutError, ConnectionRefusedError):
        latency = (time.time() - t0) * 1000
        return False, round(latency, 2)


async def _http_check(url: str, timeout: float = 5.0) -> Tuple[bool, int, float]:
    """Perform a minimal HTTP GET. Returns (success, status_code, latency_ms).

    Uses raw TCP + HTTP/1.1 to avoid external dependencies.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    # Skip HTTPS -- we only do plain HTTP health checks to avoid ssl complexity.
    if parsed.scheme == "https":
        # Fall back to TCP-only check for HTTPS endpoints.
        ok, lat = await _tcp_check(host, port, timeout)
        return ok, 0, lat

    t0 = time.time()
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        request = f"GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"
        writer.write(request.encode())
        await writer.drain()

        status_line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        latency = (time.time() - t0) * 1000
        writer.close()
        await writer.wait_closed()

        # Parse "HTTP/1.1 200 OK"
        parts = status_line.decode(errors="replace").split(None, 2)
        status_code = int(parts[1]) if len(parts) >= 2 else 0
        success = 200 <= status_code < 500
        return success, status_code, round(latency, 2)
    except (OSError, asyncio.TimeoutError, Exception):
        latency = (time.time() - t0) * 1000
        return False, 0, round(latency, 2)


class NetworkProbe(Probe):
    """Monitors network service availability and latency."""

    def __init__(
        self,
        tcp_services: Optional[List[Tuple[str, int, str]]] = None,
        http_urls: Optional[List[Tuple[str, str]]] = None,
        connect_timeout: float = 5.0,
        latency_anomaly_ms: float = 1000.0,
    ) -> None:
        """
        Args:
            tcp_services: List of (host, port, name) tuples for TCP checks.
            http_urls: List of (url, name) tuples for HTTP health checks.
            connect_timeout: Timeout in seconds for each connection attempt.
            latency_anomaly_ms: Threshold above which latency is flagged.
        """
        super().__init__()
        self._tcp_services = tcp_services or []
        self._http_urls = http_urls or []
        self._timeout = connect_timeout
        self._latency_threshold = latency_anomaly_ms
        # Previous state: service_name -> was_up
        self._prev_status: Dict[str, bool] = {}

    @property
    def name(self) -> str:
        return "network"

    @property
    def interval_seconds(self) -> float:
        return 30.0

    async def poll(self) -> ProbeResult:
        t0 = time.time()
        signals: List[EnvironmentSignal] = []

        try:
            # --- TCP checks ---
            tcp_tasks = []
            for host, port, svc_name in self._tcp_services:
                tcp_tasks.append((svc_name, host, port, _tcp_check(host, port, self._timeout)))

            # --- HTTP checks ---
            http_tasks = []
            for url, svc_name in self._http_urls:
                http_tasks.append((svc_name, url, _http_check(url, self._timeout)))

            # Run all checks in parallel.
            current_status: Dict[str, bool] = {}

            if tcp_tasks:
                tcp_coros = [t[3] for t in tcp_tasks]
                tcp_results = await asyncio.gather(*tcp_coros, return_exceptions=True)
                for (svc_name, host, port, _), result in zip(tcp_tasks, tcp_results):
                    if isinstance(result, Exception):
                        is_up, latency = False, 0.0
                    else:
                        is_up, latency = result

                    current_status[svc_name] = is_up
                    was_up = self._prev_status.get(svc_name)

                    # Emit signal on state transition.
                    if was_up is not None and was_up != is_up:
                        event = "service_up" if is_up else "service_down"
                        signals.append(
                            EnvironmentSignal(
                                source="probe.network",
                                modality="network",
                                payload={
                                    "event": event,
                                    "service": svc_name,
                                    "host": host,
                                    "port": port,
                                    "latency_ms": latency,
                                },
                                confidence=1.0,
                            )
                        )
                    elif is_up and latency > self._latency_threshold:
                        signals.append(
                            EnvironmentSignal(
                                source="probe.network",
                                modality="network",
                                payload={
                                    "event": "high_latency",
                                    "service": svc_name,
                                    "host": host,
                                    "port": port,
                                    "latency_ms": latency,
                                    "threshold_ms": self._latency_threshold,
                                },
                                confidence=0.9,
                            )
                        )

            if http_tasks:
                http_coros = [t[2] for t in http_tasks]
                http_results = await asyncio.gather(*http_coros, return_exceptions=True)
                for (svc_name, url, _), result in zip(http_tasks, http_results):
                    if isinstance(result, Exception):
                        is_up, status_code, latency = False, 0, 0.0
                    else:
                        is_up, status_code, latency = result

                    current_status[svc_name] = is_up
                    was_up = self._prev_status.get(svc_name)

                    if was_up is not None and was_up != is_up:
                        event = "service_up" if is_up else "service_down"
                        signals.append(
                            EnvironmentSignal(
                                source="probe.network",
                                modality="network",
                                payload={
                                    "event": event,
                                    "service": svc_name,
                                    "url": url,
                                    "status_code": status_code,
                                    "latency_ms": latency,
                                },
                                confidence=1.0,
                            )
                        )
                    elif is_up and latency > self._latency_threshold:
                        signals.append(
                            EnvironmentSignal(
                                source="probe.network",
                                modality="network",
                                payload={
                                    "event": "high_latency",
                                    "service": svc_name,
                                    "url": url,
                                    "status_code": status_code,
                                    "latency_ms": latency,
                                    "threshold_ms": self._latency_threshold,
                                },
                                confidence=0.9,
                            )
                        )

            self._prev_status = current_status

        except Exception as exc:
            signals.append(
                EnvironmentSignal(
                    source="probe.network",
                    modality="network",
                    payload={"error": str(exc)},
                    confidence=0.0,
                )
            )

        latency = (time.time() - t0) * 1000
        return ProbeResult(
            source=self.name,
            timestamp=t0,
            signals=signals,
            latency_ms=round(latency, 2),
        )
