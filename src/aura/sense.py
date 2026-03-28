from __future__ import annotations

from typing import Any, List

from .types import EnvironmentSignal


class SenseAdapter:
    def ingest(self, raw: Any) -> List[EnvironmentSignal]:
        raise NotImplementedError


class BasicSense(SenseAdapter):
    def __init__(self, source: str = "manual") -> None:
        self.source = source

    def ingest(self, raw: Any) -> List[EnvironmentSignal]:
        if raw is None:
            return []

        if isinstance(raw, list):
            return [self._coerce_signal(item) for item in raw]

        return [self._coerce_signal(raw)]

    def _coerce_signal(self, item: Any) -> EnvironmentSignal:
        if isinstance(item, EnvironmentSignal):
            return item
        if isinstance(item, dict):
            return EnvironmentSignal(source=self.source, payload=item)
        return EnvironmentSignal(source=self.source, payload={"value": item})
