from __future__ import annotations

import fnmatch
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional


ToolHandler = Callable[..., Any]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    handler: ToolHandler

    def execute(self, **kwargs: Any) -> Any:
        return self.handler(**kwargs)


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    name: str
    ok: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class ToolPolicy:
    def __init__(self, allow: Optional[Iterable[str]] = None, deny: Optional[Iterable[str]] = None) -> None:
        self.allow = [pattern.strip().lower() for pattern in allow or [] if str(pattern).strip()]
        self.deny = [pattern.strip().lower() for pattern in deny or [] if str(pattern).strip()]

    def is_allowed(self, name: str) -> bool:
        normalized = name.strip().lower()
        if self._matches(self.deny, normalized):
            return False
        if not self.allow:
            return True
        return self._matches(self.allow, normalized)

    @staticmethod
    def _matches(patterns: List[str], name: str) -> bool:
        for pattern in patterns:
            if pattern == "*" or fnmatch.fnmatch(name, pattern):
                return True
        return False


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool], policy: Optional[ToolPolicy] = None) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self.policy = policy
        self._retired: Dict[str, Tool] = {}  # tools removed but kept for audit

    def list(self) -> List[Tool]:
        return list(self._tools.values())

    def is_allowed(self, name: str) -> bool:
        if not self.policy:
            return True
        return self.policy.is_allowed(name)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def execute(self, call: ToolCall) -> ToolResult:
        name = call.name
        if not self.is_allowed(name):
            return ToolResult(name=name, ok=False, error="tool denied by policy")

        tool = self.get(name)
        if not tool:
            return ToolResult(name=name, ok=False, error="unknown tool")

        start = time.perf_counter()
        try:
            output = tool.execute(**(call.arguments or {}))
            duration_ms = (time.perf_counter() - start) * 1000
            return ToolResult(name=name, ok=True, output=output, duration_ms=duration_ms)
        except Exception as exc:  # pragma: no cover - defensive
            duration_ms = (time.perf_counter() - start) * 1000
            return ToolResult(name=name, ok=False, error=str(exc), duration_ms=duration_ms)

    # -- Dynamic tool management -------------------------------------------

    def register(self, tool: Tool) -> None:
        """Add a tool at runtime.  Overwrites if name already exists."""
        self._tools[tool.name] = tool
        self._retired.pop(tool.name, None)

    def retire(self, name: str) -> Optional[Tool]:
        """Remove a tool, keeping it in the retired archive for audit."""
        tool = self._tools.pop(name, None)
        if tool is not None:
            self._retired[name] = tool
        return tool

    def restore(self, name: str) -> bool:
        """Restore a previously retired tool."""
        tool = self._retired.pop(name, None)
        if tool is not None:
            self._tools[name] = tool
            return True
        return False

    def list_retired(self) -> List[Tool]:
        return list(self._retired.values())

    def has(self, name: str) -> bool:
        return name in self._tools
