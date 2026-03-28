"""Simulated agent policies for paradigm comparison experiments.

Four agent types with increasing sophistication:
- RandomAgent:      Baseline — random tool calls and generic actions
- ReactiveAgent:    Systematically queries tools, reacts to results
- AdaptiveAgent:    Uses pushed context when available, adapts behavior
- RealisticAgent:   Noisy agent with source preferences — creates room for
                    the attention tracker to learn (avoids ceiling effect)
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set

from aura.paradigm.base import AgentObservation, AgentPolicy, AgentResponse


class RandomAgent(AgentPolicy):
    """Baseline: picks random tools and generic actions."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._tools = ["system.snapshot", "git.status", "docker.status",
                        "process.list", "service.check"]
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def act(self, observation: AgentObservation) -> AgentResponse:
        self._step += 1

        # Randomly pick a tool to call
        tool = self._rng.choice(self._tools)
        tool_calls = [{"tool": tool, "args": {}}]

        actions = ["check status", "wait", "review logs", "idle", "continue work"]
        action = self._rng.choice(actions)

        if self._step > 15:
            action = "done"

        return AgentResponse(
            action=action,
            tool_calls=tool_calls,
            used_pushed_context=False,
            reasoning="random action",
        )


class ReactiveAgent(AgentPolicy):
    """Systematic reactive agent: queries tools in order, reacts to findings.

    Simulates a well-designed agent that DOESN'T have proactive context.
    It checks tools periodically and reacts when it finds issues.
    """

    def __init__(self):
        self._step = 0
        self._check_cycle = ["system.snapshot", "service.check", "git.status",
                              "process.list", "docker.status"]
        self._found_issues: List[str] = []
        self._last_tool_results: Dict = {}

    def reset(self) -> None:
        self._step = 0
        self._found_issues = []
        self._last_tool_results = {}

    def act(self, observation: AgentObservation) -> AgentResponse:
        self._step += 1
        env = observation.environment_state

        # Systematic tool check cycle
        tool_idx = (self._step - 1) % len(self._check_cycle)
        tool = self._check_cycle[tool_idx]
        tool_calls = [{"tool": tool, "args": {}}]

        # Analyze what we got from last observation (tool results from prev step)
        action = self._analyze_and_act(env)

        if self._step > 18:
            action = "done"

        return AgentResponse(
            action=action,
            tool_calls=tool_calls,
            used_pushed_context=False,
            reasoning=f"Reactive check: {tool}",
        )

    def _analyze_and_act(self, env: Dict) -> str:
        """Analyze environment state from tool results and decide action.

        Tool results are nested: {"tool_name": {result_dict}}.
        We need to look inside each tool's results.
        """
        # Flatten tool results into a unified view
        flat: Dict = {}
        for key, val in env.items():
            if isinstance(val, dict):
                # Nested tool result — merge its contents
                flat.update(val)
            else:
                flat[key] = val

        # Check for service issues
        services = flat.get("services", [])
        if isinstance(services, list):
            for svc in services:
                if isinstance(svc, dict) and svc.get("status") == "down":
                    issue_key = f"service_down:{svc.get('name')}"
                    if issue_key not in self._found_issues:
                        self._found_issues.append(issue_key)
                        return f"investigate service failure: {svc.get('name')} is down"

        # Check for suspicious processes
        processes = flat.get("processes", [])
        if isinstance(processes, list):
            for proc in processes:
                if isinstance(proc, dict) and proc.get("cpu", 0) > 80:
                    issue_key = f"suspicious:{proc.get('name')}"
                    if issue_key not in self._found_issues:
                        self._found_issues.append(issue_key)
                        return f"investigate suspicious process: {proc.get('name')}"

        # Check for alerts (may be in raw state or tool results)
        alerts = flat.get("alerts", []) or env.get("alerts", [])
        if alerts and not any("alert_seen" in i for i in self._found_issues):
            self._found_issues.append("alert_seen")
            return f"investigate alert: {alerts[-1].get('type', 'unknown')}"

        # Check for conflicts
        conflicts = flat.get("conflicts", []) or env.get("conflicts", [])
        if conflicts:
            return f"resolve file conflict: {conflicts[0]}"

        # Check resource metrics (may be nested in system.snapshot result)
        for src in [flat, env]:
            cpu = src.get("cpu", 0)
            mem = src.get("memory", 0)
            gpu = src.get("gpu_memory", 0)
            if isinstance(cpu, (int, float)) and cpu > 90:
                return "investigate resource spike: high CPU utilization detected"
            if isinstance(mem, (int, float)) and mem > 90:
                return "investigate resource spike: high memory utilization detected"
            if isinstance(gpu, (int, float)) and gpu > 90:
                return "investigate resource spike: high GPU memory utilization"

        return "continue monitoring"


class AdaptiveAgent(AgentPolicy):
    """Smart agent that uses pushed context when available.

    Simulates an agent that:
    - Reads proactive context (alerts, hints, summary) when provided
    - Acts immediately on critical alerts
    - Uses hints to guide tool selection
    - Falls back to reactive behavior when no context is pushed
    """

    def __init__(self):
        self._step = 0
        self._found_issues: List[str] = []

    def reset(self) -> None:
        self._step = 0
        self._found_issues = []

    def act(self, observation: AgentObservation) -> AgentResponse:
        self._step += 1
        ctx = observation.pushed_context

        if self._step > 18:
            return AgentResponse(action="done", used_pushed_context=False)

        # If we have pushed context, use it
        if ctx:
            return self._act_on_context(ctx, observation)

        # Fallback: basic reactive behavior
        return self._reactive_fallback(observation)

    def _act_on_context(self, ctx: Dict, observation: AgentObservation) -> AgentResponse:
        """Act based on proactive pushed context."""
        critical = ctx.get("critical_alerts", [])
        changes = ctx.get("relevant_changes", [])
        hints = ctx.get("hints", [])

        # Critical alerts — immediate action
        if critical:
            alert = critical[0]
            desc = alert.get("description", "")
            source = alert.get("source", "")

            # Determine appropriate response
            if "service" in source or "down" in desc.lower():
                action = f"investigate critical alert: {desc}; attempting restart"
                tool_calls = [{"tool": "service.check", "args": {}}]
            elif "process" in source or "suspicious" in desc.lower():
                action = f"investigate suspicious activity: {desc}; checking processes"
                tool_calls = [{"tool": "process.list", "args": {}}]
            elif "spike" in desc.lower() or "resource" in source:
                action = f"investigate resource spike: {desc}; checking system"
                tool_calls = [{"tool": "system.snapshot", "args": {}}]
            else:
                action = f"investigate critical alert: {desc}"
                tool_calls = [{"tool": "system.snapshot", "args": {}}]

            self._found_issues.append(f"critical:{desc[:50]}")
            return AgentResponse(
                action=action,
                tool_calls=tool_calls,
                used_pushed_context=True,
                reasoning=f"Responding to critical alert: {desc[:80]}",
            )

        # Relevant changes — investigate
        if changes:
            change = changes[0]
            desc = change.get("description", "")
            action = f"investigate change: {desc}"

            # Use hints for tool selection
            tool = "system.snapshot"
            if hints:
                hint = hints[0].lower()
                if "file" in hint or "re-read" in hint:
                    tool = "git.status"
                elif "service" in hint or "restart" in hint:
                    tool = "service.check"
                elif "docker" in hint or "container" in hint:
                    tool = "docker.status"

            return AgentResponse(
                action=action,
                tool_calls=[{"tool": tool, "args": {}}],
                used_pushed_context=True,
                reasoning=f"Responding to change: {desc[:80]}",
            )

        # Hints but no alerts/changes — use hints
        if hints:
            action = f"following hint: {hints[0][:80]}"
            return AgentResponse(
                action=action,
                tool_calls=[],
                used_pushed_context=True,
                reasoning="Following proactive hint",
            )

        return self._reactive_fallback(observation)

    def _reactive_fallback(self, observation: AgentObservation) -> AgentResponse:
        """Fallback when no context is pushed."""
        env = observation.environment_state

        # Quick check for obvious issues in raw state
        alerts = env.get("alerts", [])
        if alerts:
            return AgentResponse(
                action=f"investigate alert: {alerts[-1].get('type', 'unknown')}",
                tool_calls=[{"tool": "system.snapshot", "args": {}}],
                used_pushed_context=False,
            )

        conflicts = env.get("conflicts", [])
        if conflicts:
            return AgentResponse(
                action=f"resolve file conflict: {conflicts[0]}",
                tool_calls=[{"tool": "git.status", "args": {}}],
                used_pushed_context=False,
            )

        return AgentResponse(
            action="continue monitoring",
            tool_calls=[{"tool": "system.snapshot", "args": {}}],
            used_pushed_context=False,
        )


class RealisticAgent(AgentPolicy):
    """Agent with source preferences and stochastic context usage.

    Simulates a realistic LLM-based agent that:
    - Has domain-specific preferences (trusts some sources more than others)
    - Sometimes misses or ignores low-severity pushed context
    - Gradually improves at recognizing relevant sources through experience
    - Trust grows ONLY via actual reinforcement (not episode count)

    This avoids the ceiling effect of AdaptiveAgent, enabling the collaborative
    paradigm to demonstrate convergence.
    """

    def __init__(self, seed: int = 42, initial_trust: float = 0.4):
        self._rng = random.Random(seed)
        self._step = 0
        self._found_issues: List[str] = []
        self._initial_trust = initial_trust
        # Source-specific trust: grows only via reinforcement
        self._source_trust: Dict[str, float] = {}
        # Sources that led to useful actions get reinforced
        self._useful_sources: Set[str] = set()
        # Per-source reinforcement count (persists across episodes)
        self._source_reinforcements: Dict[str, int] = {}
        # Total hints used (for hint trust, grows via actual usage)
        self._hints_used: int = 0
        self._hints_offered: int = 0

    def reset(self) -> None:
        self._step = 0
        self._found_issues = []
        # NOTE: source_trust, reinforcements, hints persist across episodes

    def act(self, observation: AgentObservation) -> AgentResponse:
        self._step += 1
        ctx = observation.pushed_context

        if self._step > 18:
            return AgentResponse(action="done", used_pushed_context=False)

        if ctx:
            return self._maybe_use_context(ctx, observation)

        return self._reactive_fallback(observation)

    def _maybe_use_context(self, ctx: Dict, observation: AgentObservation) -> AgentResponse:
        """Probabilistically use pushed context based on source trust.

        Trust is source-specific and grows only through reinforcement,
        not through episode count. This creates realistic learning curves.
        """
        critical = ctx.get("critical_alerts", [])
        changes = ctx.get("relevant_changes", [])
        hints = ctx.get("hints", [])

        # Critical alerts: use with probability driven by source trust + severity
        if critical:
            alert = critical[0]
            source = alert.get("source", "unknown")
            severity = alert.get("severity", 0.5)
            trust = self._get_source_trust(source)

            # Higher severity and trust → more likely to use
            use_prob = min(0.95, trust * 0.6 + severity * 0.4)
            if self._rng.random() < use_prob:
                self._reinforce_source(source)
                return self._respond_to_alert(alert, used_context=True)
            else:
                return self._reactive_fallback(observation)

        # Relevant changes: use based on source trust AND severity
        # Low-severity changes from untrusted sources are very likely to be ignored,
        # creating the asymmetric feedback signal that drives weight differentiation
        if changes:
            change = changes[0]
            source = change.get("source", "unknown")
            severity = change.get("severity", 0.5)
            trust = self._get_source_trust(source)

            # Severity matters more for non-critical changes
            # Low trust + low severity → very likely ignored
            use_prob = min(0.85, trust * 0.4 + severity * 0.5 + 0.05)
            if self._rng.random() < use_prob:
                self._reinforce_source(source)
                desc = change.get("description", "")
                tool = self._pick_tool_for_change(change, hints)
                return AgentResponse(
                    action=f"investigate change: {desc}",
                    tool_calls=[{"tool": tool, "args": {}}],
                    used_pushed_context=True,
                    reasoning=f"Trusting source {source} (trust={trust:.2f}, sev={severity:.2f})",
                )
            else:
                return self._reactive_fallback(observation)

        # Hints only: trust grows based on actual hint usage rate
        if hints:
            self._hints_offered += 1
            # Trust based on historical success rate, not episode count
            if self._hints_offered > 0:
                hint_trust = max(
                    self._initial_trust,
                    self._hints_used / max(self._hints_offered, 1),
                )
            else:
                hint_trust = self._initial_trust
            use_prob = min(0.7, hint_trust)
            if self._rng.random() < use_prob:
                self._hints_used += 1
                return AgentResponse(
                    action=f"following hint: {hints[0][:80]}",
                    tool_calls=[],
                    used_pushed_context=True,
                    reasoning="Following proactive hint",
                )

        return self._reactive_fallback(observation)

    def _get_source_trust(self, source: str) -> float:
        """Get trust level for a source, starting at initial_trust."""
        if source not in self._source_trust:
            self._source_trust[source] = self._initial_trust
        return self._source_trust[source]

    def _reinforce_source(self, source: str) -> None:
        """Increase trust for a source that was used.

        Stronger reinforcement (+0.05) so that over 40 episodes,
        frequently-used sources can climb from 0.4 to 0.9+.
        """
        current = self._get_source_trust(source)
        self._source_trust[source] = min(0.95, current + 0.05)
        self._useful_sources.add(source)
        self._source_reinforcements[source] = self._source_reinforcements.get(source, 0) + 1

    def _respond_to_alert(self, alert: Dict, used_context: bool) -> AgentResponse:
        """Generate response to an alert."""
        desc = alert.get("description", "")
        source = alert.get("source", "")

        if "service" in source or "down" in desc.lower():
            action = f"investigate critical alert: {desc}; attempting restart"
            tool_calls = [{"tool": "service.check", "args": {}}]
        elif "process" in source or "suspicious" in desc.lower():
            action = f"investigate suspicious activity: {desc}; checking processes"
            tool_calls = [{"tool": "process.list", "args": {}}]
        elif "spike" in desc.lower() or "resource" in source:
            action = f"investigate resource spike: {desc}; checking system"
            tool_calls = [{"tool": "system.snapshot", "args": {}}]
        else:
            action = f"investigate critical alert: {desc}"
            tool_calls = [{"tool": "system.snapshot", "args": {}}]

        self._found_issues.append(f"alert:{desc[:50]}")
        return AgentResponse(
            action=action,
            tool_calls=tool_calls,
            used_pushed_context=used_context,
            reasoning=f"Alert response: {desc[:80]}",
        )

    def _pick_tool_for_change(self, change: Dict, hints: List) -> str:
        """Pick a tool based on the change type and hints."""
        desc = change.get("description", "").lower()
        source = change.get("source", "").lower()

        if hints:
            hint = hints[0].lower()
            if "file" in hint or "re-read" in hint:
                return "git.status"
            if "service" in hint or "restart" in hint:
                return "service.check"
            if "docker" in hint or "container" in hint:
                return "docker.status"

        if "service" in source or "network" in source:
            return "service.check"
        if "file" in source or "git" in source:
            return "git.status"
        if "process" in source:
            return "process.list"
        return "system.snapshot"

    def _reactive_fallback(self, observation: AgentObservation) -> AgentResponse:
        """Fallback when context is not used or not available."""
        env = observation.environment_state

        alerts = env.get("alerts", [])
        if alerts:
            return AgentResponse(
                action=f"investigate alert: {alerts[-1].get('type', 'unknown')}",
                tool_calls=[{"tool": "system.snapshot", "args": {}}],
                used_pushed_context=False,
            )

        conflicts = env.get("conflicts", [])
        if conflicts:
            return AgentResponse(
                action=f"resolve file conflict: {conflicts[0]}",
                tool_calls=[{"tool": "git.status", "args": {}}],
                used_pushed_context=False,
            )

        return AgentResponse(
            action="continue monitoring",
            tool_calls=[{"tool": "system.snapshot", "args": {}}],
            used_pushed_context=False,
        )
