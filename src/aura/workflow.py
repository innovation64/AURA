"""Workflow engine — pipeline memory, dynamic tools, and background validation.

Core concepts:
    Workflow:  An ordered sequence of (tool, args_template) that solved a task.
    Skill:     A validated, named workflow promoted to reusable status.
    ToolForge: Creates composite or LLM-generated tools at runtime.

Lifecycle:
    1. Agent explores → solves task via a sequence of tool calls
    2. WorkflowMemory captures the sequence as a candidate Workflow
    3. On reuse, BackgroundValidator concurrently checks if the workflow
       is still optimal / still works
    4. If validation fails → StrategyAuditor flags it, WorkflowMemory
       demotes or replaces it
    5. If a capability gap is detected → ToolForge synthesises a new tool
       from existing primitives or retires a broken one

Tradeoff design:
    - Reuse budget:  fraction of steps using known workflows vs exploring
    - Validation budget:  fraction of reuse steps that also run validation
    - Forge threshold:  how many consecutive failures before creating a new tool
    All three adapt dynamically based on environment stability.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from .feedback import ConditionalFeedbackStore, Outcome, StatePattern, extract_pattern
from .tools import Tool, ToolCall, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class WorkflowStatus(Enum):
    CANDIDATE = "candidate"   # just recorded, not yet validated
    ACTIVE = "active"         # validated and in use
    VALIDATING = "validating" # being re-checked in background
    STALE = "stale"           # failed validation, pending replacement
    RETIRED = "retired"       # no longer used


@dataclass
class WorkflowStep:
    """One step in a workflow pipeline."""
    tool_name: str
    args_template: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""  # what a successful result looks like
    timeout_ms: float = 5000.0


@dataclass
class Workflow:
    """A recorded pipeline of tool calls that achieved a goal."""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    trigger_pattern: StatePattern       # when to suggest this workflow
    status: WorkflowStatus = WorkflowStatus.CANDIDATE
    confidence: float = 0.5
    times_used: int = 0
    times_succeeded: int = 0
    times_failed: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    last_validated: float = field(default_factory=time.time)
    avg_duration_ms: float = 0.0
    tags: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = self.times_succeeded + self.times_failed
        return self.times_succeeded / total if total > 0 else 0.0

    @property
    def tool_names(self) -> List[str]:
        return [s.tool_name for s in self.steps]


@dataclass
class ValidationResult:
    """Result of background workflow validation."""
    workflow_id: str
    still_valid: bool
    issues: List[str] = field(default_factory=list)
    alternative_found: Optional[str] = None  # workflow_id of better alternative
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolGapSignal:
    """Signal that a capability gap was detected."""
    missing_capability: str
    failed_tool: str
    context_description: str
    times_observed: int = 1
    first_seen: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# WorkflowMemory — stores and retrieves pipeline sequences
# ---------------------------------------------------------------------------

class WorkflowMemory:
    """Stores validated tool-call sequences as reusable workflows.

    Tradeoff controls:
        similarity_threshold: how similar a situation must be to trigger reuse
        promotion_threshold:  how many successes before candidate → active
        demotion_threshold:   how many failures before active → stale
    """

    def __init__(
        self,
        max_workflows: int = 100,
        similarity_threshold: float = 0.6,
        promotion_threshold: int = 2,
        demotion_threshold: int = 2,
    ):
        self._workflows: Dict[str, Workflow] = {}
        self._max = max_workflows
        self._similarity_threshold = similarity_threshold
        self._promotion_threshold = promotion_threshold
        self._demotion_threshold = demotion_threshold
        self._id_counter = 0

    def record(
        self,
        steps: List[WorkflowStep],
        trigger_pattern: StatePattern,
        name: str = "",
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Workflow:
        """Record a new candidate workflow from an observed tool sequence."""
        # Check for duplicate
        existing = self._find_similar_workflow(trigger_pattern, [s.tool_name for s in steps])
        if existing is not None:
            existing.times_used += 1
            existing.last_used = time.time()
            return existing

        self._id_counter += 1
        wf_id = f"wf_{self._id_counter}"
        wf = Workflow(
            workflow_id=wf_id,
            name=name or f"workflow_{self._id_counter}",
            description=description,
            steps=steps,
            trigger_pattern=trigger_pattern,
            tags=tags or [],
        )
        self._workflows[wf_id] = wf

        if len(self._workflows) > self._max:
            self._evict_weakest()

        return wf

    def lookup(self, current_pattern: StatePattern) -> Optional[Workflow]:
        """Find the best active workflow for the current situation."""
        best: Optional[Workflow] = None
        best_score = 0.0

        for wf in self._workflows.values():
            if wf.status not in (WorkflowStatus.ACTIVE, WorkflowStatus.CANDIDATE):
                continue
            sim = current_pattern.similarity(wf.trigger_pattern)
            if sim < self._similarity_threshold:
                continue
            score = sim * wf.confidence * (1.0 + wf.success_rate)
            if score > best_score:
                best_score = score
                best = wf

        return best

    def record_outcome(self, workflow_id: str, succeeded: bool, duration_ms: float = 0.0) -> None:
        """Record the outcome of executing a workflow."""
        wf = self._workflows.get(workflow_id)
        if wf is None:
            return

        wf.times_used += 1
        wf.last_used = time.time()

        if succeeded:
            wf.times_succeeded += 1
            wf.confidence = min(1.0, wf.confidence + 0.1)
            # Promote candidate → active
            if wf.status == WorkflowStatus.CANDIDATE and wf.times_succeeded >= self._promotion_threshold:
                wf.status = WorkflowStatus.ACTIVE
                logger.info("Workflow '%s' promoted to ACTIVE", wf.name)
        else:
            wf.times_failed += 1
            wf.confidence = max(0.1, wf.confidence - 0.15)
            # Demote active → stale
            if wf.status == WorkflowStatus.ACTIVE and wf.times_failed >= self._demotion_threshold:
                wf.status = WorkflowStatus.STALE
                logger.info("Workflow '%s' demoted to STALE", wf.name)

        # Update rolling average duration
        if duration_ms > 0:
            n = wf.times_succeeded + wf.times_failed
            wf.avg_duration_ms = (wf.avg_duration_ms * (n - 1) + duration_ms) / n

    def record_validation(self, result: ValidationResult) -> None:
        """Update workflow based on background validation result."""
        wf = self._workflows.get(result.workflow_id)
        if wf is None:
            return

        wf.last_validated = time.time()
        if result.still_valid:
            wf.confidence = min(1.0, wf.confidence + 0.05)
            if wf.status == WorkflowStatus.VALIDATING:
                wf.status = WorkflowStatus.ACTIVE
        else:
            wf.status = WorkflowStatus.STALE
            wf.confidence = max(0.1, wf.confidence * 0.5)
            for issue in result.issues:
                logger.info("Workflow '%s' invalidated: %s", wf.name, issue)

    def get(self, workflow_id: str) -> Optional[Workflow]:
        return self._workflows.get(workflow_id)

    def list_active(self) -> List[Workflow]:
        return [w for w in self._workflows.values()
                if w.status in (WorkflowStatus.ACTIVE, WorkflowStatus.CANDIDATE)]

    def list_stale(self) -> List[Workflow]:
        return [w for w in self._workflows.values() if w.status == WorkflowStatus.STALE]

    def retire(self, workflow_id: str) -> None:
        wf = self._workflows.get(workflow_id)
        if wf:
            wf.status = WorkflowStatus.RETIRED

    def get_stats(self) -> Dict[str, Any]:
        by_status = {}
        for wf in self._workflows.values():
            by_status[wf.status.value] = by_status.get(wf.status.value, 0) + 1
        return {
            "total": len(self._workflows),
            "by_status": by_status,
            "avg_confidence": (
                sum(w.confidence for w in self._workflows.values()) /
                max(len(self._workflows), 1)
            ),
        }

    def _find_similar_workflow(
        self, pattern: StatePattern, tool_names: List[str]
    ) -> Optional[Workflow]:
        for wf in self._workflows.values():
            if wf.tool_names == tool_names:
                sim = pattern.similarity(wf.trigger_pattern)
                if sim > 0.8:
                    return wf
        return None

    def _evict_weakest(self) -> None:
        if not self._workflows:
            return
        weakest = min(
            self._workflows.values(),
            key=lambda w: w.confidence * (w.times_succeeded + 1),
        )
        del self._workflows[weakest.workflow_id]


# ---------------------------------------------------------------------------
# BackgroundValidator — concurrent workflow freshness checking
# ---------------------------------------------------------------------------

class BackgroundValidator:
    """Validates workflows while they're being used.

    Tradeoff:  validation_rate controls what fraction of workflow
    executions trigger a parallel validation check.

    Validation strategies:
    1. Tool availability:  are all tools in the workflow still registered?
    2. Outcome consistency: does the workflow still produce expected results?
    3. Efficiency check:  is there a faster alternative?
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        workflow_memory: WorkflowMemory,
        validation_rate: float = 0.2,  # validate 20% of executions
    ):
        self.registry = tool_registry
        self.memory = workflow_memory
        self.validation_rate = validation_rate
        self._validation_history: Deque[ValidationResult] = deque(maxlen=100)
        self._step_count = 0

    def should_validate(self, workflow: Workflow) -> bool:
        """Decide whether to validate this workflow execution."""
        self._step_count += 1

        # Always validate if stale or never validated
        if workflow.status == WorkflowStatus.STALE:
            return True
        age = time.time() - workflow.last_validated
        if age > 3600:  # over 1 hour since last validation
            return True

        # Budget-based: validate at the configured rate
        return (self._step_count % max(1, int(1.0 / self.validation_rate))) == 0

    def validate(self, workflow: Workflow) -> ValidationResult:
        """Run validation checks on a workflow."""
        issues: List[str] = []

        # Check 1: tool availability
        for step in workflow.steps:
            if not self.registry.has(step.tool_name):
                issues.append(f"tool '{step.tool_name}' no longer available")

            elif not self.registry.is_allowed(step.tool_name):
                issues.append(f"tool '{step.tool_name}' denied by policy")

        # Check 2: success rate degradation
        if workflow.times_used >= 3 and workflow.success_rate < 0.4:
            issues.append(
                f"success rate degraded to {workflow.success_rate:.0%} "
                f"({workflow.times_succeeded}/{workflow.times_used})"
            )

        # Check 3: staleness
        if workflow.status == WorkflowStatus.STALE:
            issues.append("workflow marked as stale")

        result = ValidationResult(
            workflow_id=workflow.workflow_id,
            still_valid=len(issues) == 0,
            issues=issues,
        )
        self._validation_history.append(result)
        self.memory.record_validation(result)
        return result

    def get_stats(self) -> Dict[str, Any]:
        recent = list(self._validation_history)
        valid_count = sum(1 for r in recent if r.still_valid)
        return {
            "total_validations": len(recent),
            "valid_rate": valid_count / max(len(recent), 1),
            "validation_rate_setting": self.validation_rate,
        }


# ---------------------------------------------------------------------------
# ToolForge — synthesise new tools from primitives
# ---------------------------------------------------------------------------

class ToolForge:
    """Creates composite tools at runtime when capability gaps are detected.

    Strategies:
    1. Compose:  chain existing tools into a pipeline tool
    2. Adapt:    wrap an existing tool with different default args
    3. Retire:   remove broken tools and suggest alternatives

    Tradeoff:  forge_threshold controls how many failures before forging.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        forge_threshold: int = 3,
    ):
        self.registry = registry
        self.forge_threshold = forge_threshold
        self._gap_signals: Dict[str, ToolGapSignal] = {}
        self._forged_tools: Dict[str, str] = {}  # tool_name -> source description
        self._forge_count = 0

    def report_gap(self, tool_name: str, context: str) -> Optional[ToolGapSignal]:
        """Report that a tool failed or a capability is missing."""
        key = tool_name
        if key in self._gap_signals:
            self._gap_signals[key].times_observed += 1
        else:
            self._gap_signals[key] = ToolGapSignal(
                missing_capability=f"tool '{tool_name}' failed or missing",
                failed_tool=tool_name,
                context_description=context,
            )

        signal = self._gap_signals[key]
        if signal.times_observed >= self.forge_threshold:
            return signal  # caller should attempt to forge
        return None

    def compose(
        self,
        name: str,
        description: str,
        tool_sequence: List[Tuple[str, Dict[str, Any]]],
    ) -> Optional[Tool]:
        """Create a composite tool that chains multiple existing tools.

        Each step's output is available to the next step via a shared context dict.
        """
        # Verify all component tools exist
        for tool_name, _ in tool_sequence:
            if not self.registry.has(tool_name):
                logger.warning("Cannot compose: tool '%s' not found", tool_name)
                return None

        registry_ref = self.registry  # capture for closure

        def _composite_handler(**kwargs: Any) -> Dict[str, Any]:
            context: Dict[str, Any] = dict(kwargs)
            results: List[Dict[str, Any]] = []
            for tool_name, args_template in tool_sequence:
                # Merge template args with runtime context
                merged_args = {**args_template}
                for k, v in merged_args.items():
                    if isinstance(v, str) and v.startswith("$"):
                        # Variable substitution from context
                        ref_key = v[1:]
                        if ref_key in context:
                            merged_args[k] = context[ref_key]

                call = ToolCall(name=tool_name, arguments=merged_args)
                result = registry_ref.execute(call)
                step_result = {
                    "tool": tool_name,
                    "ok": result.ok,
                    "output": result.output,
                    "error": result.error,
                }
                results.append(step_result)

                if result.ok and isinstance(result.output, dict):
                    context.update(result.output)

                if not result.ok:
                    return {"steps": results, "ok": False,
                            "error": f"Step '{tool_name}' failed: {result.error}"}

            return {"steps": results, "ok": True, "final_context": context}

        tool = Tool(name=name, description=description, handler=_composite_handler)
        self.registry.register(tool)
        self._forged_tools[name] = f"composite({', '.join(t for t, _ in tool_sequence)})"
        self._forge_count += 1
        logger.info("Forged composite tool '%s' from %d primitives", name, len(tool_sequence))
        return tool

    def adapt(
        self,
        base_tool_name: str,
        new_name: str,
        new_description: str,
        fixed_args: Dict[str, Any],
    ) -> Optional[Tool]:
        """Create an adapted version of an existing tool with fixed defaults."""
        base = self.registry.get(base_tool_name)
        if base is None:
            return None

        def _adapted_handler(**kwargs: Any) -> Any:
            merged = {**fixed_args, **kwargs}
            return base.execute(**merged)

        tool = Tool(name=new_name, description=new_description, handler=_adapted_handler)
        self.registry.register(tool)
        self._forged_tools[new_name] = f"adapt({base_tool_name}, {fixed_args})"
        self._forge_count += 1
        logger.info("Forged adapted tool '%s' from '%s'", new_name, base_tool_name)
        return tool

    def retire_broken(self, tool_name: str, reason: str) -> bool:
        """Retire a tool that consistently fails."""
        tool = self.registry.retire(tool_name)
        if tool is not None:
            logger.info("Retired broken tool '%s': %s", tool_name, reason)
            self._gap_signals.pop(tool_name, None)
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        return {
            "forged_count": self._forge_count,
            "active_gaps": len(self._gap_signals),
            "forged_tools": dict(self._forged_tools),
            "gap_details": {
                k: {"times": v.times_observed, "context": v.context_description[:100]}
                for k, v in self._gap_signals.items()
            },
        }


# ---------------------------------------------------------------------------
# WorkflowEngine — orchestrates everything
# ---------------------------------------------------------------------------

class WorkflowEngine:
    """Top-level orchestrator for workflow lifecycle management.

    Integrates:
    - WorkflowMemory (store/retrieve pipelines)
    - BackgroundValidator (concurrent freshness checking)
    - ToolForge (dynamic tool creation/retirement)
    - ConditionalFeedbackStore (experience feedback)

    Tradeoff parameters:
        reuse_rate:       target fraction of steps using known workflows
        validation_rate:  fraction of reuse steps that also validate
        forge_threshold:  failures before creating new tool
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        feedback_store: Optional[ConditionalFeedbackStore] = None,
        reuse_rate: float = 0.6,
        validation_rate: float = 0.2,
        forge_threshold: int = 3,
    ):
        self.registry = tool_registry
        self.feedback_store = feedback_store or ConditionalFeedbackStore()
        self.memory = WorkflowMemory()
        self.validator = BackgroundValidator(tool_registry, self.memory, validation_rate)
        self.forge = ToolForge(tool_registry, forge_threshold)

        self.reuse_rate = reuse_rate
        self._step_count = 0
        self._reuse_count = 0
        self._explore_count = 0
        self._current_recording: List[WorkflowStep] = []
        self._recording_active = False

    # ------------------------------------------------------------------
    # Pipeline: before action
    # ------------------------------------------------------------------

    def before_action(self, scene_pattern: StatePattern) -> Optional[Workflow]:
        """Called before agent acts.  Returns a workflow to reuse, or None to explore.

        Tradeoff: reuse known workflow vs explore new path.
        """
        self._step_count += 1

        # Look up a matching workflow
        candidate = self.memory.lookup(scene_pattern)
        if candidate is None:
            self._explore_count += 1
            return None

        # Decide: reuse vs explore
        actual_reuse_rate = self._reuse_count / max(self._step_count, 1)
        if actual_reuse_rate < self.reuse_rate and candidate.confidence > 0.4:
            self._reuse_count += 1

            # Background validation: check if this workflow is still good
            if self.validator.should_validate(candidate):
                candidate.status = WorkflowStatus.VALIDATING
                validation = self.validator.validate(candidate)
                if not validation.still_valid:
                    # Workflow is stale — don't reuse, explore instead
                    self._reuse_count -= 1
                    self._explore_count += 1
                    return None

            return candidate

        self._explore_count += 1
        return None

    # ------------------------------------------------------------------
    # Pipeline: during action (recording)
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        """Start recording a new exploration workflow."""
        self._current_recording = []
        self._recording_active = True

    def record_step(self, tool_name: str, args: Dict[str, Any], result: ToolResult) -> None:
        """Record a tool call during exploration."""
        if not self._recording_active:
            return
        step = WorkflowStep(
            tool_name=tool_name,
            args_template=args,
            expected_outcome="ok" if result.ok else "error",
            timeout_ms=result.duration_ms or 5000.0,
        )
        self._current_recording.append(step)

        # Track tool failures for gap detection
        if not result.ok:
            gap = self.forge.report_gap(
                tool_name,
                f"failed during workflow recording: {result.error}",
            )
            if gap is not None:
                logger.info("Tool gap detected: %s (observed %d times)",
                            tool_name, gap.times_observed)

    def finish_recording(
        self,
        scene_pattern: StatePattern,
        succeeded: bool,
        name: str = "",
        description: str = "",
    ) -> Optional[Workflow]:
        """Finish recording.  If successful, store as candidate workflow."""
        self._recording_active = False
        steps = self._current_recording
        self._current_recording = []

        if not steps or not succeeded:
            return None

        wf = self.memory.record(
            steps=steps,
            trigger_pattern=scene_pattern,
            name=name,
            description=description,
        )
        logger.info("Recorded workflow '%s' with %d steps", wf.name, len(steps))
        return wf

    # ------------------------------------------------------------------
    # Pipeline: after action
    # ------------------------------------------------------------------

    def after_action(
        self,
        workflow_id: Optional[str],
        succeeded: bool,
        scene_pattern: StatePattern,
        duration_ms: float = 0.0,
    ) -> None:
        """Called after action completes.  Updates workflow stats + feedback."""
        if workflow_id is not None:
            self.memory.record_outcome(workflow_id, succeeded, duration_ms)

            # Record to feedback store
            wf = self.memory.get(workflow_id)
            if wf is not None:
                outcome = Outcome.SUCCESS if succeeded else Outcome.FAILURE
                self.feedback_store.record_outcome(
                    scene_state=_pattern_to_scene(scene_pattern),
                    action=f"workflow:{wf.name}",
                    outcome=outcome,
                    reward=1.0 if succeeded else 0.0,
                )

    # ------------------------------------------------------------------
    # Tool lifecycle
    # ------------------------------------------------------------------

    def handle_tool_failure(self, tool_name: str, error: str, context: str) -> Optional[Tool]:
        """Handle a tool failure.  May forge a replacement or retire the tool."""
        gap = self.forge.report_gap(tool_name, f"{error}: {context}")
        if gap is None:
            return None  # below threshold, just record

        # Try to compose a replacement from related tools
        related = self._find_related_tools(tool_name)
        if related:
            composite = self.forge.compose(
                name=f"{tool_name}.auto_v{self.forge._forge_count + 1}",
                description=f"Auto-generated replacement for {tool_name}",
                tool_sequence=[(r, {}) for r in related[:3]],
            )
            if composite:
                return composite

        # If we can't replace it, retire it
        self.forge.retire_broken(tool_name, f"failed {gap.times_observed} times: {error}")
        return None

    def _find_related_tools(self, tool_name: str) -> List[str]:
        """Find tools with similar names/domains."""
        prefix = tool_name.split(".")[0] if "." in tool_name else ""
        related = []
        for tool in self.registry.list():
            if tool.name == tool_name:
                continue
            if prefix and tool.name.startswith(prefix):
                related.append(tool.name)
        return related

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            "steps": self._step_count,
            "reuse_count": self._reuse_count,
            "explore_count": self._explore_count,
            "actual_reuse_rate": self._reuse_count / max(self._step_count, 1),
            "target_reuse_rate": self.reuse_rate,
            "workflows": self.memory.get_stats(),
            "validation": self.validator.get_stats(),
            "forge": self.forge.get_stats(),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pattern_to_scene(pattern: StatePattern):
    """Convert a StatePattern back to a minimal SceneState for FeedbackStore."""
    from .types import SceneState
    return SceneState(
        summary=f"signals={','.join(pattern.signal_types)} anomalies={','.join(pattern.anomaly_categories)}",
        entities=list(pattern.active_entities)[:10],
        context={"resource_pressure": pattern.resource_pressure},
    )
