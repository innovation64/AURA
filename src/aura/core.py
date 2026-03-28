from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .act import Actor, StubActor
from .interact import BasicInteractor, Interactor
from .memory import EphemeralMemory, MemoryStore
from .reason import Reasoner, SimpleReasoner
from .scene import BasicScene, SceneModel
from .sense import BasicSense, SenseAdapter
from .types import EnvironmentSignal, Interaction, ReasoningResult
from .builtin_tools import default_tools
from .explore import Explorer, HeuristicPlanner, ExplorationOutcome
from .tools import ToolPolicy, ToolRegistry
from .guard import ExecutionGuard, InterventionLevel, GuardVerdict
from .feedback import ConditionalFeedbackStore, Outcome as FeedbackOutcome
from .auditor import StrategyAuditor
from .workflow import WorkflowEngine, WorkflowStep, Workflow
from .feedback import extract_pattern

logger = logging.getLogger(__name__)


@dataclass
class AURAConfig:
    memory_limit: int = 100
    explore_enabled: bool = True
    explore_max_steps: int = 5
    tool_policy: Optional[ToolPolicy] = None
    # Backend selection
    backend: str = "default"  # "default" | "llm" | "bmam" | "model"
    # LLM config (used by llm/bmam/model backends)
    llm_api_key: str = ""
    llm_base_url: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    # BMAM config (REST API — reads from BMAM_API_URL env var)
    bmam_base_url: Optional[str] = None
    # Persistence
    db_path: str = ":memory:"
    # Evolve config (world-level evolution)
    evolve_enabled: bool = False
    evolve_interval: int = 5
    evolve_max_mutations: int = 3
    # Proactive engine config
    proactive_enabled: bool = True
    proactive_poll_interval: float = 10.0
    proactive_relevance_threshold: float = 0.4
    probe_paths: List[str] = field(default_factory=list)
    probe_services: List[str] = field(default_factory=list)
    # Smart planner (replaces HeuristicPlanner)
    smart_planner: bool = True
    # ExecutionGuard config
    guard_enabled: bool = True
    guard_window: int = 8
    guard_threshold: float = 0.7
    # StrategyAuditor config
    auditor_enabled: bool = True
    auditor_staleness_halflife: float = 86400.0
    auditor_revalidation_budget: float = 0.1
    # WorkflowEngine config
    workflow_enabled: bool = True
    workflow_reuse_rate: float = 0.6
    workflow_validation_rate: float = 0.2
    workflow_forge_threshold: int = 3
    # Extra kwargs passed to backend builders
    backend_kwargs: Dict[str, Any] = field(default_factory=dict)


class AURAAgent:
    """
    AURA Agent — environment-aware middleware with pluggable backends.

    Supports three modes:
    - default/llm: standalone operation (no external dependencies)
    - bmam: bridged to BMAM five-brain-region system
    - model: with neural plasticity memory model

    Components can also be injected directly for full control.
    """

    def __init__(
        self,
        sense: Optional[SenseAdapter] = None,
        scene: Optional[SceneModel] = None,
        memory: Optional[MemoryStore] = None,
        reason: Optional[Reasoner] = None,
        actor: Optional[Actor] = None,
        interactor: Optional[Interactor] = None,
        explorer: Optional[Explorer] = None,
        tool_registry: Optional[ToolRegistry] = None,
        evolver: Optional[Any] = None,
        config: Optional[AURAConfig] = None,
    ) -> None:
        resolved_config = config or AURAConfig()

        # If explicit components provided, use them directly
        if any([sense, scene, memory, reason, actor, interactor]):
            self.sense = sense or BasicSense()
            self.scene = scene or BasicScene()
            self.memory = memory or EphemeralMemory(max_items=resolved_config.memory_limit)
            self.reason = reason or SimpleReasoner()
            self.actor = actor or StubActor()
            self.interactor = interactor or BasicInteractor()
        else:
            # Use backend registry
            components = self._build_from_backend(resolved_config)
            self.sense = components["sense"]
            self.scene = components["scene"]
            self.memory = components["memory"]
            self.reason = components["reason"]
            self.actor = components["actor"]
            self.interactor = components["interactor"]

        self.tool_registry = tool_registry or ToolRegistry(
            default_tools(),
            policy=resolved_config.tool_policy,
        )

        # Explorer with smart or heuristic planner
        self.explorer = explorer
        if self.explorer is None and resolved_config.explore_enabled:
            if resolved_config.smart_planner:
                try:
                    from .smart_planner import SmartPlanner
                    planner = SmartPlanner()
                except ImportError:
                    planner = HeuristicPlanner()
            else:
                planner = HeuristicPlanner()
            self.explorer = Explorer(
                planner=planner,
                registry=self.tool_registry,
                max_steps=resolved_config.explore_max_steps,
            )

        # Store plasticity engine — prefer the one embedded in memory (shared state)
        self.plasticity = None
        if not any([sense, scene, memory, reason, actor, interactor]):
            # Check if memory has an embedded plasticity engine (ModelMemory)
            if hasattr(self.memory, "plasticity") and self.memory.plasticity is not None:
                self.plasticity = self.memory.plasticity
            elif components.get("plasticity"):
                self.plasticity = components["plasticity"]

        # Store evolver — world-level cross-cutting concern
        self.evolver = evolver
        if self.evolver is None and not any([sense, scene, memory, reason, actor, interactor]):
            if components.get("evolver"):
                self.evolver = components["evolver"]

        # Proactive engine (environment monitoring)
        self._proactive_engine = None
        if resolved_config.proactive_enabled:
            self._proactive_engine = self._build_proactive_engine(resolved_config)

        # ExecutionGuard + FeedbackStore + StrategyAuditor
        self.guard: Optional[ExecutionGuard] = None
        self.feedback_store: Optional[ConditionalFeedbackStore] = None
        self.auditor: Optional[StrategyAuditor] = None
        if resolved_config.guard_enabled:
            self.guard = ExecutionGuard(
                window_size=resolved_config.guard_window,
                base_threshold=resolved_config.guard_threshold,
            )
            self.feedback_store = ConditionalFeedbackStore()
            if resolved_config.auditor_enabled:
                self.auditor = StrategyAuditor(
                    feedback_store=self.feedback_store,
                    staleness_halflife=resolved_config.auditor_staleness_halflife,
                    revalidation_budget=resolved_config.auditor_revalidation_budget,
                )

        # WorkflowEngine
        self.workflow_engine: Optional[WorkflowEngine] = None
        if resolved_config.workflow_enabled:
            self.workflow_engine = WorkflowEngine(
                tool_registry=self.tool_registry,
                feedback_store=self.feedback_store,
                reuse_rate=resolved_config.workflow_reuse_rate,
                validation_rate=resolved_config.workflow_validation_rate,
                forge_threshold=resolved_config.workflow_forge_threshold,
            )

        # Trajectory collector (training data)
        self._trajectory_collector = None
        self._config = resolved_config
        self._prev_scene: Optional[Any] = None  # for Guard state comparison

    def _build_from_backend(self, config: AURAConfig) -> Dict[str, Any]:
        """Build components from the registered backend."""
        from .backend import ensure_backends_registered, get_backend

        ensure_backends_registered()

        backend_name = config.backend

        # Prepare kwargs for backend builders
        kwargs: Dict[str, Any] = {
            "memory_limit": config.memory_limit,
            "db_path": config.db_path,
            **config.backend_kwargs,
        }

        # Create LLM engine if needed
        if backend_name in ("llm", "bmam", "model"):
            try:
                from .llm import LLMConfig, LLMEngine
                llm_config = LLMConfig(
                    api_key=config.llm_api_key,
                    base_url=config.llm_base_url,
                    model=config.llm_model,
                )
                kwargs["llm"] = LLMEngine(llm_config)
            except Exception as e:
                logger.warning("LLM engine creation failed: %s", e)

        # Pass BMAM base URL if specified
        if config.bmam_base_url:
            kwargs["bmam_base_url"] = config.bmam_base_url

        try:
            factory = get_backend(backend_name)
        except KeyError:
            logger.warning("Backend '%s' not found, falling back to 'default'", backend_name)
            factory = get_backend("default")

        components: Dict[str, Any] = {}
        for key, builder_name in [
            ("sense", "build_sense"),
            ("scene", "build_scene"),
            ("memory", "build_memory"),
            ("reason", "build_reasoner"),
            ("actor", "build_actor"),
            ("interactor", "build_interactor"),
        ]:
            builder = getattr(factory, builder_name, None)
            if builder:
                try:
                    components[key] = builder(**kwargs)
                except Exception as e:
                    logger.warning("Backend builder %s failed: %s", builder_name, e)
                    components[key] = None
            else:
                components[key] = None

        # Fill in defaults for any missing components
        if not components.get("sense"):
            components["sense"] = BasicSense()
        if not components.get("scene"):
            components["scene"] = BasicScene()
        if not components.get("memory"):
            components["memory"] = EphemeralMemory(max_items=config.memory_limit)
        if not components.get("reason"):
            components["reason"] = SimpleReasoner()
        if not components.get("actor"):
            components["actor"] = StubActor()
        if not components.get("interactor"):
            components["interactor"] = BasicInteractor()

        # Optional: plasticity engine
        if factory.build_plasticity:
            try:
                components["plasticity"] = factory.build_plasticity(**kwargs)
            except Exception as e:
                logger.warning("Plasticity engine creation failed: %s", e)

        # Optional: evolver
        if factory.build_evolver:
            try:
                components["evolver"] = factory.build_evolver(**kwargs)
            except Exception as e:
                logger.warning("Evolver creation failed: %s", e)

        return components

    def run(self, raw_input: Any, user_query: Optional[str] = None) -> Interaction:
        """Execute the full AURA pipeline: Sense → Probe → Explore → Scene → Memory → Reason → [Guard] → Act → Interact."""
        signals = self.sense.ingest(raw_input)

        # Proactive: inject probe signals if engine is available
        probe_signals: List[EnvironmentSignal] = []
        if self._proactive_engine is not None:
            probe_signals = self._collect_probe_signals()
            if probe_signals:
                signals = list(signals) + probe_signals

        # Explore: tool-based environment probing
        exploration: Optional[ExplorationOutcome] = None
        if self.explorer is not None:
            exploration = self.explorer.explore(signals, user_query=user_query, raw_input=raw_input)
            signals = list(signals) + exploration.extra_signals

        # Scene → Memory → Reason
        scene_state = self.scene.build(signals)
        self.memory.update(scene_state)
        memories = self.memory.recall(user_query)

        # WorkflowEngine: check for reusable workflow before reasoning
        reused_workflow: Optional[Workflow] = None
        scene_pattern = extract_pattern(scene_state) if scene_state else None
        if self.workflow_engine is not None and scene_pattern is not None:
            reused_workflow = self.workflow_engine.before_action(scene_pattern)
            if reused_workflow is not None:
                # Inject workflow hint into memory context for the reasoner
                memories = list(memories) + [
                    f"[workflow:{reused_workflow.name}] "
                    f"Known pipeline: {' → '.join(reused_workflow.tool_names)} "
                    f"(confidence={reused_workflow.confidence:.2f}, "
                    f"success_rate={reused_workflow.success_rate:.0%})"
                ]
            else:
                # Start recording a new exploration
                self.workflow_engine.start_recording()

        reasoning = self.reason.plan(scene_state, memories, user_query)

        # ── ExecutionGuard check ─────────────────────────────────
        guard_verdict: Optional[GuardVerdict] = None
        if self.guard is not None:
            tool_results = []
            if exploration:
                tool_results = [
                    {"name": r.name, "ok": r.ok, "output": r.output, "error": r.error}
                    for r in exploration.tool_results
                ]
            guard_verdict = self.guard.check(
                reasoning, self._prev_scene, scene_state, tool_results,
            )

            # StrategyAuditor: check environment drift
            if self.auditor is not None and self._prev_scene is not None:
                self.auditor.check_environment_drift(self._prev_scene, scene_state)

            # React to guard verdict
            if guard_verdict.level == InterventionLevel.HINT:
                # Inject informational signal into reasoning metadata
                reasoning.metadata["guard_hint"] = guard_verdict.detail

            elif guard_verdict.level == InterventionLevel.SUGGEST:
                # Query feedback store for advice
                if self.feedback_store is not None:
                    advice = self.feedback_store.query_advice(scene_state, reasoning.intent)
                    if advice and advice.suggested_alternative:
                        guard_verdict.suggested_alternative = advice.suggested_alternative
                        reasoning.metadata["guard_suggestion"] = advice.suggested_alternative
                        reasoning.metadata["guard_warning"] = advice.warning

            elif guard_verdict.level == InterventionLevel.CONSTRAIN:
                # Query feedback for alternatives and constrain
                if self.feedback_store is not None:
                    advice = self.feedback_store.query_advice(scene_state, reasoning.intent)
                    alternatives = self.feedback_store.query_alternatives(
                        scene_state, exclude_action=reasoning.intent,
                    )
                    if alternatives:
                        reasoning = ReasoningResult(
                            intent=alternatives[0],
                            rationale=f"Guard constrained: {guard_verdict.detail}. "
                                      f"Switched to alternative: {alternatives[0]}",
                            actions=reasoning.actions,
                            metadata={
                                **reasoning.metadata,
                                "guard_constrained": True,
                                "original_intent": reasoning.intent,
                            },
                        )

            elif guard_verdict.level == InterventionLevel.REDIRECT:
                reasoning = ReasoningResult(
                    intent="abort_and_report",
                    rationale=f"Guard redirect: {guard_verdict.detail}",
                    actions=["report_stuck", "request_human_input"],
                    metadata={
                        **reasoning.metadata,
                        "guard_redirected": True,
                        "original_intent": reasoning.intent,
                    },
                )

        # ── WorkflowEngine: record exploration tool calls ─────────
        if (self.workflow_engine is not None and reused_workflow is None
                and exploration is not None):
            for tr in exploration.tool_results:
                self.workflow_engine.record_step(
                    tool_name=tr.name,
                    args={},  # args not captured in ExplorationOutcome
                    result=tr,
                )

        # ── Act → Interact ───────────────────────────────────────
        action = self.actor.act(reasoning, scene_state)
        interaction = self.interactor.respond(action, scene_state)

        if exploration is not None:
            interaction.payload.setdefault("exploration", exploration.summary())

        if guard_verdict is not None:
            interaction.payload["guard"] = {
                "level": guard_verdict.level.name,
                "urgency": guard_verdict.urgency,
                "detail": guard_verdict.detail,
            }

        # ── Record outcome to FeedbackStore ──────────────────────
        ig = 0.5  # default information gain
        if self.guard is not None and self.guard.gain_estimator._gain_history:
            ig = self.guard.gain_estimator._gain_history[-1]

        if self.guard is not None and self.feedback_store is not None and self._prev_scene is not None:
            if ig > 0.3:
                fb_outcome = FeedbackOutcome.SUCCESS
            elif ig > 0.1:
                fb_outcome = FeedbackOutcome.PARTIAL
            elif guard_verdict and guard_verdict.level.value >= InterventionLevel.CONSTRAIN.value:
                fb_outcome = FeedbackOutcome.LOOP
            else:
                fb_outcome = FeedbackOutcome.STALL

            self.feedback_store.record_outcome(
                scene_state=scene_state,
                action=reasoning.intent,
                outcome=fb_outcome,
                reward=ig,
            )

            # ── Forward feedback to BMAM if available ─────────────
            # Closes the bidirectional learning loop: AURA's learned
            # outcomes feed back into BMAM's preference model.
            if hasattr(self.memory, 'feedback') and callable(getattr(self.memory, 'feedback', None)):
                try:
                    self.memory.feedback(
                        query=user_query or reasoning.intent,
                        response=interaction.message[:500] if interaction.message else reasoning.intent,
                        reward=ig,
                    )
                except Exception as e:
                    logger.debug("BMAM feedback forwarding failed: %s", e)

        # ── WorkflowEngine: finish recording / record outcome ─────
        if self.workflow_engine is not None and scene_pattern is not None:
            action_ok = ig > 0.3 if (self.guard and self.guard.gain_estimator._gain_history) else True
            if reused_workflow is not None:
                self.workflow_engine.after_action(
                    workflow_id=reused_workflow.workflow_id,
                    succeeded=action_ok,
                    scene_pattern=scene_pattern,
                )
            else:
                self.workflow_engine.finish_recording(
                    scene_pattern=scene_pattern,
                    succeeded=action_ok,
                    name=f"auto_{reasoning.intent[:30]}",
                    description=reasoning.rationale[:100] if reasoning.rationale else "",
                )

            # Handle tool failures: report to forge
            if exploration is not None:
                for tr in exploration.tool_results:
                    if not tr.ok:
                        replacement = self.workflow_engine.handle_tool_failure(
                            tr.name, tr.error or "unknown", reasoning.intent,
                        )
                        if replacement:
                            interaction.payload.setdefault("forged_tools", [])
                            interaction.payload["forged_tools"].append(replacement.name)

            # Add workflow stats to interaction payload
            interaction.payload["workflow"] = self.workflow_engine.get_stats()

        self._prev_scene = scene_state

        # Record trajectory if collector is available
        if self._trajectory_collector is not None:
            self._trajectory_collector.record_step(
                environment_state=scene_state.context if scene_state else {},
                agent_action=reasoning.intent if reasoning else "none",
                result=interaction.message[:200] if interaction else None,
                context_was_used=bool(probe_signals) if self._proactive_engine else False,
            )

        return interaction

    def _collect_probe_signals(self) -> List[EnvironmentSignal]:
        """Synchronously collect signals from the proactive engine."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return self._proactive_engine.get_cached_signals()
            coro = self._proactive_engine.poll_signals()
            return loop.run_until_complete(
                asyncio.wait_for(coro, timeout=5.0)
            )
        except RuntimeError:
            try:
                new_loop = asyncio.new_event_loop()
                coro = self._proactive_engine.poll_signals()
                return new_loop.run_until_complete(
                    asyncio.wait_for(coro, timeout=5.0)
                )
            except Exception:
                return self._proactive_engine.get_cached_signals()
        except (asyncio.TimeoutError, Exception) as e:
            logger.debug("Probe signal collection failed/timed out: %s", e)
            return self._proactive_engine.get_cached_signals()

    async def run_async(self, raw_input: Any, user_query: Optional[str] = None) -> Interaction:
        """Async version of run() — preferred when running in async context."""
        signals = self.sense.ingest(raw_input)

        # Proactive probe signals
        if self._proactive_engine is not None:
            try:
                probe_signals = await self._proactive_engine.poll_signals()
                if probe_signals:
                    signals = list(signals) + probe_signals
            except Exception as e:
                logger.debug("Async probe collection failed: %s", e)

        # Explore
        exploration: Optional[ExplorationOutcome] = None
        if self.explorer is not None:
            exploration = self.explorer.explore(signals, user_query=user_query, raw_input=raw_input)
            signals = list(signals) + exploration.extra_signals

        scene_state = self.scene.build(signals)
        self.memory.update(scene_state)
        memories = self.memory.recall(user_query)
        reasoning = self.reason.plan(scene_state, memories, user_query)
        action = self.actor.act(reasoning, scene_state)
        interaction = self.interactor.respond(action, scene_state)

        if exploration is not None:
            interaction.payload.setdefault("exploration", exploration.summary())

        return interaction

    # ── Convenience factory methods ──────────────────────────────

    @classmethod
    def from_backend(cls, backend: str, **kwargs: Any) -> "AURAAgent":
        """Create an agent with a specific backend.

        Examples:
            agent = AURAAgent.from_backend("default")
            agent = AURAAgent.from_backend("llm", llm_api_key="sk-...")
            agent = AURAAgent.from_backend("bmam")  # reads BMAM_API_URL from env
            agent = AURAAgent.from_backend("model")
        """
        config = AURAConfig(backend=backend, **kwargs)
        return cls(config=config)

    def enable_trajectory(self, storage_path: Optional[str] = None) -> None:
        """Enable trajectory collection for training data."""
        try:
            from .trajectory.collector import TrajectoryCollector
            from pathlib import Path
            path = Path(storage_path) if storage_path else None
            self._trajectory_collector = TrajectoryCollector(storage_path=path)
            self._trajectory_collector.start_episode("aura_session")
            logger.info("Trajectory collection enabled")
        except ImportError:
            logger.warning("Trajectory module not available")

    def get_environment_context(self) -> Optional[Dict[str, Any]]:
        """Get the current proactive environment context (if available)."""
        if self._proactive_engine is None:
            return None
        try:
            ctx = self._proactive_engine.get_current_context()
            return ctx if ctx else None
        except Exception:
            return None

    def _build_proactive_engine(self, config: AURAConfig) -> Optional[Any]:
        """Build the proactive engine with configured probes."""
        try:
            from .proactive.engine import ProactiveEngine, SimpleProactiveEngine
            from .probes.base import ProbeRegistry
            from .probes.system import SystemProbe
            from .probes.git import GitProbe

            registry = ProbeRegistry()
            registry.register(SystemProbe())
            registry.register(GitProbe())

            # Optional probes — only add if dependencies available
            try:
                from .probes.docker import DockerProbe
                registry.register(DockerProbe())
            except Exception:
                pass
            try:
                from .probes.filesystem import FileSystemProbe
                paths = config.probe_paths or ["."]
                registry.register(FileSystemProbe(watch_paths=paths))
            except Exception:
                pass
            try:
                from .probes.network import NetworkProbe
                if config.probe_services:
                    registry.register(NetworkProbe(urls=config.probe_services))
            except Exception:
                pass
            try:
                from .probes.process import ProcessProbe
                registry.register(ProcessProbe())
            except Exception:
                pass

            return SimpleProactiveEngine(registry)
        except Exception as e:
            logger.info("Proactive engine not available: %s", e)
            return None

    def info(self) -> Dict[str, Any]:
        """Return information about this agent's configuration."""
        probes = []
        if self._proactive_engine is not None:
            try:
                probes = self._proactive_engine.list_probes()
            except Exception:
                pass
        return {
            "backend": self._config.backend,
            "sense": type(self.sense).__name__,
            "scene": type(self.scene).__name__,
            "memory": type(self.memory).__name__,
            "reason": type(self.reason).__name__,
            "actor": type(self.actor).__name__,
            "interactor": type(self.interactor).__name__,
            "plasticity": type(self.plasticity).__name__ if self.plasticity else None,
            "explorer": type(self.explorer).__name__ if self.explorer else None,
            "evolver": type(self.evolver).__name__ if self.evolver else None,
            "proactive": self._proactive_engine is not None,
            "probes": probes,
            "trajectory": self._trajectory_collector is not None,
        }
