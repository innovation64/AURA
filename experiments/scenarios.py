"""Experiment scenarios — controlled environments for paradigm comparison.

Each scenario defines:
- initial_state: starting environment
- injections: {step_num: [changes]} — what changes at what step
- expected_actions: what a good agent should do
- inject_step: when the critical change happens (for time-to-awareness)
"""

from typing import Any, Dict, List, Tuple

from aura.paradigm.base import EnvironmentSimulator


def scenario_service_failure() -> Tuple[EnvironmentSimulator, Dict[str, Any]]:
    """Scenario 1: Database service goes down mid-task."""
    initial_state = {
        "cpu": 35, "memory": 55, "disk": 42,
        "services": [
            {"name": "database", "status": "running", "latency_ms": 12},
            {"name": "cache", "status": "running", "latency_ms": 2},
            {"name": "web", "status": "running", "latency_ms": 45},
        ],
        "errors": [],
        "task": "deploy new feature",
    }
    injections = {
        3: [
            {
                "type": "add_service_failure",
                "service_name": "database",
                "error": "Connection refused on port 5432",
            }
        ],
    }
    meta = {
        "name": "service_failure",
        "description": "Database goes down at step 3",
        "inject_step": 3,
        "expected_actions": ["detect database down", "restart database", "verify connectivity"],
    }
    return EnvironmentSimulator(initial_state, injections), meta


def scenario_file_conflict() -> Tuple[EnvironmentSimulator, Dict[str, Any]]:
    """Scenario 2: External process modifies a file the agent is editing."""
    initial_state = {
        "cpu": 25, "memory": 40, "disk": 55,
        "branch": "feature/auth",
        "uncommitted": 3,
        "files_modified": ["src/main.py", "src/auth.py"],
        "services": [],
        "errors": [],
        "task": "implement authentication",
    }
    injections = {
        2: [
            {
                "type": "add_file_conflict",
                "file": "src/main.py",
            }
        ],
    }
    meta = {
        "name": "file_conflict",
        "description": "External modification of src/main.py at step 2",
        "inject_step": 2,
        "expected_actions": ["detect file conflict", "compare changes", "resolve conflict"],
    }
    return EnvironmentSimulator(initial_state, injections), meta


def scenario_resource_exhaustion() -> Tuple[EnvironmentSimulator, Dict[str, Any]]:
    """Scenario 3: GPU memory fills up during training."""
    initial_state = {
        "cpu": 60, "memory": 50, "disk": 30, "gpu_memory": 40,
        "services": [{"name": "training_job", "status": "running"}],
        "errors": [],
        "task": "train model",
    }
    injections = {
        4: [
            {
                "type": "resource_spike",
                "metrics": {"gpu_memory": 98, "memory": 92},
            }
        ],
    }
    meta = {
        "name": "resource_exhaustion",
        "description": "GPU memory spike to 98% at step 4",
        "inject_step": 4,
        "expected_actions": ["detect resource spike", "reduce batch size", "resume training"],
    }
    return EnvironmentSimulator(initial_state, injections), meta


def scenario_security_alert() -> Tuple[EnvironmentSimulator, Dict[str, Any]]:
    """Scenario 4: Suspicious process detected."""
    initial_state = {
        "cpu": 25, "memory": 40, "disk": 55,
        "services": [{"name": "web", "status": "running"}],
        "processes": [{"name": "python", "pid": 1234, "cpu": 5}],
        "errors": [],
        "task": "routine maintenance",
    }
    injections = {
        2: [
            {
                "type": "security_alert",
                "process": {"name": "unknown_miner", "pid": 66612, "cpu": 95},
            }
        ],
    }
    meta = {
        "name": "security_alert",
        "description": "Suspicious crypto miner detected at step 2",
        "inject_step": 2,
        "expected_actions": ["investigate suspicious process", "terminate process", "scan for compromise"],
    }
    return EnvironmentSimulator(initial_state, injections), meta


def scenario_cascading_failure() -> Tuple[EnvironmentSimulator, Dict[str, Any]]:
    """Scenario 5: Multiple services fail in sequence."""
    initial_state = {
        "cpu": 30, "memory": 45, "disk": 40,
        "services": [
            {"name": "database", "status": "running", "latency_ms": 10},
            {"name": "cache", "status": "running", "latency_ms": 1},
            {"name": "api", "status": "running", "latency_ms": 50},
            {"name": "worker", "status": "running"},
        ],
        "errors": [],
        "task": "monitor production",
    }
    injections = {
        2: [
            {
                "type": "add_service_failure",
                "service_name": "database",
                "error": "Disk full",
            }
        ],
        4: [
            {
                "type": "add_service_failure",
                "service_name": "api",
                "error": "Cannot connect to database",
            }
        ],
        5: [
            {
                "type": "resource_spike",
                "metrics": {"cpu": 95, "memory": 88},
            }
        ],
    }
    meta = {
        "name": "cascading_failure",
        "description": "DB down at step 2, API down at step 4, CPU spike at step 5",
        "inject_step": 2,
        "expected_actions": ["detect database failure", "restart database",
                             "detect API cascade", "investigate root cause"],
    }
    return EnvironmentSimulator(initial_state, injections), meta


def scenario_noisy_service_failure() -> Tuple[EnvironmentSimulator, Dict[str, Any]]:
    """Scenario 6: Service failure with noisy distractors from multiple sources.

    Designed specifically for source weight differentiation:
    - probe.network → real signal (database down)
    - probe.system → noisy metrics (minor CPU bumps)
    - probe.filesystem → irrelevant file changes
    - probe.process → irrelevant process churn

    The agent should learn to trust probe.network signals and ignore the
    other sources' noise. This makes attention tracker convergence visible.
    """
    initial_state = {
        "cpu": 35, "memory": 55, "disk": 42,
        "services": [
            {"name": "database", "status": "running", "latency_ms": 12},
            {"name": "cache", "status": "running", "latency_ms": 2},
            {"name": "web", "status": "running", "latency_ms": 45},
            {"name": "logger", "status": "running", "latency_ms": 5},
        ],
        "processes": [{"name": "cron", "pid": 100, "cpu": 2}],
        "errors": [],
        "task": "deploy new feature",
    }
    injections = {
        1: [
            # Noise from probe.system: minor CPU bump
            {"type": "update", "updates": {"cpu": 52}},
            # Noise from probe.process: harmless process spawn
            {"type": "security_alert", "process": {"name": "apt-update", "pid": 200, "cpu": 15}},
        ],
        2: [
            # Noise from probe.filesystem: irrelevant file change
            {"type": "add_file_conflict", "file": "README.md"},
            # Noise from probe.system: slight memory bump
            {"type": "update", "updates": {"memory": 60}},
        ],
        3: [
            # REAL event from probe.network: database down
            {
                "type": "add_service_failure",
                "service_name": "database",
                "error": "Connection refused on port 5432",
            },
        ],
        5: [
            # More noise from probe.system
            {"type": "update", "updates": {"cpu": 48, "memory": 57}},
        ],
        7: [
            # More noise from probe.process
            {"type": "security_alert", "process": {"name": "logrotate", "pid": 300, "cpu": 8}},
        ],
    }
    meta = {
        "name": "noisy_service_failure",
        "description": "Database down at step 3 with noise from system/filesystem/process sources",
        "inject_step": 3,
        "expected_actions": ["ignore noise", "detect database down", "restart database"],
    }
    return EnvironmentSimulator(initial_state, injections), meta


def scenario_gradual_degradation() -> Tuple[EnvironmentSimulator, Dict[str, Any]]:
    """Scenario 7: Slow resource degradation that becomes critical.

    Tests whether the attention tracker can learn to pay attention to
    trend-type signals (not just spike events).
    """
    initial_state = {
        "cpu": 30, "memory": 40, "disk": 50,
        "services": [{"name": "api", "status": "running"}],
        "errors": [],
        "task": "monitor production",
    }
    injections = {
        1: [{"type": "update", "updates": {"memory": 48}}],
        2: [{"type": "update", "updates": {"memory": 55}}],
        3: [{"type": "update", "updates": {"memory": 63}}],
        4: [{"type": "update", "updates": {"memory": 72}}],
        5: [{"type": "resource_spike", "metrics": {"memory": 88, "cpu": 75}}],
        6: [{"type": "update", "updates": {"memory": 94, "cpu": 85}},
            {"type": "add_service_failure", "service_name": "api", "error": "OOM killed"}],
    }
    meta = {
        "name": "gradual_degradation",
        "description": "Memory slowly rises to critical, OOM at step 6",
        "inject_step": 5,
        "expected_actions": ["notice memory trend", "investigate before OOM", "restart after OOM"],
    }
    return EnvironmentSimulator(initial_state, injections), meta


def scenario_multi_source_noise() -> Tuple[EnvironmentSimulator, Dict[str, Any]]:
    """Scenario 8: Pure source differentiation test.

    Exactly 4 sources push signals every step. Only 1 source (probe.network)
    carries actionable information; the other 3 are pure noise. Over episodes,
    the attention tracker should learn to weight probe.network high and
    the others low.

    This is the "smoking gun" scenario for demonstrating convergence.
    """
    initial_state = {
        "cpu": 40, "memory": 50, "disk": 60,
        "services": [
            {"name": "api", "status": "running", "latency_ms": 30},
            {"name": "worker", "status": "running", "latency_ms": 5},
        ],
        "processes": [
            {"name": "node", "pid": 1000, "cpu": 10},
            {"name": "postgres", "pid": 1001, "cpu": 5},
        ],
        "errors": [],
        "task": "monitor microservices",
    }
    injections = {
        # Steps 1-3: noise from multiple sources
        1: [
            {"type": "update", "updates": {"cpu": 45}},  # probe.system noise
            {"type": "security_alert", "process": {"name": "tmpwatch", "pid": 2000, "cpu": 12}},
        ],
        2: [
            {"type": "update", "updates": {"memory": 53}},
            {"type": "add_file_conflict", "file": "package-lock.json"},
        ],
        3: [
            {"type": "update", "updates": {"cpu": 42, "memory": 51}},
        ],
        # Step 4: REAL event — api service down
        4: [
            {"type": "add_service_failure", "service_name": "api",
             "error": "Segfault in handler"},
        ],
        # Steps 5-7: more noise + cascading real event
        5: [
            {"type": "update", "updates": {"cpu": 55}},
            {"type": "security_alert", "process": {"name": "npm-audit", "pid": 2001, "cpu": 18}},
        ],
        6: [
            {"type": "add_service_failure", "service_name": "worker",
             "error": "Cannot reach api"},
            {"type": "add_file_conflict", "file": "yarn.lock"},
        ],
        7: [
            {"type": "resource_spike", "metrics": {"cpu": 92, "memory": 85}},
        ],
    }
    meta = {
        "name": "multi_source_noise",
        "description": "4 sources, only probe.network is actionable; noise from system/process/filesystem",
        "inject_step": 4,
        "expected_actions": ["learn to trust network source", "ignore system/process noise",
                             "detect api down", "detect worker cascade"],
    }
    return EnvironmentSimulator(initial_state, injections), meta


def all_scenarios() -> List[Tuple[EnvironmentSimulator, Dict[str, Any]]]:
    """Return all experiment scenarios."""
    return [
        scenario_service_failure(),
        scenario_file_conflict(),
        scenario_resource_exhaustion(),
        scenario_security_alert(),
        scenario_cascading_failure(),
        scenario_noisy_service_failure(),
        scenario_gradual_degradation(),
        scenario_multi_source_noise(),
    ]
