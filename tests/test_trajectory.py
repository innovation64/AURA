"""Tests for trajectory collection and reward signals."""
import tempfile
from pathlib import Path

from aura.trajectory.collector import TrajectoryCollector, TrajectoryStep
from aura.trajectory.reward import RewardSignal
from aura.trajectory.replay import ExperienceBuffer


class TestTrajectoryCollector:
    def test_start_episode(self):
        collector = TrajectoryCollector()
        eid = collector.start_episode("test task")
        assert isinstance(eid, str)
        assert len(eid) > 0

    def test_record_step(self):
        collector = TrajectoryCollector()
        collector.start_episode()
        step = collector.record_step(
            environment_state={"cpu": 45},
            agent_action="check status",
            result="OK",
        )
        assert isinstance(step, TrajectoryStep)
        assert step.agent_action == "check status"

    def test_end_episode(self):
        collector = TrajectoryCollector()
        collector.start_episode()
        collector.record_step({"a": 1}, "act1")
        collector.record_step({"a": 2}, "act2")
        steps = collector.end_episode()
        assert len(steps) == 2

    def test_save_and_load(self):
        collector = TrajectoryCollector()
        collector.start_episode()
        collector.record_step({"x": 1}, "test_action", result="done", reward=0.8)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)
        collector.save(path)
        loaded = TrajectoryCollector.load(path)
        assert len(loaded) == 1
        assert loaded[0].agent_action == "test_action"
        path.unlink()

    def test_statistics(self):
        collector = TrajectoryCollector()
        collector.start_episode()
        collector.record_step({}, "a", reward=0.8)
        collector.record_step({}, "b", reward=0.6)
        collector.end_episode()
        stats = collector.get_statistics()
        assert stats["total_episodes"] >= 1
        assert stats["total_steps"] >= 2


class TestRewardSignal:
    def test_compute_returns_float(self):
        reward = RewardSignal()
        score = reward.compute(
            pre_state={"error_count": 3},
            post_state={"error_count": 1},
            action="fix errors",
            task_goal="reduce errors",
        )
        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_improvement_increases_reward(self):
        reward = RewardSignal()
        improved = reward.compute(
            pre_state={"error_count": 5, "services_healthy": 2},
            post_state={"error_count": 0, "services_healthy": 5},
            action="fix",
        )
        degraded = reward.compute(
            pre_state={"error_count": 0, "services_healthy": 5},
            post_state={"error_count": 5, "services_healthy": 0},
            action="break",
        )
        assert improved > degraded


class TestExperienceBuffer:
    def test_add_and_sample(self):
        buf = ExperienceBuffer(capacity=100)
        for i in range(10):
            step = TrajectoryStep(
                environment_state={"i": i},
                agent_action=f"act_{i}",
                reward=i / 10,
            )
            buf.add(step)
        assert len(buf) == 10
        sample = buf.sample(3)
        assert len(sample) >= 1  # priority sampling with dedup may return fewer
        assert len(sample) <= 3
        assert all(isinstance(s, TrajectoryStep) for s in sample)

    def test_capacity_limit(self):
        buf = ExperienceBuffer(capacity=5)
        for i in range(10):
            buf.add(TrajectoryStep(environment_state={}, agent_action=f"a{i}"))
        assert len(buf) == 5

    def test_empty_sample(self):
        buf = ExperienceBuffer()
        sample = buf.sample(5)
        assert sample == []
