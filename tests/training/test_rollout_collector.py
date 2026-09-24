"""Unit tests for RolloutCollector.

Tests multi-environment collection, buffer filling, Commander periodic
and event-based invocation schedules, and episode outcome statistics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pytest

from src.marl.commander import CommanderPolicy
from src.marl.policies.air_escape import AirEscapePolicy
from src.marl.policies.air_fight import AirFightPolicy
from src.marl.policies.ground_defend import GroundDefendPolicy
from src.marl.policies.ground_engage import GroundEngagePolicy
from src.marl.policies.sea_defend import SeaDefendPolicy
from src.marl.policies.sea_engage import SeaEngagePolicy
from src.simulator.env import TacticalEnv
from src.simulator.scenarios import generate_scenario
from src.training.rollout_collector import RolloutCollector


class TestRolloutCollector:
    """Tests for RolloutCollector."""

    @pytest.fixture
    def setup_collector(self) -> RolloutCollector:
        """Fixture creating small 2-env RolloutCollector."""
        policies = {
            "air_fight": AirFightPolicy(variant="AC1"),
            "air_escape": AirEscapePolicy(variant="AC1"),
            "ground_engage": GroundEngagePolicy(),
            "ground_defend": GroundDefendPolicy(),
            "sea_engage": SeaEngagePolicy(),
            "sea_defend": SeaDefendPolicy(),
            "commander": CommanderPolicy(),
        }
        scenario = generate_scenario(level=1, rng=np.random.default_rng(42))
        envs = [
            TacticalEnv(scenario_config=scenario, seed=101),
            TacticalEnv(scenario_config=scenario, seed=102),
        ]
        return RolloutCollector(
            policies=policies,
            envs=envs,
            config={"buffer_size": 200, "high_level_horizon": 10, "low_level_horizon": 5},
            rng=np.random.default_rng(42),
        )

    def test_collect_fills_buffers(self, setup_collector: RolloutCollector) -> None:
        """collect() steps parallel envs and populates policy buffers."""
        collector = setup_collector
        buffers = collector.collect(steps=10)

        assert "air_fight" in buffers
        assert "commander" in buffers
        # At least some air fight transitions should be recorded
        assert len(buffers["air_fight"]) > 0

    def test_commander_invoked_periodically(self, setup_collector: RolloutCollector) -> None:
        """Commander policy is invoked when high-level horizon is exceeded."""
        collector = setup_collector
        buffers = collector.collect(steps=25)
        # With high_level_horizon=10, commander buffer should have entries
        assert len(buffers["commander"]) > 0

    def test_episode_stats_tracking(self, setup_collector: RolloutCollector) -> None:
        """Collector tracks win/loss/draw counts and episode metrics."""
        collector = setup_collector
        # Run enough steps to ensure completion or verify structure
        collector.collect(steps=15)
        stats = collector.get_episode_stats()

        assert "blue_wins" in stats
        assert "red_wins" in stats
        assert "draws" in stats
        assert "total_episodes" in stats
        assert "win_rate" in stats
        assert stats["total_episodes"] == stats["blue_wins"] + stats["red_wins"] + stats["draws"]

    def test_reset_stats_zeroes_counters(self, setup_collector: RolloutCollector) -> None:
        """reset_stats() clears all accumulated outcome counters."""
        collector = setup_collector
        collector.blue_wins = 5
        collector.red_wins = 3
        collector.draws = 1
        collector.episode_rewards = [10.0, 15.0]

        collector.reset_stats()
        stats = collector.get_episode_stats()

        assert stats["blue_wins"] == 0
        assert stats["red_wins"] == 0
        assert stats["draws"] == 0
        assert stats["total_episodes"] == 0
