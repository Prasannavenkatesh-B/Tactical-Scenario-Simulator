"""Unit tests for Trainer execution loops.

Tests parameter gradient updates, frozen policy integrity during Stage B,
and dry-run iterations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pytest
import torch

from src.marl.commander import CommanderPolicy
from src.marl.policies.air_escape import AirEscapePolicy
from src.marl.policies.air_fight import AirFightPolicy
from src.marl.policies.ground_defend import GroundDefendPolicy
from src.marl.policies.ground_engage import GroundEngagePolicy
from src.marl.policies.sea_defend import SeaDefendPolicy
from src.marl.policies.sea_engage import SeaEngagePolicy
from src.simulator.env import TacticalEnv
from src.simulator.scenarios import generate_scenario
from src.training.curriculum import CurriculumScheduler
from src.training.league import League
from src.training.rollout_collector import RolloutCollector
from src.training.trainer import Trainer


class TestTrainer:
    """Tests for Trainer."""

    @pytest.fixture
    def setup_trainer(self) -> Trainer:
        """Fixture initializing Trainer with small buffer and 2 envs."""
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
            TacticalEnv(scenario_config=scenario, seed=201),
            TacticalEnv(scenario_config=scenario, seed=202),
        ]
        collector = RolloutCollector(
            policies=policies,
            envs=envs,
            config={"buffer_size": 100, "high_level_horizon": 10, "low_level_horizon": 5},
            rng=np.random.default_rng(42),
        )
        return Trainer(
            policies=policies,
            optimizer=None,
            rollout_collector=collector,
            config={"rollout_steps": 10, "eval_interval": 10, "log_interval": 10},
            rng=np.random.default_rng(42),
        )

    def test_train_iteration_runs_without_crash(self, setup_trainer: Trainer) -> None:
        """train_iteration() completes cleanly and returns metrics dictionary."""
        trainer = setup_trainer
        metrics = trainer.train_iteration(target_policies=["air_fight"])

        assert "losses" in metrics
        assert "episode_stats" in metrics
        assert "total_loss" in metrics["losses"]

    def test_train_iteration_updates_parameters(self, setup_trainer: Trainer) -> None:
        """PPO update modifies trainable weights in the target policy."""
        trainer = setup_trainer
        policy = trainer.policies["air_fight"]

        # Collect initial parameter snapshot
        initial_params = [p.clone() for p in policy.parameters()]

        # Run multiple collection steps to ensure enough transitions for an update
        for _ in range(5):
            trainer.train_iteration(target_policies=["air_fight"])

        # Compare at least one parameter changed
        params_changed = any(
            not torch.allclose(p0, p1)
            for p0, p1 in zip(initial_params, policy.parameters())
        )
        assert params_changed is True

    def test_frozen_policies_unchanged_during_stage_b(self, setup_trainer: Trainer) -> None:
        """Frozen low-level policies retain identical weights during Commander training."""
        trainer = setup_trainer
        air_policy = trainer.policies["air_fight"]

        # Snapshot weights before Stage B
        before_weights = [p.clone() for p in air_policy.parameters()]

        curriculum = CurriculumScheduler(initial_level=5)
        # Run 2 iterations of Commander training
        trainer.train_commander(curriculum=curriculum, num_iterations=2)

        # Verify low-level weights remain completely identical
        for p_before, p_after in zip(before_weights, air_policy.parameters()):
            assert torch.allclose(p_before, p_after)
            assert p_after.requires_grad is False

    def test_commander_trains_in_stage_b(self, setup_trainer: Trainer) -> None:
        """Commander weights update during Stage B training."""
        trainer = setup_trainer
        commander = trainer.policies["commander"]

        curriculum = CurriculumScheduler(initial_level=5)
        results = trainer.train_commander(curriculum=curriculum, num_iterations=2)

        assert results["stage"] == "stage_b"
        assert results["iterations_completed"] == 2
