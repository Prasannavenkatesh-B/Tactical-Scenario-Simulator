"""Tests for behavioral entropy module."""

import numpy as np
import pytest

from src.core.actions import AirAction, GroundAction
from src.evaluation.behavioral_entropy import (
    compute_action_category_entropy,
    compute_policy_entropy_over_episode,
)
from src.marl.policies.air_fight import AirFightPolicy


class TestBehavioralEntropy:
    """Test suite for behavioral entropy across action histories and episode rollouts."""

    def test_deterministic_action_history_entropy_zero(self) -> None:
        """Deterministic policy selecting identical actions yields entropy = 0."""
        actions = [0] * 50
        res = compute_action_category_entropy(actions, action_dims={"k": 5})
        assert res["mean_entropy"] == pytest.approx(0.0, abs=1e-9)
        assert res["normalized_entropy"] == pytest.approx(0.0, abs=1e-9)

    def test_random_action_history_entropy_log_k(self) -> None:
        """Random policy uniformly visiting k categories yields entropy = ln(k)."""
        k = 4
        # Perfectly uniform sequence
        actions = [0, 1, 2, 3] * 50
        res = compute_action_category_entropy(actions, action_dims={"k": k})
        expected_h = np.log(k)
        assert res["mean_entropy"] == pytest.approx(expected_h, rel=1e-4)
        assert res["normalized_entropy"] == pytest.approx(1.0, rel=1e-4)

    def test_action_dataclass_categorization(self) -> None:
        """Works seamlessly with AirAction dataclasses."""
        actions = [
            AirAction.from_discrete(0, 4, 0, 0),
            AirAction.from_discrete(1, 4, 0, 0),
            AirAction.from_discrete(-1, 4, 0, 0),
            AirAction.from_discrete(0, 4, 1, 0),
        ]
        res = compute_action_category_entropy(actions)
        assert res["mean_entropy"] > 0.0
        assert len(res["entropy_per_category"]) == 4

    def test_ground_action_dataclass_categorization(self) -> None:
        """Works seamlessly with GroundAction dataclasses."""
        actions = [
            GroundAction(heading_delta=0.0, velocity_cmd=1, weapon_select=0, fire=0),
            GroundAction(heading_delta=0.1, velocity_cmd=1, weapon_select=1, fire=1),
        ]
        res = compute_action_category_entropy(actions)
        assert res["mean_entropy"] > 0.0
        assert len(res["entropy_per_category"]) == 2

    def test_empty_action_history_returns_zeros(self) -> None:
        """Empty action history returns zeros without error."""
        res = compute_action_category_entropy([])
        assert res["mean_entropy"] == 0.0
        assert res["normalized_entropy"] == 0.0
        assert res["entropy_per_category"] == {}

    def test_over_episode_entropy_aggregation(self) -> None:
        """Aggregates policy entropy over a sequence of observations."""
        policy = AirFightPolicy()
        obs_seq = [np.full((policy.obs_dim,), 0.1 * i, dtype=np.float32) for i in range(10)]

        res = compute_policy_entropy_over_episode(policy, obs_seq, num_samples=5)
        assert "mean_entropy" in res
        assert "std_entropy" in res
        assert "per_timestep" in res
        assert len(res["per_timestep"]) == 10
        assert res["mean_entropy"] >= 0.0
        assert res["std_entropy"] >= 0.0

    def test_over_episode_empty_observations(self) -> None:
        """Empty observation list returns empty record without error."""
        policy = AirFightPolicy()
        res = compute_policy_entropy_over_episode(policy, [])
        assert res["mean_entropy"] == 0.0
        assert res["per_timestep"] == []
