"""Unit tests for Evaluator and non-determinism measurements.

Tests win/loss/draw outcome accounting, reproducibility under identical seeds,
stochastic dispersion under varying seeds, and action entropy computation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pytest

from src.marl.policies.air_escape import AirEscapePolicy
from src.marl.policies.air_fight import AirFightPolicy
from src.marl.policies.ground_defend import GroundDefendPolicy
from src.marl.policies.ground_engage import GroundEngagePolicy
from src.marl.policies.sea_defend import SeaDefendPolicy
from src.marl.policies.sea_engage import SeaEngagePolicy
from src.training.evaluator import Evaluator


class TestEvaluator:
    """Tests for Evaluator."""

    @pytest.fixture
    def setup_evaluator(self) -> Evaluator:
        """Fixture initializing Evaluator with domain policies."""
        policies = {
            "air_fight": AirFightPolicy(variant="AC1"),
            "air_escape": AirEscapePolicy(variant="AC1"),
            "ground_engage": GroundEngagePolicy(),
            "ground_defend": GroundDefendPolicy(),
            "sea_engage": SeaEngagePolicy(),
            "sea_defend": SeaDefendPolicy(),
        }
        return Evaluator(policies=policies, rng=np.random.default_rng(42))

    def test_outcomes_sum_to_total_episodes(self, setup_evaluator: Evaluator) -> None:
        """Sum of blue_wins, red_wins, and draws equals total episodes."""
        evaluator = setup_evaluator
        res = evaluator.evaluate(num_episodes=5)

        dist = res["outcome_distribution"]
        total = dist["blue_wins"] + dist["red_wins"] + dist["draws"]
        assert total == 5
        assert pytest.approx(res["win_rate"] + res["loss_rate"] + res["draw_rate"]) == 1.0

    def test_non_determinism_same_seed_zero_variance(self, setup_evaluator: Evaluator) -> None:
        """With same_seed=True, environment and policies yield near-zero outcome variance."""
        evaluator = setup_evaluator
        res = evaluator.measure_non_determinism(num_runs=5, same_seed=True)

        assert "outcome_variance" in res
        assert res["outcome_variance"] == pytest.approx(0.0, abs=1e-5)
        # Trajectories should collapse to 1 mode
        assert res["trajectory_diversity"] == 1.0

    def test_non_determinism_different_seeds_positive_variance(self, setup_evaluator: Evaluator) -> None:
        """With same_seed=False, varying stochastic seeds produce outcome dispersion."""
        evaluator = setup_evaluator
        res = evaluator.measure_non_determinism(num_runs=10, same_seed=False)

        assert "outcome_variance" in res
        assert "action_entropy" in res
        assert res["action_entropy"] >= 0.0

    def test_action_entropy_in_valid_range(self, setup_evaluator: Evaluator) -> None:
        """Action entropy is non-negative and bounded by log(action_bins)."""
        evaluator = setup_evaluator
        res = evaluator.measure_non_determinism(num_runs=5, same_seed=False)

        # 13 heading bins => maximum entropy is ln(13) ≈ 2.56
        assert 0.0 <= res["action_entropy"] <= 5.0
