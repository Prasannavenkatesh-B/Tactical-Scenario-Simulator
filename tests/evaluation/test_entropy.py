"""Tests for entropy calculation functions."""

import numpy as np
import pytest

from src.evaluation.entropy import (
    differential_entropy_gaussian,
    normalized_shannon_entropy,
    policy_entropy,
    shannon_entropy,
)
from src.marl.policies.air_fight import AirFightPolicy


class TestEntropy:
    """Test suite for Shannon, Gaussian, and policy entropy."""

    def test_shannon_entropy_uniform_equals_log_n(self) -> None:
        """Shannon entropy of uniform distribution of size n equals ln(n)."""
        n = 10
        uniform = np.full(n, 1.0 / n)
        h = shannon_entropy(uniform)
        expected = np.log(n)
        assert h == pytest.approx(expected, rel=1e-5)

    def test_shannon_entropy_deterministic_equals_zero(self) -> None:
        """Shannon entropy of deterministic (one-hot) distribution equals 0.0."""
        deterministic = np.array([0.0, 1.0, 0.0, 0.0])
        h = shannon_entropy(deterministic)
        assert h == pytest.approx(0.0, abs=1e-9)

    def test_shannon_entropy_handles_zeros(self) -> None:
        """Zeros in distribution are handled without NaN (0 * log(0) = 0)."""
        probs = np.array([0.5, 0.0, 0.5, 0.0])
        h = shannon_entropy(probs)
        assert not np.isnan(h)
        assert h == pytest.approx(np.log(2.0), rel=1e-5)

    def test_shannon_entropy_empty_or_all_zeros(self) -> None:
        """Empty or all-zero arrays return 0.0."""
        assert shannon_entropy(np.array([])) == 0.0
        assert shannon_entropy(np.array([0.0, 0.0])) == 0.0

    def test_normalized_shannon_entropy_in_zero_to_one(self) -> None:
        """Normalized Shannon entropy returns values strictly in [0, 1]."""
        # Intermediate distribution
        probs = np.array([0.7, 0.2, 0.1])
        norm_h = normalized_shannon_entropy(probs)
        assert 0.0 <= norm_h <= 1.0

        # Max entropy is 1.0
        assert normalized_shannon_entropy(np.full(5, 0.2)) == pytest.approx(1.0, rel=1e-5)
        # Min entropy is 0.0
        assert normalized_shannon_entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-9)

    def test_normalized_shannon_entropy_single_element(self) -> None:
        """Single element returns 0.0 to avoid division by log(1)=0."""
        assert normalized_shannon_entropy(np.array([1.0])) == 0.0

    def test_gaussian_differential_entropy_scales_with_std(self) -> None:
        """Gaussian differential entropy strictly increases as std increases."""
        mean = np.zeros(3)
        std_low = np.array([0.5, 0.5, 0.5])
        std_high = np.array([2.0, 2.0, 2.0])

        h_low = differential_entropy_gaussian(mean, std_low)
        h_high = differential_entropy_gaussian(mean, std_high)

        assert h_high > h_low
        # Exact theoretical difference: d * ln(std_high / std_low)
        expected_diff = 3.0 * np.log(2.0 / 0.5)
        assert (h_high - h_low) == pytest.approx(expected_diff, rel=1e-4)

    def test_gaussian_differential_entropy_handles_near_zero_std(self) -> None:
        """Guards against zero or negative std without raising exceptions."""
        mean = np.zeros(2)
        std_zero = np.zeros(2)
        h = differential_entropy_gaussian(mean, std_zero)
        assert isinstance(h, float)
        assert not np.isnan(h)

    def test_policy_entropy_returns_required_keys(self) -> None:
        """policy_entropy returns dict with discrete_entropy, continuous_entropy, normalized."""
        policy = AirFightPolicy()
        obs = np.zeros(policy.obs_dim, dtype=np.float32)

        res = policy_entropy(policy, obs, num_samples=5)
        assert isinstance(res, dict)
        assert "discrete_entropy" in res
        assert "continuous_entropy" in res
        assert "normalized" in res

        assert isinstance(res["discrete_entropy"], float)
        assert isinstance(res["continuous_entropy"], float)
        assert isinstance(res["normalized"], float)
        assert 0.0 <= res["normalized"] <= 1.0

    def test_policy_entropy_with_mock_stochastic(self, mock_stochastic_policy) -> None:
        """policy_entropy works with custom mock policy."""
        obs = np.zeros(10, dtype=np.float32)
        res = policy_entropy(mock_stochastic_policy, obs, num_samples=50)
        assert "discrete_entropy" in res
        assert res["discrete_entropy"] > 0.0
        assert res["normalized"] > 0.0
