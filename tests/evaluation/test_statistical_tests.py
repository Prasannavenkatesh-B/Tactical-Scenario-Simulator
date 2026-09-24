"""Tests for statistical testing module."""

import numpy as np
import pytest

from src.evaluation.statistical_tests import (
    chi_square_goodness_of_fit,
    coefficient_of_variation,
    levene_variance_test,
    two_sample_ks_test,
)


class TestStatisticalTests:
    """Test suite for chi-square, Levene, CV, and KS tests."""

    def test_chi_square_uniform_counts_high_p(self) -> None:
        """Chi-square on uniform counts returns p > 0.05 (cannot reject uniform)."""
        counts = [33, 33, 34]
        res = chi_square_goodness_of_fit(counts)
        assert res["df"] == 2
        assert res["p_value"] > 0.05
        assert res["reject_null"] is False

    def test_chi_square_skewed_counts_low_p(self) -> None:
        """Chi-square on heavily skewed counts returns p < 0.05 (rejects uniform)."""
        counts = [90, 5, 5]
        res = chi_square_goodness_of_fit(counts)
        assert res["df"] == 2
        assert res["p_value"] < 0.05
        assert res["reject_null"] is True
        assert res["chi2"] > 0.0

    def test_chi_square_returns_correct_df(self) -> None:
        """Degrees of freedom equals k - 1."""
        for k in (2, 3, 5, 10):
            counts = [10] * k
            res = chi_square_goodness_of_fit(counts)
            assert res["df"] == k - 1
            assert res["p_value"] == 1.0

    def test_chi_square_edge_cases(self) -> None:
        """Empty or single count returns df=0 and p=1.0."""
        assert chi_square_goodness_of_fit([])["p_value"] == 1.0
        assert chi_square_goodness_of_fit([50])["p_value"] == 1.0
        assert chi_square_goodness_of_fit([0, 0, 0])["p_value"] == 1.0

    def test_levene_identical_variances_high_p(self) -> None:
        """Levene test on groups with identical variance returns p > 0.05."""
        rng = np.random.default_rng(42)
        group1 = rng.normal(10.0, 2.0, 100)
        group2 = rng.normal(10.0, 2.0, 100)

        res = levene_variance_test([group1, group2])
        assert res["p_value"] > 0.05
        assert res["reject_null"] is False

    def test_levene_different_variances_low_p(self) -> None:
        """Levene test on groups with different variance returns p < 0.05."""
        rng = np.random.default_rng(42)
        group1 = rng.normal(10.0, 0.5, 100)
        group2 = rng.normal(10.0, 10.0, 100)

        res = levene_variance_test([group1, group2])
        assert res["p_value"] < 0.05
        assert res["reject_null"] is True
        assert res["W"] > 0.0

    def test_levene_zero_variance_control(self) -> None:
        """Levene test comparing constant control vs variable sample returns p < 0.05."""
        control = np.array([42.0] * 50)
        stochastic = np.random.default_rng(42).normal(42.0, 5.0, 50)

        res = levene_variance_test([control, stochastic])
        assert res["p_value"] < 0.05
        assert res["reject_null"] is True

    def test_levene_edge_cases(self) -> None:
        """Fewer than 2 groups or zero variance in all groups returns p=1.0."""
        assert levene_variance_test([np.array([1, 2, 3])])["p_value"] == 1.0
        # All zero variance
        assert levene_variance_test([np.array([5, 5]), np.array([5, 5])])["p_value"] == 1.0

    def test_coefficient_of_variation_constant_array_zero(self) -> None:
        """CV of a constant array is 0.0."""
        arr = np.array([42.0, 42.0, 42.0, 42.0])
        assert coefficient_of_variation(arr) == pytest.approx(0.0, abs=1e-9)

    def test_coefficient_of_variation_varying_array(self) -> None:
        """CV of an array with std=1 and mean=10 is 0.1."""
        arr = np.array([9.0, 11.0, 9.0, 11.0])  # mean=10, std=1.0
        assert coefficient_of_variation(arr) == pytest.approx(0.1, abs=1e-5)

    def test_coefficient_of_variation_zero_mean_or_empty(self) -> None:
        """CV is 0.0 if mean is 0 or array is empty."""
        assert coefficient_of_variation(np.array([])) == 0.0
        assert coefficient_of_variation(np.array([-5.0, 5.0])) == 0.0

    def test_two_sample_ks_test_identical_distributions(self) -> None:
        """KS test on samples from the same distribution returns high p-value."""
        rng = np.random.default_rng(42)
        s1 = rng.normal(0, 1, 200)
        s2 = rng.normal(0, 1, 200)

        res = two_sample_ks_test(s1, s2)
        assert res["p_value"] > 0.05
        assert res["reject_null"] is False

    def test_two_sample_ks_test_different_distributions(self) -> None:
        """KS test on samples from different distributions returns low p-value."""
        rng = np.random.default_rng(42)
        s1 = rng.normal(0, 1, 200)
        s2 = rng.normal(5, 1, 200)

        res = two_sample_ks_test(s1, s2)
        assert res["p_value"] < 0.05
        assert res["reject_null"] is True
        assert res["statistic"] > 0.5
