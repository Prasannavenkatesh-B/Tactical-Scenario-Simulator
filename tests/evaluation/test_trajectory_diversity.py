"""Tests for trajectory diversity and spatial clustering."""

import numpy as np
import pytest

from src.evaluation.trajectory_diversity import (
    cluster_trajectories,
    compute_diversity_score,
    extract_final_positions,
)


class TestTrajectoryDiversity:
    """Test suite for trajectory endpoint extraction and clustering."""

    def test_extract_final_positions_returns_correct_shape(self) -> None:
        """Extract final positions returns array of shape (num_episodes, 3)."""
        episodes = [
            [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
            [np.array([10.0, 20.0, 30.0])],
            [np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])],
        ]
        finals = extract_final_positions(episodes)
        assert finals.shape == (3, 3)
        assert np.allclose(finals[0], [4.0, 5.0, 6.0])
        assert np.allclose(finals[1], [10.0, 20.0, 30.0])
        assert np.allclose(finals[2], [2.0, 2.0, 2.0])

    def test_extract_final_positions_handles_2d_and_empty(self) -> None:
        """Pads 2D positions with z=0.0 and handles empty episode lists."""
        episodes = [
            [np.array([5.0, 10.0])],
            [],
        ]
        finals = extract_final_positions(episodes)
        assert finals.shape == (2, 3)
        assert np.allclose(finals[0], [5.0, 10.0, 0.0])
        assert np.allclose(finals[1], [0.0, 0.0, 0.0])

    def test_cluster_trajectories_returns_k_clusters(self) -> None:
        """K-means clustering returns specified k clusters."""
        rng = np.random.default_rng(42)
        pos = rng.uniform(-100, 100, size=(50, 3))

        res = cluster_trajectories(pos, k=5)
        assert res["n_clusters"] == 5
        assert len(res["labels"]) == 50
        assert "inertia" in res
        assert "distinct_clusters" in res
        assert res["distinct_clusters"] >= 1

    def test_diversity_score_in_zero_to_one(self, multi_cluster_trajectories) -> None:
        """Diversity score is bounded within [0, 1]."""
        res = compute_diversity_score(multi_cluster_trajectories, k=5)
        assert 0.0 <= res["diversity_score"] <= 1.0
        assert res["n_clusters"] == 5
        assert res["distinct_clusters"] >= 3
        assert "silhouette_score" in res
        assert len(res["labels"]) == len(multi_cluster_trajectories)

    def test_single_trajectory_returns_score_zero(self) -> None:
        """A single trajectory has diversity score 0.0."""
        single_ep = [[np.array([10.0, 20.0, 30.0])]]
        res = compute_diversity_score(single_ep, k=5)
        assert res["diversity_score"] == 0.0
        assert res["distinct_clusters"] == 0
        assert res["silhouette_score"] == 0.0

    def test_all_identical_trajectories_low_diversity(self) -> None:
        """All identical trajectories produce 0 or 1 cluster with zero silhouette."""
        identical_eps = [[np.array([10.0, 10.0, 10.0])] for _ in range(20)]
        res = compute_diversity_score(identical_eps, k=5)
        assert res["silhouette_score"] == 0.0
