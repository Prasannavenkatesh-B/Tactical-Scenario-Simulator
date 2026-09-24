"""Shared fixtures for non-determinism evaluation tests."""

from typing import Any
import pytest
import numpy as np
import torch

from src.marl.policies.air_fight import AirFightPolicy
from src.marl.policies.ground_engage import GroundEngagePolicy
from src.marl.policies.sea_engage import SeaEngagePolicy


class MockStochasticPolicy:
    """Mock policy producing uniform random discrete actions."""

    def __init__(self, action_dim: int = 13) -> None:
        self.action_dim = action_dim

    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        if deterministic:
            return np.array([6, 4, 0, 0]), 0.0, 0.0
        val = np.random.randint(0, self.action_dim)
        return np.array([val, 4, 0, 0]), 0.0, 0.0


class MockDeterministicPolicy:
    """Mock policy always producing identical actions."""

    def act(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        return np.array([6, 4, 0, 0]), 0.0, 0.0


@pytest.fixture
def mock_stochastic_policy() -> MockStochasticPolicy:
    return MockStochasticPolicy()


@pytest.fixture
def mock_deterministic_policy() -> MockDeterministicPolicy:
    return MockDeterministicPolicy()


@pytest.fixture
def real_policies() -> dict[str, Any]:
    return {
        "air_fight": AirFightPolicy(),
        "ground_engage": GroundEngagePolicy(),
        "sea_engage": SeaEngagePolicy(),
    }


@pytest.fixture
def multi_cluster_trajectories() -> list[list[np.ndarray]]:
    """Generate 50 trajectories grouped around 5 distinct 3D endpoints."""
    trajectories: list[list[np.ndarray]] = []
    centers = [
        [100.0, 200.0, 50.0],
        [-150.0, 300.0, 80.0],
        [400.0, -100.0, 20.0],
        [-200.0, -250.0, 60.0],
        [0.0, 0.0, 100.0],
    ]
    rng = np.random.default_rng(42)

    for i in range(50):
        c = centers[i % len(centers)]
        pts = []
        for step in range(5):
            pts.append(np.array(c) * (step + 1) / 5.0 + rng.normal(0, 1.0, 3))
        trajectories.append(pts)

    return trajectories
