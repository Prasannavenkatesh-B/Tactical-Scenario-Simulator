"""Unit tests for Gymnasium single-agent adapter TSSSingleAgentEnv."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest

from src.api.model_registry import ModelRegistry
from src.integration.single_agent_env import TSSSingleAgentEnv


def test_single_agent_env_spaces(registry: ModelRegistry, protocol_path: str) -> None:
    """TSSSingleAgentEnv defines valid observation and action spaces."""
    env = TSSSingleAgentEnv(
        agent_id="blue_air_1",
        domain="air",
        variant="AC1",
        mode="in_process",
        registry_or_url=registry,
        protocol_path=protocol_path,
    )
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (13,)
    assert isinstance(env.action_space, gym.Space)


def test_single_agent_env_reset(registry: ModelRegistry, protocol_path: str) -> None:
    """env.reset() returns a 2-tuple: (observation, info)."""
    env = TSSSingleAgentEnv(
        agent_id="blue_air_1",
        domain="air",
        variant="AC1",
        mode="in_process",
        registry_or_url=registry,
        protocol_path=protocol_path,
    )
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (13,)
    assert isinstance(info, dict)


def test_single_agent_env_step_5_tuple(registry: ModelRegistry, protocol_path: str) -> None:
    """env.step() returns the standard Gymnasium 5-tuple: (obs, rew, terminated, truncated, info)."""
    env = TSSSingleAgentEnv(
        agent_id="blue_air_1",
        domain="air",
        variant="AC1",
        mode="in_process",
        registry_or_url=registry,
        protocol_path=protocol_path,
    )
    env.reset()

    action = [0, 4, 0, 0]
    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (13,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_single_agent_env_close(registry: ModelRegistry, protocol_path: str) -> None:
    """env.close() releases underlying resources cleanly."""
    env = TSSSingleAgentEnv(
        agent_id="blue_ground_1",
        domain="ground",
        mode="in_process",
        registry_or_url=registry,
        protocol_path=protocol_path,
    )
    env.reset()
    env.close()
