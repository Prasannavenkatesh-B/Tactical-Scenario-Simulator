"""Unit tests for PettingZoo TSSParallelEnv adapter."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest

from src.api.model_registry import ModelRegistry
from src.integration.parallel_env import TSSParallelEnv


def test_parallel_env_init_spaces(registry: ModelRegistry, protocol_path: str) -> None:
    """TSSParallelEnv initializes valid gym.Space for observations and actions."""
    env = TSSParallelEnv(
        mode="in_process",
        registry_or_url=registry,
        protocol_path=protocol_path,
        agent_ids=["blue_1", "red_1"],
        agent_domains={"blue_1": "air", "red_1": "ground"},
    )
    assert env.possible_agents == ["blue_1", "red_1"]
    assert env.agents == ["blue_1", "red_1"]

    obs_sp_blue = env.observation_space("blue_1")
    act_sp_blue = env.action_space("blue_1")
    assert isinstance(obs_sp_blue, spaces.Box)
    assert obs_sp_blue.shape == (13,)
    assert isinstance(act_sp_blue, gym.Space)

    obs_sp_red = env.observation_space("red_1")
    assert isinstance(obs_sp_red, spaces.Box)
    assert obs_sp_red.shape == (9,)


def test_parallel_env_reset(registry: ModelRegistry, protocol_path: str) -> None:
    """env.reset() returns matching observation and info dictionaries."""
    env = TSSParallelEnv(
        mode="in_process",
        registry_or_url=registry,
        protocol_path=protocol_path,
    )
    obs, infos = env.reset()

    assert isinstance(obs, dict)
    assert isinstance(infos, dict)
    for agent in env.agents:
        assert agent in obs
        assert agent in infos
        assert isinstance(obs[agent], np.ndarray)


def test_parallel_env_step_5_tuple(registry: ModelRegistry, protocol_path: str) -> None:
    """env.step() returns the standard PettingZoo 5-tuple."""
    env = TSSParallelEnv(
        mode="in_process",
        registry_or_url=registry,
        protocol_path=protocol_path,
    )
    env.reset()

    actions = {agent: [0, 4, 0, 0] for agent in env.agents}
    obs, rews, terms, truncs, infos = env.step(actions)

    assert isinstance(obs, dict)
    assert isinstance(rews, dict)
    assert isinstance(terms, dict)
    assert isinstance(truncs, dict)
    assert isinstance(infos, dict)

    for agent in env.agents:
        assert agent in obs
        assert isinstance(rews[agent], (int, float))
        assert isinstance(terms[agent], bool)
        assert isinstance(truncs[agent], bool)


def test_parallel_env_close(registry: ModelRegistry, protocol_path: str) -> None:
    """env.close() executes cleanly without exception."""
    env = TSSParallelEnv(
        mode="in_process",
        registry_or_url=registry,
        protocol_path=protocol_path,
    )
    env.reset()
    env.close()
