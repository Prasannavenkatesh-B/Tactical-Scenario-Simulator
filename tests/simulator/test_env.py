"""Unit tests for TacticalEnv Gymnasium environment, step order, and non-determinism."""

import numpy as np
import pytest

from src.core.actions import AirAction
from src.core.interfaces import AgentStatus, TeamSide
from src.simulator.env import TacticalEnv
from src.simulator.scenarios import generate_scenario


class TestTacticalEnv:
    """Test suite for TacticalEnv simulation loop, termination, and reproducibility."""

    def test_reset_returns_observations(self) -> None:
        """Verify reset returns valid observations for all configured agents."""
        env = TacticalEnv(generate_scenario(1), seed=42)
        obs_dict = env.reset()

        assert len(obs_dict) == 4
        for eid, obs in obs_dict.items():
            assert isinstance(obs, np.ndarray)
            assert np.all((obs >= 0.0) & (obs <= 1.0))

    def test_step_tuple_structure(self) -> None:
        """Verify step returns (obs_dict, reward_dict, done_dict, info_dict)."""
        env = TacticalEnv(generate_scenario(1), seed=42)
        env.reset()

        actions = {
            eid: AirAction(heading_delta=0.0, velocity_cmd=4, fire_cannon=0, fire_rocket=0)
            for eid in env.entities
        }
        obs, rew, done, info = env.step(actions)

        assert isinstance(obs, dict)
        assert isinstance(rew, dict)
        assert isinstance(done, dict)
        assert isinstance(info, dict)
        assert "global_done" in info
        assert "winner" in info

    def test_termination_blue_all_destroyed(self) -> None:
        """Verify episode terminates and Red wins when all Blue agents are destroyed."""
        env = TacticalEnv(generate_scenario(1), seed=42)
        env.reset()

        # Destroy all Blue agents
        for b in env.blue_entities:
            b.status = AgentStatus.DESTROYED

        actions = {eid: AirAction(0.0, 4, 0, 0) for eid in env.entities}
        obs, rew, done, info = env.step(actions)

        assert info["global_done"] is True
        assert info["winner"] == TeamSide.RED
        assert all(done.values())

    def test_termination_red_all_destroyed(self) -> None:
        """Verify episode terminates and Blue wins when all Red agents are destroyed."""
        env = TacticalEnv(generate_scenario(1), seed=42)
        env.reset()

        # Destroy all Red agents
        for r in env.red_entities:
            r.status = AgentStatus.DESTROYED

        actions = {eid: AirAction(0.0, 4, 0, 0) for eid in env.entities}
        obs, rew, done, info = env.step(actions)

        assert info["global_done"] is True
        assert info["winner"] == TeamSide.BLUE
        assert all(done.values())

    def test_termination_horizon_reached(self) -> None:
        """Verify episode ends when current step reaches episode horizon."""
        sc = generate_scenario(1)
        sc.episode_horizon = 5  # short horizon for test
        env = TacticalEnv(sc, seed=42)
        env.reset()

        actions = {eid: AirAction(0.0, 4, 0, 0) for eid in env.entities}
        for _ in range(4):
            obs, rew, done, info = env.step(actions)
            assert info["global_done"] is False

        obs, rew, done, info = env.step(actions)
        assert info["global_done"] is True

    def test_reproducibility_same_seed(self) -> None:
        """Verify two environments initialized with identical seed generate exact initial positions."""
        sc1 = generate_scenario(1, seed=12345)
        sc2 = generate_scenario(1, seed=12345)

        env1 = TacticalEnv(sc1, seed=12345)
        env2 = TacticalEnv(sc2, seed=12345)

        obs1 = env1.reset()
        obs2 = env2.reset()

        for eid in obs1:
            np.testing.assert_allclose(obs1[eid], obs2[eid], atol=1e-5)
            np.testing.assert_allclose(env1.entities[eid].position, env2.entities[eid].position, atol=1e-5)

    def test_nondeterminism_different_seeds(self) -> None:
        """Verify different seeds produce differing initial positions (NON-DETERMINISM)."""
        sc1 = generate_scenario(1, seed=1111)
        sc2 = generate_scenario(1, seed=9999)

        env1 = TacticalEnv(sc1, seed=1111)
        env2 = TacticalEnv(sc2, seed=9999)

        obs1 = env1.reset()
        obs2 = env2.reset()

        # Positions must differ
        diff_count = 0
        for eid in obs1:
            if not np.allclose(env1.entities[eid].position, env2.entities[eid].position, atol=1e-3):
                diff_count += 1
        assert diff_count > 0

    def test_trajectory_reproducibility_and_divergence(self) -> None:
        """Verify trajectory matches with same seed and diverges with different seeds."""
        sc_a = generate_scenario(1, seed=555)
        sc_b = generate_scenario(1, seed=555)
        sc_c = generate_scenario(1, seed=777)

        env_a = TacticalEnv(sc_a, seed=555)
        env_b = TacticalEnv(sc_b, seed=555)
        env_c = TacticalEnv(sc_c, seed=777)

        env_a.reset()
        env_b.reset()
        env_c.reset()

        action = AirAction(heading_delta=10.0, velocity_cmd=5, fire_cannon=0, fire_rocket=0)

        for _ in range(10):
            actions_a = {eid: action for eid in env_a.entities}
            actions_b = {eid: action for eid in env_b.entities}
            actions_c = {eid: action for eid in env_c.entities}

            obs_a, _, _, _ = env_a.step(actions_a)
            obs_b, _, _, _ = env_b.step(actions_b)
            obs_c, _, _, _ = env_c.step(actions_c)

            # A and B must match exactly
            for eid in obs_a:
                np.testing.assert_allclose(obs_a[eid], obs_b[eid], atol=1e-5)

        # A and C must have diverged
        any_diff = False
        for eid in env_a.entities:
            if not np.allclose(env_a.entities[eid].position, env_c.entities[eid].position, atol=1e-2):
                any_diff = True
                break
        assert any_diff
