"""Tests for CommanderPolicy.

4 tests covering: action space, GRU hidden persistence,
variable entity counts, domain-agnostic behavior.
"""

import sys
sys.path.insert(0, ".")

import torch
import numpy as np
import pytest

from src.marl.commander import CommanderPolicy
from src.marl.config import COMMANDER_OBS_DIM, COMMANDER_ACTION_DIM


class TestCommanderPolicy:
    """Tests for CommanderPolicy."""

    def test_action_space(self) -> None:
        """Commander produces valid discrete actions in {0, 1, 2, 3}."""
        policy = CommanderPolicy()
        obs = np.random.rand(COMMANDER_OBS_DIM).astype(np.float32)
        for _ in range(20):
            action, log_prob, value, hidden = policy.act(obs)
            assert action in {0, 1, 2, 3}
            assert isinstance(log_prob, float)
            assert isinstance(value, float)
            assert hidden is not None

    def test_gru_hidden_persistence(self) -> None:
        """GRU hidden state persists across act() calls for same agent_id."""
        policy = CommanderPolicy()
        obs1 = np.random.rand(COMMANDER_OBS_DIM).astype(np.float32)
        obs2 = np.random.rand(COMMANDER_OBS_DIM).astype(np.float32)

        # First call: no hidden
        assert "agent_A" not in policy.agent_hiddens
        _, _, _, h1 = policy.act(obs1, agent_id="agent_A")
        assert "agent_A" in policy.agent_hiddens
        assert h1 is not None

        # Second call: hidden should be carried
        _, _, _, h2 = policy.act(obs2, agent_id="agent_A")
        assert h2 is not None
        # Hidden state should have changed
        assert not torch.allclose(h1, h2)

    def test_external_hidden_state(self) -> None:
        """Passing external hidden_state overrides internal storage."""
        policy = CommanderPolicy()
        obs = np.random.rand(COMMANDER_OBS_DIM).astype(np.float32)
        custom_hidden = torch.ones(1, 1, policy.gru_hidden)
        action, lp, val, new_hidden = policy.act(
            obs, agent_id="custom", hidden_state=custom_hidden,
        )
        assert action in {0, 1, 2, 3}
        assert new_hidden is not None
        assert not torch.allclose(custom_hidden, new_hidden)

    def test_variable_entity_counts(self) -> None:
        """Commander handles different agent_ids independently."""
        policy = CommanderPolicy()
        obs = np.random.rand(COMMANDER_OBS_DIM).astype(np.float32)

        # Act for 3 different agents
        policy.act(obs, agent_id="air_1")
        policy.act(obs, agent_id="ground_1")
        policy.act(obs, agent_id="sea_1")

        assert len(policy.agent_hiddens) == 3
        assert "air_1" in policy.agent_hiddens
        assert "ground_1" in policy.agent_hiddens
        assert "sea_1" in policy.agent_hiddens

        # Reset one agent
        policy.reset_agent_hidden("air_1")
        assert "air_1" not in policy.agent_hiddens
        assert len(policy.agent_hiddens) == 2

        # Reset all
        policy.reset_all_hiddens()
        assert len(policy.agent_hiddens) == 0

    def test_domain_agnostic(self) -> None:
        """Commander accepts any valid observation regardless of source domain."""
        policy = CommanderPolicy()

        # Create observations of different "domains" — all 53 dims
        air_obs = np.zeros(COMMANDER_OBS_DIM, dtype=np.float32)
        air_obs[0:5] = [0.3, 0.5, 0.7, 0.8, 0.2]  # Air state

        ground_obs = np.zeros(COMMANDER_OBS_DIM, dtype=np.float32)
        ground_obs[0:5] = [0.1, 0.9, 0.0, 0.2, 0.6]  # Ground state

        sea_obs = np.zeros(COMMANDER_OBS_DIM, dtype=np.float32)
        sea_obs[0:5] = [0.5, 0.5, 0.0, 0.4, 0.3]  # Sea state

        # All should produce valid actions
        for obs in [air_obs, ground_obs, sea_obs]:
            action, lp, v, _ = policy.act(obs, agent_id="test")
            assert action in {0, 1, 2, 3}
            policy.reset_agent_hidden("test")

    def test_evaluate_actions(self) -> None:
        """evaluate_actions returns correct shapes for batch input."""
        policy = CommanderPolicy()
        obs = torch.randn(8, COMMANDER_OBS_DIM)
        actions = torch.randint(0, COMMANDER_ACTION_DIM, (8,))

        log_probs, values, entropy = policy.evaluate_actions(obs, actions)
        assert log_probs.shape == (8,)
        assert values.shape == (8, 1)
        assert entropy.dim() == 0  # scalar
