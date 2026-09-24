"""Tests for domain-level policies.

Tests covering: AC1/AC2 action dims, escape no attention,
ground HHAPPO, act/evaluate roundtrip, save/load weights,
parameter sharing, sea policy.
"""

import sys
sys.path.insert(0, ".")

import tempfile
import os

import torch
import numpy as np
import pytest

from src.marl.policies.air_fight import AirFightPolicy
from src.marl.policies.air_escape import AirEscapePolicy
from src.marl.policies.ground_engage import GroundEngagePolicy
from src.marl.policies.ground_defend import GroundDefendPolicy
from src.marl.policies.sea_engage import SeaEngagePolicy
from src.marl.policies.sea_defend import SeaDefendPolicy
from src.marl.config import (
    AIR_AC1_ACTION_DIM,
    AIR_AC2_ACTION_DIM,
    AIR_OBS_DIM,
    GROUND_OBS_DIM,
    GROUND_CONTINUOUS_DIM,
    GROUND_DISCRETE_DIM,
    SEA_OBS_DIM,
    SEA_CONTINUOUS_DIM,
    SEA_DISCRETE_DIM,
)


class TestAirFightPolicy:
    """Tests for AirFightPolicy."""

    def test_ac1_action_dim(self) -> None:
        """AC1 policy has action_dim = 26."""
        policy = AirFightPolicy(variant="AC1")
        assert policy.action_dim == AIR_AC1_ACTION_DIM
        assert policy.action_dim == 26

    def test_ac2_action_dim(self) -> None:
        """AC2 policy has action_dim = 24."""
        policy = AirFightPolicy(variant="AC2")
        assert policy.action_dim == AIR_AC2_ACTION_DIM
        assert policy.action_dim == 24

    def test_ac1_act_returns_valid(self) -> None:
        """AC1 act() returns action array, log_prob, value."""
        policy = AirFightPolicy(variant="AC1")
        obs = np.random.rand(AIR_OBS_DIM).astype(np.float32)
        action, log_prob, value = policy.act(obs)
        assert isinstance(action, np.ndarray)
        assert action.shape == (4,)  # 4 sub-actions for AC1
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_uses_attention(self) -> None:
        """AirFightPolicy backbone has attention block."""
        policy = AirFightPolicy(variant="AC1")
        assert policy.backbone.use_attention is True
        assert policy.backbone.attention is not None


class TestAirEscapePolicy:
    """Tests for AirEscapePolicy."""

    def test_no_attention(self) -> None:
        """AirEscapePolicy has NO attention block."""
        policy = AirEscapePolicy(variant="AC1")
        assert policy.backbone.use_attention is False
        assert policy.backbone.attention is None

    def test_no_gru(self) -> None:
        """AirEscapePolicy has NO GRU block."""
        policy = AirEscapePolicy(variant="AC1")
        assert policy.backbone.use_gru is False
        assert policy.backbone.gru is None

    def test_act(self) -> None:
        """AirEscapePolicy act() produces valid outputs."""
        policy = AirEscapePolicy(variant="AC2")
        obs = np.random.rand(AIR_OBS_DIM).astype(np.float32)
        action, log_prob, value = policy.act(obs)
        assert action.shape == (3,)  # AC2 has 3 sub-actions


class TestGroundEngagePolicy:
    """Tests for GroundEngagePolicy (HHAPPO)."""

    def test_hybrid_action_type(self) -> None:
        """GroundEngagePolicy uses hybrid action type."""
        policy = GroundEngagePolicy()
        assert policy.action_type == "hybrid"
        assert policy.continuous_dim == GROUND_CONTINUOUS_DIM
        assert policy.discrete_dim == GROUND_DISCRETE_DIM

    def test_act_returns_hybrid_action(self) -> None:
        """act() returns continuous + discrete components."""
        policy = GroundEngagePolicy()
        obs = np.random.rand(GROUND_OBS_DIM).astype(np.float32)
        action, log_prob, value = policy.act(obs)
        assert isinstance(action, np.ndarray)
        # Total dim = continuous(2) + discrete_sub_actions(2)
        assert action.shape[0] == GROUND_CONTINUOUS_DIM + 2  # [heading, velocity, weapon_select, fire]

    def test_evaluate_hybrid_actions(self) -> None:
        """evaluate_hybrid_actions returns separate cont/disc log_probs."""
        policy = GroundEngagePolicy()
        obs = torch.randn(4, GROUND_OBS_DIM)
        # Action: 2 continuous + 2 discrete indices
        actions = torch.tensor([
            [0.5, 0.3, 1.0, 0.0],
            [0.1, 0.8, 2.0, 1.0],
            [-0.2, 0.5, 0.0, 0.0],
            [0.3, 0.1, 1.0, 1.0],
        ], dtype=torch.float32)

        cont_lp, disc_lp, values, cont_ent, disc_ent = policy.evaluate_hybrid_actions(
            obs, actions,
        )
        assert cont_lp.shape == (4,)
        assert disc_lp.shape == (4,)
        assert values.shape == (4, 1)


class TestSeaEngagePolicy:
    """Tests for SeaEngagePolicy (HHAPPO)."""

    def test_hybrid_action_type(self) -> None:
        """SeaEngagePolicy uses hybrid action type."""
        policy = SeaEngagePolicy()
        assert policy.action_type == "hybrid"
        assert policy.continuous_dim == SEA_CONTINUOUS_DIM
        assert policy.discrete_dim == SEA_DISCRETE_DIM

    def test_act(self) -> None:
        """SeaEngagePolicy act() produces valid outputs."""
        policy = SeaEngagePolicy()
        obs = np.random.rand(SEA_OBS_DIM).astype(np.float32)
        action, log_prob, value = policy.act(obs)
        assert action.shape[0] == SEA_CONTINUOUS_DIM + 2  # [heading, velocity, weapon, fire]


class TestSaveLoad:
    """Tests for save/load functionality."""

    def test_save_load_roundtrip(self) -> None:
        """Save and load produces identical policy weights."""
        policy = AirFightPolicy(variant="AC1")
        obs = np.random.rand(AIR_OBS_DIM).astype(np.float32)
        action1, _, _ = policy.act(obs, deterministic=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_policy.pt")
            policy.save(path)

            policy2 = AirFightPolicy(variant="AC1")
            policy2.load(path)
            action2, _, _ = policy2.act(obs, deterministic=True)

        np.testing.assert_array_equal(action1, action2)


class TestParameterSharing:
    """Tests for parameter sharing behavior."""

    def test_same_type_agents_share_params(self) -> None:
        """Two agents using the same policy instance share parameters."""
        policy = AirFightPolicy(variant="AC1")
        obs_agent_1 = np.random.rand(AIR_OBS_DIM).astype(np.float32)
        obs_agent_2 = np.random.rand(AIR_OBS_DIM).astype(np.float32)

        # Both agents use the same policy object
        params_before = {n: p.clone() for n, p in policy.named_parameters()}
        policy.act(obs_agent_1)
        policy.act(obs_agent_2)
        # Parameters unchanged (no training step)
        for name, param in policy.named_parameters():
            assert torch.allclose(params_before[name], param)

    def test_different_type_agents_separate_params(self) -> None:
        """Different policy types have independent parameters."""
        fight_policy = AirFightPolicy(variant="AC1")
        escape_policy = AirEscapePolicy(variant="AC1")

        fight_params = set(id(p) for p in fight_policy.parameters())
        escape_params = set(id(p) for p in escape_policy.parameters())

        # No shared parameter objects
        assert fight_params.isdisjoint(escape_params)
