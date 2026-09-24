"""Tests for neural network modules.

10 tests covering: shapes, non-trivial outputs, hidden state init/carry,
distributions, parameter sharing, attention, GRU.
"""

import sys
sys.path.insert(0, ".")

import torch
import numpy as np
import pytest

from src.marl.networks import (
    EmbeddingLayer,
    SelfAttentionBlock,
    GRUBlock,
    ActorHead,
    CriticHead,
    HybridActorHead,
    MultiCategorical,
    SharedPolicyNetwork,
)
from src.marl.config import EMBED_DIM, HIDDEN_DIM, GRU_HIDDEN_DIM


class TestEmbeddingLayer:
    """Tests for EmbeddingLayer."""

    def test_output_shape(self) -> None:
        """EmbeddingLayer produces correct output shape."""
        layer = EmbeddingLayer(input_dim=13, embed_dim=100)
        x = torch.randn(4, 13)
        out = layer(x)
        assert out.shape == (4, 100)

    def test_non_trivial_output(self) -> None:
        """EmbeddingLayer output is non-zero for non-zero input."""
        layer = EmbeddingLayer(input_dim=9, embed_dim=50)
        x = torch.randn(2, 9)
        out = layer(x)
        assert out.abs().sum().item() > 0


class TestSelfAttentionBlock:
    """Tests for SelfAttentionBlock."""

    def test_output_shape(self) -> None:
        """SA block preserves input shape (B, N, D)."""
        sa = SelfAttentionBlock(embed_dim=100, num_heads=4, head_dim=32)
        x = torch.randn(2, 3, 100)
        out = sa(x)
        assert out.shape == (2, 3, 100)

    def test_residual_connection(self) -> None:
        """SA output differs from input (non-identity residual)."""
        sa = SelfAttentionBlock(embed_dim=100, num_heads=4, head_dim=32)
        x = torch.randn(2, 3, 100)
        out = sa(x)
        assert not torch.allclose(x, out, atol=1e-5)

    def test_single_token(self) -> None:
        """SA works with a single token (N=1)."""
        sa = SelfAttentionBlock(embed_dim=64, num_heads=2, head_dim=16)
        x = torch.randn(1, 1, 64)
        out = sa(x)
        assert out.shape == (1, 1, 64)


class TestGRUBlock:
    """Tests for GRUBlock."""

    def test_output_shape(self) -> None:
        """GRU produces correct output and hidden shapes."""
        gru = GRUBlock(input_dim=100, hidden_dim=64)
        x = torch.randn(2, 5, 100)  # B=2, T=5
        output, hidden = gru(x)
        assert output.shape == (2, 5, 64)
        assert hidden.shape == (1, 2, 64)

    def test_hidden_init_zeros(self) -> None:
        """GRU initializes hidden to zeros when None."""
        gru = GRUBlock(input_dim=50, hidden_dim=32)
        x = torch.randn(1, 1, 50)
        out1, h1 = gru(x, hidden=None)
        out2, h2 = gru(x, hidden=torch.zeros(1, 1, 32))
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_hidden_carry(self) -> None:
        """GRU carries hidden state across calls."""
        gru = GRUBlock(input_dim=50, hidden_dim=32)
        x1 = torch.randn(1, 1, 50)
        x2 = torch.randn(1, 1, 50)
        _, h1 = gru(x1, hidden=None)
        out_a, _ = gru(x2, hidden=h1)
        out_b, _ = gru(x2, hidden=None)
        # With carried hidden, output should differ from fresh hidden
        assert not torch.allclose(out_a, out_b, atol=1e-5)


class TestActorHead:
    """Tests for ActorHead."""

    def test_discrete_distribution(self) -> None:
        """Discrete ActorHead returns a Categorical distribution."""
        head = ActorHead(input_dim=500, action_dim=26, action_type="discrete")
        x = torch.randn(4, 500)
        dist = head(x)
        assert hasattr(dist, "sample")
        action = dist.sample()
        assert action.shape == (4,)

    def test_continuous_distribution(self) -> None:
        """Continuous ActorHead returns a Normal distribution."""
        head = ActorHead(input_dim=500, action_dim=2, action_type="continuous")
        x = torch.randn(4, 500)
        dist = head(x)
        action = dist.sample()
        assert action.shape == (4, 2)

    def test_multi_discrete(self) -> None:
        """Multi-discrete ActorHead returns MultiCategorical."""
        head = ActorHead(
            input_dim=500, action_dim=26, action_type="discrete",
            sub_action_sizes=[13, 9, 2, 2],
        )
        x = torch.randn(4, 500)
        dist = head(x)
        assert isinstance(dist, MultiCategorical)
        action = dist.sample()
        assert action.shape == (4, 4)  # 4 sub-actions


class TestCriticHead:
    """Tests for CriticHead."""

    def test_value_shape(self) -> None:
        """CriticHead produces (B, 1) output."""
        head = CriticHead(input_dim=500)
        x = torch.randn(4, 500)
        value = head(x)
        assert value.shape == (4, 1)


class TestHybridActorHead:
    """Tests for HybridActorHead."""

    def test_produces_two_distributions(self) -> None:
        """HybridActorHead returns both continuous and discrete distributions."""
        head = HybridActorHead(
            input_dim=500,
            continuous_dim=2,
            discrete_dim=5,
            discrete_sub_sizes=[3, 2],
        )
        x = torch.randn(4, 500)
        cont_dist, disc_dist = head(x)
        cont_action = cont_dist.sample()
        disc_action = disc_dist.sample()
        assert cont_action.shape == (4, 2)
        assert disc_action.shape == (4, 2)  # 2 sub-actions


class TestSharedPolicyNetwork:
    """Tests for SharedPolicyNetwork."""

    def test_backbone_shape_no_attention(self) -> None:
        """Backbone without attention produces (B, hidden_dim) features."""
        net = SharedPolicyNetwork(obs_dim=13, use_attention=False, use_gru=False)
        obs = torch.randn(4, 13)
        features, hidden = net.forward_backbone(obs)
        assert features.shape == (4, HIDDEN_DIM)
        assert hidden is None

    def test_backbone_shape_with_attention(self) -> None:
        """Backbone with attention processes (B, N, obs_dim) input."""
        net = SharedPolicyNetwork(obs_dim=13, use_attention=True, use_gru=False)
        obs = torch.randn(4, 3, 13)  # 3 entities
        features, hidden = net.forward_backbone(obs)
        assert features.shape == (4, HIDDEN_DIM)
        assert hidden is None

    def test_backbone_shape_with_gru(self) -> None:
        """Backbone with GRU produces hidden state."""
        net = SharedPolicyNetwork(obs_dim=53, use_attention=False, use_gru=True)
        obs = torch.randn(4, 53)
        features, hidden = net.forward_backbone(obs)
        assert features.shape == (4, HIDDEN_DIM)
        assert hidden is not None
        assert hidden.shape == (1, 4, GRU_HIDDEN_DIM)

    def test_parameter_sharing(self) -> None:
        """Same backbone instance shares parameters across forward calls."""
        net = SharedPolicyNetwork(obs_dim=9, use_attention=True, use_gru=False)
        params_before = {n: p.clone() for n, p in net.named_parameters()}
        obs1 = torch.randn(2, 3, 9)
        obs2 = torch.randn(3, 3, 9)
        _ = net.forward_backbone(obs1)
        _ = net.forward_backbone(obs2)
        # Parameters should be unchanged (no training step)
        for name, param in net.named_parameters():
            assert torch.allclose(params_before[name], param)
