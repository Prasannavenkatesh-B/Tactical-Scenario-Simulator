"""Neural network modules for hierarchical MARL policies.

All type-hinted with tensor shapes documented in docstrings.
Implements: EmbeddingLayer, SelfAttentionBlock, GRUBlock, ActorHead,
CriticHead, HybridActorHead, SharedPolicyNetwork.
"""

import math
import typing

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from src.marl.config import (
    ACTIVATION,
    ATTENTION_HEAD_DIM,
    ATTENTION_HEADS,
    EMBED_DIM,
    GRU_HIDDEN_DIM,
    HIDDEN_DIM,
)


def _get_activation(name: str) -> nn.Module:
    """Return activation module by name string."""
    if name == "tanh":
        return nn.Tanh()
    elif name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation: {name}")


def _init_weights_orthogonal(module: nn.Module, gain: float = 1.0) -> None:
    """Apply orthogonal initialization to linear/RNN/attention weights."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GRU):
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param, gain=gain)
            elif "bias" in name:
                nn.init.zeros_(param)


def _init_weights_xavier(module: nn.Module) -> None:
    """Apply Xavier/Glorot initialization to linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MultiCategorical Distribution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultiCategorical:
    """Factored multi-discrete distribution: independent categoricals per sub-action.

    Used for air policies with factored discrete actions
    (e.g. heading bins + velocity bins + fire flags).
    """

    def __init__(self, logits: torch.Tensor, sub_sizes: list[int]) -> None:
        """Initialize from concatenated logits split into sub-action groups.

        Args:
            logits: (B, sum(sub_sizes)) concatenated logits.
            sub_sizes: List of sub-action dimensions, e.g. [13, 9, 2, 2].
        """
        self.sub_sizes = sub_sizes
        split_logits = torch.split(logits, sub_sizes, dim=-1)
        self.distributions = [Categorical(logits=sl) for sl in split_logits]

    def sample(self) -> torch.Tensor:
        """Sample one action index per sub-action: (B, num_sub_actions)."""
        return torch.stack([d.sample() for d in self.distributions], dim=-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Sum of log-probs across independent sub-actions: (B,)."""
        lps = []
        for i, d in enumerate(self.distributions):
            lps.append(d.log_prob(actions[..., i]))
        return torch.stack(lps, dim=-1).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Sum of entropies across independent sub-actions: (B,)."""
        return torch.stack([d.entropy() for d in self.distributions], dim=-1).sum(dim=-1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EmbeddingLayer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EmbeddingLayer(nn.Module):
    """Linear embedding with activation: (B, input_dim) -> (B, embed_dim).

    Args:
        input_dim: Observation vector length.
        embed_dim: Embedding output dimension (default 100).
        activation: Activation function name (default "tanh").
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = EMBED_DIM,
        activation: str = ACTIVATION,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            _get_activation(activation),
        )
        self.net.apply(_init_weights_xavier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, input_dim) -> (B, embed_dim)."""
        return self.net(x)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SelfAttentionBlock
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SelfAttentionBlock(nn.Module):
    """Multi-head self-attention + residual + LayerNorm.

    Used by air_fight, ground_engage, ground_defend, sea_engage, sea_defend.

    Args:
        embed_dim: Token embedding dimension D.
        num_heads: Number of attention heads.
        head_dim: Dimension per head (total projection = num_heads * head_dim).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = ATTENTION_HEADS,
        head_dim: int = ATTENTION_HEAD_DIM,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        self.q_proj = nn.Linear(embed_dim, inner_dim)
        self.k_proj = nn.Linear(embed_dim, inner_dim)
        self.v_proj = nn.Linear(embed_dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.scale = math.sqrt(head_dim)

        # Orthogonal init for attention projections
        for mod in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            _init_weights_orthogonal(mod)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual: (B, N, D) -> (B, N, D).

        Args:
            x: Input tensor of shape (B, N, D) where N = sequence length.

        Returns:
            Output tensor of shape (B, N, D).
        """
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, -1)
        attn_out = self.out_proj(attn_out)

        # Residual + LayerNorm
        return self.layer_norm(x + attn_out)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRUBlock
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GRUBlock(nn.Module):
    """GRU temporal memory block. Used by commander policy only.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: GRU hidden state dimension (default 64).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = GRU_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        _init_weights_orthogonal(self.gru)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through GRU.

        Args:
            x: Input tensor (B, T, input_dim).
            hidden: Previous hidden state (1, B, hidden_dim) or None.

        Returns:
            Tuple of (output (B, T, hidden_dim), new_hidden (1, B, hidden_dim)).
        """
        if hidden is None:
            hidden = torch.zeros(1, x.size(0), self.hidden_dim, device=x.device)
        output, new_hidden = self.gru(x, hidden)
        return output, new_hidden


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ActorHead
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ActorHead(nn.Module):
    """Final projection to action distribution.

    Args:
        input_dim: Backbone output dimension.
        action_dim: Total action dimension (sum of sub-action sizes for multi-discrete).
        action_type: "continuous" or "discrete".
        sub_action_sizes: Optional list of sub-action sizes for factored multi-discrete.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        action_type: str,
        sub_action_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.action_type = action_type
        self.action_dim = action_dim
        self.sub_action_sizes = sub_action_sizes

        if action_type == "continuous":
            self.mean_layer = nn.Linear(input_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
            _init_weights_xavier(self.mean_layer)
        elif action_type == "discrete":
            self.logits_layer = nn.Linear(input_dim, action_dim)
            _init_weights_xavier(self.logits_layer)
        else:
            raise ValueError(f"Unknown action_type: {action_type}")

    def forward(self, x: torch.Tensor) -> typing.Any:
        """Produce action distribution from backbone features.

        Args:
            x: (B, input_dim) backbone output.

        Returns:
            Distribution: Normal for continuous, Categorical/MultiCategorical for discrete.
        """
        if self.action_type == "continuous":
            mean = self.mean_layer(x)
            std = self.log_std.exp().expand_as(mean)
            return Normal(mean, std)
        else:
            logits = self.logits_layer(x)
            if self.sub_action_sizes is not None:
                return MultiCategorical(logits, self.sub_action_sizes)
            return Categorical(logits=logits)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CriticHead
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CriticHead(nn.Module):
    """Value function head: (B, input_dim) -> (B, 1).

    Args:
        input_dim: Backbone output dimension.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.value_layer = nn.Linear(input_dim, 1)
        _init_weights_xavier(self.value_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, input_dim) -> (B, 1)."""
        return self.value_layer(x)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HybridActorHead
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HybridActorHead(nn.Module):
    """HHAPPO actor: two sub-heads — one continuous, one discrete.

    Args:
        input_dim: Backbone output dimension.
        continuous_dim: Number of continuous action components (e.g. 2 for heading+velocity).
        discrete_dim: Total discrete logits dimension.
        discrete_sub_sizes: Sub-action sizes for factored discrete.
    """

    def __init__(
        self,
        input_dim: int,
        continuous_dim: int,
        discrete_dim: int,
        discrete_sub_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.continuous_head = ActorHead(input_dim, continuous_dim, "continuous")
        self.discrete_head = ActorHead(
            input_dim, discrete_dim, "discrete",
            sub_action_sizes=discrete_sub_sizes,
        )

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[typing.Any, typing.Any]:
        """Produce continuous and discrete distributions.

        Args:
            x: (B, input_dim) backbone output.

        Returns:
            Tuple of (continuous_dist, discrete_dist).
        """
        cont_dist = self.continuous_head(x)
        disc_dist = self.discrete_head(x)
        return cont_dist, disc_dist


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SharedPolicyNetwork
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SharedPolicyNetwork(nn.Module):
    """Backbone shared between actor and critic with optional SA/GRU.

    Parameter sharing rules:
    - Same-type agents (all AC1) share one policy network instance.
    - Actor and critic share EmbeddingLayer and SA/GRU backbone.
    - Only final ActorHead and CriticHead are separate.

    Args:
        obs_dim: Observation vector length.
        embed_dim: Embedding dimension.
        hidden_dim: Hidden layer dimension.
        use_attention: Whether to use SelfAttentionBlock.
        use_gru: Whether to use GRUBlock.
        attention_heads: Number of SA heads.
        gru_hidden: GRU hidden dim.
        activation: Activation function name.
    """

    def __init__(
        self,
        obs_dim: int,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        use_attention: bool = False,
        use_gru: bool = False,
        attention_heads: int = ATTENTION_HEADS,
        gru_hidden: int = GRU_HIDDEN_DIM,
        activation: str = ACTIVATION,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.use_gru = use_gru
        self.embed_dim = embed_dim

        self.embedding = EmbeddingLayer(obs_dim, embed_dim, activation)

        current_dim = embed_dim
        self.attention: SelfAttentionBlock | None = None
        if use_attention:
            self.attention = SelfAttentionBlock(
                embed_dim, attention_heads, ATTENTION_HEAD_DIM,
            )

        self.gru: GRUBlock | None = None
        if use_gru:
            self.gru = GRUBlock(current_dim, gru_hidden)
            current_dim = gru_hidden

        self.hidden_layer = nn.Sequential(
            nn.Linear(current_dim, hidden_dim),
            _get_activation(activation),
        )
        self.hidden_layer.apply(_init_weights_xavier)

        self.output_dim = hidden_dim

    def forward_backbone(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Shared forward pass through embedding, optional SA/GRU, and hidden layer.

        Args:
            obs: Observation tensor. Shape depends on use_attention:
                - Without SA: (B, obs_dim)
                - With SA: (B, N, obs_dim) where N = number of entities in attention
            hidden: Optional GRU hidden state (1, B, gru_hidden).

        Returns:
            Tuple of (features (B, hidden_dim), new_hidden or None).
        """
        new_hidden: torch.Tensor | None = None

        if self.use_attention and obs.dim() == 3:
            # SA path: embed each token, attend, mean-pool
            B, N, D = obs.shape
            embedded = self.embedding(obs.view(B * N, D)).view(B, N, self.embed_dim)
            assert self.attention is not None
            attended = self.attention(embedded)
            # Mean-pool over sequence dimension
            features = attended.mean(dim=1)  # (B, embed_dim)
        else:
            features = self.embedding(obs)  # (B, embed_dim)

        if self.use_gru and self.gru is not None:
            # GRU expects (B, T, D); treat features as single timestep
            if features.dim() == 2:
                features = features.unsqueeze(1)  # (B, 1, embed_dim)
            gru_out, new_hidden = self.gru(features, hidden)
            features = gru_out[:, -1, :]  # (B, gru_hidden)

        features = self.hidden_layer(features)  # (B, hidden_dim)
        return features, new_hidden
