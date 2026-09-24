"""Commander policy: high-level hierarchical policy using GRU + PPO.

Operates at a coarser temporal resolution (every HIGH_LEVEL_HORIZON steps).
Observes CommanderObservation (53 dims), outputs discrete commands
{0=ESCAPE, 1=FIGHT target 1, 2=FIGHT target 2, 3=FIGHT target 3}.
Maintains per-agent GRU hidden state tracking.
"""

import typing
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.core.interfaces import BasePolicy
from src.marl.config import (
    CLIP_EPSILON,
    COMMANDER_ACTION_DIM,
    COMMANDER_OBS_DIM,
    ENTROPY_COEF,
    GAMMA,
    GAE_LAMBDA,
    GRU_HIDDEN_DIM,
    LR_ACTOR,
    MAX_GRAD_NORM,
    MINIBATCH_SIZE,
    UPDATE_EPOCHS,
    VALUE_COEF,
)
from src.marl.networks import (
    ActorHead,
    CriticHead,
    EmbeddingLayer,
    GRUBlock,
    SharedPolicyNetwork,
)
from src.marl.ppo import ppo_update_step
from src.marl.rollout_buffer import RolloutBuffer


class CommanderPolicy(BasePolicy, nn.Module):
    """High-level commander policy with GRU temporal memory + PPO.

    Per-agent hidden state tracking: each agent's GRU hidden is stored
    in a dict keyed by agent_id. This allows variable entity counts
    and domain-agnostic behavior.

    Args:
        obs_dim: Commander observation dimension (default 53).
        action_dim: Discrete action space size (default 4).
        gru_hidden: GRU hidden dimension.
        lr: Learning rate.
    """

    def __init__(
        self,
        obs_dim: int = COMMANDER_OBS_DIM,
        action_dim: int = COMMANDER_ACTION_DIM,
        gru_hidden: int = GRU_HIDDEN_DIM,
        config: dict[str, typing.Any] | None = None,
        lr: float = LR_ACTOR,
        num_agents: int = 4,
    ) -> None:
        nn.Module.__init__(self)

        if config is not None:
            lr = config.get("lr", lr)
            obs_dim = config.get("obs_dim", obs_dim)
            action_dim = config.get("action_dim", action_dim)
            gru_hidden = config.get("gru_hidden", gru_hidden)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gru_hidden = gru_hidden
        self.num_agents = num_agents

        # Backbone with GRU, NO attention
        self.backbone = SharedPolicyNetwork(
            obs_dim=obs_dim,
            use_attention=False,
            use_gru=True,
            gru_hidden=gru_hidden,
        )
        backbone_out = self.backbone.output_dim

        # Discrete action head
        self.actor_head = ActorHead(
            input_dim=backbone_out,
            action_dim=action_dim,
            action_type="discrete",
        )
        self.critic_head = CriticHead(backbone_out)

        # Per-agent hidden state tracking
        self.agent_hiddens: dict[str, torch.Tensor] = {}

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def reset_agent_hidden(self, agent_id: str) -> None:
        """Reset GRU hidden state for a specific agent.

        Args:
            agent_id: The agent identifier to reset.
        """
        if agent_id in self.agent_hiddens:
            del self.agent_hiddens[agent_id]

    def reset_all_hiddens(self) -> None:
        """Reset all per-agent GRU hidden states."""
        self.agent_hiddens.clear()

    def act(
        self,
        observation: np.ndarray | torch.Tensor,
        agent_id: str = "commander",
        hidden_state: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[int, float, float, torch.Tensor | None]:
        """Select a commander action for a specific agent.

        Args:
            observation: Commander observation (obs_dim,) or (1, obs_dim).
            agent_id: Identifier for per-agent hidden state tracking.
            hidden_state: Optional external GRU hidden state. If provided, used
                          directly; otherwise falls back to internal self.agent_hiddens.
            deterministic: If True, use argmax instead of sample.

        Returns:
            Tuple of (action_index, log_prob, value_estimate, new_hidden).
        """
        with torch.no_grad():
            if isinstance(observation, np.ndarray):
                obs = torch.as_tensor(observation, dtype=torch.float32)
            else:
                obs = observation
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)

            # Use passed hidden_state if provided; otherwise fall back to per-agent storage
            hidden = hidden_state if hidden_state is not None else self.agent_hiddens.get(agent_id)

            features, new_hidden = self.backbone.forward_backbone(obs, hidden)
            value = self.critic_head(features).squeeze(-1)

            # Store updated hidden state back to internal storage
            if new_hidden is not None:
                self.agent_hiddens[agent_id] = new_hidden.detach()

            dist = self.actor_head(features)
            if deterministic:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)

            return action.item(), log_prob.item(), value.item(), new_hidden

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log_probs, values, and entropy for a batch.

        Note: During batch evaluation we do NOT use per-agent hidden states.
        GRU gets zero-initialized hidden (single timestep per sample).

        Args:
            obs: (B, obs_dim) observation batch.
            actions: (B,) or (B,1) action indices.

        Returns:
            Tuple of (log_probs (B,), values (B,1), entropy_mean scalar).
        """
        features, _ = self.backbone.forward_backbone(obs, hidden=None)
        value = self.critic_head(features)

        dist = self.actor_head(features)
        actions = actions.squeeze(-1).long()
        log_probs = dist.log_prob(actions)
        ent = dist.entropy().mean()

        return log_probs, value, ent

    def update(
        self,
        batch: dict[str, typing.Any] | RolloutBuffer,
    ) -> dict[str, float]:
        """Update commander policy from rollout buffer or batch dict.

        Args:
            batch: RolloutBuffer or dict with obs, actions, etc.

        Returns:
            Dict of loss metrics averaged over all epochs/minibatches.
        """
        if isinstance(batch, RolloutBuffer):
            return self._update_from_buffer(batch)
        elif isinstance(batch, dict):
            return ppo_update_step(self, self.optimizer, batch)
        else:
            raise TypeError(f"Expected RolloutBuffer or dict, got {type(batch)}")

    def _update_from_buffer(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run multiple epochs of minibatch updates from a RolloutBuffer."""
        all_losses: dict[str, list[float]] = {}

        for _ in range(UPDATE_EPOCHS):
            for mb in buffer.get_minibatches(MINIBATCH_SIZE):
                losses = ppo_update_step(self, self.optimizer, mb)
                for k, v in losses.items():
                    all_losses.setdefault(k, []).append(v)

        return {k: float(np.mean(v)) for k, v in all_losses.items()}

    def save(self, path: str | Path) -> None:
        """Save commander policy state dict.

        Args:
            path: File path for checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        """Load commander policy state dict.

        Args:
            path: File path for checkpoint.
        """
        state_dict = torch.load(path, weights_only=True)
        self.load_state_dict(state_dict)
