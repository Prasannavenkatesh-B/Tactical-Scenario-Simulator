"""Base low-level policy: shared logic for all domain-level MARL policies.

Subclasses BasePolicy (from src.core.interfaces) AND nn.Module.
Provides act(), evaluate_actions(), evaluate_hybrid_actions(), update(),
save(), load() shared across all 6 domain policies.
"""

import typing
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.core.interfaces import BasePolicy
from src.marl.config import (
    CLIP_EPSILON,
    EMBED_DIM,
    ENTROPY_COEF,
    GAMMA,
    GAE_LAMBDA,
    HIDDEN_DIM,
    LR_ACTOR,
    LR_CRITIC,
    MAX_GRAD_NORM,
    MINIBATCH_SIZE,
    UPDATE_EPOCHS,
    VALUE_COEF,
)
from src.marl.networks import (
    ActorHead,
    CriticHead,
    HybridActorHead,
    MultiCategorical,
    SharedPolicyNetwork,
)
from src.marl.ppo import ppo_update_step
from src.marl.hhappo import hhappo_update_step
from src.marl.rollout_buffer import RolloutBuffer


class BaseLowLevelPolicy(BasePolicy, nn.Module):
    """Shared base class for all domain-level MARL policies.

    Inherits from BasePolicy (src.core) for interface compliance
    AND nn.Module for PyTorch integration.

    Subclasses must set self.backbone, self.actor_head, self.critic_head,
    self.action_type, and self.action_dim before calling super().__init_policy__().

    Args:
        obs_dim: Observation vector dimension.
        action_dim: Total action dimension.
        action_type: "discrete" or "hybrid".
        use_attention: Whether backbone uses SelfAttentionBlock.
        use_gru: Whether backbone uses GRUBlock.
        lr: Learning rate.
        sub_action_sizes: For multi-discrete, list of sub-action sizes.
        continuous_dim: For hybrid, number of continuous action components.
        discrete_dim: For hybrid, total discrete logits dimension.
        discrete_sub_sizes: For hybrid, sub-action sizes for factored discrete.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_type: str = "discrete",
        use_attention: bool = False,
        use_gru: bool = False,
        config: dict[str, typing.Any] | None = None,
        lr: float = LR_ACTOR,
        sub_action_sizes: list[int] | None = None,
        continuous_dim: int = 0,
        discrete_dim: int = 0,
        discrete_sub_sizes: list[int] | None = None,
    ) -> None:
        # nn.Module must be initialized first
        nn.Module.__init__(self)
        # BasePolicy has no __init__, so no call needed

        embed_dim = EMBED_DIM
        hidden_dim = HIDDEN_DIM
        if config is not None:
            lr = config.get("lr", lr)
            embed_dim = config.get("embed_dim", embed_dim)
            hidden_dim = config.get("hidden_dim", hidden_dim)

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_type = action_type
        self.continuous_dim = continuous_dim
        self.discrete_dim = discrete_dim

        # Build backbone
        self.backbone = SharedPolicyNetwork(
            obs_dim=obs_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            use_attention=use_attention,
            use_gru=use_gru,
        )
        backbone_out = self.backbone.output_dim

        # Build heads based on action type
        self.actor_head: ActorHead | HybridActorHead
        if action_type == "hybrid":
            self.actor_head = HybridActorHead(
                input_dim=backbone_out,
                continuous_dim=continuous_dim,
                discrete_dim=discrete_dim,
                discrete_sub_sizes=discrete_sub_sizes,
            )
        else:
            self.actor_head = ActorHead(
                input_dim=backbone_out,
                action_dim=action_dim,
                action_type=action_type,
                sub_action_sizes=sub_action_sizes,
            )

        self.critic_head = CriticHead(backbone_out)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def act(
        self,
        observation: np.ndarray | torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        """Select an action given an observation.

        Args:
            observation: Observation vector (obs_dim,) or (1, obs_dim).
            deterministic: If True, use mode instead of sample.

        Returns:
            Tuple of (action_array, log_prob, value_estimate).
        """
        with torch.no_grad():
            if isinstance(observation, np.ndarray):
                obs = torch.as_tensor(observation, dtype=torch.float32)
            else:
                obs = observation
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)

            features, _ = self.backbone.forward_backbone(obs)
            value = self.critic_head(features).squeeze(-1)

            if self.action_type == "hybrid":
                cont_dist, disc_dist = self.actor_head(features)
                if deterministic:
                    cont_action = cont_dist.mean
                    if isinstance(disc_dist, MultiCategorical):
                        disc_action = torch.stack(
                            [d.probs.argmax(dim=-1) for d in disc_dist.distributions],
                            dim=-1,
                        )
                    else:
                        disc_action = disc_dist.probs.argmax(dim=-1, keepdim=True)
                else:
                    cont_action = cont_dist.sample()
                    disc_action = disc_dist.sample()

                if disc_action.dim() == 1:
                    disc_action = disc_action.unsqueeze(-1)

                action = torch.cat([cont_action, disc_action.float()], dim=-1)
                log_prob = (
                    cont_dist.log_prob(cont_action).sum(dim=-1)
                    + disc_dist.log_prob(disc_action.squeeze(-1) if disc_action.shape[-1] == 1 else disc_action)
                )
            else:
                dist = self.actor_head(features)
                if deterministic:
                    if isinstance(dist, MultiCategorical):
                        action = torch.stack(
                            [d.probs.argmax(dim=-1) for d in dist.distributions],
                            dim=-1,
                        ).float()
                    else:
                        action = dist.probs.argmax(dim=-1, keepdim=True).float()
                else:
                    action = dist.sample()
                    if isinstance(dist, MultiCategorical):
                        pass  # already (B, num_sub_actions)
                    else:
                        action = action.unsqueeze(-1)
                    action = action.float()

                if isinstance(dist, MultiCategorical):
                    log_prob = dist.log_prob(action.long())
                else:
                    log_prob = dist.log_prob(action.squeeze(-1).long())

            return (
                action.squeeze(0).cpu().numpy(),
                log_prob.item(),
                value.item(),
            )

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log_probs, values, and entropy for a batch.

        Used by ppo_update_step for discrete-only policies.

        Args:
            obs: (B, obs_dim) observation batch.
            actions: (B, action_dim) action batch.

        Returns:
            Tuple of (log_probs (B,), values (B,1), entropy_mean scalar).
        """
        features, _ = self.backbone.forward_backbone(obs)
        value = self.critic_head(features)

        dist = self.actor_head(features)
        if isinstance(dist, MultiCategorical):
            log_probs = dist.log_prob(actions.long())
            ent = dist.entropy().mean()
        else:
            log_probs = dist.log_prob(actions.squeeze(-1).long())
            ent = dist.entropy().mean()

        return log_probs, value, ent

    def evaluate_hybrid_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate hybrid actions for HHAPPO.

        Args:
            obs: (B, obs_dim) observation batch.
            actions: (B, total_action_dim) where first continuous_dim columns
                     are continuous, rest are discrete indices.

        Returns:
            Tuple of (cont_log_probs, disc_log_probs, values, cont_entropy, disc_entropy).
        """
        features, _ = self.backbone.forward_backbone(obs)
        value = self.critic_head(features)

        cont_dist, disc_dist = self.actor_head(features)

        cont_actions = actions[:, :self.continuous_dim]
        disc_actions = actions[:, self.continuous_dim:]

        cont_log_probs = cont_dist.log_prob(cont_actions).sum(dim=-1)
        cont_entropy = cont_dist.entropy().sum(dim=-1).mean()

        if isinstance(disc_dist, MultiCategorical):
            disc_log_probs = disc_dist.log_prob(disc_actions.long())
            disc_entropy = disc_dist.entropy().mean()
        else:
            disc_log_probs = disc_dist.log_prob(disc_actions.squeeze(-1).long())
            disc_entropy = disc_dist.entropy().mean()

        return cont_log_probs, disc_log_probs, value, cont_entropy, disc_entropy

    def update(
        self,
        batch: dict[str, typing.Any] | RolloutBuffer,
    ) -> dict[str, float]:
        """Update policy from rollout buffer or batch dict.

        Routes to PPO or HHAPPO based on self.action_type.

        Args:
            batch: RolloutBuffer or dict with obs, actions, etc.

        Returns:
            Dict of loss metrics averaged over all epochs/minibatches.
        """
        if isinstance(batch, RolloutBuffer):
            return self._update_from_buffer(batch)
        elif isinstance(batch, dict):
            if self.action_type == "hybrid":
                return hhappo_update_step(self, self.optimizer, batch)
            return ppo_update_step(self, self.optimizer, batch)
        else:
            raise TypeError(f"Expected RolloutBuffer or dict, got {type(batch)}")

    def _update_from_buffer(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run multiple epochs of minibatch updates from a RolloutBuffer."""
        all_losses: dict[str, list[float]] = {}

        for _ in range(UPDATE_EPOCHS):
            for mb in buffer.get_minibatches(MINIBATCH_SIZE):
                if self.action_type == "hybrid":
                    losses = hhappo_update_step(self, self.optimizer, mb)
                else:
                    losses = ppo_update_step(self, self.optimizer, mb)

                for k, v in losses.items():
                    all_losses.setdefault(k, []).append(v)

        return {k: float(np.mean(v)) for k, v in all_losses.items()}

    def save(self, path: str | Path) -> None:
        """Save policy state dict to file.

        Args:
            path: File path for the checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        """Load policy state dict from file.

        Args:
            path: File path for the checkpoint.
        """
        state_dict = torch.load(path, weights_only=True)
        self.load_state_dict(state_dict)
