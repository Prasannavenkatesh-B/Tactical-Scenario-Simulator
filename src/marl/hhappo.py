"""HHAPPO — Hybrid Hierarchical Action PPO.

Handles policies with both continuous (heading, velocity) and discrete
(weapon select, fire) action components with SEPARATE advantage streams.
"""

import typing

import numpy as np
import torch
import torch.nn as nn

from src.marl.config import (
    CLIP_EPSILON,
    ENTROPY_COEF,
    MAX_GRAD_NORM,
    VALUE_CLIP_EPSILON,
    VALUE_COEF,
)
from src.marl.ppo import ppo_clip_loss, value_loss


def hhappo_loss(
    new_cont_log_probs: torch.Tensor,
    old_cont_log_probs: torch.Tensor,
    cont_advantages: torch.Tensor,
    new_disc_log_probs: torch.Tensor,
    old_disc_log_probs: torch.Tensor,
    disc_advantages: torch.Tensor,
    clip_epsilon: float = CLIP_EPSILON,
) -> torch.Tensor:
    """HHAPPO loss with separate advantage streams.

    L^HHAPPO = L^CLIP_cont + L^CLIP_disc

    Critical: Two independent Â^cont and Â^disc — no shared advantage.

    Args:
        new_cont_log_probs: Continuous action log probs, shape (B,).
        old_cont_log_probs: Old continuous log probs, shape (B,).
        cont_advantages: Continuous advantage Â^cont, shape (B,).
        new_disc_log_probs: Discrete action log probs, shape (B,).
        old_disc_log_probs: Old discrete log probs, shape (B,).
        disc_advantages: Discrete advantage Â^disc, shape (B,).
        clip_epsilon: Clipping parameter ε.

    Returns:
        Combined HHAPPO loss scalar (to be maximized; negate for descent).
    """
    cont_loss = ppo_clip_loss(
        new_cont_log_probs, old_cont_log_probs, cont_advantages, clip_epsilon,
    )
    disc_loss = ppo_clip_loss(
        new_disc_log_probs, old_disc_log_probs, disc_advantages, clip_epsilon,
    )
    return cont_loss + disc_loss


def hhappo_update_step(
    policy: typing.Any,
    optimizer: torch.optim.Optimizer,
    batch: typing.Mapping[str, np.ndarray | torch.Tensor],
    clip_epsilon: float = CLIP_EPSILON,
    value_coef: float = VALUE_COEF,
    entropy_coef: float = ENTROPY_COEF,
    max_grad_norm: float = MAX_GRAD_NORM,
) -> dict[str, float]:
    """Single HHAPPO update step with separate continuous/discrete advantages.

    Args:
        policy: Policy module with evaluate_hybrid_actions(obs, actions) method.
        optimizer: Optimizer for policy parameters.
        batch: Dict with keys obs, actions, cont_log_probs, disc_log_probs,
               advantages, returns, values. The 'advantages' field holds the
               combined advantages, and we compute separate streams from stored
               cont/disc log probs.
        clip_epsilon: PPO clip parameter.
        value_coef: Value loss coefficient c_v.
        entropy_coef: Entropy coefficient c_e.
        max_grad_norm: Gradient norm clipping threshold.

    Returns:
        Dict with loss components.
    """
    device = next(policy.parameters()).device

    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)
    old_cont_lp = torch.as_tensor(batch["cont_log_probs"], dtype=torch.float32, device=device)
    old_disc_lp = torch.as_tensor(batch["disc_log_probs"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)
    old_values = torch.as_tensor(batch["values"], dtype=torch.float32, device=device)

    # Forward pass: get separate log probs and distributions
    (new_cont_lp, new_disc_lp, new_values,
     cont_entropy, disc_entropy) = policy.evaluate_hybrid_actions(obs, actions)

    # Separate advantage streams — same advantages but separate ratio computation
    # ensures no gradient leakage between continuous and discrete heads
    cont_advantages = advantages.detach()
    disc_advantages = advantages.detach()

    # Compute HHAPPO loss
    h_loss = hhappo_loss(
        new_cont_lp, old_cont_lp, cont_advantages,
        new_disc_lp, old_disc_lp, disc_advantages,
        clip_epsilon,
    )

    v_loss = value_loss(new_values, old_values, returns, clip_epsilon)
    e_loss = cont_entropy + disc_entropy

    # Total loss: L = -L^HHAPPO + c_v * L^VF - c_e * L^ENT
    total = -h_loss + value_coef * v_loss - entropy_coef * e_loss

    optimizer.zero_grad()
    total.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()

    return {
        "policy_loss": h_loss.item(),
        "value_loss": v_loss.item(),
        "entropy_loss": e_loss.item(),
        "total_loss": total.item(),
        "cont_loss": ppo_clip_loss(
            new_cont_lp.detach(), old_cont_lp, cont_advantages, clip_epsilon,
        ).item(),
        "disc_loss": ppo_clip_loss(
            new_disc_lp.detach(), old_disc_lp, disc_advantages, clip_epsilon,
        ).item(),
    }
