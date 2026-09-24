"""PPO-Clip loss functions, GAE computation, and update step.

Pure functions taking tensors and returning loss scalars.
Implements the exact mathematical formulations from the spec.
"""

import typing

import numpy as np
import torch
import torch.nn as nn

from src.marl.config import (
    CLIP_EPSILON,
    ENTROPY_COEF,
    MAX_GRAD_NORM,
    UPDATE_EPOCHS,
    VALUE_CLIP_EPSILON,
    VALUE_COEF,
)


def ppo_clip_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = CLIP_EPSILON,
) -> torch.Tensor:
    """PPO-Clip surrogate objective.

    L^CLIP(θ) = E_t [ min( r_t(θ) * Â_t, clip(r_t(θ), 1-ε, 1+ε) * Â_t ) ]

    Args:
        new_log_probs: Log π_θ(a_t|s_t), shape (B,).
        old_log_probs: Log π_θ_old(a_t|s_t), shape (B,).
        advantages: Â_t, shape (B,).
        clip_epsilon: Clipping parameter ε (default 0.2).

    Returns:
        Scalar PPO-Clip loss (to be maximized; negate for gradient descent).
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    return torch.min(surr1, surr2).mean()


def value_loss(
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_epsilon: float = VALUE_CLIP_EPSILON,
) -> torch.Tensor:
    """Clipped value function loss.

    L^VF(φ) = E_t [ max( (V_φ(s_t) - R̂_t)², (clip(V_φ, V_old-ε, V_old+ε) - R̂_t)² ) ]

    Args:
        new_values: V_φ(s_t), shape (B,) or (B,1).
        old_values: V_φ_old(s_t), shape (B,) or (B,1).
        returns: R̂_t, shape (B,) or (B,1).
        clip_epsilon: Value clipping parameter ε_v (default 0.2).

    Returns:
        Scalar value loss.
    """
    new_values = new_values.squeeze(-1)
    old_values = old_values.squeeze(-1)
    returns = returns.squeeze(-1)

    v_clipped = old_values + torch.clamp(
        new_values - old_values, -clip_epsilon, clip_epsilon,
    )
    loss_unclipped = (new_values - returns).pow(2)
    loss_clipped = (v_clipped - returns).pow(2)
    return torch.max(loss_unclipped, loss_clipped).mean()


def entropy_bonus(dist: typing.Any) -> torch.Tensor:
    """Entropy bonus from action distribution.

    L^ENT = E_t [ H(π_θ(·|s_t)) ]

    Args:
        dist: Action distribution with .entropy() method.

    Returns:
        Mean entropy scalar.
    """
    return dist.entropy().mean()


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation.

    δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    Â_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}
    Terminal state: bootstrap_value = 0 when done=True.

    Args:
        rewards: (T,) reward array.
        values: (T,) value estimates.
        dones: (T,) done flags (1.0 = terminal).
        last_value: V(s_T) bootstrap value.
        gamma: Discount factor.
        gae_lambda: GAE lambda.

    Returns:
        Tuple of (advantages (T,), returns (T,)).
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]

        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def ppo_update_step(
    policy: typing.Any,
    optimizer: torch.optim.Optimizer,
    batch: typing.Mapping[str, np.ndarray | torch.Tensor],
    clip_epsilon: float = CLIP_EPSILON,
    value_coef: float = VALUE_COEF,
    entropy_coef: float = ENTROPY_COEF,
    max_grad_norm: float = MAX_GRAD_NORM,
) -> dict[str, float]:
    """Single PPO update step: forward → losses → backward → grad clip → step.

    Args:
        policy: Policy module with evaluate_actions(obs, actions) method.
        optimizer: Optimizer for policy parameters.
        batch: Dict with keys obs, actions, log_probs, advantages, returns, values.
        clip_epsilon: PPO clip parameter.
        value_coef: Value loss coefficient c_v.
        entropy_coef: Entropy coefficient c_e.
        max_grad_norm: Gradient norm clipping threshold.

    Returns:
        Dict with loss components: policy_loss, value_loss, entropy_loss, total_loss.
    """
    device = next(policy.parameters()).device

    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)
    old_values = torch.as_tensor(batch["values"], dtype=torch.float32, device=device)

    # Forward pass through policy
    new_log_probs, new_values, dist_entropy = policy.evaluate_actions(obs, actions)

    # Compute losses
    p_loss = ppo_clip_loss(new_log_probs, old_log_probs, advantages, clip_epsilon)
    v_loss = value_loss(new_values, old_values, returns, clip_epsilon)
    e_loss = dist_entropy

    # Total loss: L = -L^CLIP + c_v * L^VF - c_e * L^ENT
    total = -p_loss + value_coef * v_loss - entropy_coef * e_loss

    optimizer.zero_grad()
    total.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()

    return {
        "policy_loss": p_loss.item(),
        "value_loss": v_loss.item(),
        "entropy_loss": e_loss.item(),
        "total_loss": total.item(),
    }
