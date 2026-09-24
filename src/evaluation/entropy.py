"""Entropy calculation utilities for discrete, continuous, and hybrid MARL policies.

Provides:
- shannon_entropy: Discrete Shannon entropy H(π) = -Σ p log(p)
- normalized_shannon_entropy: Normalized discrete entropy in [0, 1]
- differential_entropy_gaussian: Gaussian differential entropy h(π) = 0.5 Σ log(2πe σ_i^2)
- policy_entropy: Extracts or samples policy distributions and computes entropy metrics
"""

from typing import Any
import numpy as np
import torch


def shannon_entropy(action_probs: np.ndarray | list[float]) -> float:
    """Compute discrete Shannon entropy: H(π) = -Σ p·log(p).

    Uses natural logarithm base e. Handles p=0 correctly (0·log0 = 0).

    Args:
        action_probs: Array of probabilities summing to 1.

    Returns:
        Shannon entropy in nats (float >= 0.0).
    """
    probs = np.asarray(action_probs, dtype=np.float64).flatten()
    if probs.size == 0:
        return 0.0

    # Ensure positive probabilities (filter out zeros and negatives)
    p = probs[probs > 0.0]
    if p.size == 0:
        return 0.0

    return float(-np.sum(p * np.log(p)))


def normalized_shannon_entropy(action_probs: np.ndarray | list[float]) -> float:
    """Compute normalized Shannon entropy: H_norm = H(π) / log(len(p)).

    Returns:
        Float value in [0, 1]. Returns 0.0 if len(p) <= 1.
    """
    probs = np.asarray(action_probs, dtype=np.float64).flatten()
    n = len(probs)
    if n <= 1:
        return 0.0

    h = shannon_entropy(probs)
    denom = np.log(float(n))
    if denom <= 0.0:
        return 0.0

    norm = h / denom
    return float(np.clip(norm, 0.0, 1.0))


def differential_entropy_gaussian(
    mean: np.ndarray | list[float],
    std: np.ndarray | list[float],
) -> float:
    """Compute differential entropy of a multidimensional Gaussian policy.

    Formula:
        h = 0.5 * log((2πe)^d · det(Σ))
        For diagonal covariance: h = 0.5 * Σ log(2πe · σ_i^2)

    Args:
        mean: Mean vector (unused in differential entropy, preserved for signature).
        std: Standard deviation vector σ.

    Returns:
        Differential entropy in nats.
    """
    s = np.asarray(std, dtype=np.float64).flatten()
    if s.size == 0:
        return 0.0

    # Guard against non-positive standard deviations
    s = np.clip(s, 1e-12, None)
    term = np.log(2.0 * np.pi * np.e * (s ** 2))
    return float(0.5 * np.sum(term))


def policy_entropy(
    policy: Any,
    observation: np.ndarray,
    num_samples: int = 1,
) -> dict[str, float]:
    """Sample the policy or extract its distribution to compute action entropy.

    Supports both analytical evaluation (if policy has actor network) and
    empirical sampling (if policy only exposes .act()).

    Args:
        policy: MARL policy instance (BaseLowLevelPolicy, CommanderPolicy, or duck-typed).
        observation: Observation vector matching policy input dimension.
        num_samples: Number of stochastic samples for empirical estimation if analytical
                     distribution is not accessible or if num_samples > 1 is requested.

    Returns:
        dict containing:
            - 'discrete_entropy': float >= 0.0
            - 'continuous_entropy': float
            - 'normalized': float in [0, 1]
    """
    # 1. Attempt analytical distribution extraction if policy has backbone & actor_head
    if hasattr(policy, "actor_head") and hasattr(policy, "backbone"):
        try:
            with torch.no_grad():
                obs = torch.as_tensor(observation, dtype=torch.float32)
                if obs.dim() == 1:
                    obs = obs.unsqueeze(0)

                features, _ = policy.backbone.forward_backbone(obs)
                action_type = getattr(policy, "action_type", "discrete")

                if action_type == "hybrid":
                    cont_dist, disc_dist = policy.actor_head(features)
                    cont_ent = float(cont_dist.entropy().sum().item())
                    if hasattr(disc_dist, "distributions"):
                        disc_ent = float(sum(d.entropy().sum().item() for d in disc_dist.distributions))
                        max_disc = float(sum(np.log(d.probs.shape[-1]) for d in disc_dist.distributions))
                    elif hasattr(disc_dist, "probs"):
                        disc_ent = float(disc_dist.entropy().sum().item())
                        max_disc = float(np.log(disc_dist.probs.shape[-1]))
                    else:
                        disc_ent = 0.0
                        max_disc = 1.0

                    norm = float(np.clip(disc_ent / max(max_disc, 1e-6), 0.0, 1.0)) if max_disc > 0 else 0.0
                    return {
                        "discrete_entropy": disc_ent,
                        "continuous_entropy": cont_ent,
                        "normalized": norm,
                    }
                else:
                    dist = policy.actor_head(features)
                    if hasattr(dist, "distributions"):
                        disc_ent = float(sum(d.entropy().sum().item() for d in dist.distributions))
                        max_disc = float(sum(np.log(d.probs.shape[-1]) for d in dist.distributions))
                    elif hasattr(dist, "probs"):
                        disc_ent = float(dist.entropy().sum().item())
                        max_disc = float(np.log(dist.probs.shape[-1]))
                    else:
                        disc_ent = 0.0
                        max_disc = 1.0

                    norm = float(np.clip(disc_ent / max(max_disc, 1e-6), 0.0, 1.0)) if max_disc > 0 else 0.0
                    return {
                        "discrete_entropy": disc_ent,
                        "continuous_entropy": 0.0,
                        "normalized": norm,
                    }
        except Exception:
            pass

    # 2. Empirical sampling fallback via policy.act()
    sample_count = max(num_samples, 20 if num_samples == 1 else num_samples)
    sampled_actions = []

    for _ in range(sample_count):
        if hasattr(policy, "act"):
            try:
                # Most policies accept deterministic=False
                act_res = policy.act(observation, deterministic=False)
                act_val = act_res[0] if isinstance(act_res, tuple) else act_res
            except TypeError:
                act_res = policy.act(observation)
                act_val = act_res[0] if isinstance(act_res, tuple) else act_res
        elif callable(policy):
            act_val = policy(observation)
        else:
            act_val = 0

        if isinstance(act_val, torch.Tensor):
            act_val = act_val.detach().cpu().numpy()
        sampled_actions.append(np.asarray(act_val).flatten())

    samples_arr = np.array(sampled_actions)  # shape (N, action_dim)
    
    # Check if samples are discrete or continuous
    is_integer_like = np.all(np.equal(np.mod(samples_arr, 1), 0))
    if is_integer_like:
        # Convert each row to tuple for categorical frequency
        unique_rows, counts = np.unique(samples_arr, axis=0, return_counts=True)
        probs = counts / float(len(samples_arr))
        disc_ent = shannon_entropy(probs)
        num_categories = len(unique_rows)
        norm = normalized_shannon_entropy(probs)
        return {
            "discrete_entropy": disc_ent,
            "continuous_entropy": 0.0,
            "normalized": norm,
        }
    else:
        # Continuous: estimate std
        mean = np.mean(samples_arr, axis=0)
        std = np.std(samples_arr, axis=0)
        cont_ent = differential_entropy_gaussian(mean, std)
        # Normalization proxy based on std
        norm = float(np.clip(np.mean(std) / (1.0 + np.mean(std)), 0.0, 1.0))
        return {
            "discrete_entropy": 0.0,
            "continuous_entropy": cont_ent,
            "normalized": norm,
        }
