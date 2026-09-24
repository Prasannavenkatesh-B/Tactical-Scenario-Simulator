"""Behavioral entropy evaluation over action sequences and episode timesteps.

Provides:
- compute_action_category_entropy: Evaluates entropy of agent action history across timesteps
- compute_policy_entropy_over_episode: Evaluates policy distribution entropy across episode observations
"""

from typing import Any
import numpy as np

from src.evaluation.entropy import policy_entropy, shannon_entropy


def _extract_action_key(action: Any) -> Any:
    """Extract a hashable key from an action object or primitive."""
    if hasattr(action, "heading_delta") and hasattr(action, "velocity_cmd"):
        # AirAction, GroundAction, or SeaAction dataclass
        fire_attrs = []
        for attr in ("fire_cannon", "fire_rocket", "weapon_select", "fire", "cannon_fire", "rocket_fire"):
            if hasattr(action, attr):
                fire_attrs.append(getattr(action, attr))
        return (
            getattr(action, "heading_delta"),
            getattr(action, "velocity_cmd"),
            tuple(fire_attrs),
        )
    if isinstance(action, np.ndarray):
        return tuple(action.flatten().tolist())
    if isinstance(action, (list, tuple)):
        return tuple(action)
    return action


def compute_action_category_entropy(
    action_history: list[Any],
    action_dims: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bucket a sequence of actions into categories and compute entropy over time.

    Args:
        action_history: Sequence of actions chosen across an episode.
        action_dims: Optional specification of category dimensions or count k
                     (e.g., {'k': 4} or {'heading': 13, 'velocity': 9}).

    Returns:
        dict containing:
            - 'entropy_per_category': dict mapping category/component to entropy float
            - 'mean_entropy': float (Shannon entropy across the distribution)
            - 'normalized_entropy': float in [0.0, 1.0]
    """
    if not action_history:
        return {
            "entropy_per_category": {},
            "mean_entropy": 0.0,
            "normalized_entropy": 0.0,
        }

    # Extract hashable categories
    keys = [_extract_action_key(a) for a in action_history]
    total_actions = len(keys)

    # Compute frequencies using Counter
    from collections import Counter

    counts_map = Counter(keys)
    unique_keys = list(counts_map.keys())
    counts = np.array(list(counts_map.values()), dtype=np.float64)
    probs = counts / float(total_actions)

    overall_entropy = shannon_entropy(probs)

    # Build per-category distribution breakdown
    entropy_per_cat: dict[str, float] = {}
    for k_val, p in zip(unique_keys, probs):
        # Category marginal entropy contribution: -p * log(p)
        cat_str = str(k_val)
        contrib = float(-p * np.log(p)) if p > 0 else 0.0
        entropy_per_cat[cat_str] = contrib

    # Determine denominator for normalization
    k = len(unique_keys)
    if action_dims is not None:
        if "k" in action_dims:
            k = int(action_dims["k"])
        elif "total" in action_dims:
            k = int(action_dims["total"])
        elif "categories" in action_dims:
            k = len(action_dims["categories"])

    if k > 1:
        max_entropy = np.log(float(k))
        norm_entropy = float(np.clip(overall_entropy / max_entropy, 0.0, 1.0))
    else:
        norm_entropy = 0.0

    return {
        "entropy_per_category": entropy_per_cat,
        "mean_entropy": float(overall_entropy),
        "normalized_entropy": norm_entropy,
    }


def compute_policy_entropy_over_episode(
    policy: Any,
    observations: list[np.ndarray],
    num_samples: int = 10,
) -> dict[str, Any]:
    """Sample the policy at each timestep observation and compute entropy over time.

    Args:
        policy: MARL policy instance.
        observations: List of observation arrays encountered during an episode.
        num_samples: Number of samples per observation step.

    Returns:
        dict containing:
            - 'mean_entropy': float (average entropy across timesteps)
            - 'std_entropy': float (standard deviation of entropy across timesteps)
            - 'per_timestep': list[float] (entropy for each timestep)
    """
    if not observations:
        return {
            "mean_entropy": 0.0,
            "std_entropy": 0.0,
            "per_timestep": [],
        }

    per_timestep: list[float] = []
    for obs in observations:
        ent_res = policy_entropy(policy, obs, num_samples=num_samples)
        # Use discrete entropy if positive, else continuous entropy
        step_ent = (
            ent_res["discrete_entropy"]
            if ent_res["discrete_entropy"] > 0.0
            else ent_res["continuous_entropy"]
        )
        per_timestep.append(float(step_ent))

    mean_ent = float(np.mean(per_timestep)) if per_timestep else 0.0
    std_ent = float(np.std(per_timestep)) if per_timestep else 0.0

    return {
        "mean_entropy": mean_ent,
        "std_entropy": std_ent,
        "per_timestep": per_timestep,
    }
