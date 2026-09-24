"""Sea Engage policy: π_engage for naval units.

Uses SelfAttentionBlock + HHAPPO with hybrid action space.
Continuous: heading_delta + velocity_cmd (2 dims)
Discrete: weapon_select(2) + fire(2) = 4 logits, factored [2, 2]
"""

import numpy as np
import torch

from src.marl.config import (
    SEA_CONTINUOUS_DIM,
    SEA_DISCRETE_DIM,
    SEA_DISCRETE_SIZES,
    SEA_OBS_DIM,
    LR_ACTOR,
)
from src.marl.policies.base_policy import BaseLowLevelPolicy


class SeaEngagePolicy(BaseLowLevelPolicy):
    """Engage policy for Sea domain.

    Uses SelfAttentionBlock + HHAPPO.
    Continuous head: heading_delta, velocity_cmd (2 dims).
    Discrete head: weapon_select (2 options) + fire (2 options).

    Args:
        lr: Learning rate.
    """

    def __init__(
        self,
        config: dict | None = None,
        lr: float = LR_ACTOR,
    ) -> None:
        if isinstance(config, (int, float)):
            lr = float(config)
            config = None
        if config is not None:
            lr = config.get("lr", lr)

        total_action_dim = SEA_CONTINUOUS_DIM + SEA_DISCRETE_DIM
        super().__init__(
            obs_dim=SEA_OBS_DIM,
            action_dim=total_action_dim,
            action_type="hybrid",
            use_attention=True,
            use_gru=False,
            config=config,
            lr=lr,
            continuous_dim=SEA_CONTINUOUS_DIM,
            discrete_dim=SEA_DISCRETE_DIM,
            discrete_sub_sizes=SEA_DISCRETE_SIZES,
        )
