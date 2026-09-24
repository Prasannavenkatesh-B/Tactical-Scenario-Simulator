"""Ground Engage policy: π_engage for ground units.

Uses SelfAttentionBlock + HHAPPO with hybrid action space.
Continuous: heading_delta + velocity_cmd (2 dims)
Discrete: weapon_select(3) + fire(2) = 5 logits, factored [3, 2]
"""

import numpy as np
import torch

from src.marl.config import (
    GROUND_CONTINUOUS_DIM,
    GROUND_DISCRETE_DIM,
    GROUND_DISCRETE_SIZES,
    GROUND_OBS_DIM,
    LR_ACTOR,
)
from src.marl.policies.base_policy import BaseLowLevelPolicy


class GroundEngagePolicy(BaseLowLevelPolicy):
    """Engage policy for Ground domain.

    Uses SelfAttentionBlock + HHAPPO.
    Continuous head: heading_delta, velocity_cmd (2 dims).
    Discrete head: weapon_select (3 options) + fire (2 options).

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

        total_action_dim = GROUND_CONTINUOUS_DIM + GROUND_DISCRETE_DIM
        super().__init__(
            obs_dim=GROUND_OBS_DIM,
            action_dim=total_action_dim,
            action_type="hybrid",
            use_attention=True,
            use_gru=False,
            config=config,
            lr=lr,
            continuous_dim=GROUND_CONTINUOUS_DIM,
            discrete_dim=GROUND_DISCRETE_DIM,
            discrete_sub_sizes=GROUND_DISCRETE_SIZES,
        )
