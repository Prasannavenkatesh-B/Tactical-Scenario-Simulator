"""Ground Defend policy: π_defend for ground units.

Uses SelfAttentionBlock + HHAPPO with hybrid action space.
Same action dimensions as GroundEngagePolicy.
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


class GroundDefendPolicy(BaseLowLevelPolicy):
    """Defend policy for Ground domain.

    Uses SelfAttentionBlock + HHAPPO.
    Same hybrid action space as GroundEngagePolicy.

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
