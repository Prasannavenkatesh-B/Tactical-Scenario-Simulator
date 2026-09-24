"""Air Escape policy: π_escape for AC1 and AC2 aircraft.

No attention, no GRU — fast evasive policy.
Same action space as AirFightPolicy.
"""

import numpy as np
import torch

from src.marl.config import (
    AIR_AC1_ACTION_DIM,
    AIR_AC1_ACTION_SIZES,
    AIR_AC2_ACTION_DIM,
    AIR_AC2_ACTION_SIZES,
    AIR_OBS_DIM,
    LR_ACTOR,
)
from src.marl.policies.base_policy import BaseLowLevelPolicy


class AirEscapePolicy(BaseLowLevelPolicy):
    """Escape policy for Air domain (AC1 / AC2).

    NO attention, NO GRU — lightweight for fast evasive maneuvers.
    Same factored discrete action space as AirFightPolicy.

    Args:
        variant: "AC1" or "AC2" to determine action space.
        lr: Learning rate.
    """

    def __init__(
        self,
        variant: str = "AC1",
        config: dict | None = None,
        lr: float = LR_ACTOR,
    ) -> None:
        if isinstance(config, (int, float)):
            lr = float(config)
            config = None
        if config is not None:
            lr = config.get("lr", lr)
            variant = config.get("variant", variant)

        if variant == "AC1":
            action_dim = AIR_AC1_ACTION_DIM
            sub_action_sizes = AIR_AC1_ACTION_SIZES
        elif variant == "AC2":
            action_dim = AIR_AC2_ACTION_DIM
            sub_action_sizes = AIR_AC2_ACTION_SIZES
        else:
            raise ValueError(f"Unknown variant: {variant}, expected AC1 or AC2")

        self.variant = variant

        super().__init__(
            obs_dim=AIR_OBS_DIM,
            action_dim=action_dim,
            action_type="discrete",
            use_attention=False,
            use_gru=False,
            config=config,
            lr=lr,
            sub_action_sizes=sub_action_sizes,
        )
