"""Hyperparameter configuration for the hierarchical MARL engine.

All named constants — no magic numbers elsewhere in the MARL module.
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PPO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LR_ACTOR: float = 1e-4
LR_CRITIC: float = 1e-4
GAMMA: float = 0.95
GAE_LAMBDA: float = 0.95
CLIP_EPSILON: float = 0.2
VALUE_CLIP_EPSILON: float = 0.2
ENTROPY_COEF: float = 0.01
VALUE_COEF: float = 0.5
MAX_GRAD_NORM: float = 0.5

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Batches
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOW_LEVEL_BATCH_SIZE: int = 2000
HIGH_LEVEL_BATCH_SIZE: int = 1000
UPDATE_EPOCHS: int = 10
MINIBATCH_SIZE: int = 256

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Network sizes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EMBED_DIM: int = 256
HIDDEN_DIM: int = 1024
GRU_HIDDEN_DIM: int = 128
ATTENTION_HEADS: int = 4
ATTENTION_HEAD_DIM: int = 64

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Commander
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HIGH_LEVEL_HORIZON: int = 40
LOW_LEVEL_HORIZON: int = 10
COMMANDER_NUM_OPPONENTS: int = 3
COMMANDER_NUM_FRIENDLIES: int = 2
COMMANDER_OBS_DIM: int = 53  # from CommanderObservation.to_padded_array()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Observation dimensions (from src/core/observations.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AIR_OBS_DIM: int = 13
GROUND_OBS_DIM: int = 9
SEA_OBS_DIM: int = 9

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Action dimensions (factored multi-discrete sums)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AIR_AC1_ACTION_DIM: int = 26   # heading(13) + velocity(9) + cannon(2) + rocket(2)
AIR_AC2_ACTION_DIM: int = 24   # heading(13) + velocity(9) + cannon(2)
AIR_AC1_ACTION_SIZES: list[int] = [13, 9, 2, 2]
AIR_AC2_ACTION_SIZES: list[int] = [13, 9, 2]
COMMANDER_ACTION_DIM: int = 4  # discrete {0,1,2,3}

# Ground/Sea HHAPPO dimensions
GROUND_CONTINUOUS_DIM: int = 2   # heading_delta, velocity_cmd
GROUND_DISCRETE_DIM: int = 5    # weapon_select(3) + fire(2)
GROUND_DISCRETE_SIZES: list[int] = [3, 2]
SEA_CONTINUOUS_DIM: int = 2      # heading_delta, velocity_cmd
SEA_DISCRETE_DIM: int = 4       # weapon_select(2) + fire(2)
SEA_DISCRETE_SIZES: list[int] = [2, 2]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Activation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACTIVATION: str = "tanh"
