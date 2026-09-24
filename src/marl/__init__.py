"""Hierarchical MARL engine for multi-domain tactical scenario simulation.

Exports all policy classes, commander, network modules, PPO/HHAPPO functions,
and rollout buffer.
"""

from src.marl.config import (
    AIR_AC1_ACTION_DIM,
    AIR_AC1_ACTION_SIZES,
    AIR_AC2_ACTION_DIM,
    AIR_AC2_ACTION_SIZES,
    AIR_OBS_DIM,
    CLIP_EPSILON,
    COMMANDER_ACTION_DIM,
    COMMANDER_OBS_DIM,
    EMBED_DIM,
    ENTROPY_COEF,
    GAMMA,
    GAE_LAMBDA,
    GROUND_CONTINUOUS_DIM,
    GROUND_DISCRETE_DIM,
    GROUND_OBS_DIM,
    GRU_HIDDEN_DIM,
    HIDDEN_DIM,
    HIGH_LEVEL_HORIZON,
    LOW_LEVEL_BATCH_SIZE,
    LR_ACTOR,
    LR_CRITIC,
    MAX_GRAD_NORM,
    MINIBATCH_SIZE,
    SEA_CONTINUOUS_DIM,
    SEA_DISCRETE_DIM,
    SEA_OBS_DIM,
    UPDATE_EPOCHS,
    VALUE_COEF,
)
from src.marl.networks import (
    ActorHead,
    CriticHead,
    EmbeddingLayer,
    GRUBlock,
    HybridActorHead,
    MultiCategorical,
    SelfAttentionBlock,
    SharedPolicyNetwork,
)
from src.marl.rollout_buffer import RolloutBuffer
from src.marl.ppo import (
    compute_gae,
    entropy_bonus,
    ppo_clip_loss,
    ppo_update_step,
    value_loss,
)
from src.marl.hhappo import hhappo_loss, hhappo_update_step
from src.marl.commander import CommanderPolicy
from src.marl.policies import (
    AirEscapePolicy,
    AirFightPolicy,
    BaseLowLevelPolicy,
    GroundDefendPolicy,
    GroundEngagePolicy,
    SeaDefendPolicy,
    SeaEngagePolicy,
)

__all__ = [
    # Config
    "AIR_OBS_DIM",
    "GROUND_OBS_DIM",
    "SEA_OBS_DIM",
    "AIR_AC1_ACTION_DIM",
    "AIR_AC2_ACTION_DIM",
    "AIR_AC1_ACTION_SIZES",
    "AIR_AC2_ACTION_SIZES",
    "COMMANDER_ACTION_DIM",
    "COMMANDER_OBS_DIM",
    "GAMMA",
    "GAE_LAMBDA",
    "CLIP_EPSILON",
    "ENTROPY_COEF",
    "VALUE_COEF",
    "LR_ACTOR",
    "LR_CRITIC",
    "MAX_GRAD_NORM",
    "UPDATE_EPOCHS",
    "MINIBATCH_SIZE",
    "LOW_LEVEL_BATCH_SIZE",
    "EMBED_DIM",
    "HIDDEN_DIM",
    "GRU_HIDDEN_DIM",
    "HIGH_LEVEL_HORIZON",
    "GROUND_CONTINUOUS_DIM",
    "GROUND_DISCRETE_DIM",
    "SEA_CONTINUOUS_DIM",
    "SEA_DISCRETE_DIM",
    # Networks
    "EmbeddingLayer",
    "SelfAttentionBlock",
    "GRUBlock",
    "ActorHead",
    "CriticHead",
    "HybridActorHead",
    "SharedPolicyNetwork",
    "MultiCategorical",
    # Buffer
    "RolloutBuffer",
    # PPO
    "ppo_clip_loss",
    "value_loss",
    "entropy_bonus",
    "compute_gae",
    "ppo_update_step",
    # HHAPPO
    "hhappo_loss",
    "hhappo_update_step",
    # Policies
    "BaseLowLevelPolicy",
    "AirFightPolicy",
    "AirEscapePolicy",
    "GroundEngagePolicy",
    "GroundDefendPolicy",
    "SeaEngagePolicy",
    "SeaDefendPolicy",
    "CommanderPolicy",
]
