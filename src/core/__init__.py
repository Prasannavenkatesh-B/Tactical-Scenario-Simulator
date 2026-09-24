"""Core interfaces, observations, actions, and rewards for the multi-domain MARL system."""

from .interfaces import (
    AgentStatus,
    BaseEntity,
    BaseEnvironment,
    BasePolicy,
    BaseReward,
    DomainType,
    PolicyType,
    TeamSide,
    WeaponType,
)
from .observations import (
    AirObservation,
    CommanderObservation,
    FullObservation,
    GroundObservation,
    NORMALIZATION_CONFIG,
    SeaObservation,
)
from .actions import (
    ACTION_CONFIG,
    AirAction,
    CommanderAction,
    GroundAction,
    SeaAction,
)
from .rewards import (
    MultiDomainRewardWeights,
    REWARD_CONFIG,
    compute_AA,
    compute_ATA,
    compute_boundary_penalty,
    compute_commander_reward,
    compute_composite_reward,
    compute_death_penalty,
    compute_distance,
    compute_escape_reward,
    compute_fight_reward,
    compute_friendly_fire_penalty,
    compute_heading_off,
    compute_kill_reward,
    normalize_angle,
)

__all__ = [
    # Enums
    "DomainType",
    "TeamSide",
    "AgentStatus",
    "PolicyType",
    "WeaponType",
    # Abstract Base Classes
    "BaseEntity",
    "BasePolicy",
    "BaseReward",
    "BaseEnvironment",
    # Observations
    "AirObservation",
    "GroundObservation",
    "SeaObservation",
    "CommanderObservation",
    "FullObservation",
    "NORMALIZATION_CONFIG",
    # Actions
    "AirAction",
    "GroundAction",
    "SeaAction",
    "CommanderAction",
    "ACTION_CONFIG",
    # Rewards
    "MultiDomainRewardWeights",
    "REWARD_CONFIG",
    "compute_fight_reward",
    "compute_escape_reward",
    "compute_commander_reward",
    "compute_boundary_penalty",
    "compute_friendly_fire_penalty",
    "compute_kill_reward",
    "compute_death_penalty",
    "compute_composite_reward",
    "normalize_angle",
    "compute_distance",
    "compute_ATA",
    "compute_AA",
    "compute_heading_off",
]
