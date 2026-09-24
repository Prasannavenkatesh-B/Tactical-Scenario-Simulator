"""Reward calculation functions and tactical geometry routines.

Implements domain-specific rewards, hierarchical commander rewards, penalty
functions, composite multi-domain aggregation, and spatial orientation metrics
for DRDO tactical scenario simulation.
"""

from dataclasses import dataclass
import math
import typing
import numpy as np


# Module-level configuration constants for rewards and thresholds
REWARD_FIGHT_MIN: float = 1.0
REWARD_FIGHT_MAX: float = 2.0

ESCAPE_DIST_MIN_KM: float = 6.0
ESCAPE_DIST_MAX_KM: float = 13.0
ESCAPE_PENALTY_VALUE: float = -0.01
ESCAPE_REWARD_VALUE: float = 0.01

COMMANDER_REWARD_VALUE: float = 0.1
COMMANDER_DIST_THRESHOLD_KM: float = 5.0
COMMANDER_ATA_THRESHOLD_DEG: float = 30.0
COMMANDER_AA_THRESHOLD_DEG: float = 50.0

PENALTY_BOUNDARY_VALUE: float = -5.0
PENALTY_FRIENDLY_FIRE_VALUE: float = -2.0
REWARD_KILL_VALUE: float = 1.0
PENALTY_DEATH_VALUE: float = -1.0

DEFAULT_WEIGHT_AIR: float = 1.0
DEFAULT_WEIGHT_GROUND: float = 1.0
DEFAULT_WEIGHT_SEA: float = 1.0
DEFAULT_WEIGHT_MISSION: float = 0.5

TWO_PI: float = 2.0 * math.pi
PI: float = math.pi

REWARD_CONFIG: dict[str, float] = {
    "reward_fight_min": REWARD_FIGHT_MIN,
    "reward_fight_max": REWARD_FIGHT_MAX,
    "escape_dist_min_km": ESCAPE_DIST_MIN_KM,
    "escape_dist_max_km": ESCAPE_DIST_MAX_KM,
    "escape_penalty": ESCAPE_PENALTY_VALUE,
    "escape_reward": ESCAPE_REWARD_VALUE,
    "commander_reward": COMMANDER_REWARD_VALUE,
    "commander_dist_threshold_km": COMMANDER_DIST_THRESHOLD_KM,
    "commander_ata_threshold_deg": COMMANDER_ATA_THRESHOLD_DEG,
    "commander_aa_threshold_deg": COMMANDER_AA_THRESHOLD_DEG,
    "penalty_boundary": PENALTY_BOUNDARY_VALUE,
    "penalty_friendly_fire": PENALTY_FRIENDLY_FIRE_VALUE,
    "reward_kill": REWARD_KILL_VALUE,
    "penalty_death": PENALTY_DEATH_VALUE,
    "default_weight_air": DEFAULT_WEIGHT_AIR,
    "default_weight_ground": DEFAULT_WEIGHT_GROUND,
    "default_weight_sea": DEFAULT_WEIGHT_SEA,
    "default_weight_mission": DEFAULT_WEIGHT_MISSION,
}


def compute_fight_reward(
    alpha_ATA_a: float,
    c_max: float,
    c_rem: float,
    clamp_range: bool = False,
) -> float:
    r"""Compute fight policy reward according to base paper Eq. 1.

    Mathematical Formula:
        $$r_{\text{fight}} = \alpha_{\text{ATA}, a} + \frac{c_{\text{max}} - c_{\text{rem}}}{c_{\text{max}}}$$

    where:
        $\alpha_{\text{ATA}, a}$ = antenna train angle of opponent to agent (normalized in $[0, 1]$)
        $c_{\text{max}}$ = maximum ammunition capacity (cannon + rockets, $c_{\text{max}} > 0$)
        $c_{\text{rem}}$ = remaining ammunition ($0 \le c_{\text{rem}} \le c_{\text{max}}$)

    Range:
        $[1, 2]$ under valid tactical fight engagements (where opponent is tracked
        and ammo is expended). No per-timestep penalty.

    Args:
        alpha_ATA_a: Normalized antenna train angle of opponent to agent [0, 1].
        c_max: Maximum ammunition (cannon + rockets).
        c_rem: Remaining ammunition.
        clamp_range: If True, explicitly clamp output to [1.0, 2.0].

    Returns:
        Scalar fight policy reward.
    """
    alpha_clamped = float(np.clip(alpha_ATA_a, 0.0, 1.0))

    if c_max <= 0:
        ammo_term = 0.0
    else:
        c_rem_clamped = float(np.clip(c_rem, 0.0, c_max))
        ammo_term = (c_max - c_rem_clamped) / c_max

    reward = alpha_clamped + ammo_term

    if clamp_range:
        reward = float(np.clip(reward, REWARD_FIGHT_MIN, REWARD_FIGHT_MAX))

    return reward


def compute_escape_reward(
    distance: float,
    in_km: bool = True,
) -> float:
    r"""Compute escape policy reward according to base paper Eq. 2.

    Mathematical Formula:
        $$r_{\text{escape}} = \begin{cases}
            -0.01 & \text{if } d < 6\text{ km} \\
            +0.01 & \text{if } d > 13\text{ km} \\
            0 & \text{otherwise}
        \end{cases}$$

    where $d$ is the distance to the closest opponent.

    Args:
        distance: Distance to closest opponent (in km if in_km=True, or in meters).
        in_km: Flag indicating whether distance is supplied in kilometers.

    Returns:
        Scalar escape policy reward (-0.01, +0.01, or 0.0).
    """
    d_km = distance if in_km else (distance / 1000.0)

    if d_km < ESCAPE_DIST_MIN_KM:
        return ESCAPE_PENALTY_VALUE
    elif d_km > ESCAPE_DIST_MAX_KM:
        return ESCAPE_REWARD_VALUE
    return 0.0


def compute_commander_reward(
    d_o: float,
    alpha_ATA_o: float,
    alpha_AA_o: float,
    a_c: int,
    dist_in_km: bool = True,
    angle_in_degrees: bool = True,
) -> float:
    r"""Compute commander favorable situation reward according to base paper Eq. 3.

    Mathematical Formula:
        $$r_{\text{commander}} = \begin{cases}
            +0.1 & \text{if } (d_o < 5\text{ km}) \land (\alpha_{\text{ATA}, o} < 30^\circ)
                   \land (\alpha_{\text{AA}, o} < 50^\circ) \land (a_c > 0) \\
            0 & \text{otherwise}
        \end{cases}$$

    where:
        $d_o$ = distance to opponent
        $\alpha_{\text{ATA}, o}$ = antenna train angle of opponent
        $\alpha_{\text{AA}, o}$ = aspect angle of opponent
        $a_c$ = commander action (must be > 0, i.e., not escape)

    Args:
        d_o: Distance to opponent.
        alpha_ATA_o: Antenna train angle of opponent.
        alpha_AA_o: Aspect angle of opponent.
        a_c: Commander action index (0=ESCAPE, 1,2,3=FIGHT target 1,2,3).
        dist_in_km: True if d_o is in kilometers, False if in meters.
        angle_in_degrees: True if angles are in degrees, False if in radians.

    Returns:
        +0.1 if all tactical dominance criteria are met, 0.0 otherwise.
    """
    d_km = d_o if dist_in_km else (d_o / 1000.0)
    ata_deg = alpha_ATA_o if angle_in_degrees else math.degrees(alpha_ATA_o)
    aa_deg = alpha_AA_o if angle_in_degrees else math.degrees(alpha_AA_o)

    if (
        d_km < COMMANDER_DIST_THRESHOLD_KM
        and ata_deg < COMMANDER_ATA_THRESHOLD_DEG
        and aa_deg < COMMANDER_AA_THRESHOLD_DEG
        and a_c > 0
    ):
        return COMMANDER_REWARD_VALUE

    return 0.0


def compute_boundary_penalty(out_of_bounds: bool) -> float:
    r"""Compute penalty for exiting map boundaries.

    Mathematical Formula:
        $$r_{\text{boundary}} = \begin{cases}
            -5.0 & \text{if agent exits map boundary} \\
            0.0 & \text{otherwise}
        \end{cases}$$

    Args:
        out_of_bounds: True if entity has exceeded scenario boundary.

    Returns:
        -5.0 if out of bounds, 0.0 otherwise.
    """
    return PENALTY_BOUNDARY_VALUE if out_of_bounds else 0.0


def compute_friendly_fire_penalty(friendly_fire: bool) -> float:
    r"""Compute penalty for friendly fire destruction.

    Mathematical Formula:
        $$r_{\text{friendly\_fire}} = \begin{cases}
            -2.0 & \text{if agent destroys a friendly aircraft} \\
            0.0 & \text{otherwise}
        \end{cases}$$

    Args:
        friendly_fire: True if agent damaged/destroyed a friendly entity.

    Returns:
        -2.0 if friendly fire occurred, 0.0 otherwise.
    """
    return PENALTY_FRIENDLY_FIRE_VALUE if friendly_fire else 0.0


def compute_kill_reward(kill_count: int = 1) -> float:
    r"""Compute reward for destroying an opponent entity.

    Mathematical Formula:
        $$r_{\text{kill}} = +1.0 \times \text{kill\_count}$$

    Args:
        kill_count: Number of opponents eliminated in this step.

    Returns:
        Positive scalar kill reward.
    """
    return REWARD_KILL_VALUE * float(kill_count)


def compute_death_penalty(is_killed: bool) -> float:
    r"""Compute penalty when an agent is destroyed.

    Mathematical Formula:
        $$r_{\text{death}} = \begin{cases}
            -1.0 & \text{if agent is killed} \\
            0.0 & \text{otherwise}
        \end{cases}$$

    Args:
        is_killed: True if agent was destroyed.

    Returns:
        -1.0 if killed, 0.0 otherwise.
    """
    return PENALTY_DEATH_VALUE if is_killed else 0.0


@dataclass
class MultiDomainRewardWeights:
    """Configurable weights for multi-domain composite reward aggregation.

    Attributes:
        w_air: Weight for air domain rewards (default: 1.0).
        w_ground: Weight for ground domain rewards (default: 1.0).
        w_sea: Weight for sea domain rewards (default: 1.0).
        w_mission: Weight for scenario mission rewards (default: 0.5).
    """
    w_air: float = DEFAULT_WEIGHT_AIR
    w_ground: float = DEFAULT_WEIGHT_GROUND
    w_sea: float = DEFAULT_WEIGHT_SEA
    w_mission: float = DEFAULT_WEIGHT_MISSION


def compute_composite_reward(
    r_air: float = 0.0,
    r_ground: float = 0.0,
    r_sea: float = 0.0,
    r_mission: float = 0.0,
    r_boundary: float = 0.0,
    r_friendly_fire: float = 0.0,
    weights: MultiDomainRewardWeights | None = None,
) -> float:
    r"""Compute composite multi-domain reward.

    Mathematical Formula:
        $$r_{\text{total}} = w_{\text{air}} r_{\text{air}} + w_{\text{ground}} r_{\text{ground}}
                           + w_{\text{sea}} r_{\text{sea}} + w_{\text{mission}} r_{\text{mission}}
                           + r_{\text{boundary}} + r_{\text{friendly\_fire}}$$

    Default weights:
        $w_{\text{air}} = 1.0, w_{\text{ground}} = 1.0, w_{\text{sea}} = 1.0, w_{\text{mission}} = 0.5$.

    Args:
        r_air: Air domain reward component.
        r_ground: Ground domain reward component.
        r_sea: Sea domain reward component.
        r_mission: High-level mission objective reward component.
        r_boundary: Boundary violation penalty component.
        r_friendly_fire: Friendly fire penalty component.
        weights: Optional MultiDomainRewardWeights instance.

    Returns:
        Aggregated composite scalar reward.
    """
    w = weights if weights is not None else MultiDomainRewardWeights()
    return (
        w.w_air * r_air
        + w.w_ground * r_ground
        + w.w_sea * r_sea
        + w.w_mission * r_mission
        + r_boundary
        + r_friendly_fire
    )


# =============================================================================
# Spatial & Angular Geometry Utilities
# =============================================================================

def normalize_angle(angle_rad: float) -> float:
    r"""Normalize an angle in radians to the interval [0, 2pi).

    Mathematical Formula:
        $$\theta_{\text{norm}} = \theta \pmod{2\pi}$$

    Args:
        angle_rad: Arbitrary angle in radians.

    Returns:
        Normalized angle strictly in [0.0, 2pi).
    """
    wrapped = angle_rad % TWO_PI
    if wrapped < 0.0:
        wrapped += TWO_PI
    # Guard against float rounding giving exactly TWO_PI
    if wrapped >= TWO_PI:
        wrapped = 0.0
    return float(wrapped)


def compute_distance(
    pos1: typing.Sequence[float] | np.ndarray,
    pos2: typing.Sequence[float] | np.ndarray,
) -> float:
    r"""Compute Euclidean distance between two spatial positions.

    Mathematical Formula:
        $$d = \|\mathbf{p}_1 - \mathbf{p}_2\|_2 = \sqrt{\sum_{i=1}^n (p_{1, i} - p_{2, i})^2}$$

    Args:
        pos1: Coordinate vector for position 1 (2D or 3D).
        pos2: Coordinate vector for position 2 (same dimension as pos1).

    Returns:
        Euclidean distance (0.0 if pos1 == pos2).
    """
    p1 = np.asarray(pos1, dtype=np.float64)
    p2 = np.asarray(pos2, dtype=np.float64)
    return float(np.linalg.norm(p1 - p2))


def compute_ATA(
    observer_pos: typing.Sequence[float] | np.ndarray,
    observer_heading: float,
    target_pos: typing.Sequence[float] | np.ndarray,
) -> float:
    r"""Compute Antenna Train Angle (ATA) in radians in [0, pi].

    The angle between the observer's heading vector and the line of sight (LOS)
    vector toward the target. Returns 0.0 when the observer points directly at the target.

    Mathematical Formula:
        $$\theta_{\text{LOS}} = \operatorname{atan2}(y_{\text{target}} - y_{\text{obs}},
                                                     x_{\text{target}} - x_{\text{obs}})$$
        $$\Delta\theta = (\theta_{\text{LOS}} - \theta_{\text{obs}} + \pi) \pmod{2\pi} - \pi$$
        $$\alpha_{\text{ATA}} = |\Delta\theta| \in [0, \pi]$$

    Args:
        observer_pos: 2D or 3D position of observer.
        observer_heading: Observer heading angle in radians.
        target_pos: 2D or 3D position of target.

    Returns:
        ATA angle magnitude in radians in [0.0, pi].
    """
    dx = float(target_pos[0]) - float(observer_pos[0])
    dy = float(target_pos[1]) - float(observer_pos[1])

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0

    los_angle = math.atan2(dy, dx)
    diff = (los_angle - observer_heading + PI) % TWO_PI - PI
    return float(abs(diff))


def compute_AA(
    observer_pos: typing.Sequence[float] | np.ndarray,
    observer_heading: float,
    target_pos: typing.Sequence[float] | np.ndarray,
    target_heading: float,
) -> float:
    r"""Compute Aspect Angle (AA) in radians in [0, pi].

    The angle between the target's heading vector and the line of sight vector
    from the observer to the target. Returns 0.0 when the observer is directly
    behind the target (tail-chasing / 6 o'clock position).

    Mathematical Formula:
        $$\theta_{\text{LOS}} = \operatorname{atan2}(y_{\text{target}} - y_{\text{obs}},
                                                     x_{\text{target}} - x_{\text{obs}})$$
        $$\Delta\theta = (\text{target\_heading} - \theta_{\text{LOS}} + \pi) \pmod{2\pi} - \pi$$
        $$\alpha_{\text{AA}} = |\Delta\theta| \in [0, \pi]$$

    Args:
        observer_pos: Position of observer.
        observer_heading: Observer heading in radians (reserved for 3D extension).
        target_pos: Position of target.
        target_heading: Target heading angle in radians.

    Returns:
        Aspect angle magnitude in radians in [0.0, pi].
    """
    dx = float(target_pos[0]) - float(observer_pos[0])
    dy = float(target_pos[1]) - float(observer_pos[1])

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0

    los_angle = math.atan2(dy, dx)
    diff = (target_heading - los_angle + PI) % TWO_PI - PI
    return float(abs(diff))


def compute_heading_off(
    observer_heading: float,
    target_pos: typing.Sequence[float] | np.ndarray,
    observer_pos: typing.Sequence[float] | np.ndarray,
) -> float:
    r"""Compute heading-off angle to target in radians in [0, pi].

    The angular displacement between the observer's heading vector and the line
    of sight pointing toward the target.

    Mathematical Formula:
        $$\theta_{\text{LOS}} = \operatorname{atan2}(y_{\text{target}} - y_{\text{obs}},
                                                     x_{\text{target}} - x_{\text{obs}})$$
        $$\Delta\theta = (\theta_{\text{LOS}} - \text{observer\_heading} + \pi) \pmod{2\pi} - \pi$$
        $$\alpha_{\text{off}} = |\Delta\theta| \in [0, \pi]$$

    Args:
        observer_heading: Observer heading angle in radians.
        target_pos: Position of target.
        observer_pos: Position of observer.

    Returns:
        Heading-off angle magnitude in radians in [0.0, pi].
    """
    dx = float(target_pos[0]) - float(observer_pos[0])
    dy = float(target_pos[1]) - float(observer_pos[1])

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0

    los_angle = math.atan2(dy, dx)
    diff = (los_angle - observer_heading + PI) % TWO_PI - PI
    return float(abs(diff))
