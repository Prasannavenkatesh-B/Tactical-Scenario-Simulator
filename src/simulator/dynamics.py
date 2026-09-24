"""Kinematic dynamic integration for air, ground, and sea domain entities.

Implements physics-based motion integration, turn-rate clamping, acceleration limits,
terrain slope traversal checks, and hydro-meteorological sea state degradation.
"""

import math
import typing
import numpy as np

from src.core.actions import AirAction, GroundAction, SeaAction
from src.simulator.config import (
    AC1_ACCELERATION_KNOTS_S,
    AC2_ACCELERATION_KNOTS_S,
    DEG_TO_RAD,
    GROUND_ACCELERATION_KMH_S,
    GROUND_MAX_SLOPE,
    KMH_TO_KMS,
    KNOTS_TO_KMH,
    RAD_TO_DEG,
    SEA_ACCELERATION_KNOTS_S,
    TWO_PI,
)

if typing.TYPE_CHECKING:
    from src.simulator.entities.air import AirEntity
    from src.simulator.entities.ground import GroundEntity
    from src.simulator.entities.sea import SeaEntity
    from src.simulator.map import Map2D


def integrate_air(
    entity: "AirEntity",
    action: AirAction,
    dt: float,
    map_ref: typing.Optional["Map2D"] = None,
) -> None:
    """Integrate kinematic state forward for an AirEntity.

    Steps:
        1. Turn rate clamped to entity's maximum angular velocity (* dt).
        2. Heading updated and normalized to [0, 2pi).
        3. Target velocity mapped from discrete command {0..8} and bounded by acceleration.
        4. Spatial position updated via speed, heading, and climb rate.

    Args:
        entity: AirEntity instance being updated.
        action: AirAction command.
        dt: Integration time step in seconds.
        map_ref: Optional Map2D reference.
    """
    # 1. Heading update
    max_turn_rad = entity.max_turn_rate_deg * DEG_TO_RAD * dt
    desired_delta_rad = action.heading_delta * DEG_TO_RAD
    clamped_delta_rad = float(np.clip(desired_delta_rad, -max_turn_rad, max_turn_rad))
    entity.heading = entity.heading + clamped_delta_rad

    # 2. Velocity command and linear acceleration
    is_ac1 = entity.aircraft_type.upper() == "AC1"
    accel_knots_s = AC1_ACCELERATION_KNOTS_S if is_ac1 else AC2_ACCELERATION_KNOTS_S
    accel_kmh_s = accel_knots_s * KNOTS_TO_KMH
    max_dv = accel_kmh_s * dt

    cmd_frac = float(np.clip(action.velocity_cmd / 8.0, 0.0, 1.0))
    target_speed = entity.min_speed_kmh + cmd_frac * (entity.max_speed_kmh - entity.min_speed_kmh)

    speed_err = target_speed - entity.speed
    dv = float(np.clip(speed_err, -max_dv, max_dv))
    entity.speed = float(np.clip(entity.speed + dv, entity.min_speed_kmh, entity.max_speed_kmh))

    # 3. Position update
    v_kms = entity.speed * KMH_TO_KMS
    dx = v_kms * dt * math.cos(entity.heading)
    dy = v_kms * dt * math.sin(entity.heading)
    dz = 0.0  # Level flight simplification unless climb commanded

    new_x = float(entity.position[0] + dx)
    new_y = float(entity.position[1] + dy)
    new_z = float(entity.position[2] + dz) if len(entity.position) > 2 else 5.0
    entity.position = np.array([new_x, new_y, new_z], dtype=np.float64)


def integrate_ground(
    entity: "GroundEntity",
    action: GroundAction,
    dt: float,
    map_ref: typing.Optional["Map2D"] = None,
) -> None:
    """Integrate kinematic state forward for a GroundEntity.

    Steps:
        1. Turn rate clamped to ground angular velocity (* dt).
        2. Velocity mapped from {0..5} and acceleration-limited.
        3. Proposed coordinate evaluated against terrain slope: movement blocked
           if slope exceeds GROUND_MAX_SLOPE.
        4. Altitude matched to surface terrain elevation.

    Args:
        entity: GroundEntity instance being updated.
        action: GroundAction command.
        dt: Integration time step in seconds.
        map_ref: Optional Map2D reference for elevation & slope queries.
    """
    # 1. Heading update
    max_turn_rad = entity.max_turn_rate_deg * DEG_TO_RAD * dt
    desired_delta_rad = action.heading_delta * DEG_TO_RAD
    clamped_delta_rad = float(np.clip(desired_delta_rad, -max_turn_rad, max_turn_rad))
    entity.heading = entity.heading + clamped_delta_rad

    # 2. Velocity command and linear acceleration
    max_dv = GROUND_ACCELERATION_KMH_S * dt
    cmd_frac = float(np.clip(action.velocity_cmd / 5.0, 0.0, 1.0))
    target_speed = cmd_frac * entity.max_speed_kmh

    speed_err = target_speed - entity.speed
    dv = float(np.clip(speed_err, -max_dv, max_dv))
    entity.speed = float(np.clip(entity.speed + dv, entity.min_speed_kmh, entity.max_speed_kmh))

    # 3. Terrain slope evaluation
    v_kms = entity.speed * KMH_TO_KMS
    dx = v_kms * dt * math.cos(entity.heading)
    dy = v_kms * dt * math.sin(entity.heading)
    new_x = float(entity.position[0] + dx)
    new_y = float(entity.position[1] + dy)

    if map_ref is not None:
        cur_elev = map_ref.elevation_at(entity.position[0], entity.position[1])
        new_elev = map_ref.elevation_at(new_x, new_y)
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 1e-6:
            slope = abs(new_elev - cur_elev) / dist
            if slope > GROUND_MAX_SLOPE:
                # Steep grade encountered: halt forward translation
                entity.speed = 0.0
                return
        new_z = new_elev
    else:
        new_z = float(entity.position[2]) if len(entity.position) > 2 else 0.0

    entity.position = np.array([new_x, new_y, new_z], dtype=np.float64)


def integrate_sea(
    entity: "SeaEntity",
    action: SeaAction,
    dt: float,
    sea_state: float = 0.0,
    map_ref: typing.Optional["Map2D"] = None,
) -> None:
    """Integrate kinematic state forward for a SeaEntity.

    Steps:
        1. Turn rate and max speed degraded by Beaufort sea state: (1 - sea_state / 12).
        2. Heading updated with clamped angular rate.
        3. Velocity commanded and acceleration-limited.
        4. Surface coordinate updated at z = 0.

    Args:
        entity: SeaEntity instance being updated.
        action: SeaAction command.
        dt: Integration time step in seconds.
        sea_state: Current sea state (0-9 Beaufort scale).
        map_ref: Optional Map2D reference.
    """
    # Hydrodynamic degradation based on sea state
    sea_factor = float(np.clip(1.0 - (sea_state / 12.0), 0.2, 1.0))
    effective_max_turn = entity.max_turn_rate_deg * sea_factor
    effective_max_speed = entity.max_speed_kmh * sea_factor

    # 1. Heading update
    max_turn_rad = effective_max_turn * DEG_TO_RAD * dt
    desired_delta_rad = action.heading_delta * DEG_TO_RAD
    clamped_delta_rad = float(np.clip(desired_delta_rad, -max_turn_rad, max_turn_rad))
    entity.heading = entity.heading + clamped_delta_rad

    # 2. Velocity command
    accel_kmh_s = SEA_ACCELERATION_KNOTS_S * KNOTS_TO_KMH
    max_dv = accel_kmh_s * dt

    cmd_frac = float(np.clip(action.velocity_cmd / 5.0, 0.0, 1.0))
    target_speed = cmd_frac * effective_max_speed

    speed_err = target_speed - entity.speed
    dv = float(np.clip(speed_err, -max_dv, max_dv))
    entity.speed = float(np.clip(entity.speed + dv, entity.min_speed_kmh, effective_max_speed))

    # 3. Position update on sea surface
    v_kms = entity.speed * KMH_TO_KMS
    dx = v_kms * dt * math.cos(entity.heading)
    dy = v_kms * dt * math.sin(entity.heading)

    new_x = float(entity.position[0] + dx)
    new_y = float(entity.position[1] + dy)
    entity.position = np.array([new_x, new_y, 0.0], dtype=np.float64)
