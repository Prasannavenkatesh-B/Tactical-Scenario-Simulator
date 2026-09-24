"""Air domain combat entity implementing AC1 and AC2 flight models."""

import math
import typing
import numpy as np

from src.core.actions import AirAction
from src.core.interfaces import DomainType, TeamSide, WeaponType
from src.core.observations import AirObservation
from src.core.rewards import compute_AA, compute_ATA, compute_distance, compute_heading_off
from src.simulator.config import (
    AC1_AMMO_CANNON,
    AC1_AMMO_ROCKET,
    AC1_CLIMB_RATE_KMS,
    AC1_MAX_TURN_RATE_DEG,
    AC1_SPEED_RANGE_KNOTS,
    AC2_AMMO_CANNON,
    AC2_AMMO_ROCKET,
    AC2_CLIMB_RATE_KMS,
    AC2_MAX_TURN_RATE_DEG,
    AC2_SPEED_RANGE_KNOTS,
    DEFAULT_MAP_SIZE_KM,
    KNOTS_TO_KMH,
    MAX_ALTITUDE_KM,
    TWO_PI,
)
from src.simulator.entities.base import ObservationArray, SimEntity


class AirEntity(SimEntity):
    """Air domain entity modeling AC1 (agile dogfighter) and AC2 (interceptor)."""

    def __init__(
        self,
        entity_id: str,
        team: TeamSide,
        aircraft_type: str = "AC1",
        position: typing.Sequence[float] | np.ndarray = (0.0, 0.0, 5.0),
        heading: float = 0.0,
        speed: float = 300.0 * KNOTS_TO_KMH,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize AirEntity.

        Args:
            entity_id: Unique entity ID.
            team: TeamSide (BLUE, RED).
            aircraft_type: "AC1" or "AC2".
            position: Initial (x, y, z) in km.
            heading: Heading angle in radians.
            speed: Initial speed in km/h.
            rng: Seeded numpy Generator.
        """
        super().__init__(
            entity_id=entity_id,
            domain=DomainType.AIR,
            team=team,
            aircraft_type=aircraft_type,
            position=position,
            heading=heading,
            speed=speed,
            rng=rng,
        )

        is_ac1 = self.aircraft_type.upper() == "AC1"
        self.max_turn_rate_deg: float = AC1_MAX_TURN_RATE_DEG if is_ac1 else AC2_MAX_TURN_RATE_DEG
        speed_range_knots = AC1_SPEED_RANGE_KNOTS if is_ac1 else AC2_SPEED_RANGE_KNOTS

        self.min_speed_kmh: float = speed_range_knots[0] * KNOTS_TO_KMH
        self.max_speed_kmh: float = speed_range_knots[1] * KNOTS_TO_KMH
        self.climb_rate_kms: float = AC1_CLIMB_RATE_KMS if is_ac1 else AC2_CLIMB_RATE_KMS

        # Clamp initial speed to operational envelope
        self._speed = float(np.clip(self._speed, self.min_speed_kmh, self.max_speed_kmh))

        # Initial ammunition
        self.ammo[WeaponType.CANNON] = AC1_AMMO_CANNON if is_ac1 else AC2_AMMO_CANNON
        self.ammo[WeaponType.ROCKET] = AC1_AMMO_ROCKET if is_ac1 else AC2_AMMO_ROCKET
        self.max_ammo_cannon: int = self.ammo[WeaponType.CANNON]
        self.max_ammo_rocket: int = self.ammo[WeaponType.ROCKET]

        self.is_shooting: bool = False

    def reset(
        self,
        position: typing.Sequence[float] | np.ndarray,
        heading: float,
        speed: float,
    ) -> None:
        """Reset air entity spatial coordinates and replenish ammo."""
        super().reset(position, heading, speed)
        self._speed = float(np.clip(self._speed, self.min_speed_kmh, self.max_speed_kmh))
        self.ammo[WeaponType.CANNON] = self.max_ammo_cannon
        self.ammo[WeaponType.ROCKET] = self.max_ammo_rocket
        self.is_shooting = False

    def _integrate_dynamics(self, dt: float, action: typing.Any = None) -> None:
        """Perform kinematic integration using dynamics module."""
        from src.simulator.dynamics import integrate_air

        if action is None:
            # Default straight level flight
            action = AirAction(heading_delta=0.0, velocity_cmd=4, fire_cannon=0, fire_rocket=0)
        elif not isinstance(action, AirAction):
            action = AirAction.from_array(action)

        integrate_air(self, action, dt)

    def get_observation(
        self,
        opponents: list[SimEntity] | None = None,
        friendlies: list[SimEntity] | None = None,
        map_ref: typing.Any = None,
    ) -> np.ndarray:
        """Compute normalized AirObservation array (dimension = 13).

        Args:
            opponents: Active opponent entities.
            friendlies: Active friendly entities.
            map_ref: Reference to active Map2D.

        Returns:
            1D float32 numpy array of length 13.
        """
        map_size = map_ref.size_km if map_ref is not None else DEFAULT_MAP_SIZE_KM
        max_alt = map_ref.z_bounds[1] if map_ref is not None else MAX_ALTITUDE_KM

        # Normalized coordinates [0, 1]
        norm_x = self.position[0] / max(map_size, 1e-6)
        norm_y = self.position[1] / max(map_size, 1e-6)
        norm_z = self.position[2] / max(max_alt, 1e-6)
        norm_v = (self.speed - self.min_speed_kmh) / max(self.max_speed_kmh - self.min_speed_kmh, 1e-6)
        norm_h = self.heading / TWO_PI

        # Opponent relative geometry
        closest_opp: SimEntity | None = None
        min_dist = float("inf")

        if opponents:
            for opp in opponents:
                if opp.is_alive():
                    d = compute_distance(self.position[:2], opp.position[:2])
                    if d < min_dist:
                        min_dist = d
                        closest_opp = opp

        if closest_opp is not None:
            norm_do = min_dist / max(map_size * math.sqrt(2), 1e-6)
            ata = compute_ATA(self.position[:2], self.heading, closest_opp.position[:2])
            aa = compute_AA(self.position[:2], self.heading, closest_opp.position[:2], closest_opp.heading)
            head_off = compute_heading_off(self.heading, closest_opp.position[:2], self.position[:2])

            norm_ata = ata / math.pi
            norm_aa = aa / math.pi
            norm_off = head_off / math.pi
        else:
            norm_do = 1.0
            norm_ata = 0.5
            norm_aa = 0.5
            norm_off = 0.5

        # Ammunition status
        norm_c1 = (self.ammo.get(WeaponType.CANNON, 0) / max(self.max_ammo_cannon, 1))
        norm_c2 = (self.ammo.get(WeaponType.ROCKET, 0) / max(self.max_ammo_rocket, 1)
                   if self.max_ammo_rocket > 0 else 0.0)

        rocket_ready = 1.0 if self.ammo.get(WeaponType.ROCKET, 0) > 0 else 0.0
        shooting_flag = 1.0 if self.is_shooting else 0.0

        obs = AirObservation(
            x=norm_x,
            y=norm_y,
            z=norm_z,
            v=norm_v,
            alpha_h=norm_h,
            alpha_off=norm_off,
            alpha_AA=norm_aa,
            alpha_ATA=norm_ata,
            d_o=norm_do,
            c1=norm_c1,
            c2=norm_c2,
            w=rocket_ready,
            s_r=shooting_flag,
        )
        return obs.to_array().view(ObservationArray)
