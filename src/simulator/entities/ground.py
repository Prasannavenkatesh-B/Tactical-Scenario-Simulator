"""Ground domain combat entity modeling mobile units, SAMs, and artillery sites."""

import math
import typing
import numpy as np

from src.core.actions import GroundAction
from src.core.interfaces import DomainType, TeamSide, WeaponType
from src.core.observations import GroundObservation
from src.core.rewards import compute_ATA, compute_distance
from src.simulator.config import (
    DEFAULT_MAP_SIZE_KM,
    GROUND_AMMO,
    GROUND_MAX_TURN_RATE_DEG,
    GROUND_SENSOR_RANGE_KM,
    GROUND_SPEED_RANGE_KMPH,
    MAX_TERRAIN_ELEVATION_KM,
    TWO_PI,
)
from src.simulator.entities.base import ObservationArray, SimEntity


class GroundEntity(SimEntity):
    """Ground domain combat entity constrained by terrain elevation and slope."""

    def __init__(
        self,
        entity_id: str,
        team: TeamSide,
        aircraft_type: str = "GROUND",
        position: typing.Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
        heading: float = 0.0,
        speed: float = 20.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize GroundEntity.

        Args:
            entity_id: Unique entity ID.
            team: TeamSide (BLUE, RED).
            aircraft_type: "GROUND", "SAM", or "ARTILLERY".
            position: Initial (x, y, z) in km.
            heading: Initial heading in radians.
            speed: Initial speed in km/h (clamped to [0, 60]).
            rng: Seeded numpy Generator.
        """
        super().__init__(
            entity_id=entity_id,
            domain=DomainType.GROUND,
            team=team,
            aircraft_type=aircraft_type,
            position=position,
            heading=heading,
            speed=speed,
            rng=rng,
        )

        self.min_speed_kmh: float = GROUND_SPEED_RANGE_KMPH[0]
        self.max_speed_kmh: float = GROUND_SPEED_RANGE_KMPH[1]
        self.max_turn_rate_deg: float = GROUND_MAX_TURN_RATE_DEG
        self.sensor_range_km: float = GROUND_SENSOR_RANGE_KM

        self._speed = float(np.clip(self._speed, self.min_speed_kmh, self.max_speed_kmh))

        self.ammo[WeaponType.SAM] = GROUND_AMMO
        self.ammo[WeaponType.CANNON] = GROUND_AMMO
        self.max_ammo: int = GROUND_AMMO

    def reset(
        self,
        position: typing.Sequence[float] | np.ndarray,
        heading: float,
        speed: float,
    ) -> None:
        """Reset ground entity coordinates and ammunition."""
        super().reset(position, heading, speed)
        self._speed = float(np.clip(self._speed, self.min_speed_kmh, self.max_speed_kmh))
        self.ammo[WeaponType.SAM] = self.max_ammo
        self.ammo[WeaponType.CANNON] = self.max_ammo

    def _integrate_dynamics(self, dt: float, action: typing.Any = None) -> None:
        """Perform kinematic integration using dynamics module."""
        from src.simulator.dynamics import integrate_ground

        if action is None:
            action = GroundAction(heading_delta=0.0, velocity_cmd=2, weapon_select=0, fire=0)
        elif not isinstance(action, GroundAction):
            action = GroundAction.from_array(action)

        integrate_ground(self, action, dt)

    def get_observation(
        self,
        opponents: list[SimEntity] | None = None,
        friendlies: list[SimEntity] | None = None,
        map_ref: typing.Any = None,
    ) -> np.ndarray:
        """Compute normalized GroundObservation array (dimension = 9).

        Args:
            opponents: Active opponent entities.
            friendlies: Active friendly entities.
            map_ref: Reference to active Map2D.

        Returns:
            1D float32 numpy array of length 9.
        """
        map_size = map_ref.size_km if map_ref is not None else DEFAULT_MAP_SIZE_KM

        norm_x = self.position[0] / max(map_size, 1e-6)
        norm_y = self.position[1] / max(map_size, 1e-6)
        norm_v = self.speed / max(self.max_speed_kmh, 1e-6)
        norm_theta = self.heading / TWO_PI

        # Terrain elevation
        elev = map_ref.elevation_at(self.position[0], self.position[1]) if map_ref is not None else self.position[2]
        norm_elev = elev / max(MAX_TERRAIN_ELEVATION_KM, 1e-6)

        # Distance & ATA to closest opponent
        min_dist = float("inf")
        closest_opp: SimEntity | None = None

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
            norm_ata = ata / math.pi
        else:
            norm_do = 1.0
            norm_ata = 0.5

        # Ammo and sensor
        total_ammo = sum(self.ammo.values())
        norm_c1 = total_ammo / max(self.max_ammo * 2, 1)
        norm_sensor = self.sensor_range_km / max(map_size, 1e-6)

        obs = GroundObservation(
            x=norm_x,
            y=norm_y,
            v=norm_v,
            theta=norm_theta,
            elev=norm_elev,
            alpha_ATA=norm_ata,
            d_o=norm_do,
            c1=norm_c1,
            sensor_range=norm_sensor,
        )
        return obs.to_array().view(ObservationArray)
