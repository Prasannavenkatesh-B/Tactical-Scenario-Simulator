"""Sea domain combat entity modeling surface combatants and naval vessels."""

import math
import typing
import numpy as np

from src.core.actions import SeaAction
from src.core.interfaces import DomainType, TeamSide, WeaponType
from src.core.observations import SeaObservation
from src.core.rewards import compute_ATA, compute_distance
from src.simulator.config import (
    DEFAULT_MAP_SIZE_KM,
    KNOTS_TO_KMH,
    SEA_AMMO,
    SEA_MAX_TURN_RATE_DEG,
    SEA_RADAR_RANGE_KM,
    SEA_SPEED_RANGE_KNOTS,
    TWO_PI,
)
from src.simulator.entities.base import ObservationArray, SimEntity


class SeaEntity(SimEntity):
    """Sea domain surface combatant subject to marine hydrodynamics and sea states."""

    def __init__(
        self,
        entity_id: str,
        team: TeamSide,
        aircraft_type: str = "SEA",
        position: typing.Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
        heading: float = 0.0,
        speed: float = 15.0 * KNOTS_TO_KMH,
        sea_state: float = 2.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize SeaEntity.

        Args:
            entity_id: Unique entity ID.
            team: TeamSide (BLUE, RED).
            aircraft_type: "SEA" or "FRIGATE".
            position: Initial (x, y, z=0) in km.
            heading: Initial heading in radians.
            speed: Initial speed in km/h.
            sea_state: Beaufort scale value [0, 9].
            rng: Seeded numpy Generator.
        """
        super().__init__(
            entity_id=entity_id,
            domain=DomainType.SEA,
            team=team,
            aircraft_type=aircraft_type,
            position=position,
            heading=heading,
            speed=speed,
            rng=rng,
        )

        self.min_speed_kmh: float = SEA_SPEED_RANGE_KNOTS[0] * KNOTS_TO_KMH
        self.max_speed_kmh: float = SEA_SPEED_RANGE_KNOTS[1] * KNOTS_TO_KMH
        self.max_turn_rate_deg: float = SEA_MAX_TURN_RATE_DEG
        self.radar_range_km: float = SEA_RADAR_RANGE_KM
        self.sea_state: float = float(np.clip(sea_state, 0.0, 9.0))

        # Clamp speed
        self._speed = float(np.clip(self._speed, self.min_speed_kmh, self.max_speed_kmh))

        self.ammo[WeaponType.MISSILE] = SEA_AMMO
        self.ammo[WeaponType.CANNON] = SEA_AMMO
        self.max_ammo: int = SEA_AMMO

    def reset(
        self,
        position: typing.Sequence[float] | np.ndarray,
        heading: float,
        speed: float,
    ) -> None:
        """Reset sea entity coordinates and ammunition."""
        super().reset(position, heading, speed)
        self._speed = float(np.clip(self._speed, self.min_speed_kmh, self.max_speed_kmh))
        self.ammo[WeaponType.MISSILE] = self.max_ammo
        self.ammo[WeaponType.CANNON] = self.max_ammo

    def _integrate_dynamics(self, dt: float, action: typing.Any = None) -> None:
        """Perform kinematic integration using dynamics module."""
        from src.simulator.dynamics import integrate_sea

        if action is None:
            action = SeaAction(heading_delta=0.0, velocity_cmd=3, weapon_select=0, fire=0)
        elif not isinstance(action, SeaAction):
            action = SeaAction.from_array(action)

        integrate_sea(self, action, dt, sea_state=self.sea_state)

    def get_observation(
        self,
        opponents: list[SimEntity] | None = None,
        friendlies: list[SimEntity] | None = None,
        map_ref: typing.Any = None,
    ) -> np.ndarray:
        """Compute normalized SeaObservation array (dimension = 9).

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
        norm_sea = self.sea_state / 9.0

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

        total_ammo = sum(self.ammo.values())
        norm_c1 = total_ammo / max(self.max_ammo * 2, 1)
        norm_radar = self.radar_range_km / max(map_size, 1e-6)

        obs = SeaObservation(
            x=norm_x,
            y=norm_y,
            v=norm_v,
            theta=norm_theta,
            sea_state=norm_sea,
            alpha_ATA=norm_ata,
            d_o=norm_do,
            c1=norm_c1,
            radar_range=norm_radar,
        )
        return obs.to_array().view(ObservationArray)
