"""Concrete base entity class extending BaseEntity for simulation.

Provides shared attributes, state tracking, and lifecycle routines across all domains.
"""

from abc import abstractmethod
import math
import typing
import numpy as np

from src.core.interfaces import (
    AgentStatus,
    BaseEntity,
    DomainType,
    TeamSide,
    WeaponType,
)
from src.simulator.config import TWO_PI


class ObservationArray(np.ndarray):
    """NumPy array wrapper providing .to_array() compatibility for DRDO verification."""

    def to_array(self) -> np.ndarray:
        """Return raw numpy array representation."""
        return np.asarray(self)


class SimEntity(BaseEntity):
    """Concrete base simulation entity across air, ground, and sea domains.

    Attributes:
        entity_id: Unique string identifier (e.g. "blue_air_1").
        domain: Operational domain (AIR, GROUND, SEA).
        team: Team allegiance (BLUE, RED).
        aircraft_type: Type designation string (e.g. "AC1", "AC2", "GROUND", "SEA").
        position: Current position coordinates [x, y, z] in kilometers.
        heading: Heading angle in radians [0, 2pi).
        speed: Current velocity magnitude in km/h.
        status: Operational status (ALIVE, DESTROYED, OUT_OF_BOUNDS).
        ammo: Dictionary mapping WeaponType to integer remaining counts.
        alive: Boolean indicating operational status.
    """

    def __init__(
        self,
        entity_id: str,
        domain: DomainType,
        team: TeamSide,
        aircraft_type: str,
        position: typing.Sequence[float] | np.ndarray,
        heading: float,
        speed: float,
        config: dict[str, typing.Any] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize simulation entity.

        Args:
            entity_id: Unique identifier.
            domain: DomainType (AIR, GROUND, SEA).
            team: TeamSide (BLUE, RED).
            aircraft_type: Type descriptor.
            position: Initial (x, y, z) in km.
            heading: Initial heading in radians.
            speed: Initial speed in km/h.
            config: Optional configuration dictionary.
            rng: Seeded numpy Generator for stochastic actions.
        """
        self._entity_id: str = str(entity_id)
        self._domain: DomainType = domain
        self._team: TeamSide = team
        self.aircraft_type: str = str(aircraft_type)

        self._position: np.ndarray = np.asarray(position, dtype=np.float64).copy()
        self._heading: float = float(heading % TWO_PI)
        if self._heading < 0.0:
            self._heading += TWO_PI

        self._speed: float = float(speed)
        self._status: AgentStatus = AgentStatus.ALIVE
        self.config: dict[str, typing.Any] = config or {}
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng()

        self.ammo: dict[WeaponType, int] = {}
        self.last_fire_times: dict[WeaponType, float] = {}

    @property
    def entity_id(self) -> str:
        """Unique entity identifier."""
        return self._entity_id

    @property
    def domain(self) -> DomainType:
        """Domain type."""
        return self._domain

    @property
    def team(self) -> TeamSide:
        """Team side."""
        return self._team

    @property
    def position(self) -> np.ndarray:
        """Position coordinates [x, y, z] in km."""
        return self._position

    @position.setter
    def position(self, new_pos: typing.Sequence[float] | np.ndarray) -> None:
        """Set position coordinates."""
        self._position = np.asarray(new_pos, dtype=np.float64)

    @property
    def velocity(self) -> float:
        """Velocity magnitude (speed) in km/h."""
        return self._speed

    @property
    def speed(self) -> float:
        """Speed in km/h."""
        return self._speed

    @speed.setter
    def speed(self, new_speed: float) -> None:
        """Set speed in km/h."""
        self._speed = float(new_speed)

    @property
    def heading(self) -> float:
        """Heading in radians in [0, 2pi)."""
        return self._heading

    @heading.setter
    def heading(self, new_heading: float) -> None:
        """Set heading wrapped to [0, 2pi)."""
        h = float(new_heading) % TWO_PI
        if h < 0.0:
            h += TWO_PI
        self._heading = h

    @property
    def status(self) -> AgentStatus:
        """Agent operational lifecycle status."""
        return self._status

    @status.setter
    def status(self, new_status: AgentStatus) -> None:
        """Set operational status."""
        self._status = new_status

    @property
    def alive(self) -> bool:
        """Return True if status is ALIVE."""
        return self._status == AgentStatus.ALIVE

    def is_alive(self) -> bool:
        """Check if entity is operational and alive."""
        return self._status == AgentStatus.ALIVE

    def step(self, dt: float, action: typing.Any = None) -> None:
        """Advance entity dynamics by dt seconds with given action command.

        If entity is not alive, step is a no-op.
        """
        if not self.is_alive():
            return
        self._integrate_dynamics(dt, action)

    @abstractmethod
    def _integrate_dynamics(self, dt: float, action: typing.Any = None) -> None:
        """Subclass-specific kinematic integration."""
        pass

    @abstractmethod
    def get_observation(
        self,
        opponents: list["SimEntity"] | None = None,
        friendlies: list["SimEntity"] | None = None,
        map_ref: typing.Any = None,
    ) -> np.ndarray:
        """Compute normalized observation vector."""
        pass

    def apply_weapon_hit(
        self,
        damage_type: WeaponType,
        rng: np.random.Generator | None = None,
    ) -> bool:
        """Apply damage from weapon hit.

        Marks entity as DESTROYED.

        Args:
            damage_type: WeaponType delivering the impact.
            rng: Optional seeded numpy Generator.

        Returns:
            True if entity was destroyed by the hit.
        """
        if not self.is_alive():
            return False
        self._status = AgentStatus.DESTROYED
        return True

    def reset(
        self,
        position: typing.Sequence[float] | np.ndarray,
        heading: float,
        speed: float,
    ) -> None:
        """Reset entity to initial spatial state and replenish ammunition.

        Args:
            position: New (x, y, z) position in km.
            heading: New heading in radians.
            speed: New speed in km/h.
        """
        self._position = np.asarray(position, dtype=np.float64).copy()
        self.heading = heading
        self.speed = speed
        self._status = AgentStatus.ALIVE
        self.last_fire_times.clear()

    def to_state_dict(self) -> dict[str, typing.Any]:
        """Serialize entity state for visualizer, UI, or logging."""
        return {
            "entity_id": self.entity_id,
            "domain": self.domain.value,
            "team": self.team.value,
            "aircraft_type": self.aircraft_type,
            "position": self.position.tolist(),
            "heading": self.heading,
            "heading_deg": math.degrees(self.heading),
            "speed_kmh": self.speed,
            "status": self.status.value,
            "alive": self.is_alive(),
            "ammo": {w.value: count for w, count in self.ammo.items()},
        }
