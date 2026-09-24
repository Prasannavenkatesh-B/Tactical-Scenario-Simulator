"""Core interfaces and abstract base classes for the multi-domain MARL system.

Submitted to DRDO for tactical scenario simulation across air, ground, and sea domains.
"""

from abc import ABC, abstractmethod
from enum import Enum
import typing
import numpy as np


class DomainType(Enum):
    """Operational domains for tactical scenario simulation."""
    AIR = "AIR"
    GROUND = "GROUND"
    SEA = "SEA"


class TeamSide(Enum):
    """Team allegiance for agents in the scenario."""
    BLUE = "BLUE"   # Friendly forces
    RED = "RED"     # Opponent forces


class AgentStatus(Enum):
    """Operational lifecycle status of an agent."""
    ALIVE = "ALIVE"
    DESTROYED = "DESTROYED"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"


class PolicyType(Enum):
    """Hierarchical policy operational modes."""
    FIGHT = "FIGHT"
    ESCAPE = "ESCAPE"
    ENGAGE = "ENGAGE"
    DEFEND = "DEFEND"
    COMMANDER = "COMMANDER"


class WeaponType(Enum):
    """Weapons available across domains."""
    CANNON = "CANNON"
    ROCKET = "ROCKET"
    MISSILE = "MISSILE"
    SAM = "SAM"
    ARTILLERY = "ARTILLERY"


class BaseEntity(ABC):
    """Abstract base class representing an operational tactical entity."""

    @property
    @abstractmethod
    def entity_id(self) -> str:
        """Unique identifier of the entity."""
        pass

    @property
    @abstractmethod
    def domain(self) -> DomainType:
        """Domain type of the entity (AIR, GROUND, SEA)."""
        pass

    @property
    @abstractmethod
    def team(self) -> TeamSide:
        """Team allegiance of the entity (BLUE, RED)."""
        pass

    @property
    @abstractmethod
    def position(self) -> tuple[float, float, float] | np.ndarray:
        """Position coordinates (x, y, z) in meters."""
        pass

    @property
    @abstractmethod
    def velocity(self) -> float | tuple[float, float, float] | np.ndarray:
        """Velocity magnitude (speed in m/s) or velocity vector."""
        pass

    @property
    @abstractmethod
    def heading(self) -> float:
        """Heading angle in radians in [0, 2pi]."""
        pass

    @property
    @abstractmethod
    def status(self) -> AgentStatus:
        """Operational status of the entity."""
        pass

    @abstractmethod
    def step(self, dt: float) -> None:
        """Advance entity dynamics by dt seconds."""
        pass

    @abstractmethod
    def is_alive(self) -> bool:
        """Return True if the entity is operational and alive."""
        pass

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """Return the normalized observation vector for this entity."""
        pass


class BasePolicy(ABC):
    """Abstract base class for MARL policies (domain and commander levels)."""

    @abstractmethod
    def act(self, observation: np.ndarray | typing.Any) -> typing.Any:
        """Select an action given an observation vector or dataclass."""
        pass

    @abstractmethod
    def update(self, batch: dict[str, typing.Any] | typing.Any) -> dict[str, float]:
        """Update policy parameters given a transition batch and return loss dictionary."""
        pass


class BaseReward(ABC):
    """Abstract base class for state-action-transition reward computation."""

    @abstractmethod
    def compute(
        self,
        state: typing.Any,
        action: typing.Any,
        next_state: typing.Any,
    ) -> float:
        """Compute transition scalar reward."""
        pass


class BaseEnvironment(ABC):
    """Abstract base class for the multi-domain tactical simulation environment."""

    @abstractmethod
    def reset(
        self,
        scenario_config: dict[str, typing.Any] | None = None,
    ) -> dict[str, typing.Any]:
        """Reset scenario simulation and return initial observations dictionary."""
        pass

    @abstractmethod
    def step(
        self,
        action_dict: dict[str, typing.Any],
    ) -> tuple[
        dict[str, typing.Any],
        dict[str, float],
        dict[str, bool],
        dict[str, typing.Any],
    ]:
        """Step the simulation environment forward with the provided actions.

        Returns:
            Tuple of (obs_dict, reward_dict, done_dict, info_dict).
        """
        pass

    @abstractmethod
    def render(self, mode: str = "human") -> np.ndarray | None:
        """Render the environment state; optionally returns an RGB frame array."""
        pass
