"""Action dataclasses for multi-domain tactical agents and commander policy.

Implements hybrid action spaces (continuous heading delta + discrete commands)
with bidirectional numpy array conversions (to_array and from_array).
"""

from dataclasses import dataclass
import typing
import numpy as np


# Module-level configuration constants for action bounds
AIR_HEADING_DELTA_MIN: float = -90.0
AIR_HEADING_DELTA_MAX: float = 90.0
AIR_HEADING_STEP_DEG: float = 15.0
AIR_HEADING_DISCRETE_MIN: int = -6
AIR_HEADING_DISCRETE_MAX: int = 6
AIR_VELOCITY_CMD_MIN: int = 0
AIR_VELOCITY_CMD_MAX: int = 8

GROUND_HEADING_DELTA_MIN: float = -45.0
GROUND_HEADING_DELTA_MAX: float = 45.0
GROUND_VELOCITY_CMD_MIN: int = 0
GROUND_VELOCITY_CMD_MAX: int = 5
GROUND_WEAPON_SELECT_MIN: int = 0
GROUND_WEAPON_SELECT_MAX: int = 2  # 0=cannon, 1=missile, 2=SAM

SEA_HEADING_DELTA_MIN: float = -30.0
SEA_HEADING_DELTA_MAX: float = 30.0
SEA_VELOCITY_CMD_MIN: int = 0
SEA_VELOCITY_CMD_MAX: int = 5
SEA_WEAPON_SELECT_MIN: int = 0
SEA_WEAPON_SELECT_MAX: int = 1  # 0=cannon, 1=missile

COMMANDER_ACTION_MIN: int = 0
COMMANDER_ACTION_MAX: int = 3  # 0=ESCAPE, 1=FIGHT target 1, 2=FIGHT target 2, 3=FIGHT target 3

BINARY_ACTION_MIN: int = 0
BINARY_ACTION_MAX: int = 1

ACTION_CONFIG: dict[str, typing.Any] = {
    "air_heading_delta_min": AIR_HEADING_DELTA_MIN,
    "air_heading_delta_max": AIR_HEADING_DELTA_MAX,
    "air_heading_step_deg": AIR_HEADING_STEP_DEG,
    "air_heading_discrete_min": AIR_HEADING_DISCRETE_MIN,
    "air_heading_discrete_max": AIR_HEADING_DISCRETE_MAX,
    "air_velocity_cmd_min": AIR_VELOCITY_CMD_MIN,
    "air_velocity_cmd_max": AIR_VELOCITY_CMD_MAX,
    "ground_heading_delta_min": GROUND_HEADING_DELTA_MIN,
    "ground_heading_delta_max": GROUND_HEADING_DELTA_MAX,
    "ground_velocity_cmd_min": GROUND_VELOCITY_CMD_MIN,
    "ground_velocity_cmd_max": GROUND_VELOCITY_CMD_MAX,
    "ground_weapon_select_min": GROUND_WEAPON_SELECT_MIN,
    "ground_weapon_select_max": GROUND_WEAPON_SELECT_MAX,
    "sea_heading_delta_min": SEA_HEADING_DELTA_MIN,
    "sea_heading_delta_max": SEA_HEADING_DELTA_MAX,
    "sea_velocity_cmd_min": SEA_VELOCITY_CMD_MIN,
    "sea_velocity_cmd_max": SEA_VELOCITY_CMD_MAX,
    "sea_weapon_select_min": SEA_WEAPON_SELECT_MIN,
    "sea_weapon_select_max": SEA_WEAPON_SELECT_MAX,
    "commander_action_min": COMMANDER_ACTION_MIN,
    "commander_action_max": COMMANDER_ACTION_MAX,
}


@dataclass
class AirAction:
    """Hybrid action command for Air domain agents.

    Attributes:
        heading_delta: Continuous heading change in degrees [-90.0, 90.0],
                       mapped to {-6..+6} discrete steps of 15 degrees.
        velocity_cmd: Discrete velocity command index {0..8}.
        fire_cannon: Discrete binary flag {0, 1} for cannon weapon.
        fire_rocket: Discrete binary flag {0, 1} for rocket weapon.
    """
    heading_delta: float
    velocity_cmd: int
    fire_cannon: int
    fire_rocket: int

    DIM: typing.ClassVar[int] = 4

    def __post_init__(self) -> None:
        """Validate and clamp action values within valid operational ranges."""
        self.heading_delta = float(np.clip(
            float(self.heading_delta),
            AIR_HEADING_DELTA_MIN,
            AIR_HEADING_DELTA_MAX,
        ))
        self.velocity_cmd = int(np.clip(
            int(round(float(self.velocity_cmd))),
            AIR_VELOCITY_CMD_MIN,
            AIR_VELOCITY_CMD_MAX,
        ))
        self.fire_cannon = int(np.clip(
            int(round(float(self.fire_cannon))),
            BINARY_ACTION_MIN,
            BINARY_ACTION_MAX,
        ))
        self.fire_rocket = int(np.clip(
            int(round(float(self.fire_rocket))),
            BINARY_ACTION_MIN,
            BINARY_ACTION_MAX,
        ))

    @property
    def discrete_heading_step(self) -> int:
        """Map continuous heading delta to nearest discrete step in {-6..+6}."""
        raw_step = round(self.heading_delta / AIR_HEADING_STEP_DEG)
        return int(np.clip(raw_step, AIR_HEADING_DISCRETE_MIN, AIR_HEADING_DISCRETE_MAX))

    def to_array(self) -> np.ndarray:
        """Convert action dataclass to 1D float64 numpy array."""
        return np.array([
            self.heading_delta,
            float(self.velocity_cmd),
            float(self.fire_cannon),
            float(self.fire_rocket),
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray | typing.Sequence[float]) -> "AirAction":
        """Reconstruct AirAction from an array representation."""
        if len(arr) != cls.DIM:
            raise ValueError(f"Expected array of length {cls.DIM}, got {len(arr)}")
        return cls(
            heading_delta=float(arr[0]),
            velocity_cmd=int(round(float(arr[1]))),
            fire_cannon=int(round(float(arr[2]))),
            fire_rocket=int(round(float(arr[3]))),
        )

    @classmethod
    def from_discrete(
        cls,
        heading_step: int,
        velocity_cmd: int,
        fire_cannon: int = 0,
        fire_rocket: int = 0,
    ) -> "AirAction":
        """Instantiate AirAction from discrete heading step index {-6..+6}."""
        clamped_step = int(np.clip(heading_step, AIR_HEADING_DISCRETE_MIN, AIR_HEADING_DISCRETE_MAX))
        heading_delta = float(clamped_step * AIR_HEADING_STEP_DEG)
        return cls(
            heading_delta=heading_delta,
            velocity_cmd=velocity_cmd,
            fire_cannon=fire_cannon,
            fire_rocket=fire_rocket,
        )


@dataclass
class GroundAction:
    """Hybrid action command for Ground domain agents.

    Attributes:
        heading_delta: Continuous heading change in degrees [-45.0, 45.0].
        velocity_cmd: Discrete velocity command index {0..5}.
        weapon_select: Discrete weapon selection index {0=cannon, 1=missile, 2=SAM}.
        fire: Discrete binary flag {0, 1} to engage weapon.
    """
    heading_delta: float
    velocity_cmd: int
    weapon_select: int
    fire: int

    DIM: typing.ClassVar[int] = 4

    def __post_init__(self) -> None:
        """Validate and clamp action values within valid operational ranges."""
        self.heading_delta = float(np.clip(
            float(self.heading_delta),
            GROUND_HEADING_DELTA_MIN,
            GROUND_HEADING_DELTA_MAX,
        ))
        self.velocity_cmd = int(np.clip(
            int(round(float(self.velocity_cmd))),
            GROUND_VELOCITY_CMD_MIN,
            GROUND_VELOCITY_CMD_MAX,
        ))
        self.weapon_select = int(np.clip(
            int(round(float(self.weapon_select))),
            GROUND_WEAPON_SELECT_MIN,
            GROUND_WEAPON_SELECT_MAX,
        ))
        self.fire = int(np.clip(
            int(round(float(self.fire))),
            BINARY_ACTION_MIN,
            BINARY_ACTION_MAX,
        ))

    def to_array(self) -> np.ndarray:
        """Convert action dataclass to 1D float64 numpy array."""
        return np.array([
            self.heading_delta,
            float(self.velocity_cmd),
            float(self.weapon_select),
            float(self.fire),
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray | typing.Sequence[float]) -> "GroundAction":
        """Reconstruct GroundAction from an array representation."""
        if len(arr) != cls.DIM:
            raise ValueError(f"Expected array of length {cls.DIM}, got {len(arr)}")
        return cls(
            heading_delta=float(arr[0]),
            velocity_cmd=int(round(float(arr[1]))),
            weapon_select=int(round(float(arr[2]))),
            fire=int(round(float(arr[3]))),
        )


@dataclass
class SeaAction:
    """Hybrid action command for Sea domain agents.

    Attributes:
        heading_delta: Continuous heading change in degrees [-30.0, 30.0].
        velocity_cmd: Discrete velocity command index {0..5}.
        weapon_select: Discrete weapon selection index {0=cannon, 1=missile}.
        fire: Discrete binary flag {0, 1} to engage weapon.
    """
    heading_delta: float
    velocity_cmd: int
    weapon_select: int
    fire: int

    DIM: typing.ClassVar[int] = 4

    def __post_init__(self) -> None:
        """Validate and clamp action values within valid operational ranges."""
        self.heading_delta = float(np.clip(
            float(self.heading_delta),
            SEA_HEADING_DELTA_MIN,
            SEA_HEADING_DELTA_MAX,
        ))
        self.velocity_cmd = int(np.clip(
            int(round(float(self.velocity_cmd))),
            SEA_VELOCITY_CMD_MIN,
            SEA_VELOCITY_CMD_MAX,
        ))
        self.weapon_select = int(np.clip(
            int(round(float(self.weapon_select))),
            SEA_WEAPON_SELECT_MIN,
            SEA_WEAPON_SELECT_MAX,
        ))
        self.fire = int(np.clip(
            int(round(float(self.fire))),
            BINARY_ACTION_MIN,
            BINARY_ACTION_MAX,
        ))

    def to_array(self) -> np.ndarray:
        """Convert action dataclass to 1D float64 numpy array."""
        return np.array([
            self.heading_delta,
            float(self.velocity_cmd),
            float(self.weapon_select),
            float(self.fire),
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray | typing.Sequence[float]) -> "SeaAction":
        """Reconstruct SeaAction from an array representation."""
        if len(arr) != cls.DIM:
            raise ValueError(f"Expected array of length {cls.DIM}, got {len(arr)}")
        return cls(
            heading_delta=float(arr[0]),
            velocity_cmd=int(round(float(arr[1]))),
            weapon_select=int(round(float(arr[2]))),
            fire=int(round(float(arr[3]))),
        )


@dataclass
class CommanderAction:
    """Discrete action command for high-level Commander policy.

    Attributes:
        action: Discrete command index {0, 1, 2, 3}:
                0 = activate ESCAPE policy for this agent
                1 = activate FIGHT policy, target opponent 1
                2 = activate FIGHT policy, target opponent 2
                3 = activate FIGHT policy, target opponent 3
    """
    action: int

    DIM: typing.ClassVar[int] = 1

    def __post_init__(self) -> None:
        """Validate and clamp commander command index to {0..3}."""
        self.action = int(np.clip(
            int(round(float(self.action))),
            COMMANDER_ACTION_MIN,
            COMMANDER_ACTION_MAX,
        ))

    def to_array(self) -> np.ndarray:
        """Convert action dataclass to 1D float64 numpy array."""
        return np.array([float(self.action)], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray | typing.Sequence[float]) -> "CommanderAction":
        """Reconstruct CommanderAction from an array representation."""
        if len(arr) != cls.DIM:
            raise ValueError(f"Expected array of length {cls.DIM}, got {len(arr)}")
        return cls(action=int(round(float(arr[0]))))
