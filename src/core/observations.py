"""Observation dataclasses for multi-domain MARL agents.

All observations are normalized to [0, 1] prior to storage in accordance with
the DRDO multi-domain tactical scenario simulation specifications.
"""

from dataclasses import dataclass, field
import math
import typing
import numpy as np


# Module-level configuration constants for normalization bounds
MAP_BOUNDS_X: float = 100_000.0   # 100 km
MAP_BOUNDS_Y: float = 100_000.0   # 100 km
MAP_BOUNDS_Z: float = 15_000.0    # 15 km

MAX_SPEED_AIR: float = 600.0       # m/s (~Mach 1.8)
MAX_SPEED_GROUND: float = 30.0     # m/s (~108 km/h)
MAX_SPEED_SEA: float = 25.0        # m/s (~50 knots)

MAX_DISTANCE: float = 100_000.0    # 100 km
MAX_ELEVATION: float = 5_000.0     # 5 km
MAX_BEAUFORT_SEA_STATE: float = 9.0  # Beaufort scale max
MAX_SENSOR_RANGE: float = 50_000.0 # 50 km
MAX_RADAR_RANGE: float = 100_000.0 # 100 km

MAX_CANNON_AMMO_AIR: float = 500.0
MAX_ROCKET_AMMO_AIR: float = 8.0
MAX_AMMO_GROUND: float = 100.0
MAX_AMMO_SEA: float = 200.0

TWO_PI: float = 2.0 * math.pi
PI: float = math.pi

# Named configuration mapping for runtime parameterization
NORMALIZATION_CONFIG: dict[str, float] = {
    "map_bounds_x": MAP_BOUNDS_X,
    "map_bounds_y": MAP_BOUNDS_Y,
    "map_bounds_z": MAP_BOUNDS_Z,
    "max_speed_air": MAX_SPEED_AIR,
    "max_speed_ground": MAX_SPEED_GROUND,
    "max_speed_sea": MAX_SPEED_SEA,
    "max_distance": MAX_DISTANCE,
    "max_elevation": MAX_ELEVATION,
    "max_beaufort_sea_state": MAX_BEAUFORT_SEA_STATE,
    "max_sensor_range": MAX_SENSOR_RANGE,
    "max_radar_range": MAX_RADAR_RANGE,
    "max_cannon_ammo_air": MAX_CANNON_AMMO_AIR,
    "max_rocket_ammo_air": MAX_ROCKET_AMMO_AIR,
    "max_ammo_ground": MAX_AMMO_GROUND,
    "max_ammo_sea": MAX_AMMO_SEA,
    "two_pi": TWO_PI,
}


def _clamp_0_1(value: float) -> float:
    """Clamp a floating-point value strictly to [0.0, 1.0]."""
    return float(np.clip(float(value), 0.0, 1.0))


@dataclass
class AirObservation:
    """Normalized observation vector for Air agent (dimension = 13).

    All values are normalized and clamped to [0.0, 1.0] before storage.

    Attributes:
        x: Position X normalized to map bounds [0, 1].
        y: Position Y normalized to map bounds [0, 1].
        z: Position Z (altitude) normalized to map bounds [0, 1].
        v: Speed normalized to [0, max_speed].
        alpha_h: Heading angle normalized to [0, 2pi].
        alpha_off: Heading-off angle to opponent normalized to [0, 1].
        alpha_AA: Aspect angle normalized to [0, 1].
        alpha_ATA: Antenna train angle normalized to [0, 1].
        d_o: Distance to closest opponent normalized to [0, 1].
        c1: Remaining cannon ammo normalized to [0, 1].
        c2: Remaining rockets normalized to [0, 1].
        w: Rocket ready flag (0.0 or 1.0).
        s_r: Currently shooting flag (0.0 or 1.0).
    """
    x: float
    y: float
    z: float
    v: float
    alpha_h: float
    alpha_off: float
    alpha_AA: float
    alpha_ATA: float
    d_o: float
    c1: float
    c2: float
    w: float
    s_r: float

    DIM: typing.ClassVar[int] = 13

    def __post_init__(self) -> None:
        """Clamp all observation components to [0.0, 1.0]."""
        self.x = _clamp_0_1(self.x)
        self.y = _clamp_0_1(self.y)
        self.z = _clamp_0_1(self.z)
        self.v = _clamp_0_1(self.v)
        self.alpha_h = _clamp_0_1(self.alpha_h)
        self.alpha_off = _clamp_0_1(self.alpha_off)
        self.alpha_AA = _clamp_0_1(self.alpha_AA)
        self.alpha_ATA = _clamp_0_1(self.alpha_ATA)
        self.d_o = _clamp_0_1(self.d_o)
        self.c1 = _clamp_0_1(self.c1)
        self.c2 = _clamp_0_1(self.c2)
        self.w = _clamp_0_1(self.w)
        self.s_r = _clamp_0_1(self.s_r)

    def to_array(self) -> np.ndarray:
        """Convert observation dataclass to 1D float32 numpy array of length 13."""
        return np.array([
            self.x, self.y, self.z, self.v,
            self.alpha_h, self.alpha_off, self.alpha_AA, self.alpha_ATA,
            self.d_o, self.c1, self.c2, self.w, self.s_r
        ], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray | typing.Sequence[float]) -> "AirObservation":
        """Instantiate AirObservation from array, clamping values to [0.0, 1.0]."""
        if len(arr) != cls.DIM:
            raise ValueError(f"Expected array of length {cls.DIM}, got {len(arr)}")
        return cls(*[float(v) for v in arr[:cls.DIM]])

    @classmethod
    def from_raw(
        cls,
        x: float,
        y: float,
        z: float,
        v: float,
        heading_rad: float,
        heading_off_rad: float,
        aspect_angle_rad: float,
        antenna_train_angle_rad: float,
        distance_to_opponent: float,
        cannon_ammo: float,
        rocket_ammo: float,
        rocket_ready: bool | float,
        is_shooting: bool | float,
        bounds_x: float = MAP_BOUNDS_X,
        bounds_y: float = MAP_BOUNDS_Y,
        bounds_z: float = MAP_BOUNDS_Z,
        max_speed: float = MAX_SPEED_AIR,
        max_dist: float = MAX_DISTANCE,
        max_cannon: float = MAX_CANNON_AMMO_AIR,
        max_rockets: float = MAX_ROCKET_AMMO_AIR,
    ) -> "AirObservation":
        """Instantiate normalized AirObservation from unnormalized physical quantities."""
        norm_h = (heading_rad % TWO_PI) / TWO_PI
        norm_off = abs(heading_off_rad) / PI
        norm_aa = abs(aspect_angle_rad) / PI
        norm_ata = abs(antenna_train_angle_rad) / PI

        return cls(
            x=x / max(bounds_x, 1e-6),
            y=y / max(bounds_y, 1e-6),
            z=z / max(bounds_z, 1e-6),
            v=v / max(max_speed, 1e-6),
            alpha_h=norm_h,
            alpha_off=norm_off,
            alpha_AA=norm_aa,
            alpha_ATA=norm_ata,
            d_o=distance_to_opponent / max(max_dist, 1e-6),
            c1=cannon_ammo / max(max_cannon, 1e-6),
            c2=rocket_ammo / max(max_rockets, 1e-6),
            w=1.0 if rocket_ready else 0.0,
            s_r=1.0 if is_shooting else 0.0,
        )


@dataclass
class GroundObservation:
    """Normalized observation vector for Ground agent (dimension = 9).

    All values are normalized and clamped to [0.0, 1.0] before storage.

    Attributes:
        x: Position X normalized to map bounds [0, 1].
        y: Position Y normalized to map bounds [0, 1].
        v: Speed normalized to [0, max_speed].
        theta: Heading angle normalized to [0, 2pi].
        elev: Terrain elevation normalized to [0, max_elevation].
        alpha_ATA: Antenna train angle to target normalized to [0, 1].
        d_o: Distance to closest opponent normalized to [0, 1].
        c1: Remaining ammo normalized to [0, 1].
        sensor_range: Current sensor detection range normalized to [0, 1].
    """
    x: float
    y: float
    v: float
    theta: float
    elev: float
    alpha_ATA: float
    d_o: float
    c1: float
    sensor_range: float

    DIM: typing.ClassVar[int] = 9

    def __post_init__(self) -> None:
        """Clamp all observation components to [0.0, 1.0]."""
        self.x = _clamp_0_1(self.x)
        self.y = _clamp_0_1(self.y)
        self.v = _clamp_0_1(self.v)
        self.theta = _clamp_0_1(self.theta)
        self.elev = _clamp_0_1(self.elev)
        self.alpha_ATA = _clamp_0_1(self.alpha_ATA)
        self.d_o = _clamp_0_1(self.d_o)
        self.c1 = _clamp_0_1(self.c1)
        self.sensor_range = _clamp_0_1(self.sensor_range)

    def to_array(self) -> np.ndarray:
        """Convert observation dataclass to 1D float32 numpy array of length 9."""
        return np.array([
            self.x, self.y, self.v, self.theta, self.elev,
            self.alpha_ATA, self.d_o, self.c1, self.sensor_range
        ], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray | typing.Sequence[float]) -> "GroundObservation":
        """Instantiate GroundObservation from array, clamping values to [0.0, 1.0]."""
        if len(arr) != cls.DIM:
            raise ValueError(f"Expected array of length {cls.DIM}, got {len(arr)}")
        return cls(*[float(v) for v in arr[:cls.DIM]])

    @classmethod
    def from_raw(
        cls,
        x: float,
        y: float,
        v: float,
        heading_rad: float,
        elevation: float,
        antenna_train_angle_rad: float,
        distance_to_opponent: float,
        ammo: float,
        sensor_range: float,
        bounds_x: float = MAP_BOUNDS_X,
        bounds_y: float = MAP_BOUNDS_Y,
        max_speed: float = MAX_SPEED_GROUND,
        max_elev: float = MAX_ELEVATION,
        max_dist: float = MAX_DISTANCE,
        max_ammo: float = MAX_AMMO_GROUND,
        max_sensor: float = MAX_SENSOR_RANGE,
    ) -> "GroundObservation":
        """Instantiate normalized GroundObservation from unnormalized physical quantities."""
        norm_theta = (heading_rad % TWO_PI) / TWO_PI
        norm_ata = abs(antenna_train_angle_rad) / PI

        return cls(
            x=x / max(bounds_x, 1e-6),
            y=y / max(bounds_y, 1e-6),
            v=v / max(max_speed, 1e-6),
            theta=norm_theta,
            elev=elevation / max(max_elev, 1e-6),
            alpha_ATA=norm_ata,
            d_o=distance_to_opponent / max(max_dist, 1e-6),
            c1=ammo / max(max_ammo, 1e-6),
            sensor_range=sensor_range / max(max_sensor, 1e-6),
        )


@dataclass
class SeaObservation:
    """Normalized observation vector for Sea agent (dimension = 9).

    All values are normalized and clamped to [0.0, 1.0] before storage.

    Attributes:
        x: Position X normalized to map bounds [0, 1].
        y: Position Y normalized to map bounds [0, 1].
        v: Speed normalized to [0, max_speed].
        theta: Heading angle normalized to [0, 2pi].
        sea_state: Sea state value (0-9 Beaufort scale) normalized to [0, 1].
        alpha_ATA: Antenna train angle to target normalized to [0, 1].
        d_o: Distance to closest opponent normalized to [0, 1].
        c1: Remaining ammo normalized to [0, 1].
        radar_range: Current radar detection range normalized to [0, 1].
    """
    x: float
    y: float
    v: float
    theta: float
    sea_state: float
    alpha_ATA: float
    d_o: float
    c1: float
    radar_range: float

    DIM: typing.ClassVar[int] = 9

    def __post_init__(self) -> None:
        """Clamp all observation components to [0.0, 1.0]."""
        self.x = _clamp_0_1(self.x)
        self.y = _clamp_0_1(self.y)
        self.v = _clamp_0_1(self.v)
        self.theta = _clamp_0_1(self.theta)
        self.sea_state = _clamp_0_1(self.sea_state)
        self.alpha_ATA = _clamp_0_1(self.alpha_ATA)
        self.d_o = _clamp_0_1(self.d_o)
        self.c1 = _clamp_0_1(self.c1)
        self.radar_range = _clamp_0_1(self.radar_range)

    def to_array(self) -> np.ndarray:
        """Convert observation dataclass to 1D float32 numpy array of length 9."""
        return np.array([
            self.x, self.y, self.v, self.theta, self.sea_state,
            self.alpha_ATA, self.d_o, self.c1, self.radar_range
        ], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray | typing.Sequence[float]) -> "SeaObservation":
        """Instantiate SeaObservation from array, clamping values to [0.0, 1.0]."""
        if len(arr) != cls.DIM:
            raise ValueError(f"Expected array of length {cls.DIM}, got {len(arr)}")
        return cls(*[float(v) for v in arr[:cls.DIM]])

    @classmethod
    def from_raw(
        cls,
        x: float,
        y: float,
        v: float,
        heading_rad: float,
        sea_state_beaufort: float,
        antenna_train_angle_rad: float,
        distance_to_opponent: float,
        ammo: float,
        radar_range: float,
        bounds_x: float = MAP_BOUNDS_X,
        bounds_y: float = MAP_BOUNDS_Y,
        max_speed: float = MAX_SPEED_SEA,
        max_sea_state: float = MAX_BEAUFORT_SEA_STATE,
        max_dist: float = MAX_DISTANCE,
        max_ammo: float = MAX_AMMO_SEA,
        max_radar: float = MAX_RADAR_RANGE,
    ) -> "SeaObservation":
        """Instantiate normalized SeaObservation from unnormalized physical quantities."""
        norm_theta = (heading_rad % TWO_PI) / TWO_PI
        norm_ata = abs(antenna_train_angle_rad) / PI

        return cls(
            x=x / max(bounds_x, 1e-6),
            y=y / max(bounds_y, 1e-6),
            v=v / max(max_speed, 1e-6),
            theta=norm_theta,
            sea_state=sea_state_beaufort / max(max_sea_state, 1e-6),
            alpha_ATA=norm_ata,
            d_o=distance_to_opponent / max(max_dist, 1e-6),
            c1=ammo / max(max_ammo, 1e-6),
            radar_range=radar_range / max(max_radar, 1e-6),
        )


@dataclass
class CommanderObservation:
    """Observation vector for high-level Commander policy (variable dimension).

    Combines the commander's own state, up to 3 opponent states (with domain ID appended),
    up to 2 friendly states (with domain ID appended), and one-hot domain encodings.

    All numerical values are clamped to [0.0, 1.0].

    Attributes:
        own_state: Commander's own state vector [x, y, z, v, heading] (normalized [0, 1]).
        opponent_states: List of up to 3 closest opponent state vectors (each with domain_id appended).
        friendly_states: List of up to 2 closest friendly state vectors (each with domain_id appended).
        domain_ids: Array of one-hot domain encodings for each entity observed.
    """
    own_state: np.ndarray
    opponent_states: list[np.ndarray] = field(default_factory=list)
    friendly_states: list[np.ndarray] = field(default_factory=list)
    domain_ids: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))

    MAX_OPPONENTS: typing.ClassVar[int] = 3
    MAX_FRIENDLIES: typing.ClassVar[int] = 2
    NUM_DOMAINS: typing.ClassVar[int] = 3

    def __post_init__(self) -> None:
        """Validate and clamp arrays to [0.0, 1.0]."""
        self.own_state = np.clip(np.asarray(self.own_state, dtype=np.float32), 0.0, 1.0)
        self.opponent_states = [
            np.clip(np.asarray(s, dtype=np.float32), 0.0, 1.0)
            for s in self.opponent_states[:self.MAX_OPPONENTS]
        ]
        self.friendly_states = [
            np.clip(np.asarray(s, dtype=np.float32), 0.0, 1.0)
            for s in self.friendly_states[:self.MAX_FRIENDLIES]
        ]
        self.domain_ids = np.clip(np.asarray(self.domain_ids, dtype=np.float32), 0.0, 1.0)

    def to_array(self) -> np.ndarray:
        """Flatten and concatenate all observation components into a 1D float32 numpy array."""
        parts: list[np.ndarray] = [self.own_state.flatten()]

        for opp in self.opponent_states:
            parts.append(opp.flatten())

        for fr in self.friendly_states:
            parts.append(fr.flatten())

        if self.domain_ids.size > 0:
            parts.append(self.domain_ids.flatten())

        return np.concatenate(parts, dtype=np.float32)

    def to_padded_array(self, state_dim: int = 5) -> np.ndarray:
        """Produce a fixed-size 1D float32 numpy array padded with zeros.

        Layout:
            - own_state: state_dim
            - 3 x opponent_states: 3 * (state_dim + 1)
            - 2 x friendly_states: 2 * (state_dim + 1)
            - domain_ids: 6 * 3 = 18 (one-hot domain encoding for 1 self + 3 opp + 2 fr)
        """
        padded_parts: list[np.ndarray] = []

        # Own state (padded or truncated to state_dim)
        own = np.zeros(state_dim, dtype=np.float32)
        dim = min(len(self.own_state), state_dim)
        own[:dim] = self.own_state[:dim]
        padded_parts.append(own)

        # Opponents (3 slots, each state_dim + 1)
        entity_slot_dim = state_dim + 1
        for i in range(self.MAX_OPPONENTS):
            opp_buf = np.zeros(entity_slot_dim, dtype=np.float32)
            if i < len(self.opponent_states):
                s = self.opponent_states[i]
                d = min(len(s), entity_slot_dim)
                opp_buf[:d] = s[:d]
            padded_parts.append(opp_buf)

        # Friendlies (2 slots, each state_dim + 1)
        for i in range(self.MAX_FRIENDLIES):
            fr_buf = np.zeros(entity_slot_dim, dtype=np.float32)
            if i < len(self.friendly_states):
                s = self.friendly_states[i]
                d = min(len(s), entity_slot_dim)
                fr_buf[:d] = s[:d]
            padded_parts.append(fr_buf)

        # Domain IDs: 6 entities * 3 domains = 18 floats
        total_entities = 1 + self.MAX_OPPONENTS + self.MAX_FRIENDLIES
        domains_buf = np.zeros(total_entities * self.NUM_DOMAINS, dtype=np.float32)
        if self.domain_ids.size > 0:
            flat_d = self.domain_ids.flatten()
            num_d = min(len(flat_d), len(domains_buf))
            domains_buf[:num_d] = flat_d[:num_d]
        padded_parts.append(domains_buf)

        return np.concatenate(padded_parts, dtype=np.float32)


@dataclass
class FullObservation:
    """Aggregated container for all agent observations in a scenario step.

    Facilitates structured observation dispatching across domain policies
    and the high-level commander policy.
    """
    air_observations: dict[str, AirObservation] = field(default_factory=dict)
    ground_observations: dict[str, GroundObservation] = field(default_factory=dict)
    sea_observations: dict[str, SeaObservation] = field(default_factory=dict)
    commander_observations: dict[str, CommanderObservation] = field(default_factory=dict)
    global_state: np.ndarray | None = None

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert all nested entity observations into a dictionary of numpy arrays."""
        result: dict[str, np.ndarray] = {}
        for agent_id, air_obs in self.air_observations.items():
            result[agent_id] = air_obs.to_array()
        for agent_id, ground_obs in self.ground_observations.items():
            result[agent_id] = ground_obs.to_array()
        for agent_id, sea_obs in self.sea_observations.items():
            result[agent_id] = sea_obs.to_array()
        for agent_id, cmd_obs in self.commander_observations.items():
            result[agent_id] = cmd_obs.to_array()
        if self.global_state is not None:
            result["__global__"] = np.asarray(self.global_state, dtype=np.float32)
        return result

    def to_array(self) -> np.ndarray:
        """Concatenate all entity observations into a single 1D float32 numpy array."""
        obs_dict = self.to_dict()
        if not obs_dict:
            return np.zeros(0, dtype=np.float32)
        sorted_keys = sorted(obs_dict.keys())
        return np.concatenate([obs_dict[k].flatten() for k in sorted_keys], dtype=np.float32)
