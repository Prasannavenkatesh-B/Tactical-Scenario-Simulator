"""Stochastic sensor perception models.

Implements probabilistic detection falloff, aspect sensitivity, Gaussian noise
injection, and false alarm ghost contacts to model real-world radar/sensor uncertainty.
"""

from dataclasses import dataclass
import math
import typing
import numpy as np

from src.core.rewards import compute_AA, compute_distance
from src.simulator.config import (
    DEG_TO_RAD,
    RAD_TO_DEG,
    SENSOR_ASPECT_PENALTY,
    SENSOR_BASE_PD,
    SENSOR_BEARING_NOISE_DEG,
    SENSOR_FALSE_ALARM_RATE,
    SENSOR_RANGE_FALLOFF,
    SENSOR_RANGE_NOISE_SIGMA_FRAC,
    TWO_PI,
)
from src.simulator.entities.base import SimEntity


@dataclass
class SensorContact:
    """Perception report produced by a sensor detection.

    Attributes:
        target_id: String ID of detected entity (or ghost identifier).
        estimated_position: Noisy 3D position vector in km.
        estimated_velocity: Noisy velocity magnitude in km/h.
        estimated_heading: Noisy heading angle in radians [0, 2pi).
        confidence: Probability of detection (Pd) associated with this contact.
        timestamp: Simulation timestamp (seconds) when detected.
    """
    target_id: str
    estimated_position: np.ndarray
    estimated_velocity: float
    estimated_heading: float
    confidence: float
    timestamp: float


class Sensor:
    """Stochastic sensor providing probabilistic, noisy detections."""

    def __init__(
        self,
        sensor_type: str,
        max_range_km: float,
        rng: np.random.Generator,
        base_pd: float = SENSOR_BASE_PD,
        range_falloff: float = SENSOR_RANGE_FALLOFF,
        aspect_penalty: float = SENSOR_ASPECT_PENALTY,
        false_alarm_rate: float = SENSOR_FALSE_ALARM_RATE,
    ) -> None:
        """Initialize stochastic sensor.

        Args:
            sensor_type: Sensor classification string (e.g. "RADAR", "EO/IR").
            max_range_km: Maximum detection horizon in km.
            rng: Seeded numpy Generator for stochastic rolls.
            base_pd: Peak nominal detection probability.
            range_falloff: Exponent governing range degradation.
            aspect_penalty: Penalty multiplier for unfavorable aspect geometry.
            false_alarm_rate: Probability of generating a ghost contact per tick.
        """
        self.sensor_type: str = str(sensor_type)
        self.max_range_km: float = float(max_range_km)
        self.rng: np.random.Generator = rng
        self.base_pd: float = float(base_pd)
        self.range_falloff: float = float(range_falloff)
        self.aspect_penalty: float = float(aspect_penalty)
        self.false_alarm_rate: float = float(false_alarm_rate)

    def compute_pd(
        self,
        target: SimEntity,
        observer: SimEntity,
        environment: dict[str, typing.Any] | None = None,
    ) -> float:
        """Calculate the probability of detection (Pd) for a given target-observer pair.

        Formula:
            Pd = base_pd * (1 - d/R)^falloff * aspect_factor * environment_factor

        Args:
            target: Target entity being scanned.
            observer: Observer entity hosting this sensor.
            environment: Optional environmental condition dict (e.g. weather).

        Returns:
            Computed detection probability in [0.0, 1.0].
        """
        d = compute_distance(observer.position, target.position)
        R = self.max_range_km

        if d >= R or R <= 0.0:
            return 0.0

        # Range falloff
        range_factor = (1.0 - (d / R)) ** self.range_falloff

        # Aspect factor: penalty based on aspect angle
        aspect_angle = compute_AA(
            observer_pos=observer.position[:2],
            observer_heading=observer.heading,
            target_pos=target.position[:2],
            target_heading=target.heading,
        )
        aspect_factor = 1.0 - (self.aspect_penalty * abs(math.cos(aspect_angle)))

        # Environmental degradation
        env_factor = 1.0
        if environment:
            weather = str(environment.get("weather", "clear")).lower()
            if weather == "rain":
                env_factor = 0.85
            elif weather == "fog":
                env_factor = 0.70

        pd = self.base_pd * range_factor * aspect_factor * env_factor
        return float(np.clip(pd, 0.0, 1.0))

    def detect(
        self,
        target: SimEntity,
        observer: SimEntity,
        environment: dict[str, typing.Any] | None = None,
        current_time: float = 0.0,
    ) -> SensorContact | None:
        """Perform a stochastic detection attempt against a target entity.

        Carries out a Bernoulli trial with probability Pd. If successful, injects
        range and bearing Gaussian noise to simulate sensor measurement errors.

        Args:
            target: Entity being scanned.
            observer: Entity hosting sensor.
            environment: Environmental metadata dict.
            current_time: Current simulation elapsed time in seconds.

        Returns:
            SensorContact if detected, None if undetected.
        """
        if not target.is_alive():
            return None

        pd = self.compute_pd(target, observer, environment)
        if pd <= 0.0:
            return None

        # Stochastic Bernoulli trial
        if self.rng.random() >= pd:
            return None

        # Target detected -> synthesize noisy contact
        obs_pos = observer.position
        tgt_pos = target.position

        dx = float(tgt_pos[0]) - float(obs_pos[0])
        dy = float(tgt_pos[1]) - float(obs_pos[1])
        true_dist = math.sqrt(dx * dx + dy * dy)
        true_bearing = math.atan2(dy, dx)

        # Gaussian measurement noise
        range_noise_sigma = SENSOR_RANGE_NOISE_SIGMA_FRAC * self.max_range_km
        range_noise = float(self.rng.normal(0.0, range_noise_sigma))
        bearing_noise = float(self.rng.normal(0.0, SENSOR_BEARING_NOISE_DEG * DEG_TO_RAD))

        noisy_dist = max(0.01, true_dist + range_noise)
        noisy_bearing = (true_bearing + bearing_noise) % TWO_PI

        est_x = float(obs_pos[0] + noisy_dist * math.cos(noisy_bearing))
        est_y = float(obs_pos[1] + noisy_dist * math.sin(noisy_bearing))

        # Altitude noise for 3D
        if len(tgt_pos) > 2:
            alt_noise = float(self.rng.normal(0.0, 0.05))
            est_z = max(0.0, float(tgt_pos[2] + alt_noise))
            est_pos = np.array([est_x, est_y, est_z], dtype=np.float64)
        else:
            est_pos = np.array([est_x, est_y, 0.0], dtype=np.float64)

        heading_noise = float(self.rng.normal(0.0, 2.0 * DEG_TO_RAD))
        est_heading = (target.heading + heading_noise) % TWO_PI

        vel_noise = float(self.rng.normal(0.0, 0.05 * target.velocity))
        est_velocity = max(0.0, target.velocity + vel_noise)

        return SensorContact(
            target_id=target.entity_id,
            estimated_position=est_pos,
            estimated_velocity=est_velocity,
            estimated_heading=est_heading,
            confidence=pd,
            timestamp=current_time,
        )

    def sample_false_alarm(
        self,
        observer: SimEntity,
        current_time: float = 0.0,
    ) -> SensorContact | None:
        """Sample potential stochastic false alarm (ghost contact).

        Args:
            observer: Observing entity.
            current_time: Current simulation time.

        Returns:
            Ghost SensorContact if false alarm triggered, None otherwise.
        """
        if self.rng.random() >= self.false_alarm_rate:
            return None

        # Synthesize ghost contact inside sensor coverage
        ghost_dist = float(self.rng.uniform(0.1, self.max_range_km))
        ghost_bearing = float(self.rng.uniform(0.0, TWO_PI))

        gx = float(observer.position[0] + ghost_dist * math.cos(ghost_bearing))
        gy = float(observer.position[1] + ghost_dist * math.sin(ghost_bearing))
        gz = float(observer.position[2]) if len(observer.position) > 2 else 0.0

        ghost_id = f"ghost_{int(self.rng.integers(1000, 9999))}"
        return SensorContact(
            target_id=ghost_id,
            estimated_position=np.array([gx, gy, gz], dtype=np.float64),
            estimated_velocity=float(self.rng.uniform(100.0, 500.0)),
            estimated_heading=float(self.rng.uniform(0.0, TWO_PI)),
            confidence=float(self.rng.uniform(0.1, 0.4)),
            timestamp=current_time,
        )
