"""Stochastic weapon models with Weapon Engagement Zone (WEZ) and PK calculations.

Implements firing cone verification, probabilistic kill (Pk) scaling with range,
miss distance dispersion, and weapon cooldown enforcement.
"""

from dataclasses import dataclass
import math
import typing
import numpy as np

from src.core.interfaces import WeaponType
from src.core.rewards import compute_ATA, compute_distance
from src.simulator.config import (
    COOLDOWN_CANNON_SECONDS,
    COOLDOWN_MISSILE_SECONDS,
    COOLDOWN_ROCKET_SECONDS,
    COOLDOWN_SAM_SECONDS,
    DEG_TO_RAD,
    RAD_TO_DEG,
    WEAPON_MISS_DISTANCE_SIGMA_KM,
    WEAPON_PK_SCALE,
)
from src.simulator.entities.base import SimEntity


@dataclass
class WeaponEffect:
    """Outcome report for a weapon firing event.

    Attributes:
        hit: True if the weapon successfully impacted the target.
        miss_distance_km: Miss distance in kilometers (0.0 if hit).
        target_destroyed: True if target was eliminated.
        weapon_type: Type of weapon fired.
        shooter_id: Entity ID of the firing agent.
        target_id: Entity ID of the targeted agent.
        timestamp: Simulation timestamp of engagement.
    """
    hit: bool
    miss_distance_km: float
    target_destroyed: bool
    weapon_type: WeaponType
    shooter_id: str
    target_id: str
    timestamp: float = 0.0


class Weapon:
    """Stochastic weapon model with WEZ cone geometry and distance-scaled PK."""

    def __init__(
        self,
        weapon_type: WeaponType,
        max_range_km: float,
        wez_angle_deg: float,
        base_pk: float,
        rng: np.random.Generator,
        cooldown_seconds: float | None = None,
        miss_sigma_km: float = WEAPON_MISS_DISTANCE_SIGMA_KM,
        pk_scale: float = WEAPON_PK_SCALE,
    ) -> None:
        """Initialize weapon system.

        Args:
            weapon_type: WeaponType enum (CANNON, ROCKET, MISSILE, SAM).
            max_range_km: Maximum effective engagement range in km.
            wez_angle_deg: Total angular width of weapon engagement cone (degrees).
            base_pk: Nominal probability of kill at point-blank range.
            rng: Seeded numpy Generator for stochastic hit rolls.
            cooldown_seconds: Minimum reload/cooldown duration between firings.
            miss_sigma_km: Standard deviation for Gaussian miss distance dispersion.
            pk_scale: Calibration multiplier for kill probability.
        """
        self.weapon_type: WeaponType = weapon_type
        self.max_range_km: float = float(max_range_km)
        self.wez_angle_deg: float = float(wez_angle_deg)
        self.base_pk: float = float(base_pk)
        self.rng: np.random.Generator = rng
        self.miss_sigma_km: float = float(miss_sigma_km)
        self.pk_scale: float = float(pk_scale)

        if cooldown_seconds is not None:
            self.cooldown_seconds: float = float(cooldown_seconds)
        else:
            if weapon_type == WeaponType.CANNON:
                self.cooldown_seconds = COOLDOWN_CANNON_SECONDS
            elif weapon_type == WeaponType.ROCKET:
                self.cooldown_seconds = COOLDOWN_ROCKET_SECONDS
            elif weapon_type == WeaponType.MISSILE:
                self.cooldown_seconds = COOLDOWN_MISSILE_SECONDS
            elif weapon_type == WeaponType.SAM:
                self.cooldown_seconds = COOLDOWN_SAM_SECONDS
            else:
                self.cooldown_seconds = 1.0

    def is_in_wez(self, shooter: SimEntity, target: SimEntity) -> bool:
        """Check whether the target lies inside the Weapon Engagement Zone (WEZ) cone.

        Conditions:
            1. Distance d <= max_range_km
            2. Line-of-sight angle to target within +/- (wez_angle_deg / 2) of shooter heading

        Args:
            shooter: Firing entity.
            target: Target entity.

        Returns:
            True if target is inside the WEZ cone.
        """
        d = compute_distance(shooter.position, target.position)
        if d > self.max_range_km:
            return False

        ata_rad = compute_ATA(shooter.position[:2], shooter.heading, target.position[:2])
        ata_deg = ata_rad * RAD_TO_DEG
        half_cone = self.wez_angle_deg / 2.0
        return ata_deg <= half_cone

    def can_fire(
        self,
        shooter: SimEntity,
        target: SimEntity,
        current_time: float = 0.0,
    ) -> bool:
        """Check whether the weapon can legally discharge against the target.

        Verifies:
            - Shooter and target are alive
            - Ammunition for this weapon type > 0
            - Firing cooldown has elapsed
            - Target is within WEZ range and angular cone

        Args:
            shooter: Firing entity.
            target: Targeted entity.
            current_time: Current simulation timestamp.

        Returns:
            True if all firing conditions are satisfied.
        """
        if not shooter.is_alive() or not target.is_alive():
            return False

        current_ammo = shooter.ammo.get(self.weapon_type, 0)
        if current_ammo <= 0:
            return False

        last_fired = shooter.last_fire_times.get(self.weapon_type, -999.0)
        if (current_time - last_fired) < self.cooldown_seconds:
            return False

        return self.is_in_wez(shooter, target)

    def compute_effective_pk(self, shooter: SimEntity, target: SimEntity) -> float:
        """Compute the range-dependent effective Probability of Kill (Pk).

        Formula:
            effective_pk = base_pk * pk_scale * (1 - 0.5 * (d / R))

        Args:
            shooter: Firing entity.
            target: Targeted entity.

        Returns:
            Effective Pk in [0.0, 1.0].
        """
        d = compute_distance(shooter.position, target.position)
        R = max(self.max_range_km, 1e-6)
        range_factor = max(0.0, 1.0 - 0.5 * (d / R))
        pk = self.base_pk * self.pk_scale * range_factor
        return float(np.clip(pk, 0.0, 1.0))

    def fire(
        self,
        shooter: SimEntity,
        target: SimEntity,
        current_time: float = 0.0,
    ) -> WeaponEffect:
        """Execute a weapon discharge event against a target.

        Decrements shooter ammunition, logs cooldown timestamp, performs a stochastic
        Bernoulli trial with effective Pk, and applies damage if hit.

        Args:
            shooter: Firing entity.
            target: Target entity.
            current_time: Current simulation timestamp.

        Returns:
            WeaponEffect describing outcome, miss distance, and casualties.
        """
        # Decrement ammunition
        shooter.ammo[self.weapon_type] = max(0, shooter.ammo.get(self.weapon_type, 0) - 1)
        shooter.last_fire_times[self.weapon_type] = current_time

        effective_pk = self.compute_effective_pk(shooter, target)
        hit_roll = self.rng.random()

        if hit_roll < effective_pk:
            # Direct hit
            destroyed = target.apply_weapon_hit(self.weapon_type, self.rng)
            return WeaponEffect(
                hit=True,
                miss_distance_km=0.0,
                target_destroyed=destroyed,
                weapon_type=self.weapon_type,
                shooter_id=shooter.entity_id,
                target_id=target.entity_id,
                timestamp=current_time,
            )
        else:
            # Miss: sample stochastic miss distance dispersion
            miss_dist = abs(float(self.rng.normal(0.0, self.miss_sigma_km)))
            return WeaponEffect(
                hit=False,
                miss_distance_km=miss_dist,
                target_destroyed=False,
                weapon_type=self.weapon_type,
                shooter_id=shooter.entity_id,
                target_id=target.entity_id,
                timestamp=current_time,
            )
