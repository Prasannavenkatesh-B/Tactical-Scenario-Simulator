"""Unit tests for stochastic weapon models, WEZ geometries, and kill probabilities."""

import math
import numpy as np
import pytest

from src.core.interfaces import TeamSide, WeaponType
from src.simulator.entities.air import AirEntity
from src.simulator.weapons import Weapon


class TestWeapons:
    """Test suite for Weapon firing logic and stochastic effects."""

    def test_can_fire_out_of_wez_angle(self) -> None:
        """Verify weapon cannot discharge when target is outside WEZ cone."""
        rng = np.random.default_rng(42)
        # 10 degree total cone (+/- 5 degrees)
        weapon = Weapon(WeaponType.CANNON, max_range_km=2.0, wez_angle_deg=10.0, base_pk=0.8, rng=rng)

        shooter = AirEntity("s", TeamSide.BLUE, "AC1", position=(0.0, 0.0, 5.0), heading=0.0)

        # Target at 15 degrees off heading (outside 5 deg half-cone)
        tgt_x = 1.0 * math.cos(math.radians(15.0))
        tgt_y = 1.0 * math.sin(math.radians(15.0))
        target_outside = AirEntity("t_out", TeamSide.RED, "AC1", position=(tgt_x, tgt_y, 5.0))

        assert not weapon.is_in_wez(shooter, target_outside)
        assert not weapon.can_fire(shooter, target_outside)

        # Target at 2 degrees off heading (inside 5 deg half-cone)
        tgt_in_x = 1.0 * math.cos(math.radians(2.0))
        tgt_in_y = 1.0 * math.sin(math.radians(2.0))
        target_inside = AirEntity("t_in", TeamSide.RED, "AC1", position=(tgt_in_x, tgt_in_y, 5.0))

        assert weapon.is_in_wez(shooter, target_inside)
        assert weapon.can_fire(shooter, target_inside)

    def test_can_fire_out_of_wez_range(self) -> None:
        """Verify weapon cannot fire beyond max_range_km."""
        rng = np.random.default_rng(42)
        weapon = Weapon(WeaponType.CANNON, max_range_km=2.0, wez_angle_deg=10.0, base_pk=0.8, rng=rng)

        shooter = AirEntity("s", TeamSide.BLUE, "AC1", position=(0.0, 0.0, 5.0), heading=0.0)
        target_too_far = AirEntity("t_far", TeamSide.RED, "AC1", position=(2.5, 0.0, 5.0))

        assert not weapon.is_in_wez(shooter, target_too_far)
        assert not weapon.can_fire(shooter, target_too_far)

    def test_can_fire_ammo_zero(self) -> None:
        """Verify weapon cannot fire when ammunition is exhausted."""
        rng = np.random.default_rng(42)
        weapon = Weapon(WeaponType.CANNON, max_range_km=2.0, wez_angle_deg=10.0, base_pk=0.8, rng=rng)

        shooter = AirEntity("s", TeamSide.BLUE, "AC1", position=(0.0, 0.0, 5.0), heading=0.0)
        target = AirEntity("t", TeamSide.RED, "AC1", position=(1.0, 0.0, 5.0))

        shooter.ammo[WeaponType.CANNON] = 0
        assert not weapon.can_fire(shooter, target)

    def test_pk_scales_with_range(self) -> None:
        """Verify effective Pk is higher at closer ranges."""
        rng = np.random.default_rng(42)
        weapon = Weapon(WeaponType.CANNON, max_range_km=4.0, wez_angle_deg=20.0, base_pk=0.8, rng=rng)

        shooter = AirEntity("s", TeamSide.BLUE, "AC1", position=(0.0, 0.0, 5.0), heading=0.0)
        t_near = AirEntity("t_near", TeamSide.RED, "AC1", position=(0.5, 0.0, 5.0))
        t_mid = AirEntity("t_mid", TeamSide.RED, "AC1", position=(2.0, 0.0, 5.0))
        t_far = AirEntity("t_far", TeamSide.RED, "AC1", position=(3.8, 0.0, 5.0))

        pk_near = weapon.compute_effective_pk(shooter, t_near)
        pk_mid = weapon.compute_effective_pk(shooter, t_mid)
        pk_far = weapon.compute_effective_pk(shooter, t_far)

        assert pk_near > pk_mid > pk_far > 0.0

    def test_miss_distance_stochastic(self) -> None:
        """Verify miss distances exhibit stochastic Gaussian dispersion."""
        rng = np.random.default_rng(42)
        # base_pk = 0.0 guarantees miss
        weapon = Weapon(WeaponType.CANNON, max_range_km=5.0, wez_angle_deg=20.0, base_pk=0.0, rng=rng)

        shooter = AirEntity("s", TeamSide.BLUE, "AC1", position=(0.0, 0.0, 5.0), heading=0.0)
        target = AirEntity("t", TeamSide.RED, "AC1", position=(1.0, 0.0, 5.0))

        misses = [weapon.fire(shooter, target, current_time=i * 1.0).miss_distance_km for i in range(20)]
        assert len(set(misses)) > 1
        assert all(m > 0.0 for m in misses)

    def test_cooldown_prevents_rapid_fire(self) -> None:
        """Verify cooldown timer prevents firing consecutively before duration elapses."""
        rng = np.random.default_rng(42)
        weapon = Weapon(WeaponType.CANNON, max_range_km=2.0, wez_angle_deg=10.0, base_pk=0.5,
                        rng=rng, cooldown_seconds=0.5)

        shooter = AirEntity("s", TeamSide.BLUE, "AC1", position=(0.0, 0.0, 5.0), heading=0.0)
        target = AirEntity("t", TeamSide.RED, "AC1", position=(1.0, 0.0, 5.0))

        assert weapon.can_fire(shooter, target, current_time=0.0)
        weapon.fire(shooter, target, current_time=0.0)

        # 0.2s later (cooldown not expired)
        assert not weapon.can_fire(shooter, target, current_time=0.2)

        # 0.5s later (cooldown expired)
        assert weapon.can_fire(shooter, target, current_time=0.5)

    def test_weapon_fire_decrements_ammo(self) -> None:
        """Verify firing weapon decreases remaining ammunition."""
        rng = np.random.default_rng(42)
        weapon = Weapon(WeaponType.CANNON, max_range_km=2.0, wez_angle_deg=10.0, base_pk=0.5, rng=rng)
        shooter = AirEntity("s", TeamSide.BLUE, "AC1", position=(0.0, 0.0, 5.0), heading=0.0)
        target = AirEntity("t", TeamSide.RED, "AC1", position=(1.0, 0.0, 5.0))

        initial_ammo = shooter.ammo[WeaponType.CANNON]
        weapon.fire(shooter, target, current_time=0.0)
        assert shooter.ammo[WeaponType.CANNON] == initial_ammo - 1

    def test_weapon_rocket_longer_range(self) -> None:
        """Verify rocket weapon has longer WEZ reach than cannon."""
        rng = np.random.default_rng(42)
        cannon = Weapon(WeaponType.CANNON, max_range_km=2.0, wez_angle_deg=10.0, base_pk=0.7, rng=rng)
        rocket = Weapon(WeaponType.ROCKET, max_range_km=6.0, wez_angle_deg=15.0, base_pk=0.65, rng=rng)

        shooter = AirEntity("s", TeamSide.BLUE, "AC1", position=(0.0, 0.0, 5.0), heading=0.0)
        target_4km = AirEntity("t_4k", TeamSide.RED, "AC1", position=(4.0, 0.0, 5.0))

        # At 4 km, cannon cannot fire but rocket can
        assert not cannon.can_fire(shooter, target_4km)
        assert rocket.can_fire(shooter, target_4km)
