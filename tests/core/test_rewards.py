"""Unit tests for reward functions and spatial/angular geometry routines."""

import math
import numpy as np
import pytest

from src.core.rewards import (
    MultiDomainRewardWeights,
    compute_AA,
    compute_ATA,
    compute_boundary_penalty,
    compute_commander_reward,
    compute_composite_reward,
    compute_death_penalty,
    compute_distance,
    compute_escape_reward,
    compute_fight_reward,
    compute_friendly_fire_penalty,
    compute_heading_off,
    compute_kill_reward,
    normalize_angle,
)


class TestFightReward:
    """Test suite for fight policy reward (base paper Eq. 1)."""

    def test_r_fight_range_valid_inputs(self) -> None:
        """Verify r_fight returns values in [1, 2] for all valid tactical inputs."""
        valid_tactical_cases = [
            # (alpha_ATA_a, c_max, c_rem, expected)
            (1.0, 10.0, 10.0, 1.0),  # full ammo, opponent looking directly away
            (1.0, 10.0, 0.0, 2.0),   # all ammo expended, opponent looking away
            (0.5, 10.0, 5.0, 1.0),   # half angle, half ammo spent
            (0.8, 10.0, 2.0, 1.6),   # 0.8 + 8/10 = 1.6
            (0.6, 100.0, 40.0, 1.2), # 0.6 + 60/100 = 1.2
            (0.9, 8.0, 1.0, 0.9 + 7.0 / 8.0), # in [1, 2]
        ]
        for alpha, c_max, c_rem, expected in valid_tactical_cases:
            reward = compute_fight_reward(alpha, c_max, c_rem)
            assert 1.0 <= reward <= 2.0
            assert pytest.approx(reward, 1e-5) == expected

    def test_r_fight_boundary_clamp(self) -> None:
        """Verify clamp_range option guarantees [1, 2] for arbitrary inputs."""
        reward_low = compute_fight_reward(alpha_ATA_a=0.1, c_max=10.0, c_rem=10.0, clamp_range=True)
        assert reward_low == 1.0

        reward_high = compute_fight_reward(alpha_ATA_a=1.0, c_max=10.0, c_rem=0.0, clamp_range=True)
        assert reward_high == 2.0

    def test_r_fight_zero_c_max_protection(self) -> None:
        """Verify graceful division by zero protection when c_max <= 0."""
        reward = compute_fight_reward(alpha_ATA_a=1.0, c_max=0.0, c_rem=0.0)
        assert reward == 1.0


class TestEscapeReward:
    """Test suite for escape policy reward (base paper Eq. 2)."""

    def test_r_escape_distance_thresholds(self) -> None:
        """Verify -0.01 for d < 6 km, +0.01 for d > 13 km, and 0.0 otherwise."""
        # Danger zone (< 6 km)
        assert compute_escape_reward(0.0) == -0.01
        assert compute_escape_reward(2.5) == -0.01
        assert compute_escape_reward(5.99) == -0.01

        # Safe distance (> 13 km)
        assert compute_escape_reward(13.01) == 0.01
        assert compute_escape_reward(20.0) == 0.01
        assert compute_escape_reward(100.0) == 0.01

        # Intermediate zone (6 km <= d <= 13 km)
        assert compute_escape_reward(6.0) == 0.0
        assert compute_escape_reward(9.5) == 0.0
        assert compute_escape_reward(13.0) == 0.0

    def test_r_escape_meters_input(self) -> None:
        """Verify escape reward handles meter inputs with in_km=False."""
        assert compute_escape_reward(5_000.0, in_km=False) == -0.01
        assert compute_escape_reward(10_000.0, in_km=False) == 0.0
        assert compute_escape_reward(15_000.0, in_km=False) == 0.01


class TestCommanderReward:
    """Test suite for commander favorable situation reward (base paper Eq. 3)."""

    def test_r_commander_all_conditions_met(self) -> None:
        """Verify +0.1 returned when d_o < 5km, ATA < 30 deg, AA < 50 deg, a_c > 0."""
        reward = compute_commander_reward(
            d_o=3.0,
            alpha_ATA_o=20.0,
            alpha_AA_o=40.0,
            a_c=1,
            dist_in_km=True,
            angle_in_degrees=True,
        )
        assert reward == 0.1

    def test_r_commander_distance_fail(self) -> None:
        """Verify 0 returned when d_o >= 5 km."""
        assert compute_commander_reward(d_o=5.0, alpha_ATA_o=10.0, alpha_AA_o=20.0, a_c=1) == 0.0
        assert compute_commander_reward(d_o=8.0, alpha_ATA_o=10.0, alpha_AA_o=20.0, a_c=1) == 0.0

    def test_r_commander_ata_fail(self) -> None:
        """Verify 0 returned when alpha_ATA_o >= 30 deg."""
        assert compute_commander_reward(d_o=3.0, alpha_ATA_o=30.0, alpha_AA_o=20.0, a_c=1) == 0.0
        assert compute_commander_reward(d_o=3.0, alpha_ATA_o=45.0, alpha_AA_o=20.0, a_c=1) == 0.0

    def test_r_commander_aa_fail(self) -> None:
        """Verify 0 returned when alpha_AA_o >= 50 deg."""
        assert compute_commander_reward(d_o=3.0, alpha_ATA_o=20.0, alpha_AA_o=50.0, a_c=1) == 0.0
        assert compute_commander_reward(d_o=3.0, alpha_ATA_o=20.0, alpha_AA_o=75.0, a_c=1) == 0.0

    def test_r_commander_action_fail(self) -> None:
        """Verify 0 returned when a_c == 0 (escape action)."""
        assert compute_commander_reward(d_o=3.0, alpha_ATA_o=20.0, alpha_AA_o=30.0, a_c=0) == 0.0

    def test_r_commander_radians_and_meters(self) -> None:
        """Verify commander reward with SI meters and radians."""
        # 30 deg ~ 0.523 rad, 50 deg ~ 0.872 rad
        reward = compute_commander_reward(
            d_o=4_000.0,
            alpha_ATA_o=math.radians(25.0),
            alpha_AA_o=math.radians(45.0),
            a_c=2,
            dist_in_km=False,
            angle_in_degrees=False,
        )
        assert reward == 0.1


class TestPenaltyAndCompositeRewards:
    """Test suite for penalties and multi-domain composite rewards."""

    def test_penalties_and_kills(self) -> None:
        """Verify individual penalty and kill reward functions."""
        assert compute_boundary_penalty(True) == -5.0
        assert compute_boundary_penalty(False) == 0.0

        assert compute_friendly_fire_penalty(True) == -2.0
        assert compute_friendly_fire_penalty(False) == 0.0

        assert compute_kill_reward(1) == 1.0
        assert compute_kill_reward(3) == 3.0

        assert compute_death_penalty(True) == -1.0
        assert compute_death_penalty(False) == 0.0

    def test_composite_reward_default_weights(self) -> None:
        """Verify composite multi-domain aggregation with default weights."""
        total = compute_composite_reward(
            r_air=1.5,
            r_ground=2.0,
            r_sea=1.0,
            r_mission=4.0,  # 4.0 * 0.5 = 2.0
            r_boundary=-5.0,
            r_friendly_fire=-2.0,
        )
        # 1.0*1.5 + 1.0*2.0 + 1.0*1.0 + 0.5*4.0 - 5.0 - 2.0 = 1.5 + 2.0 + 1.0 + 2.0 - 7.0 = -0.5
        assert pytest.approx(total, 1e-5) == -0.5

    def test_composite_reward_custom_weights(self) -> None:
        """Verify configurable weights can be applied."""
        custom_weights = MultiDomainRewardWeights(w_air=2.0, w_ground=0.5, w_sea=0.0, w_mission=1.0)
        total = compute_composite_reward(
            r_air=1.0,
            r_ground=2.0,
            r_sea=5.0,
            r_mission=1.0,
            weights=custom_weights,
        )
        # 2.0*1.0 + 0.5*2.0 + 0.0*5.0 + 1.0*1.0 = 2.0 + 1.0 + 0.0 + 1.0 = 4.0
        assert pytest.approx(total, 1e-5) == 4.0


class TestGeometryRoutines:
    """Test suite for spatial and angular geometry functions."""

    def test_normalize_angle(self) -> None:
        """Verify angle normalization into [0, 2pi)."""
        assert normalize_angle(0.0) == 0.0
        assert pytest.approx(normalize_angle(2 * math.pi), 1e-7) == 0.0
        assert pytest.approx(normalize_angle(3 * math.pi), 1e-7) == math.pi
        assert pytest.approx(normalize_angle(-math.pi / 2), 1e-7) == 1.5 * math.pi
        assert pytest.approx(normalize_angle(-2 * math.pi), 1e-7) == 0.0
        assert pytest.approx(normalize_angle(4 * math.pi), 1e-7) == 0.0

    def test_compute_distance(self) -> None:
        """Verify Euclidean distance calculation and zero distance for same position."""
        pos = (100.0, 200.0, 300.0)
        assert compute_distance(pos, pos) == 0.0

        p1 = (0.0, 0.0)
        p2 = (3.0, 4.0)
        assert pytest.approx(compute_distance(p1, p2), 1e-6) == 5.0

        p3d_1 = (0.0, 0.0, 0.0)
        p3d_2 = (1.0, 2.0, 2.0)
        assert pytest.approx(compute_distance(p3d_1, p3d_2), 1e-6) == 3.0

    def test_compute_ATA(self) -> None:
        """Verify ATA returns 0 when observer points directly at target."""
        # Observer at origin pointing along +X (heading = 0)
        obs_pos = (0.0, 0.0)
        obs_heading = 0.0

        # Target directly in front along +X
        target_front = (100.0, 0.0)
        assert pytest.approx(compute_ATA(obs_pos, obs_heading, target_front), 1e-7) == 0.0

        # Target directly to the left (+Y) -> 90 degrees
        target_left = (0.0, 50.0)
        assert pytest.approx(compute_ATA(obs_pos, obs_heading, target_left), 1e-7) == math.pi / 2

        # Target directly behind (-X) -> 180 degrees
        target_behind = (-100.0, 0.0)
        assert pytest.approx(compute_ATA(obs_pos, obs_heading, target_behind), 1e-7) == math.pi

    def test_compute_AA(self) -> None:
        """Verify Aspect Angle calculation (0 when observer is behind target)."""
        obs_pos = (0.0, 0.0)
        obs_heading = 0.0
        target_pos = (100.0, 0.0)

        # Target is flying along +X (heading 0.0); observer is at (0, 0) directly behind target
        target_heading_tail = 0.0
        assert pytest.approx(compute_AA(obs_pos, obs_heading, target_pos, target_heading_tail), 1e-7) == 0.0

        # Target is flying along -X (heading pi); target flying toward observer (head on)
        target_heading_head_on = math.pi
        assert pytest.approx(compute_AA(obs_pos, obs_heading, target_pos, target_heading_head_on), 1e-7) == math.pi

    def test_compute_heading_off(self) -> None:
        """Verify heading-off angle calculation."""
        obs_pos = (0.0, 0.0)
        obs_heading = 0.0
        target_pos = (50.0, 50.0)  # LOS is 45 degrees
        assert pytest.approx(compute_heading_off(obs_heading, target_pos, obs_pos), 1e-7) == math.pi / 4
