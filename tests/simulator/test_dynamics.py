"""Unit tests for kinematic dynamic integrations across domains."""

import math
import numpy as np
import pytest

from src.core.actions import AirAction, GroundAction, SeaAction
from src.core.interfaces import TeamSide
from src.simulator.config import KNOTS_TO_KMH
from src.simulator.dynamics import integrate_air, integrate_ground, integrate_sea
from src.simulator.entities.air import AirEntity
from src.simulator.entities.ground import GroundEntity
from src.simulator.entities.sea import SeaEntity
from src.simulator.map import Map2D


class TestDynamics:
    """Test suite for kinematic integration routines."""

    def test_air_turn_rate_limit(self) -> None:
        """Verify AirEntity turn rate is bounded by max angular velocity."""
        # AC1 max turn rate is 5.0 deg/s
        ac = AirEntity("ac", TeamSide.BLUE, "AC1", heading=0.0)

        # Commanded large heading change: 90 degrees
        act = AirAction(heading_delta=90.0, velocity_cmd=4, fire_cannon=0, fire_rocket=0)
        dt = 0.1  # 0.1s -> max turn = 5.0 * 0.1 = 0.5 degrees = 0.0087266 rad
        max_turn_deg = 5.0 * dt

        integrate_air(ac, act, dt)
        assert pytest.approx(math.degrees(ac.heading), 1e-4) == max_turn_deg

    def test_ground_movement_blocked_on_steep_slope(self) -> None:
        """Verify ground entity halts forward translation when terrain slope is excessive."""
        # Create map and artificially insert a sharp cliff
        m = Map2D(size_km=30.0, terrain_seed=42)
        # Create cliff: elevation jumps by 2.0 km across adjacent cells
        m.terrain_grid[10, :] = 0.0
        m.terrain_grid[11, :] = 2.5

        ge = GroundEntity("ge", TeamSide.BLUE, position=(3.0, 3.0, 0.0), heading=math.pi / 2, speed=30.0)
        # Force position right at base of cliff
        ge.position = np.array([3.0, 3.0, 0.0], dtype=np.float64)

        # Mock map with steep slope check
        class MockCliffMap:
            def elevation_at(self, x: float, y: float) -> float:
                return 3.0 if y > 3.01 else 0.0

        mock_map = MockCliffMap()
        act = GroundAction(heading_delta=0.0, velocity_cmd=5, weapon_select=0, fire=0)

        integrate_ground(ge, act, dt=1.0, map_ref=mock_map)  # type: ignore
        # Movement should be blocked and speed zeroed
        assert ge.speed == 0.0
        assert ge.position[1] == 3.0

    def test_sea_turn_rate_reduced_by_sea_state(self) -> None:
        """Verify ship turn rate is lower at higher Beaufort sea states."""
        se_calm = SeaEntity("se_c", TeamSide.BLUE, heading=0.0, sea_state=0.0)
        se_rough = SeaEntity("se_r", TeamSide.BLUE, heading=0.0, sea_state=8.0)

        act = SeaAction(heading_delta=30.0, velocity_cmd=3, weapon_select=0, fire=0)
        dt = 0.5

        integrate_sea(se_calm, act, dt, sea_state=0.0)
        integrate_sea(se_rough, act, dt, sea_state=8.0)

        # Calm sea should turn faster than rough sea
        assert se_calm.heading > se_rough.heading

    def test_air_acceleration_limit(self) -> None:
        """Verify air speed change is bounded by physical acceleration."""
        ac = AirEntity("ac", TeamSide.BLUE, "AC1", speed=300.0 * KNOTS_TO_KMH)
        initial_speed = ac.speed

        # Command max speed (velocity_cmd = 8)
        act = AirAction(heading_delta=0.0, velocity_cmd=8, fire_cannon=0, fire_rocket=0)
        dt = 0.1
        integrate_air(ac, act, dt)

        # Max dv = 50 knots/s * 1.852 * 0.1s = 9.26 km/h
        max_dv = 50.0 * KNOTS_TO_KMH * dt
        assert pytest.approx(ac.speed - initial_speed, 1e-4) == max_dv

    def test_ground_turn_rate_limit(self) -> None:
        """Verify ground entity turn rate is clamped to max_turn_rate."""
        ge = GroundEntity("ge", TeamSide.BLUE, heading=0.0)
        act = GroundAction(heading_delta=45.0, velocity_cmd=2, weapon_select=0, fire=0)
        dt = 0.1
        integrate_ground(ge, act, dt)

        # Max turn = 15 deg/s * 0.1s = 1.5 deg
        max_turn_rad = math.radians(15.0 * dt)
        assert pytest.approx(ge.heading, 1e-5) == max_turn_rad
