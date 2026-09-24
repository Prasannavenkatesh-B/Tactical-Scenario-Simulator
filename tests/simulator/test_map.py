"""Unit tests for Map2D, procedural terrain generation, boundaries, and team zones."""

import numpy as np
import pytest

from src.core.interfaces import TeamSide
from src.simulator.config import DEFAULT_MAP_SIZE_KM, MAX_ALTITUDE_KM, MIN_ALTITUDE_KM
from src.simulator.map import Map2D


class TestMap2D:
    """Test suite for Map2D spatial representation."""

    def test_map_bounds(self) -> None:
        """Verify default bounds configuration."""
        m = Map2D(size_km=30.0)
        assert m.x_bounds == (0.0, 30.0)
        assert m.y_bounds == (0.0, 30.0)
        assert m.z_bounds == (MIN_ALTITUDE_KM, MAX_ALTITUDE_KM)

    def test_in_and_out_of_bounds(self) -> None:
        """Verify is_in_bounds for valid and invalid coordinates."""
        m = Map2D(size_km=30.0)
        # Inside
        assert m.is_in_bounds((15.0, 15.0, 5.0))
        assert m.is_in_bounds((0.0, 0.0, 0.0))
        assert m.is_in_bounds((30.0, 30.0, 15.0))

        # Outside X/Y
        assert not m.is_in_bounds((-1.0, 15.0, 5.0))
        assert not m.is_in_bounds((35.0, 15.0, 5.0))
        assert not m.is_in_bounds((15.0, -0.5, 5.0))
        assert not m.is_in_bounds((15.0, 31.0, 5.0))

        # Outside Z
        assert not m.is_in_bounds((15.0, 15.0, -0.5))
        assert not m.is_in_bounds((15.0, 15.0, 20.0))

    def test_terrain_elevation_determinism(self) -> None:
        """Verify identical terrain seed produces exact same elevation grid."""
        m1 = Map2D(size_km=30.0, terrain_seed=42)
        m2 = Map2D(size_km=30.0, terrain_seed=42)
        m_diff = Map2D(size_km=30.0, terrain_seed=999)

        np.testing.assert_array_equal(m1.terrain_grid, m2.terrain_grid)
        assert not np.array_equal(m1.terrain_grid, m_diff.terrain_grid)

        # Elevation at specific point
        elev1 = m1.elevation_at(10.5, 20.3)
        elev2 = m2.elevation_at(10.5, 20.3)
        assert pytest.approx(elev1, 1e-6) == elev2

    def test_team_zone_sampling(self) -> None:
        """Verify sampled positions lie strictly within the assigned team halves."""
        m = Map2D(size_km=40.0)
        rng = np.random.default_rng(123)

        # Default: Blue is left half [0, 20], Red is right half [20, 40]
        m.assign_team_zones(randomize=False)

        for _ in range(50):
            bx, by, bz = m.sample_position(TeamSide.BLUE, rng)
            assert 0.0 <= bx <= 20.0
            assert 0.0 <= by <= 40.0
            assert bz >= 0.0

            rx, ry, rz = m.sample_position(TeamSide.RED, rng)
            assert 20.0 <= rx <= 40.0
            assert 0.0 <= ry <= 40.0
            assert rz >= 0.0

    def test_set_borders(self) -> None:
        """Verify dynamic border adjustment."""
        m = Map2D(size_km=30.0)
        m.set_borders(5.0, 25.0, 10.0, 20.0)
        assert m.x_bounds == (5.0, 25.0)
        assert m.y_bounds == (10.0, 20.0)

        assert not m.is_in_bounds((2.0, 15.0, 5.0))
        assert m.is_in_bounds((10.0, 15.0, 5.0))

        with pytest.raises(ValueError):
            m.set_borders(20.0, 10.0, 0.0, 10.0)
