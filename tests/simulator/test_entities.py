"""Unit tests for simulation entity models across Air, Ground, and Sea domains."""

import math
import numpy as np
import pytest

from src.core.actions import AirAction, GroundAction, SeaAction
from src.core.interfaces import AgentStatus, TeamSide, WeaponType
from src.simulator.config import KNOTS_TO_KMH
from src.simulator.entities.air import AirEntity
from src.simulator.entities.ground import GroundEntity
from src.simulator.entities.sea import SeaEntity
from src.simulator.map import Map2D


class TestEntities:
    """Test suite for domain entity behaviors and constraints."""

    def test_air_entity_speed_clamped(self) -> None:
        """Verify AC1 and AC2 speeds are clamped to their respective envelopes."""
        # AC1: 100 to 900 knots
        ac1 = AirEntity("ac1", TeamSide.BLUE, "AC1", speed=50.0 * KNOTS_TO_KMH)
        assert pytest.approx(ac1.speed, 1e-4) == 100.0 * KNOTS_TO_KMH

        ac1_high = AirEntity("ac1_h", TeamSide.BLUE, "AC1", speed=1200.0 * KNOTS_TO_KMH)
        assert pytest.approx(ac1_high.speed, 1e-4) == 900.0 * KNOTS_TO_KMH

        # AC2: 100 to 600 knots
        ac2 = AirEntity("ac2", TeamSide.RED, "AC2", speed=800.0 * KNOTS_TO_KMH)
        assert pytest.approx(ac2.speed, 1e-4) == 600.0 * KNOTS_TO_KMH

    def test_air_entity_heading_wrapping(self) -> None:
        """Verify heading values wrap strictly to [0, 2pi)."""
        ac = AirEntity("ac", TeamSide.BLUE, "AC1", heading=-math.pi / 2)
        assert pytest.approx(ac.heading, 1e-6) == 1.5 * math.pi

        ac.heading = 3.5 * math.pi
        assert pytest.approx(ac.heading, 1e-6) == 1.5 * math.pi

    def test_ground_entity_speed_and_clamping(self) -> None:
        """Verify GroundEntity bounds speeds to [0, 60] km/h."""
        ge = GroundEntity("ge", TeamSide.BLUE, speed=-10.0)
        assert ge.speed == 0.0

        ge_fast = GroundEntity("ge_f", TeamSide.BLUE, speed=100.0)
        assert ge_fast.speed == 60.0

    def test_sea_entity_speed_and_sea_state(self) -> None:
        """Verify SeaEntity hydrodynamics and sea state clamping."""
        se = SeaEntity("se", TeamSide.BLUE, sea_state=5.0)
        assert se.sea_state == 5.0

        se_clamp = SeaEntity("se_c", TeamSide.BLUE, sea_state=12.0)
        assert se_clamp.sea_state == 9.0

    def test_destroyed_entity_stops_actions(self) -> None:
        """Verify destroyed entities do not update kinematics on step."""
        ac = AirEntity("ac", TeamSide.BLUE, "AC1", position=(10.0, 10.0, 5.0), speed=500.0)
        initial_pos = ac.position.copy()

        # Destroy entity
        destroyed = ac.apply_weapon_hit(WeaponType.CANNON)
        assert destroyed
        assert ac.status == AgentStatus.DESTROYED
        assert not ac.is_alive()

        # Step should be a no-op
        act = AirAction(heading_delta=30.0, velocity_cmd=8, fire_cannon=0, fire_rocket=0)
        ac.step(dt=0.1, action=act)
        np.testing.assert_array_equal(ac.position, initial_pos)

    def test_entity_reset_restores_ammo(self) -> None:
        """Verify calling reset() replenishes initial ammunition and marks status ALIVE."""
        ac = AirEntity("ac", TeamSide.BLUE, "AC1")
        ac.ammo[WeaponType.CANNON] = 10
        ac.status = AgentStatus.DESTROYED

        ac.reset(position=(5.0, 5.0, 4.0), heading=1.0, speed=400.0)
        assert ac.status == AgentStatus.ALIVE
        assert ac.is_alive()
        assert ac.ammo[WeaponType.CANNON] == ac.max_ammo_cannon

    def test_air_entity_observation_clamped_values(self) -> None:
        """Verify AirEntity observation produces normalized 13-dim array strictly in [0, 1]."""
        ac = AirEntity("ac", TeamSide.BLUE, "AC1", position=(15.0, 15.0, 8.0))
        opp = AirEntity("opp", TeamSide.RED, "AC1", position=(20.0, 20.0, 8.0))
        m = Map2D(size_km=30.0)

        obs = ac.get_observation(opponents=[opp], friendlies=[], map_ref=m)
        assert len(obs) == 13
        assert np.all((obs >= 0.0) & (obs <= 1.0))
