"""Unit tests for action dataclasses and bidirectional array conversions."""

import numpy as np
import pytest

from src.core.actions import (
    AirAction,
    CommanderAction,
    GroundAction,
    SeaAction,
)


class TestAirAction:
    """Test suite for AirAction dataclass."""

    def test_round_trip(self) -> None:
        """Verify to_array -> from_array reconstructs the identical object."""
        test_cases = [
            AirAction(heading_delta=0.0, velocity_cmd=0, fire_cannon=0, fire_rocket=0),
            AirAction(heading_delta=45.0, velocity_cmd=5, fire_cannon=1, fire_rocket=0),
            AirAction(heading_delta=-75.0, velocity_cmd=8, fire_cannon=0, fire_rocket=1),
            AirAction(heading_delta=90.0, velocity_cmd=4, fire_cannon=1, fire_rocket=1),
            AirAction(heading_delta=-90.0, velocity_cmd=2, fire_cannon=0, fire_rocket=0),
        ]
        for original in test_cases:
            arr = original.to_array()
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (4,)
            reconstructed = AirAction.from_array(arr)
            assert reconstructed == original

    def test_discrete_step_mapping(self) -> None:
        """Verify heading delta maps correctly to discrete steps {-6..+6}."""
        assert AirAction(heading_delta=0.0, velocity_cmd=0, fire_cannon=0, fire_rocket=0).discrete_heading_step == 0
        assert AirAction(heading_delta=15.0, velocity_cmd=0, fire_cannon=0, fire_rocket=0).discrete_heading_step == 1
        assert AirAction(heading_delta=-15.0, velocity_cmd=0, fire_cannon=0, fire_rocket=0).discrete_heading_step == -1
        assert AirAction(heading_delta=90.0, velocity_cmd=0, fire_cannon=0, fire_rocket=0).discrete_heading_step == 6
        assert AirAction(heading_delta=-90.0, velocity_cmd=0, fire_cannon=0, fire_rocket=0).discrete_heading_step == -6
        # Intermediate continuous angles map to nearest step
        assert AirAction(heading_delta=22.0, velocity_cmd=0, fire_cannon=0, fire_rocket=0).discrete_heading_step == 1
        assert AirAction(heading_delta=23.0, velocity_cmd=0, fire_cannon=0, fire_rocket=0).discrete_heading_step == 2

    def test_from_discrete(self) -> None:
        """Verify constructor from discrete step produces correct continuous angle."""
        act = AirAction.from_discrete(heading_step=3, velocity_cmd=6, fire_cannon=1, fire_rocket=0)
        assert act.heading_delta == 45.0
        assert act.velocity_cmd == 6
        assert act.fire_cannon == 1
        assert act.fire_rocket == 0

    def test_clamping(self) -> None:
        """Verify out-of-range action components are clamped."""
        act = AirAction(heading_delta=180.0, velocity_cmd=20, fire_cannon=5, fire_rocket=-2)
        assert act.heading_delta == 90.0
        assert act.velocity_cmd == 8
        assert act.fire_cannon == 1
        assert act.fire_rocket == 0

        act_low = AirAction(heading_delta=-180.0, velocity_cmd=-5, fire_cannon=0, fire_rocket=0)
        assert act_low.heading_delta == -90.0
        assert act_low.velocity_cmd == 0


class TestGroundAction:
    """Test suite for GroundAction dataclass."""

    def test_round_trip(self) -> None:
        """Verify to_array -> from_array reconstructs the identical object."""
        test_cases = [
            GroundAction(heading_delta=0.0, velocity_cmd=0, weapon_select=0, fire=0),
            GroundAction(heading_delta=30.0, velocity_cmd=3, weapon_select=1, fire=1),
            GroundAction(heading_delta=-45.0, velocity_cmd=5, weapon_select=2, fire=1),
        ]
        for original in test_cases:
            arr = original.to_array()
            assert arr.shape == (4,)
            reconstructed = GroundAction.from_array(arr)
            assert reconstructed == original

    def test_clamping(self) -> None:
        """Verify out-of-range values for GroundAction are bounded properly."""
        act = GroundAction(heading_delta=90.0, velocity_cmd=10, weapon_select=5, fire=3)
        assert act.heading_delta == 45.0
        assert act.velocity_cmd == 5
        assert act.weapon_select == 2
        assert act.fire == 1

        act_low = GroundAction(heading_delta=-90.0, velocity_cmd=-1, weapon_select=-1, fire=-1)
        assert act_low.heading_delta == -45.0
        assert act_low.velocity_cmd == 0
        assert act_low.weapon_select == 0
        assert act_low.fire == 0


class TestSeaAction:
    """Test suite for SeaAction dataclass."""

    def test_round_trip(self) -> None:
        """Verify to_array -> from_array reconstructs the identical object."""
        test_cases = [
            SeaAction(heading_delta=0.0, velocity_cmd=0, weapon_select=0, fire=0),
            SeaAction(heading_delta=25.0, velocity_cmd=4, weapon_select=1, fire=1),
            SeaAction(heading_delta=-30.0, velocity_cmd=5, weapon_select=0, fire=1),
        ]
        for original in test_cases:
            arr = original.to_array()
            assert arr.shape == (4,)
            reconstructed = SeaAction.from_array(arr)
            assert reconstructed == original

    def test_clamping(self) -> None:
        """Verify out-of-range values for SeaAction are bounded properly."""
        act = SeaAction(heading_delta=60.0, velocity_cmd=10, weapon_select=3, fire=2)
        assert act.heading_delta == 30.0
        assert act.velocity_cmd == 5
        assert act.weapon_select == 1
        assert act.fire == 1

        act_low = SeaAction(heading_delta=-60.0, velocity_cmd=-2, weapon_select=-1, fire=-1)
        assert act_low.heading_delta == -30.0
        assert act_low.velocity_cmd == 0
        assert act_low.weapon_select == 0
        assert act_low.fire == 0


class TestCommanderAction:
    """Test suite for CommanderAction dataclass."""

    def test_round_trip(self) -> None:
        """Verify to_array -> from_array reconstructs the identical object."""
        for action_idx in range(4):
            original = CommanderAction(action=action_idx)
            arr = original.to_array()
            assert arr.shape == (1,)
            reconstructed = CommanderAction.from_array(arr)
            assert reconstructed == original

    def test_clamping(self) -> None:
        """Verify action index is clamped to {0..3}."""
        assert CommanderAction(action=-1).action == 0
        assert CommanderAction(action=10).action == 3
        assert CommanderAction(action=2).action == 2
