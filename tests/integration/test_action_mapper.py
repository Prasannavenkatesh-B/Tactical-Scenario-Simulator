"""Unit tests for TSSActionMapper translating policy decisions to TSS command packets."""

import pytest

from src.core.actions import AirAction, CommanderAction, GroundAction, SeaAction
from src.integration.tss_action_mapper import TSSActionMapper
from src.integration.tss_observation_mapper import TSSMappingError


@pytest.fixture
def action_mapper(protocol_path: str) -> TSSActionMapper:
    return TSSActionMapper(protocol_path=protocol_path)


def test_air_action_mapping(action_mapper: TSSActionMapper) -> None:
    """AirAction maps to flight controls (turn degrees, speed knots, binary weapons)."""
    action = AirAction(heading_delta=15.0, velocity_cmd=4, fire_cannon=1, fire_rocket=0)
    cmd = action_mapper.from_air_action(action)

    assert "turn" in cmd
    assert cmd["turn"] == 15.0
    assert "set_speed" in cmd
    assert cmd["set_speed"] > 0.0
    assert cmd["fire_cannon"] is True
    assert cmd["fire_rocket"] is False


def test_ground_action_mapping(action_mapper: TSSActionMapper) -> None:
    """GroundAction maps to vehicle steering, throttle, weapon select, and fire."""
    action = GroundAction(heading_delta=-20.0, velocity_cmd=3, weapon_select=1, fire=1)
    cmd = action_mapper.from_ground_action(action)

    assert cmd["turn"] == -20.0
    assert cmd["set_speed"] > 0.0
    assert cmd["weapon_select"] == "missile"
    assert cmd["fire"] is True


def test_sea_action_mapping(action_mapper: TSSActionMapper) -> None:
    """SeaAction maps to rudder angle, engine throttle, weapon selector, and fire."""
    action = SeaAction(heading_delta=10.0, velocity_cmd=2, weapon_select=0, fire=0)
    cmd = action_mapper.from_sea_action(action)

    assert cmd["turn"] == 10.0
    assert cmd["set_speed"] > 0.0
    assert cmd["weapon_select"] == "deck_gun"
    assert cmd["fire"] is False


def test_commander_action_mapping(action_mapper: TSSActionMapper) -> None:
    """CommanderAction maps to tactical directive and target priority index."""
    # 0 = Escape
    cmd_escape = action_mapper.from_commander_action(CommanderAction(action=0))
    assert cmd_escape["activate_policy"] == "escape"
    assert cmd_escape["target_index"] == 0

    # 1 = Fight target 0
    cmd_fight1 = action_mapper.from_commander_action(CommanderAction(action=1))
    assert cmd_fight1["activate_policy"] == "fight"
    assert cmd_fight1["target_index"] == 0

    # 3 = Fight target 2
    cmd_fight3 = action_mapper.from_commander_action(CommanderAction(action=3))
    assert cmd_fight3["activate_policy"] == "fight"
    assert cmd_fight3["target_index"] == 2


def test_excessive_turn_raises_mapping_error(action_mapper: TSSActionMapper) -> None:
    """Turn command exceeding agent-specified max_turn_rate raises TSSMappingError."""
    action = AirAction(heading_delta=60.0, velocity_cmd=4, fire_cannon=0, fire_rocket=0)
    tss_state = {"max_turn_rate": 30.0}  # Aircraft limited to 30 deg/step

    with pytest.raises(TSSMappingError, match="exceeds vehicle maximum turn rate"):
        action_mapper.from_air_action(action, tss_agent_state=tss_state)
