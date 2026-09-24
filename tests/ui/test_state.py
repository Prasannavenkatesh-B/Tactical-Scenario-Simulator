"""Tests for UIState simulation management, entity lifecycle, and persistence."""

from src.core.interfaces import TeamSide
from src.ui.config import TRAIL_LENGTH
from src.ui.state import UIState


def test_place_entity_adds_to_state(ui_state: UIState) -> None:
    """Verify placing an entity via UIState adds it to active environment and trajectories."""
    init_count = len(ui_state.entities)
    ui_state.pending_placement = {"domain": "air", "variant": "AC1", "team": TeamSide.BLUE}

    eid = ui_state.place_entity(15.0, 15.0)
    assert eid is not None
    assert eid in ui_state.entities
    assert len(ui_state.entities) == init_count + 1
    assert eid in ui_state.trajectories
    assert ui_state.selected_entity_id == eid


def test_delete_entity_removes_from_state(ui_state: UIState) -> None:
    """Verify deleting an entity cleans up all references in state and env."""
    eid = list(ui_state.entities.keys())[0]
    ui_state.select_entity(eid)
    ui_state.manual_control_entity = eid

    ui_state.delete_entity(eid)
    assert eid not in ui_state.entities
    assert eid not in ui_state.env.entities
    assert eid not in ui_state.trajectories
    assert ui_state.selected_entity_id is None
    assert ui_state.manual_control_entity is None


def test_step_simulation_advances_step_counter(ui_state: UIState) -> None:
    """Verify stepping the simulation updates steps, observations, and trajectories."""
    assert ui_state.current_step == 0
    ui_state.step_simulation()
    assert ui_state.current_step == 1
    assert len(ui_state.observations) > 0


def test_trajectory_length_bounded(ui_state: UIState) -> None:
    """Verify trajectory history does not exceed TRAIL_LENGTH."""
    for _ in range(TRAIL_LENGTH + 20):
        ui_state.step_simulation()

    for eid, trail in ui_state.trajectories.items():
        assert len(trail) <= TRAIL_LENGTH


def test_reset_simulation_restores_initial_step(ui_state: UIState) -> None:
    """Verify reset resets step count to 0 and clears transient effects."""
    ui_state.step_simulation()
    ui_state.step_simulation()
    assert ui_state.current_step == 2

    ui_state.reset_simulation()
    assert ui_state.current_step == 0
    assert len(ui_state.explosions) == 0


def test_save_and_load_scenario(ui_state: UIState) -> None:
    """Verify saving a scenario to DB returns a valid positive ID and can be loaded."""
    sc_id = ui_state.save_scenario("Test_Interactive_Scenario")
    assert sc_id > 0

    ui_state.load_scenario(sc_id)
    assert ui_state.scenario_config.name == "Test_Interactive_Scenario"
