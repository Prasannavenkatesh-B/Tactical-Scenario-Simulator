"""Tests for ScenarioIO database serialization and persistence edge cases."""

from src.database.db import Database
from src.simulator.scenarios import generate_scenario
from src.ui.scenario_io import ScenarioIO
from src.ui.state import UIState


def test_save_load_round_trip(test_db: Database, ui_state: UIState) -> None:
    """Verify saving a scenario and loading it round-trips entity types and locations."""
    io = ScenarioIO(test_db)
    sc_id = io.save(ui_state.scenario_config, "Roundtrip_Scenario", ui_state.entities)
    assert sc_id > 0

    loaded_cfg, raw_row = io.load(sc_id)
    assert loaded_cfg.name == "Roundtrip_Scenario"
    assert loaded_cfg.map_size_km == ui_state.scenario_config.map_size_km
    assert len(raw_row["blue_entities"]) + len(raw_row["red_entities"]) == len(ui_state.entities)


def test_save_with_no_db_returns_negative_one(ui_state: UIState) -> None:
    """Verify attempting to save when database is None returns -1 safely."""
    io_nodb = ScenarioIO(None)
    res = io_nodb.save(ui_state.scenario_config, "No_DB_Test", ui_state.entities)
    assert res == -1


def test_list_scenarios_empty_when_no_db() -> None:
    """Verify listing scenarios with disabled DB returns an empty list."""
    io_nodb = ScenarioIO(None)
    assert io_nodb.list_scenarios() == []


def test_delete_removes_scenario(test_db: Database, ui_state: UIState) -> None:
    """Verify deleting a scenario purges it from database."""
    io = ScenarioIO(test_db)
    sc_id = io.save(ui_state.scenario_config, "Delete_Scenario_Test", ui_state.entities)
    assert sc_id > 0

    io.delete(sc_id)
    scenarios = io.list_scenarios()
    ids = [s["scenario_id"] for s in scenarios]
    assert sc_id not in ids
