"""Tests for ScenarioRepository CRUD and ScenarioConfig conversions."""

import sqlite3
import pytest

from src.core.interfaces import DomainType
from src.database.db import Database
from src.database.scenario_repo import ScenarioRepository
from src.simulator.scenarios import ScenarioConfig


def test_create_and_get_scenario(db: Database) -> None:
    """Verify creating a scenario stores and parses JSON entities properly."""
    repo = ScenarioRepository(db)
    blue_entities = [{"domain": "AIR", "type": "AC1", "count": 1}]
    red_entities = [{"domain": "AIR", "type": "AC2", "count": 1}]

    sc_id = repo.create(
        name="Dogfight_Alpha",
        level=1,
        map_size_km=30.0,
        episode_horizon=200,
        blue_entities=blue_entities,
        red_entities=red_entities,
        weather="clear",
        notes="Test scenario",
    )
    assert sc_id > 0

    sc = repo.get(sc_id)
    assert sc is not None
    assert sc["name"] == "Dogfight_Alpha"
    assert sc["level"] == 1
    assert sc["map_size_km"] == 30.0
    assert sc["episode_horizon"] == 200
    assert sc["weather"] == "clear"
    assert sc["notes"] == "Test scenario"
    assert isinstance(sc["blue_entities"], list)
    assert sc["blue_entities"] == blue_entities
    assert sc["red_entities"] == red_entities


def test_get_by_name(db: Database) -> None:
    """Verify retrieval by unique scenario name."""
    repo = ScenarioRepository(db)
    repo.create(
        name="Named_Scenario",
        level=2,
        map_size_km=30.0,
        episode_horizon=200,
        blue_entities=[],
        red_entities=[],
    )

    sc = repo.get_by_name("Named_Scenario")
    assert sc is not None
    assert sc["level"] == 2


def test_get_nonexistent(db: Database) -> None:
    """Verify querying nonexistent scenarios returns None."""
    repo = ScenarioRepository(db)
    assert repo.get(9999) is None
    assert repo.get_by_name("Nonexistent") is None


def test_list_by_level(db: Database) -> None:
    """Verify filtering scenarios by curriculum level."""
    repo = ScenarioRepository(db)
    repo.create("L1_A", 1, 30.0, 200, [], [])
    repo.create("L1_B", 1, 30.0, 200, [], [])
    repo.create("L2_A", 2, 30.0, 200, [], [])

    l1_scenarios = repo.list_by_level(1)
    assert len(l1_scenarios) == 2
    assert {s["name"] for s in l1_scenarios} == {"L1_A", "L1_B"}

    l2_scenarios = repo.list_by_level(2)
    assert len(l2_scenarios) == 1
    assert l2_scenarios[0]["name"] == "L2_A"

    assert repo.list_by_level(3) == []


def test_update_scenario(db: Database) -> None:
    """Verify modifying scenario attributes."""
    repo = ScenarioRepository(db)
    sc_id = repo.create("Update_Target", 1, 30.0, 200, [], [])

    new_blue = [{"domain": "GROUND", "type": "SAM", "count": 2}]
    repo.update(sc_id, map_size_km=45.0, weather="rain", blue_entities=new_blue)

    updated = repo.get(sc_id)
    assert updated is not None
    assert updated["map_size_km"] == 45.0
    assert updated["weather"] == "rain"
    assert updated["blue_entities"] == new_blue


def test_delete_scenario(db: Database) -> None:
    """Verify deleting a scenario."""
    repo = ScenarioRepository(db)
    sc_id = repo.create("To_Delete", 1, 30.0, 200, [], [])
    assert repo.get(sc_id) is not None

    repo.delete(sc_id)
    assert repo.get(sc_id) is None


def test_to_scenario_config(db: Database) -> None:
    """Verify converting a stored scenario to a simulator ScenarioConfig."""
    repo = ScenarioRepository(db)
    blue = [{"domain": "AIR", "type": "AC1", "count": 1}]
    red = [{"domain": "GROUND", "type": "SAM", "count": 1}]

    sc_id = repo.create("Conversion_Test", 4, 30.0, 250, blue, red, weather="fog")
    config = repo.to_scenario_config(sc_id)

    assert isinstance(config, ScenarioConfig)
    assert config.name == "Conversion_Test"
    assert config.map_size_km == 30.0
    assert config.episode_horizon == 250
    assert config.weather == "fog"
    assert config.blue_entities[0]["domain"] == DomainType.AIR
    assert config.red_entities[0]["domain"] == DomainType.GROUND


def test_to_scenario_config_missing_raises(db: Database) -> None:
    """Verify converting nonexistent scenario ID raises KeyError."""
    repo = ScenarioRepository(db)
    with pytest.raises(KeyError):
        repo.to_scenario_config(9999)


def test_unique_scenario_name_constraint(db: Database) -> None:
    """Verify unique constraint on scenario name."""
    repo = ScenarioRepository(db)
    repo.create("Unique_Name", 1, 30.0, 200, [], [])
    with pytest.raises(sqlite3.IntegrityError):
        repo.create("Unique_Name", 1, 30.0, 200, [], [])
