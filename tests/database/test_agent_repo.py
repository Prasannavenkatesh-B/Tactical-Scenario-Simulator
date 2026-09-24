"""Tests for AgentRepository upsert and query functions."""

from src.database.agent_repo import AgentRepository
from src.database.db import Database


def test_upsert_insert(db: Database) -> None:
    """Verify inserting new agent parameters via upsert."""
    repo = AgentRepository(db)
    repo.upsert(
        domain="air",
        variant="AC1",
        max_speed=900.0,
        min_speed=100.0,
        turn_rate_max=5.0,
        sensor_range_km=20.0,
        weapon_range_km=6.0,
        hit_probability=0.70,
        ammo_cannon=200,
        ammo_rocket=5,
    )

    p = repo.get("air", "AC1")
    assert p is not None
    assert p["domain"] == "air"
    assert p["variant"] == "AC1"
    assert p["max_speed"] == 900.0
    assert p["ammo_rocket"] == 5


def test_upsert_update(db: Database) -> None:
    """Verify upserting an existing domain/variant updates existing record without duplicate."""
    repo = AgentRepository(db)
    repo.upsert("air", "AC1", max_speed=900.0, min_speed=100.0)
    # Update max_speed to 950.0
    repo.upsert("air", "AC1", max_speed=950.0, min_speed=120.0)

    p = repo.get("air", "AC1")
    assert p is not None
    assert p["max_speed"] == 950.0
    assert p["min_speed"] == 120.0

    # Ensure still only 1 record
    all_params = repo.list_all()
    assert len(all_params) == 1


def test_get_case_insensitive(db: Database) -> None:
    """Verify get query is normalized regardless of casing."""
    repo = AgentRepository(db)
    repo.upsert("AIR", "ac2", max_speed=600.0)

    p1 = repo.get("air", "AC2")
    p2 = repo.get("AIR", "ac2")
    assert p1 is not None and p2 is not None
    assert p1["param_id"] == p2["param_id"]


def test_list_all(db: Database) -> None:
    """Verify listing all agent parameter entries."""
    repo = AgentRepository(db)
    repo.upsert("air", "AC1")
    repo.upsert("air", "AC2")
    repo.upsert("ground", "DEFAULT")

    items = repo.list_all()
    assert len(items) == 3
    pairs = [(i["domain"], i["variant"]) for i in items]
    assert ("air", "AC1") in pairs
    assert ("air", "AC2") in pairs
    assert ("ground", "DEFAULT") in pairs


def test_delete(db: Database) -> None:
    """Verify deleting an agent parameter entry."""
    repo = AgentRepository(db)
    repo.upsert("sea", "DEFAULT", max_speed=35.0)
    assert repo.get("sea", "DEFAULT") is not None

    repo.delete("sea", "DEFAULT")
    assert repo.get("sea", "DEFAULT") is None
