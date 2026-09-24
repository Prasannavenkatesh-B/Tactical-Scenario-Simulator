"""Tests for CheckpointRepository operations."""

import sqlite3
import pytest

from src.database.checkpoint_repo import CheckpointRepository
from src.database.db import Database


def test_register_and_get(db: Database) -> None:
    """Verify registering a checkpoint and retrieving by ID."""
    repo = CheckpointRepository(db)
    ckpt_id = repo.register(
        path="checkpoints/air_iter_00100.pt",
        policy_name="AirFightPolicy",
        stage="stage_a",
        iteration=100,
        win_rate=0.75,
        kill_death_ratio=2.5,
        param_count=235000,
        file_size_bytes=1048576,
        notes="First milestone",
    )
    assert ckpt_id > 0

    ckpt = repo.get(ckpt_id)
    assert ckpt is not None
    assert ckpt["path"] == "checkpoints/air_iter_00100.pt"
    assert ckpt["policy_name"] == "AirFightPolicy"
    assert ckpt["stage"] == "stage_a"
    assert ckpt["iteration"] == 100
    assert ckpt["win_rate"] == 0.75
    assert ckpt["param_count"] == 235000


def test_get_by_path(db: Database) -> None:
    """Verify retrieving checkpoint by unique file path."""
    repo = CheckpointRepository(db)
    repo.register(
        path="checkpoints/commander_0050.pt",
        policy_name="CommanderPolicy",
        stage="stage_b",
        iteration=50,
    )

    ckpt = repo.get_by_path("checkpoints/commander_0050.pt")
    assert ckpt is not None
    assert ckpt["policy_name"] == "CommanderPolicy"
    assert repo.get_by_path("nonexistent.pt") is None


def test_list_by_policy(db: Database) -> None:
    """Verify listing checkpoints by policy in descending order of iteration."""
    repo = CheckpointRepository(db)
    repo.register("ckpt_10.pt", "AirFight", "stage_a", 10)
    repo.register("ckpt_30.pt", "AirFight", "stage_a", 30)
    repo.register("ckpt_20.pt", "AirFight", "stage_a", 20)
    repo.register("ckpt_ground.pt", "GroundEngage", "stage_a", 15)

    ckpts = repo.list_by_policy("AirFight")
    assert len(ckpts) == 3
    assert [c["iteration"] for c in ckpts] == [30, 20, 10]


def test_list_latest(db: Database) -> None:
    """Verify listing the latest checkpoint per policy."""
    repo = CheckpointRepository(db)
    repo.register("air_10.pt", "AirFight", "stage_a", 10)
    repo.register("air_20.pt", "AirFight", "stage_a", 20)
    repo.register("ground_5.pt", "GroundEngage", "stage_a", 5)

    latest = repo.list_latest()
    latest_map = {c["policy_name"]: c["iteration"] for c in latest}
    assert latest_map["AirFight"] == 20
    assert latest_map["GroundEngage"] == 5


def test_delete_checkpoint(db: Database) -> None:
    """Verify deleting a checkpoint."""
    repo = CheckpointRepository(db)
    ckpt_id = repo.register("del.pt", "SeaEngage", "stage_a", 1)
    assert repo.get(ckpt_id) is not None

    repo.delete(ckpt_id)
    assert repo.get(ckpt_id) is None


def test_duplicate_path_raises(db: Database) -> None:
    """Verify unique constraint on checkpoint file path."""
    repo = CheckpointRepository(db)
    repo.register("duplicate.pt", "AirFight", "stage_a", 10)
    with pytest.raises(sqlite3.IntegrityError):
        repo.register("duplicate.pt", "AirEscape", "stage_a", 20)
