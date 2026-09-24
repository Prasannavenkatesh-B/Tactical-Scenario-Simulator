"""Tests for database schema creation, constraints, and PRAGMAs."""

import sqlite3
from pathlib import Path
import pytest

from src.database.db import Database
from src.database.migrations import SchemaManager


def test_tables_exist(db: Database) -> None:
    """Verify all 6 core tables are properly created."""
    expected_tables = {
        "schema_version",
        "scenarios",
        "agent_params",
        "model_checkpoints",
        "training_runs",
        "metrics_history",
    }
    rows = db.fetch_all("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = {r["name"] for r in rows}
    for t in expected_tables:
        assert t in table_names, f"Table {t} missing from schema"


def test_indexes_exist(db: Database) -> None:
    """Verify performance indexes exist."""
    expected_indexes = {
        "idx_metrics_run_iter",
        "idx_checkpoints_policy",
        "idx_scenarios_level",
    }
    rows = db.fetch_all("SELECT name FROM sqlite_master WHERE type='index';")
    index_names = {r["name"] for r in rows}
    for idx in expected_indexes:
        assert idx in index_names, f"Index {idx} missing from database"


def test_foreign_keys_enforced(db: Database) -> None:
    """Verify foreign key constraint rejects child records with missing parent."""
    # Attempting to insert into metrics_history with non-existent run_id=999
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            """
            INSERT INTO metrics_history (run_id, iteration, loss)
            VALUES (?, ?, ?);
            """,
            (999, 1, 0.5),
        )


def test_schema_version_record(db: Database) -> None:
    """Verify schema_version table records current version 1."""
    row = db.fetch_one("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1;")
    assert row is not None
    assert row["version"] == 1


def test_file_db_wal_mode(tmp_path: Path) -> None:
    """Verify file-based databases enable WAL mode and foreign keys."""
    db_file = tmp_path / "test_wal.db"
    db = Database(str(db_file))
    SchemaManager(db).apply_migrations()

    try:
        journal_row = db.fetch_one("PRAGMA journal_mode;")
        assert journal_row is not None
        assert journal_row["journal_mode"].lower() == "wal"

        fk_row = db.fetch_one("PRAGMA foreign_keys;")
        assert fk_row is not None
        assert fk_row["foreign_keys"] == 1
    finally:
        db.close()
