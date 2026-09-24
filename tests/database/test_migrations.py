"""Tests for SchemaManager and migration idempotence."""

from pathlib import Path

from src.database.config import SCHEMA_VERSION, SCHEMA_VERSION_FILE
from src.database.db import Database
from src.database.migrations import SchemaManager


def test_fresh_migration(unmigrated_db: Database) -> None:
    """Verify version starts at 0 and increments to SCHEMA_VERSION."""
    mgr = SchemaManager(unmigrated_db)
    assert mgr.current_version == 0
    mgr.apply_migrations()
    assert mgr.current_version == SCHEMA_VERSION


def test_idempotent_migration(unmigrated_db: Database) -> None:
    """Verify applying migrations multiple times succeeds and preserves version."""
    mgr = SchemaManager(unmigrated_db)
    mgr.apply_migrations()
    assert mgr.current_version == 1

    # Second run should be a no-op
    mgr.apply_migrations()
    assert mgr.current_version == 1


def test_schema_version_file_matches() -> None:
    """Verify the version declared in configs/schema_version.txt matches SCHEMA_VERSION."""
    ver_path = Path(SCHEMA_VERSION_FILE)
    assert ver_path.exists()
    content = ver_path.read_text(encoding="utf-8").strip()
    assert int(content) == SCHEMA_VERSION


def test_table_data_preserved_across_reapply(db: Database) -> None:
    """Verify existing rows are not lost when migrations are re-applied."""
    db.execute(
        """
        INSERT INTO scenarios (name, level, map_size_km, episode_horizon, blue_entities, red_entities)
        VALUES ('Preserved_Scenario', 1, 30.0, 200, '[]', '[]');
        """
    )
    row1 = db.fetch_one("SELECT COUNT(*) as cnt FROM scenarios;")
    assert row1 is not None and row1["cnt"] == 1

    mgr = SchemaManager(db)
    mgr.apply_migrations()

    row2 = db.fetch_one("SELECT COUNT(*) as cnt FROM scenarios;")
    assert row2 is not None and row2["cnt"] == 1
