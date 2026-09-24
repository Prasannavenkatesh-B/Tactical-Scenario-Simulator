"""Schema version tracking and database migration manager."""

import sqlite3
from typing import Any

from src.database.db import Database
from src.database.schema import SCHEMA_V1


class VersionInt(int):
    """Integer that can also be called like a method for full API compatibility."""

    def __call__(self) -> int:
        return int(self)


class SchemaManager:
    """Manages schema migrations and version tracking."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def _get_version(self) -> int:
        """Internal helper to read current version from database."""
        try:
            row = self.db.fetch_one(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version';"
            )
            if not row:
                return 0

            v_row = self.db.fetch_one("SELECT MAX(version) AS max_v FROM schema_version;")
            if v_row and v_row["max_v"] is not None:
                return int(v_row["max_v"])
            return 0
        except sqlite3.OperationalError:
            return 0

    @property
    def current_version(self) -> VersionInt:
        """Read the currently applied schema version.

        Returns a VersionInt which behaves as an integer and can also be called
        as a function (e.g. current_version() or current_version == 1).
        """
        return VersionInt(self._get_version())

    def apply_migrations(self) -> None:
        """Apply all outstanding migrations idempotently.

        Safe to invoke repeatedly without side effects or duplicate errors.
        """
        cur_v = self._get_version()

        if cur_v < 1:
            with self.db.transaction() as conn:
                for statement in SCHEMA_V1:
                    stmt = statement.strip()
                    if stmt:
                        conn.execute(stmt)

                # Record version 1 if not already recorded
                row = conn.execute("SELECT 1 FROM schema_version WHERE version = 1;").fetchone()
                if not row:
                    conn.execute("INSERT INTO schema_version (version) VALUES (1);")
