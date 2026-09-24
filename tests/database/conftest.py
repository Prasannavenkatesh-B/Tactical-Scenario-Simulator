"""Pytest fixtures for database testing."""

from typing import Generator
import pytest

from src.database.db import Database
from src.database.migrations import SchemaManager


@pytest.fixture
def db() -> Generator[Database, None, None]:
    """Yield an isolated, migrated in-memory Database instance."""
    database = Database(db_path=":memory:")
    SchemaManager(database).apply_migrations()
    try:
        yield database
    finally:
        database.close()


@pytest.fixture
def unmigrated_db() -> Generator[Database, None, None]:
    """Yield an empty, unmigrated in-memory Database instance."""
    database = Database(db_path=":memory:")
    try:
        yield database
    finally:
        database.close()
