"""SQLite connection and transaction management.

Provides context managers for safe connection scoping and ACID transaction
rollbacks, strict foreign key enforcement, WAL mode, and dict-like row access.
"""

from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterator, Sequence

from src.database.config import DB_PATH


class Database:
    """SQLite connection and transaction manager.

    Supports both file-based persistent databases and in-memory databases
    for isolated unit testing.
    """

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        self._is_memory = (db_path == ":memory:")

        if not self._is_memory:
            target_dir = Path(db_path).parent
            target_dir.mkdir(parents=True, exist_ok=True)
            self._memory_conn: sqlite3.Connection | None = None
        else:
            # Keep a persistent connection for in-memory database lifetime
            self._memory_conn = sqlite3.connect(":memory:")
            self._memory_conn.row_factory = sqlite3.Row
            self._configure_connection(self._memory_conn)

    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        """Apply required performance and integrity PRAGMAs."""
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        if not self._is_memory:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield an active SQLite connection with PRAGMAs configured."""
        if self._is_memory and self._memory_conn is not None:
            yield self._memory_conn
        else:
            conn = sqlite3.connect(self.db_path)
            try:
                self._configure_connection(conn)
                yield conn
            finally:
                conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection inside an explicit ACID transaction.

        Commits changes upon clean exit, or rolls back changes if an exception occurs.
        """
        with self.connection() as conn:
            try:
                # Python sqlite3 manages transactions automatically with 'with conn:'
                with conn:
                    yield conn
            except Exception:
                raise

    def execute(self, query: str, params: tuple[Any, ...] | list[Any] = ()) -> int:
        """Execute a single query within a transaction.

        Returns:
            The cursor's lastrowid if non-zero, else rowcount.
        """
        with self.transaction() as conn:
            cursor = conn.execute(query, params)
            return int(cursor.lastrowid if cursor.lastrowid else cursor.rowcount)

    def execute_many(self, query: str, params_list: Sequence[Sequence[Any]]) -> None:
        """Execute a batch of queries within a transaction."""
        if not params_list:
            return
        with self.transaction() as conn:
            conn.executemany(query, params_list)

    def fetch_one(self, query: str, params: tuple[Any, ...] | list[Any] = ()) -> dict[str, Any] | None:
        """Fetch a single row as a dictionary, or None if no result."""
        with self.connection() as conn:
            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row is not None else None

    def fetch_all(self, query: str, params: tuple[Any, ...] | list[Any] = ()) -> list[dict[str, Any]]:
        """Fetch all query results as a list of dictionaries."""
        with self.connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def close(self) -> None:
        """Close any persistent in-memory database connection."""
        if self._memory_conn is not None:
            self._memory_conn.close()
            self._memory_conn = None
