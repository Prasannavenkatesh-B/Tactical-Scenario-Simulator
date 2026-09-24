"""Database configuration paths and default constants.

Defines database file paths, schema versioning metadata, batch insert sizes,
and JSON serialization formats.
"""

from typing import Final

DB_PATH: Final[str] = "data/tactical_marl.db"
SCHEMA_VERSION: Final[int] = 1
SCHEMA_VERSION_FILE: Final[str] = "configs/schema_version.txt"
BATCH_INSERT_SIZE: Final[int] = 500
JSON_SEPARATORS: Final[tuple[str, str]] = (",", ":")
