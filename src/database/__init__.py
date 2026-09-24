"""Database layer for the Tactical Simulation System (TSS).

Provides SQLite persistent and in-memory database management, schema migrations,
and domain repositories for scenarios, agent parameters, model checkpoints,
training runs, and metrics history.
"""

from src.database.agent_repo import AgentRepository
from src.database.checkpoint_repo import CheckpointRepository
from src.database.config import DB_PATH, SCHEMA_VERSION
from src.database.db import Database
from src.database.metrics_repo import MetricsRepository
from src.database.migrations import SchemaManager
from src.database.run_repo import RunRepository
from src.database.scenario_repo import ScenarioRepository
from src.database.seed import seed_all, seed_default_agent_params, seed_default_scenarios

__all__ = [
    "AgentRepository",
    "CheckpointRepository",
    "DB_PATH",
    "Database",
    "MetricsRepository",
    "RunRepository",
    "ScenarioRepository",
    "SCHEMA_VERSION",
    "SchemaManager",
    "seed_all",
    "seed_default_agent_params",
    "seed_default_scenarios",
]
