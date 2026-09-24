"""Scenario repository providing CRUD operations and ScenarioConfig conversions."""

import json
from typing import Any

from src.core.interfaces import DomainType
from src.database.config import JSON_SEPARATORS
from src.database.db import Database
from src.simulator.scenarios import ScenarioConfig


class ScenarioRepository:
    """Repository for querying, creating, and modifying simulation scenario records."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def _parse_row(self, row: dict[str, Any] | None) -> dict[str, Any] | None:
        """Parse JSON entity strings in scenario row."""
        if row is None:
            return None
        res = dict(row)
        if isinstance(res.get("blue_entities"), str):
            res["blue_entities"] = json.loads(res["blue_entities"])
        if isinstance(res.get("red_entities"), str):
            res["red_entities"] = json.loads(res["red_entities"])
        return res

    def create(
        self,
        name: str,
        level: int,
        map_size_km: float,
        episode_horizon: int,
        blue_entities: list[dict[str, Any]],
        red_entities: list[dict[str, Any]],
        weather: str = "clear",
        notes: str = "",
    ) -> int:
        """Create a new scenario record in the database."""
        blue_json = json.dumps(blue_entities, separators=JSON_SEPARATORS, default=str)
        red_json = json.dumps(red_entities, separators=JSON_SEPARATORS, default=str)

        query = """
        INSERT INTO scenarios (
            name, level, map_size_km, episode_horizon, weather, blue_entities, red_entities, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (name, level, float(map_size_km), int(episode_horizon), weather, blue_json, red_json, notes)
        return self.db.execute(query, params)

    def get(self, scenario_id: int) -> dict[str, Any] | None:
        """Retrieve a scenario record by ID with entity lists parsed."""
        query = "SELECT * FROM scenarios WHERE scenario_id = ?;"
        row = self.db.fetch_one(query, (scenario_id,))
        return self._parse_row(row)

    def get_by_name(self, name: str) -> dict[str, Any] | None:
        """Retrieve a scenario record by unique name."""
        query = "SELECT * FROM scenarios WHERE name = ?;"
        row = self.db.fetch_one(query, (name,))
        return self._parse_row(row)

    def list_by_level(self, level: int) -> list[dict[str, Any]]:
        """List all scenarios matching a specific curriculum level."""
        query = "SELECT * FROM scenarios WHERE level = ? ORDER BY scenario_id ASC;"
        rows = self.db.fetch_all(query, (level,))
        return [self._parse_row(r) for r in rows if r is not None]  # type: ignore

    def update(self, scenario_id: int, **kwargs: Any) -> None:
        """Update fields of an existing scenario record."""
        if not kwargs:
            return

        set_clauses = []
        params = []

        for key, val in kwargs.items():
            if key in ("blue_entities", "red_entities") and isinstance(val, (list, dict)):
                val = json.dumps(val, separators=JSON_SEPARATORS, default=str)
            set_clauses.append(f"{key} = ?")
            params.append(val)

        params.append(scenario_id)
        query = f"UPDATE scenarios SET {', '.join(set_clauses)} WHERE scenario_id = ?;"
        self.db.execute(query, tuple(params))

    def delete(self, scenario_id: int) -> None:
        """Delete a scenario record by ID."""
        query = "DELETE FROM scenarios WHERE scenario_id = ?;"
        self.db.execute(query, (scenario_id,))

    def to_scenario_config(self, scenario_id: int) -> ScenarioConfig:
        """Convert a stored database scenario row to a simulator ScenarioConfig."""
        row = self.get(scenario_id)
        if row is None:
            raise KeyError(f"Scenario ID {scenario_id} not found.")

        # Ensure entity domain types are handled cleanly
        blue_entities = []
        for e in row["blue_entities"]:
            item = dict(e)
            if isinstance(item.get("domain"), str):
                d_str = item["domain"].split(".")[-1].upper()
                try:
                    item["domain"] = DomainType[d_str]
                except KeyError:
                    pass
            blue_entities.append(item)

        red_entities = []
        for e in row["red_entities"]:
            item = dict(e)
            if isinstance(item.get("domain"), str):
                d_str = item["domain"].split(".")[-1].upper()
                try:
                    item["domain"] = DomainType[d_str]
                except KeyError:
                    pass
            red_entities.append(item)

        return ScenarioConfig(
            name=row["name"],
            map_size_km=float(row["map_size_km"]),
            episode_horizon=int(row["episode_horizon"]),
            blue_entities=blue_entities,
            red_entities=red_entities,
            weather=row.get("weather", "clear"),
            randomize_positions=True,
            randomize_team_zones=True,
        )
