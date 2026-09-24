"""Scenario persistence and database serialization for the interactive UI."""

from typing import Any

from src.core.interfaces import TeamSide
from src.database.db import Database
from src.database.scenario_repo import ScenarioRepository
from src.simulator.entities.air import AirEntity
from src.simulator.entities.base import SimEntity
from src.simulator.scenarios import ScenarioConfig


class ScenarioIO:
    """Save/load scenarios to and from SQLite database via ScenarioRepository."""

    def __init__(self, db: Database | None = None) -> None:
        self.db: Database | None = db
        self.repo: ScenarioRepository | None = ScenarioRepository(db) if db is not None else None

    def save(
        self,
        scenario_config: ScenarioConfig,
        name: str,
        entities: dict[str, SimEntity],
    ) -> int:
        """Serialize entity types + initial positions to JSON and persist in DB.

        Returns:
            scenario_id if saved, or -1 if database is disabled.
        """
        if self.db is None or self.repo is None:
            return -1

        blue_list: list[dict[str, Any]] = []
        red_list: list[dict[str, Any]] = []

        for entity in entities.values():
            variant = "DEFAULT"
            if isinstance(entity, AirEntity):
                variant = entity.aircraft_type

            spec: dict[str, Any] = {
                "entity_id": entity.entity_id,
                "domain": entity.domain.value,
                "type": variant,
                "team": entity.team.value,
                "position": [float(p) for p in entity.position],
                "heading": float(entity.heading),
                "speed": float(entity.speed),
                "count": 1,
            }

            if entity.team == TeamSide.BLUE:
                blue_list.append(spec)
            else:
                red_list.append(spec)

        existing = self.repo.get_by_name(name)
        if existing is not None:
            sc_id = int(existing["scenario_id"])
            self.repo.update(
                sc_id,
                map_size_km=scenario_config.map_size_km,
                episode_horizon=scenario_config.episode_horizon,
                blue_entities=blue_list,
                red_entities=red_list,
                weather=scenario_config.weather,
            )
            return sc_id

        # Determine level or default to 1
        level = getattr(scenario_config, "level", 1)
        if not isinstance(level, int):
            level = 1

        sc_id = self.repo.create(
            name=name,
            level=level,
            map_size_km=scenario_config.map_size_km,
            episode_horizon=scenario_config.episode_horizon,
            blue_entities=blue_list,
            red_entities=red_list,
            weather=scenario_config.weather,
            notes="Saved from 2D Tactical UI",
        )
        return sc_id

    def load(self, scenario_id: int) -> tuple[ScenarioConfig, dict[str, Any]]:
        """Load scenario from DB. Returns (scenario_config, initial_entity_specs)."""
        if self.db is None or self.repo is None:
            raise ValueError("Database connection is not configured.")

        row = self.repo.get(scenario_id)
        if row is None:
            raise KeyError(f"Scenario ID {scenario_id} not found in database.")

        scenario_config = self.repo.to_scenario_config(scenario_id)
        return scenario_config, row

    def list_scenarios(self) -> list[dict[str, Any]]:
        """List summary info for all available scenarios."""
        if self.db is None:
            return []
        query = "SELECT scenario_id, name, level, created_at FROM scenarios ORDER BY scenario_id ASC;"
        return self.db.fetch_all(query)

    def delete(self, scenario_id: int) -> None:
        """Delete scenario record by ID."""
        if self.db is None or self.repo is None:
            return
        self.repo.delete(scenario_id)
