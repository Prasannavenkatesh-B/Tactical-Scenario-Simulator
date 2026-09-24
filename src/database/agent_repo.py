"""Agent parameters repository with SQLite upsert capabilities."""

from typing import Any

from src.database.db import Database


class AgentRepository:
    """Repository for querying, updating, and inserting agent kinematic and tactical parameters."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert(
        self,
        domain: str,
        variant: str,
        max_speed: float = 0.0,
        min_speed: float = 0.0,
        turn_rate_max: float = 0.0,
        sensor_range_km: float = 0.0,
        weapon_range_km: float = 0.0,
        hit_probability: float = 0.0,
        ammo_cannon: int = 0,
        ammo_rocket: int = 0,
        **kwargs: Any,
    ) -> None:
        """Insert or update agent configuration parameters for a given domain and variant."""
        query = """
        INSERT INTO agent_params (
            domain, variant, max_speed, min_speed, turn_rate_max,
            sensor_range_km, weapon_range_km, hit_probability,
            ammo_cannon, ammo_rocket
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(domain, variant) DO UPDATE SET
            max_speed = excluded.max_speed,
            min_speed = excluded.min_speed,
            turn_rate_max = excluded.turn_rate_max,
            sensor_range_km = excluded.sensor_range_km,
            weapon_range_km = excluded.weapon_range_km,
            hit_probability = excluded.hit_probability,
            ammo_cannon = excluded.ammo_cannon,
            ammo_rocket = excluded.ammo_rocket;
        """
        params = (
            domain.lower(),
            variant.upper(),
            float(max_speed),
            float(min_speed),
            float(turn_rate_max),
            float(sensor_range_km),
            float(weapon_range_km),
            float(hit_probability),
            int(ammo_cannon),
            int(ammo_rocket),
        )
        self.db.execute(query, params)

    def get(self, domain: str, variant: str) -> dict[str, Any] | None:
        """Retrieve agent parameters for a specific domain and variant."""
        query = "SELECT * FROM agent_params WHERE domain = ? AND variant = ?;"
        return self.db.fetch_one(query, (domain.lower(), variant.upper()))

    def list_all(self) -> list[dict[str, Any]]:
        """List all registered agent parameter records."""
        query = "SELECT * FROM agent_params ORDER BY domain, variant;"
        return self.db.fetch_all(query)

    def delete(self, domain: str, variant: str) -> None:
        """Delete agent parameters for a specific domain and variant."""
        query = "DELETE FROM agent_params WHERE domain = ? AND variant = ?;"
        self.db.execute(query, (domain.lower(), variant.upper()))
