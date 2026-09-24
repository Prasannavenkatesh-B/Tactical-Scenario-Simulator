"""Database seeding utilities for default scenarios and agent specifications."""

from enum import Enum
from typing import Any

from src.database.agent_repo import AgentRepository
from src.database.db import Database
from src.database.scenario_repo import ScenarioRepository
from src.simulator.config import (
    AC1_AMMO_CANNON,
    AC1_AMMO_ROCKET,
    AC1_HIT_PROB_CANNON,
    AC1_MAX_TURN_RATE_DEG,
    AC1_ROCKET_WEZ_RANGE_KM,
    AC1_SPEED_RANGE_KNOTS,
    AC2_AMMO_CANNON,
    AC2_AMMO_ROCKET,
    AC2_HIT_PROB_CANNON,
    AC2_MAX_TURN_RATE_DEG,
    AC2_SPEED_RANGE_KNOTS,
    AC2_WEZ_RANGE_KM,
    GROUND_AMMO,
    GROUND_HIT_PROB,
    GROUND_MAX_TURN_RATE_DEG,
    GROUND_SENSOR_RANGE_KM,
    GROUND_SPEED_RANGE_KMPH,
    GROUND_WEZ_RANGE_KM,
    KNOTS_TO_KMH,
    SEA_AMMO,
    SEA_HIT_PROB,
    SEA_MAX_TURN_RATE_DEG,
    SEA_RADAR_RANGE_KM,
    SEA_SPEED_RANGE_KNOTS,
    SEA_WEZ_RANGE_KM,
)
from src.simulator.scenarios import generate_scenario


def _clean_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert any Enum attributes to string representation for database storage."""
    cleaned = []
    for ent in entities:
        item = dict(ent)
        for k, v in item.items():
            if isinstance(v, Enum):
                item[k] = v.value
        cleaned.append(item)
    return cleaned


def seed_default_scenarios(db: Database) -> None:
    """Seed curriculum levels 1-5 into scenarios table idempotently."""
    repo = ScenarioRepository(db)
    for level in range(1, 6):
        sc_cfg = generate_scenario(level=level, seed=42)
        existing = repo.get_by_name(sc_cfg.name)
        if existing is None:
            repo.create(
                name=sc_cfg.name,
                level=level,
                map_size_km=sc_cfg.map_size_km,
                episode_horizon=sc_cfg.episode_horizon,
                blue_entities=_clean_entities(sc_cfg.blue_entities),
                red_entities=_clean_entities(sc_cfg.red_entities),
                weather=sc_cfg.weather,
                notes=f"Default curriculum Level {level} scenario.",
            )


def seed_default_agent_params(db: Database) -> None:
    """Seed baseline agent kinematic and weapon parameters for all domains."""
    repo = AgentRepository(db)

    # AC1 Aircraft
    repo.upsert(
        domain="air",
        variant="AC1",
        max_speed=AC1_SPEED_RANGE_KNOTS[1] * KNOTS_TO_KMH,
        min_speed=AC1_SPEED_RANGE_KNOTS[0] * KNOTS_TO_KMH,
        turn_rate_max=AC1_MAX_TURN_RATE_DEG,
        sensor_range_km=20.0,
        weapon_range_km=AC1_ROCKET_WEZ_RANGE_KM,
        hit_probability=AC1_HIT_PROB_CANNON,
        ammo_cannon=AC1_AMMO_CANNON,
        ammo_rocket=AC1_AMMO_ROCKET,
    )

    # AC2 Aircraft
    repo.upsert(
        domain="air",
        variant="AC2",
        max_speed=AC2_SPEED_RANGE_KNOTS[1] * KNOTS_TO_KMH,
        min_speed=AC2_SPEED_RANGE_KNOTS[0] * KNOTS_TO_KMH,
        turn_rate_max=AC2_MAX_TURN_RATE_DEG,
        sensor_range_km=25.0,
        weapon_range_km=AC2_WEZ_RANGE_KM,
        hit_probability=AC2_HIT_PROB_CANNON,
        ammo_cannon=AC2_AMMO_CANNON,
        ammo_rocket=AC2_AMMO_ROCKET,
    )

    # Ground Unit
    repo.upsert(
        domain="ground",
        variant="DEFAULT",
        max_speed=GROUND_SPEED_RANGE_KMPH[1],
        min_speed=GROUND_SPEED_RANGE_KMPH[0],
        turn_rate_max=GROUND_MAX_TURN_RATE_DEG,
        sensor_range_km=GROUND_SENSOR_RANGE_KM,
        weapon_range_km=GROUND_WEZ_RANGE_KM,
        hit_probability=GROUND_HIT_PROB,
        ammo_cannon=GROUND_AMMO,
        ammo_rocket=0,
    )

    # Sea Surface Combatant
    repo.upsert(
        domain="sea",
        variant="DEFAULT",
        max_speed=SEA_SPEED_RANGE_KNOTS[1] * KNOTS_TO_KMH,
        min_speed=SEA_SPEED_RANGE_KNOTS[0] * KNOTS_TO_KMH,
        turn_rate_max=SEA_MAX_TURN_RATE_DEG,
        sensor_range_km=SEA_RADAR_RANGE_KM,
        weapon_range_km=SEA_WEZ_RANGE_KM,
        hit_probability=SEA_HIT_PROB,
        ammo_cannon=SEA_AMMO,
        ammo_rocket=0,
    )


def seed_all(db: Database) -> None:
    """Run all database seeding procedures."""
    seed_default_scenarios(db)
    seed_default_agent_params(db)
