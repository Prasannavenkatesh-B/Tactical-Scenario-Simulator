"""Tactical scenario definitions and curriculum scenario generator.

Supports progressive MARL training from 2v2 basic dogfights (Level 1) to
complex multi-domain tri-service tactical engagements (Level 5).
"""

from dataclasses import dataclass, field
import math
import typing
import numpy as np

from src.core.interfaces import DomainType, TeamSide
from src.simulator.config import (
    DEFAULT_EPISODE_HORIZON,
    DEFAULT_LEVEL5_HORIZON,
    DEFAULT_MAP_SIZE_KM,
    HIGH_LEVEL_MAP_SIZE_KM,
    KNOTS_TO_KMH,
)
from src.simulator.entities.air import AirEntity
from src.simulator.entities.base import SimEntity
from src.simulator.entities.ground import GroundEntity
from src.simulator.entities.sea import SeaEntity

if typing.TYPE_CHECKING:
    from src.simulator.map import Map2D


@dataclass
class ScenarioConfig:
    """Configuration specification for a tactical simulation scenario.

    Attributes:
        name: Human-readable scenario name.
        map_size_km: Width and height of scenario boundary in km.
        episode_horizon: Maximum simulation steps before truncation.
        blue_entities: List of entity specification dictionaries for Blue team.
        red_entities: List of entity specification dictionaries for Red team.
        randomize_positions: If True, spawn locations are randomized within team zones.
        randomize_team_zones: If True, team zone layout (West/East vs South/North) alternates.
        weather: Atmospheric conditions ("clear", "rain", "fog").
        seed: Deterministic random seed, or None for non-deterministic execution.
    """
    name: str
    map_size_km: float = DEFAULT_MAP_SIZE_KM
    episode_horizon: int = DEFAULT_EPISODE_HORIZON
    blue_entities: list[dict[str, typing.Any]] = field(default_factory=list)
    red_entities: list[dict[str, typing.Any]] = field(default_factory=list)
    randomize_positions: bool = True
    randomize_team_zones: bool = True
    weather: str = "clear"
    seed: int | None = None


def generate_scenario(
    level: int,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> ScenarioConfig:
    """Generate a curriculum scenario configuration for the given tier level (1 to 5).

    Curriculum Tiers:
        - Level 1: 2 Blue Air (1 AC1, 1 AC2) vs 2 Red Air (static opponents).
        - Level 2: 2 vs 2 Air dogfight with random maneuver opponents.
        - Level 3: 2 vs 2 Air dogfight with tactical scripted opponents.
        - Level 4: Joint Air-Ground (2 Blue Air + 1 Blue Ground vs 2 Red Air + 1 Red Ground).
        - Level 5: Full Multi-Domain Tri-Service (2 Air + 1 Ground + 1 Sea per team, 50 km map).

    Args:
        level: Curriculum level (1 to 5).
        rng: Optional seeded numpy Generator.
        seed: Optional explicit random seed.

    Returns:
        ScenarioConfig ready for environment initialization.
    """
    cur_seed = seed
    if cur_seed is None and rng is not None:
        cur_seed = int(rng.integers(0, 2**31 - 1))

    if level == 1:
        # Level 1: 2v2 air dogfight with static opponents
        return ScenarioConfig(
            name="Level_1_Air_2v2_Static",
            map_size_km=DEFAULT_MAP_SIZE_KM,
            episode_horizon=DEFAULT_EPISODE_HORIZON,
            blue_entities=[
                {"domain": DomainType.AIR, "type": "AC1", "count": 1},
                {"domain": DomainType.AIR, "type": "AC2", "count": 1},
            ],
            red_entities=[
                {"domain": DomainType.AIR, "type": "AC1", "count": 1, "script": "static"},
                {"domain": DomainType.AIR, "type": "AC2", "count": 1, "script": "static"},
            ],
            randomize_positions=True,
            randomize_team_zones=True,
            weather="clear",
            seed=cur_seed,
        )

    elif level == 2:
        # Level 2: 2v2 air dogfight with random opponents
        return ScenarioConfig(
            name="Level_2_Air_2v2_Random",
            map_size_km=DEFAULT_MAP_SIZE_KM,
            episode_horizon=DEFAULT_EPISODE_HORIZON,
            blue_entities=[
                {"domain": DomainType.AIR, "type": "AC1", "count": 1},
                {"domain": DomainType.AIR, "type": "AC2", "count": 1},
            ],
            red_entities=[
                {"domain": DomainType.AIR, "type": "AC1", "count": 1, "script": "random"},
                {"domain": DomainType.AIR, "type": "AC2", "count": 1, "script": "random"},
            ],
            randomize_positions=True,
            randomize_team_zones=True,
            weather="clear",
            seed=cur_seed,
        )

    elif level == 3:
        # Level 3: 2v2 air dogfight with tactical scripted opponents
        return ScenarioConfig(
            name="Level_3_Air_2v2_Tactical",
            map_size_km=DEFAULT_MAP_SIZE_KM,
            episode_horizon=DEFAULT_EPISODE_HORIZON,
            blue_entities=[
                {"domain": DomainType.AIR, "type": "AC1", "count": 1},
                {"domain": DomainType.AIR, "type": "AC2", "count": 1},
            ],
            red_entities=[
                {"domain": DomainType.AIR, "type": "AC1", "count": 1, "script": "pursuit"},
                {"domain": DomainType.AIR, "type": "AC2", "count": 1, "script": "pursuit"},
            ],
            randomize_positions=True,
            randomize_team_zones=True,
            weather="clear",
            seed=cur_seed,
        )

    elif level == 4:
        # Level 4: 2 Air + 1 Ground per side
        return ScenarioConfig(
            name="Level_4_Joint_Air_Ground",
            map_size_km=DEFAULT_MAP_SIZE_KM,
            episode_horizon=DEFAULT_EPISODE_HORIZON,
            blue_entities=[
                {"domain": DomainType.AIR, "type": "AC1", "count": 1},
                {"domain": DomainType.AIR, "type": "AC2", "count": 1},
                {"domain": DomainType.GROUND, "type": "SAM", "count": 1},
            ],
            red_entities=[
                {"domain": DomainType.AIR, "type": "AC1", "count": 1, "script": "pursuit"},
                {"domain": DomainType.AIR, "type": "AC2", "count": 1, "script": "pursuit"},
                {"domain": DomainType.GROUND, "type": "SAM", "count": 1, "script": "defend"},
            ],
            randomize_positions=True,
            randomize_team_zones=True,
            weather="clear",
            seed=cur_seed,
        )

    elif level == 5:
        # Level 5: Full Multi-Domain Tri-Service (Air, Ground, Sea)
        return ScenarioConfig(
            name="Level_5_Multi_Domain_Tri_Service",
            map_size_km=HIGH_LEVEL_MAP_SIZE_KM,
            episode_horizon=DEFAULT_LEVEL5_HORIZON,
            blue_entities=[
                {"domain": DomainType.AIR, "type": "AC1", "count": 1},
                {"domain": DomainType.AIR, "type": "AC2", "count": 1},
                {"domain": DomainType.GROUND, "type": "SAM", "count": 1},
                {"domain": DomainType.SEA, "type": "FRIGATE", "count": 1},
            ],
            red_entities=[
                {"domain": DomainType.AIR, "type": "AC1", "count": 1, "script": "pursuit"},
                {"domain": DomainType.AIR, "type": "AC2", "count": 1, "script": "pursuit"},
                {"domain": DomainType.GROUND, "type": "SAM", "count": 1, "script": "defend"},
                {"domain": DomainType.SEA, "type": "FRIGATE", "count": 1, "script": "patrol"},
            ],
            randomize_positions=True,
            randomize_team_zones=True,
            weather="clear",
            seed=cur_seed,
        )

    else:
        raise ValueError(f"Unsupported scenario level {level}. Supported levels: 1..5")


def spawn_entities(
    config: ScenarioConfig,
    map_ref: "Map2D",
    rng: np.random.Generator,
) -> tuple[list[SimEntity], list[SimEntity]]:
    """Instantiate and spawn initial Blue and Red entities inside assigned team zones.

    Args:
        config: Scenario configuration.
        map_ref: Active Map2D instance with assigned team zones.
        rng: Seeded numpy Generator.

    Returns:
        Tuple of (blue_entities, red_entities).
    """
    blue_list: list[SimEntity] = []
    red_list: list[SimEntity] = []

    def _spawn_group(specs: list[dict[str, typing.Any]], team: TeamSide) -> list[SimEntity]:
        entities: list[SimEntity] = []
        team_str = "blue" if team == TeamSide.BLUE else "red"

        for spec in specs:
            domain = spec["domain"]
            type_str = spec["type"]
            count = spec.get("count", 1)

            for i in range(count):
                idx = len([e for e in entities if e.domain == domain]) + 1
                entity_id = f"{team_str}_{domain.value.lower()}_{idx}"
                entity: SimEntity
                if domain == DomainType.AIR:
                    if config.name.startswith("Level_3"):
                        bz = map_ref.blue_zone
                        rz = map_ref.red_zone
                        is_vert = abs(bz[1] - rz[0]) < 1e-3 or abs(rz[1] - bz[0]) < 1e-3
                        alt = 5.0 + float(rng.uniform(-0.1, 0.1))
                        d_bound = 1.4 + float(rng.uniform(-0.1, 0.1))
                        trans_offset = (idx - 1.5) * 1.0 + float(rng.uniform(-0.05, 0.05))
                        if is_vert:
                            x_div = bz[1] if abs(bz[1] - rz[0]) < 1e-3 else rz[1]
                            blue_on_left = abs(bz[1] - rz[0]) < 1e-3
                            mid_y = (bz[2] + bz[3]) / 2.0
                            if team == TeamSide.BLUE:
                                px = (x_div - d_bound) if blue_on_left else (x_div + d_bound)
                                heading = 0.0 if blue_on_left else math.pi
                            else:
                                px = (x_div + d_bound) if blue_on_left else (x_div - d_bound)
                                heading = math.pi if blue_on_left else 0.0
                            pos = (px, mid_y + trans_offset, alt)
                        else:
                            y_div = bz[3] if abs(bz[3] - rz[2]) < 1e-3 else rz[3]
                            blue_below = abs(bz[3] - rz[2]) < 1e-3
                            mid_x = (bz[0] + bz[1]) / 2.0
                            if team == TeamSide.BLUE:
                                py = (y_div - d_bound) if blue_below else (y_div + d_bound)
                                heading = math.pi / 2 if blue_below else 3 * math.pi / 2
                            else:
                                py = (y_div + d_bound) if blue_below else (y_div - d_bound)
                                heading = 3 * math.pi / 2 if blue_below else math.pi / 2
                            pos = (mid_x + trans_offset, py, alt)
                    else:
                        alt = float(rng.uniform(3.0, 10.0))
                        pos = map_ref.sample_position(team, rng, altitude_km=alt)
                        heading = map_ref.sample_heading(team, rng)
                    speed = 350.0 * KNOTS_TO_KMH
                    entity = AirEntity(
                        entity_id=entity_id,
                        team=team,
                        aircraft_type=type_str,
                        position=pos,
                        heading=heading,
                        speed=speed,
                        rng=rng,
                    )
                elif domain == DomainType.GROUND:
                    pos_2d = map_ref.sample_position(team, rng, altitude_km=0.0)
                    elev = map_ref.elevation_at(pos_2d[0], pos_2d[1])
                    pos = (pos_2d[0], pos_2d[1], elev)
                    heading = map_ref.sample_heading(team, rng)
                    speed = 25.0
                    entity = GroundEntity(
                        entity_id=entity_id,
                        team=team,
                        aircraft_type=type_str,
                        position=pos,
                        heading=heading,
                        speed=speed,
                        rng=rng,
                    )
                elif domain == DomainType.SEA:
                    pos_2d = map_ref.sample_position(team, rng, altitude_km=0.0)
                    pos = (pos_2d[0], pos_2d[1], 0.0)
                    heading = map_ref.sample_heading(team, rng)
                    speed = 18.0 * KNOTS_TO_KMH
                    entity = SeaEntity(
                        entity_id=entity_id,
                        team=team,
                        aircraft_type=type_str,
                        position=pos,
                        heading=heading,
                        speed=speed,
                        rng=rng,
                    )
                else:
                    raise ValueError(f"Unknown domain {domain}")

                # Store script descriptor if present
                if "script" in spec:
                    entity.config["script"] = spec["script"]

                entities.append(entity)
        return entities

    blue_list = _spawn_group(config.blue_entities, TeamSide.BLUE)
    red_list = _spawn_group(config.red_entities, TeamSide.RED)
    return blue_list, red_list
