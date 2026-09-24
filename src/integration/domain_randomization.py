"""Sim-to-Real domain randomization for closing physics and sensor fidelity gaps."""

import copy
from typing import Any
import numpy as np

from src.integration.config import (
    AGENT_SPEED_JITTER,
    DT_RANGE_S,
    MAP_SIZE_RANGE_KM,
    SENSOR_RANGE_JITTER,
    WEAPON_PK_JITTER,
)
from src.simulator.scenarios import ScenarioConfig


class DomainRandomizer:
    """Randomizes physical, environmental, and kinematic parameters within defined tolerances.

    Ensures policies trained in our simulation generalize reliably across DRDO TSS's
    underlying flight dynamics models and sensor simulation engines.
    """

    def __init__(
        self,
        sensor_range_jitter: float = SENSOR_RANGE_JITTER,
        weapon_pk_jitter: float = WEAPON_PK_JITTER,
        agent_speed_jitter: float = AGENT_SPEED_JITTER,
        dt_range: tuple[float, float] = DT_RANGE_S,
        map_size_range: tuple[float, float] = MAP_SIZE_RANGE_KM,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.sensor_range_jitter = sensor_range_jitter
        self.weapon_pk_jitter = weapon_pk_jitter
        self.agent_speed_jitter = agent_speed_jitter
        self.dt_range = dt_range
        self.map_size_range = map_size_range
        self.rng = rng if rng is not None else np.random.default_rng()

    def randomize_agent_params(self, base_params: dict[str, Any]) -> dict[str, Any]:
        """Apply bounded multiplicative Gaussian/uniform jitter to vehicle and weapon parameters."""
        p = copy.deepcopy(base_params)

        # 1. Sensor range jitter
        if "sensor_range" in p:
            factor = 1.0 + float(self.rng.uniform(-self.sensor_range_jitter, self.sensor_range_jitter))
            p["sensor_range"] = max(1.0, float(p["sensor_range"]) * factor)

        if "radar_range" in p:
            factor = 1.0 + float(self.rng.uniform(-self.sensor_range_jitter, self.sensor_range_jitter))
            p["radar_range"] = max(1.0, float(p["radar_range"]) * factor)

        # 2. Weapon Pk jitter
        if "weapon_pk" in p:
            factor = 1.0 + float(self.rng.uniform(-self.weapon_pk_jitter, self.weapon_pk_jitter))
            p["weapon_pk"] = float(np.clip(float(p["weapon_pk"]) * factor, 0.01, 0.99))

        if "pk" in p:
            factor = 1.0 + float(self.rng.uniform(-self.weapon_pk_jitter, self.weapon_pk_jitter))
            p["pk"] = float(np.clip(float(p["pk"]) * factor, 0.01, 0.99))

        # 3. Agent speed jitter
        if "speed" in p:
            factor = 1.0 + float(self.rng.uniform(-self.agent_speed_jitter, self.agent_speed_jitter))
            p["speed"] = max(0.1, float(p["speed"]) * factor)

        if "max_speed" in p:
            factor = 1.0 + float(self.rng.uniform(-self.agent_speed_jitter, self.agent_speed_jitter))
            p["max_speed"] = max(0.1, float(p["max_speed"]) * factor)

        return p

    def randomize_dt(self) -> float:
        """Sample a randomized simulation timestep dt within [dt_min, dt_max]."""
        return float(self.rng.uniform(self.dt_range[0], self.dt_range[1]))

    def randomize_map_size(self) -> float:
        """Sample a randomized map boundary size within [map_min_km, map_max_km]."""
        return float(self.rng.uniform(self.map_size_range[0], self.map_size_range[1]))

    def sample_scenario_config(self, base_config: ScenarioConfig) -> ScenarioConfig:
        """Return a randomized copy of base_config for Sim-to-Real robustness."""
        rand_map_size = self.randomize_map_size()

        rand_blue = [self.randomize_agent_params(e) for e in base_config.blue_entities]
        rand_red = [self.randomize_agent_params(e) for e in base_config.red_entities]

        return ScenarioConfig(
            name=f"{base_config.name}_domain_rand",
            map_size_km=rand_map_size,
            episode_horizon=base_config.episode_horizon,
            blue_entities=rand_blue,
            red_entities=rand_red,
            randomize_positions=base_config.randomize_positions,
            randomize_team_zones=base_config.randomize_team_zones,
            weather=base_config.weather,
            seed=base_config.seed,
        )
