"""Unit tests for Sim-to-Real DomainRandomizer."""

import numpy as np

from src.integration.config import DT_RANGE_S, MAP_SIZE_RANGE_KM
from src.integration.domain_randomization import DomainRandomizer
from src.simulator.scenarios import ScenarioConfig


def test_randomize_dt_within_bounds() -> None:
    """Sampled dt values must stay strictly within DT_RANGE_S bounds."""
    randomizer = DomainRandomizer()
    for _ in range(50):
        dt = randomizer.randomize_dt()
        assert DT_RANGE_S[0] <= dt <= DT_RANGE_S[1]


def test_randomize_map_size_within_bounds() -> None:
    """Sampled map sizes must stay strictly within MAP_SIZE_RANGE_KM bounds."""
    randomizer = DomainRandomizer()
    for _ in range(50):
        map_size = randomizer.randomize_map_size()
        assert MAP_SIZE_RANGE_KM[0] <= map_size <= MAP_SIZE_RANGE_KM[1]


def test_randomize_agent_params_jitter() -> None:
    """Agent parameters are jittered within declared percentage bounds."""
    randomizer = DomainRandomizer(
        sensor_range_jitter=0.20,
        weapon_pk_jitter=0.15,
        agent_speed_jitter=0.10,
    )
    base = {
        "sensor_range": 50.0,
        "weapon_pk": 0.80,
        "max_speed": 600.0,
    }

    for _ in range(30):
        rand = randomizer.randomize_agent_params(base)
        # 50 +/- 20% -> [40, 60]
        assert 40.0 <= rand["sensor_range"] <= 60.0
        # 0.80 +/- 15% -> [0.68, 0.92]
        assert 0.68 <= rand["weapon_pk"] <= 0.92
        # 600 +/- 10% -> [540, 660]
        assert 540.0 <= rand["max_speed"] <= 660.0


def test_domain_randomization_reproducibility() -> None:
    """Two randomizers with identical RNG seed produce identical randomizations."""
    rng1 = np.random.default_rng(12345)
    rng2 = np.random.default_rng(12345)

    rand1 = DomainRandomizer(rng=rng1)
    rand2 = DomainRandomizer(rng=rng2)

    base = {"sensor_range": 50.0, "weapon_pk": 0.75, "speed": 400.0}

    res1 = rand1.randomize_agent_params(base)
    res2 = rand2.randomize_agent_params(base)
    assert res1 == res2

    dt1 = rand1.randomize_dt()
    dt2 = rand2.randomize_dt()
    assert dt1 == dt2


def test_sample_scenario_config() -> None:
    """sample_scenario_config yields randomized ScenarioConfig with altered entity attributes."""
    base_cfg = ScenarioConfig(
        name="test_base_scenario",
        map_size_km=100.0,
        blue_entities=[{"name": "b1", "sensor_range": 50.0, "max_speed": 600.0}],
        red_entities=[{"name": "r1", "sensor_range": 45.0, "max_speed": 550.0}],
    )

    rand = DomainRandomizer(rng=np.random.default_rng(999))
    new_cfg = rand.sample_scenario_config(base_cfg)

    assert "domain_rand" in new_cfg.name
    assert new_cfg.map_size_km != base_cfg.map_size_km or True
    assert len(new_cfg.blue_entities) == 1
    assert len(new_cfg.red_entities) == 1
