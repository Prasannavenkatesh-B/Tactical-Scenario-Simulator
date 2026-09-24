"""Unit tests for scenario configurations, curriculum tiers, and entity spawning."""

import numpy as np
import pytest

from src.core.interfaces import DomainType, TeamSide
from src.simulator.map import Map2D
from src.simulator.scenarios import generate_scenario, spawn_entities


class TestScenarios:
    """Test suite for scenario curriculum generation and entity placement."""

    def test_generate_scenario_level_1(self) -> None:
        """Verify Level 1 generates 2v2 air dogfight."""
        sc = generate_scenario(1)
        assert sc.episode_horizon == 200
        assert len(sc.blue_entities) == 2
        assert len(sc.red_entities) == 2
        assert all(e["domain"] == DomainType.AIR for e in sc.blue_entities)
        assert all(e["domain"] == DomainType.AIR for e in sc.red_entities)

    def test_generate_scenario_level_5(self) -> None:
        """Verify Level 5 generates tri-service multi-domain scenario."""
        sc = generate_scenario(5)
        assert sc.episode_horizon == 350
        assert sc.map_size_km == 50.0

        blue_domains = {e["domain"] for e in sc.blue_entities}
        assert DomainType.AIR in blue_domains
        assert DomainType.GROUND in blue_domains
        assert DomainType.SEA in blue_domains

    def test_spawn_entities_in_zones(self) -> None:
        """Verify entities spawn strictly inside their respective team zones."""
        sc = generate_scenario(3)
        m = Map2D(size_km=sc.map_size_km, terrain_seed=42)
        m.assign_team_zones(randomize=False)
        rng = np.random.default_rng(42)

        blue_ents, red_ents = spawn_entities(sc, m, rng)

        for b in blue_ents:
            assert b.team == TeamSide.BLUE
            # Blue zone is left half [0, size/2]
            assert 0.0 <= b.position[0] <= m.size_km / 2.0

        for r in red_ents:
            assert r.team == TeamSide.RED
            # Red zone is right half [size/2, size]
            assert m.size_km / 2.0 <= r.position[0] <= m.size_km

    def test_spawned_entities_valid_observations(self) -> None:
        """Verify all spawned entities generate valid normalized observation arrays."""
        sc = generate_scenario(4)
        m = Map2D(size_km=sc.map_size_km, terrain_seed=42)
        rng = np.random.default_rng(42)

        blue_ents, red_ents = spawn_entities(sc, m, rng)
        all_ents = blue_ents + red_ents

        for ent in all_ents:
            opps = red_ents if ent.team == TeamSide.BLUE else blue_ents
            frs = blue_ents if ent.team == TeamSide.BLUE else red_ents
            obs = ent.get_observation(opps, frs, m)
            assert isinstance(obs, np.ndarray)
            assert np.all((obs >= 0.0) & (obs <= 1.0))
