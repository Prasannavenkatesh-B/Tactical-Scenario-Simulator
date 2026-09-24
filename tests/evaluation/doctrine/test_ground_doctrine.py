"""Tests for Ground combat tactical doctrine detectors (Patterns 8-10)."""

import numpy as np
import pytest

from src.evaluation.doctrine.base_detector import EpisodeData
from src.evaluation.doctrine.ground_doctrine import (
    EngagementRangeDisciplineDetector,
    MutualSupportDetector,
    TerrainCoverDetector,
)
from src.simulator.scenarios import generate_scenario


class TestGroundDoctrine:
    """Test suite for ground tactical doctrine patterns."""

    def test_terrain_cover_detected(self, synthetic_ground_episode: EpisodeData) -> None:
        """TerrainCoverDetector detects ground forces holding high-elevation terrain."""
        detector = TerrainCoverDetector()
        res = detector.detect(synthetic_ground_episode)
        assert res.doctrine_name == "terrain_cover"
        assert res.detected is True
        assert res.confidence >= 0.50

    def test_terrain_cover_rejects_low_elevation(self) -> None:
        """TerrainCoverDetector rejects positions held in open depression with low elevation."""
        cfg = generate_scenario(level=3, rng=np.random.default_rng(42))
        g_traj = [{"t": t, "position": np.array([20.0, 30.0, 0.0]), "heading": 0.0, "speed": 10.0, "status": "ALIVE"} for t in range(10)]
        r_traj = [{"t": t, "position": np.array([25.0, 30.0, 0.0]), "heading": 0.0, "speed": 0.0, "status": "ALIVE"} for t in range(10)]

        ep = EpisodeData(
            entity_trajectories={"blue_ground_1": g_traj, "red_ground_1": r_traj},
            engagements=[],
            observations={},
            commander_decisions=[],
            scenario_config=cfg,
            episode_length=10,
        )
        res = TerrainCoverDetector().detect(ep)
        assert res.confidence < 0.50

    def test_mutual_support_detected(self, synthetic_ground_episode: EpisodeData) -> None:
        """MutualSupportDetector confirms ground units advance within 15 km support radius."""
        detector = MutualSupportDetector()
        res = detector.detect(synthetic_ground_episode)
        assert res.doctrine_name == "mutual_support"
        assert res.detected is True
        assert res.confidence >= 0.80

    def test_mutual_support_rejects_excessive_separation(self) -> None:
        """MutualSupportDetector rejects units separated by over 15 km."""
        cfg = generate_scenario(level=3, rng=np.random.default_rng(42))
        g1 = [{"t": t, "position": np.array([10.0, 10.0, 0.0]), "heading": 0.0, "speed": 0.0, "status": "ALIVE"} for t in range(5)]
        g2 = [{"t": t, "position": np.array([60.0, 60.0, 0.0]), "heading": 0.0, "speed": 0.0, "status": "ALIVE"} for t in range(5)]

        ep = EpisodeData(
            entity_trajectories={"blue_ground_1": g1, "blue_ground_2": g2},
            engagements=[],
            observations={},
            commander_decisions=[],
            scenario_config=cfg,
            episode_length=5,
        )
        res = MutualSupportDetector().detect(ep)
        assert res.detected is False
        assert res.confidence == 0.0

    def test_engagement_range_discipline_detected(self, synthetic_ground_episode: EpisodeData) -> None:
        """EngagementRangeDisciplineDetector verifies shots fired within effective range (< 80% max)."""
        detector = EngagementRangeDisciplineDetector()
        res = detector.detect(synthetic_ground_episode)
        assert res.doctrine_name == "engagement_range_discipline"
        assert res.detected is True
        assert res.confidence >= 0.50

    def test_engagement_range_discipline_rejects_out_of_envelope_fire(self) -> None:
        """Rejects ground firing executed near extreme maximum range (> 12 km for 15km weapon)."""
        cfg = generate_scenario(level=3, rng=np.random.default_rng(42))
        engs = [
            {"t": 5, "shooter_id": "blue_ground_1", "target_id": "red_1", "weapon": "cannon", "hit": False, "distance_km": 14.8},
            {"t": 8, "shooter_id": "blue_ground_1", "target_id": "red_1", "weapon": "cannon", "hit": False, "distance_km": 14.5},
        ]
        ep = EpisodeData(
            entity_trajectories={},
            engagements=engs,
            observations={},
            commander_decisions=[],
            scenario_config=cfg,
            episode_length=10,
        )
        res = EngagementRangeDisciplineDetector().detect(ep)
        assert res.detected is False
        assert res.confidence == 0.0

    def test_ground_detectors_handle_empty_episode(self, empty_episode: EpisodeData) -> None:
        """All ground detectors handle empty episodes gracefully."""
        for d in [TerrainCoverDetector(), MutualSupportDetector(), EngagementRangeDisciplineDetector()]:
            res = d.detect(empty_episode)
            assert isinstance(res.confidence, float)
            assert 0.0 <= res.confidence <= 1.0
