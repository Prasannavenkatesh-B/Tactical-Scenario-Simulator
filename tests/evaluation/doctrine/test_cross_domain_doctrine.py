"""Tests for Cross-Domain joint combat doctrine detectors (Patterns 14-16)."""

import numpy as np
import pytest

from src.evaluation.doctrine.base_detector import EpisodeData
from src.evaluation.doctrine.cross_domain_doctrine import (
    AirGroundCoordinationDetector,
    MaritimePatrolDetector,
    SEADSupportDetector,
)
from src.simulator.scenarios import generate_scenario


class TestCrossDomainDoctrine:
    """Test suite for joint cross-domain operational doctrines."""

    def test_air_ground_coordination_detected(self, synthetic_joint_episode: EpisodeData) -> None:
        """AirGroundCoordinationDetector detects CAS aircraft operating within 5 km of ground engagement."""
        detector = AirGroundCoordinationDetector()
        res = detector.detect(synthetic_joint_episode)
        assert res.doctrine_name == "air_ground_coordination"
        assert res.detected is True
        assert res.confidence >= 0.50

    def test_sead_support_detected(self, synthetic_joint_episode: EpisodeData) -> None:
        """SEADSupportDetector verifies air assets prioritize hostile SAM suppression."""
        detector = SEADSupportDetector()
        res = detector.detect(synthetic_joint_episode)
        assert res.doctrine_name == "sead_support"
        assert res.detected is True
        assert res.confidence >= 0.50

    def test_maritime_patrol_detected(self, synthetic_joint_episode: EpisodeData) -> None:
        """MaritimePatrolDetector confirms naval vessel stationing along contested zone boundary (< 30 km)."""
        detector = MaritimePatrolDetector()
        res = detector.detect(synthetic_joint_episode)
        assert res.doctrine_name == "maritime_patrol"
        assert res.detected is True
        assert res.confidence >= 0.50

    def test_maritime_patrol_rejects_far_offshore_drift(self) -> None:
        """MaritimePatrolDetector rejects vessels straying far from the contested boundary."""
        cfg = generate_scenario(level=5, rng=np.random.default_rng(42))
        cfg.map_size_km = 100.0
        # Boundary at Y=50. Vessel at Y=5 (45 km away, exceeding 30 km radius)
        s_traj = [{"t": t, "position": np.array([10.0, 5.0, 0.0]), "heading": 0.0, "speed": 15.0, "status": "ALIVE"} for t in range(10)]
        ep = EpisodeData(
            entity_trajectories={"blue_sea_1": s_traj},
            engagements=[],
            observations={},
            commander_decisions=[],
            scenario_config=cfg,
            episode_length=10,
        )
        res = MaritimePatrolDetector().detect(ep)
        assert res.detected is False
        assert res.confidence == 0.0

    def test_cross_domain_detectors_handle_empty_episode(self, empty_episode: EpisodeData) -> None:
        """All cross-domain detectors handle empty episodes gracefully."""
        for d in [AirGroundCoordinationDetector(), SEADSupportDetector(), MaritimePatrolDetector()]:
            res = d.detect(empty_episode)
            assert isinstance(res.confidence, float)
            assert 0.0 <= res.confidence <= 1.0
