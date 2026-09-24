"""Tests for Maritime / Naval combat doctrine detectors (Patterns 11-13)."""

import numpy as np
import pytest

from src.evaluation.doctrine.base_detector import EpisodeData
from src.evaluation.doctrine.sea_doctrine import (
    EvasiveManeuverDetector,
    ScreenFormationDetector,
    StandoffEngagementDetector,
)
from src.simulator.scenarios import generate_scenario


class TestSeaDoctrine:
    """Test suite for maritime tactical doctrine patterns."""

    def test_standoff_engagement_detected(self, synthetic_sea_episode: EpisodeData) -> None:
        """StandoffEngagementDetector confirms long-range naval missile fire (> 60% range)."""
        detector = StandoffEngagementDetector()
        res = detector.detect(synthetic_sea_episode)
        assert res.doctrine_name == "standoff_engagement"
        assert res.detected is True
        assert res.confidence >= 0.50

    def test_standoff_engagement_rejects_point_blank_fire(self) -> None:
        """Rejects engagements occurring at close point-blank range (< 60% range)."""
        cfg = generate_scenario(level=4, rng=np.random.default_rng(42))
        engs = [
            {"t": 5, "shooter_id": "blue_sea_1", "target_id": "red_sea_1", "weapon": "missile", "hit": True, "distance_km": 5.0}
        ]
        ep = EpisodeData(
            entity_trajectories={},
            engagements=engs,
            observations={},
            commander_decisions=[],
            scenario_config=cfg,
            episode_length=10,
        )
        res = StandoffEngagementDetector().detect(ep)
        assert res.detected is False
        assert res.confidence == 0.0

    def test_screen_formation_detected(self, synthetic_sea_episode: EpisodeData) -> None:
        """ScreenFormationDetector confirms parallel/barrier line formation between vessels."""
        detector = ScreenFormationDetector()
        res = detector.detect(synthetic_sea_episode)
        assert res.doctrine_name == "screen_formation"
        assert res.detected is True
        assert res.confidence >= 0.60

    def test_evasive_maneuver_detected(self, synthetic_sea_episode: EpisodeData) -> None:
        """EvasiveManeuverDetector detects ship turn rate under hostile threat proximity."""
        detector = EvasiveManeuverDetector()
        res = detector.detect(synthetic_sea_episode)
        assert res.doctrine_name == "evasive_maneuver"
        assert res.detected is True
        assert res.confidence >= 0.50

    def test_sea_detectors_handle_empty_episode(self, empty_episode: EpisodeData) -> None:
        """All naval detectors handle empty episodes without crashing."""
        for d in [StandoffEngagementDetector(), ScreenFormationDetector(), EvasiveManeuverDetector()]:
            res = d.detect(empty_episode)
            assert isinstance(res.confidence, float)
            assert 0.0 <= res.confidence <= 1.0
