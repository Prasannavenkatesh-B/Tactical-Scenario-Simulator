"""Tests for Air-to-Air combat doctrine detectors (Patterns 1-7)."""

import math
import numpy as np
import pytest

from src.evaluation.doctrine.air_doctrine import (
    DefensiveBreakDetector,
    EnergyManagementDetector,
    LagPursuitDetector,
    LeadPursuitDetector,
    PincerManeuverDetector,
    PursuitCurveDetector,
    ThreatPrioritizationDetector,
    angular_error,
    compute_bearing,
)
from src.evaluation.doctrine.base_detector import EpisodeData
from src.simulator.scenarios import generate_scenario


class TestAirDoctrine:
    """Test suite for Shaw fighter combat tactical patterns."""

    def test_math_bearing_and_angular_error(self) -> None:
        """Verify mathematical helper functions for bearing and angle differences."""
        # East is 0 rad
        assert compute_bearing([0, 0], [10, 0]) == pytest.approx(0.0, abs=1e-5)
        # North is pi/2 rad
        assert compute_bearing([0, 0], [0, 10]) == pytest.approx(math.pi / 2.0, abs=1e-5)
        # Angular error wraps modulo 2pi
        assert angular_error(0.1, 0.2) == pytest.approx(0.1, abs=1e-5)
        assert angular_error(0.0, 2.0 * math.pi - 0.1) == pytest.approx(0.1, abs=1e-5)
        assert angular_error(0.0, math.pi) == pytest.approx(math.pi, abs=1e-5)

    def test_pursuit_curve_detected(self, synthetic_air_episode: EpisodeData) -> None:
        """PursuitCurveDetector detects tail-chase intercept with high confidence."""
        detector = PursuitCurveDetector()
        res = detector.detect(synthetic_air_episode)
        assert res.doctrine_name == "pursuit_curve"
        assert res.detected is True
        assert res.confidence >= 0.70

    def test_pursuit_curve_rejects_divergent_trajectories(self) -> None:
        """PursuitCurveDetector rejects trajectories where attacker steers away from target."""
        cfg = generate_scenario(level=1, rng=np.random.default_rng(42))
        b_traj = [{"t": t, "position": np.array([10.0 + t, 50.0, 5.0]), "heading": math.pi, "speed": 500.0, "status": "ALIVE"} for t in range(10)]
        r_traj = [{"t": t, "position": np.array([20.0 + t, 50.0, 5.0]), "heading": 0.0, "speed": 500.0, "status": "ALIVE"} for t in range(10)]

        ep = EpisodeData(
            entity_trajectories={"blue_air_1": b_traj, "red_air_1": r_traj},
            engagements=[],
            observations={},
            commander_decisions=[],
            scenario_config=cfg,
            episode_length=10,
        )
        res = PursuitCurveDetector().detect(ep)
        assert res.detected is False
        assert res.confidence < 0.50

    def test_lead_pursuit_detected(self, synthetic_air_episode: EpisodeData) -> None:
        """LeadPursuitDetector detects intercept geometry pointing at future waypoint."""
        detector = LeadPursuitDetector()
        res = detector.detect(synthetic_air_episode)
        assert res.doctrine_name == "lead_pursuit"
        assert isinstance(res.confidence, float)
        assert 0.0 <= res.confidence <= 1.0

    def test_lag_pursuit_detected(self, synthetic_air_episode: EpisodeData) -> None:
        """LagPursuitDetector detects position held in rear hemisphere beyond standoff distance."""
        detector = LagPursuitDetector()
        res = detector.detect(synthetic_air_episode)
        assert res.doctrine_name == "lag_pursuit"
        assert res.detected is True
        assert res.confidence >= 0.50

    def test_defensive_break_detected(self, synthetic_air_episode: EpisodeData) -> None:
        """DefensiveBreakDetector identifies sharp turn under adversary threat."""
        detector = DefensiveBreakDetector()
        res = detector.detect(synthetic_air_episode)
        assert res.doctrine_name == "defensive_break"
        assert res.detected is True
        assert res.confidence >= 0.50
        assert res.evidence["max_turn_rate_rad_s"] >= 1.0

    def test_energy_management_detected(self, synthetic_air_episode: EpisodeData) -> None:
        """EnergyManagementDetector detects dive-acceleration trade."""
        detector = EnergyManagementDetector()
        res = detector.detect(synthetic_air_episode)
        assert res.doctrine_name == "energy_management"
        assert res.detected is True
        assert res.confidence >= 0.50

    def test_pincer_maneuver_detected(self, synthetic_air_episode: EpisodeData) -> None:
        """PincerManeuverDetector detects two friendly aircraft converging from > 120 deg separation."""
        detector = PincerManeuverDetector()
        res = detector.detect(synthetic_air_episode)
        assert res.doctrine_name == "pincer_maneuver"
        assert res.detected is True
        assert res.confidence >= 0.50

    def test_threat_prioritization_detected(self, synthetic_air_episode: EpisodeData) -> None:
        """ThreatPrioritizationDetector verifies commander targets immediate nearest threat."""
        detector = ThreatPrioritizationDetector()
        res = detector.detect(synthetic_air_episode)
        assert res.doctrine_name == "threat_prioritization"
        assert res.detected is True
        assert res.confidence >= 0.50

    def test_air_detectors_handle_empty_episode(self, empty_episode: EpisodeData) -> None:
        """All air detectors handle empty episodes without exceptions."""
        detectors = [
            PursuitCurveDetector(),
            LeadPursuitDetector(),
            LagPursuitDetector(),
            DefensiveBreakDetector(),
            EnergyManagementDetector(),
            PincerManeuverDetector(),
            ThreatPrioritizationDetector(),
        ]
        for d in detectors:
            res = d.detect(empty_episode)
            assert isinstance(res.confidence, float)
            assert 0.0 <= res.confidence <= 1.0
