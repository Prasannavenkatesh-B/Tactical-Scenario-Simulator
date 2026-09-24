"""Tests for DoctrineRegistry aggregator and multi-detector execution."""

import pytest

from src.evaluation.doctrine.base_detector import DoctrineResult, EpisodeData
from src.evaluation.doctrine.doctrine_registry import DoctrineRegistry


class TestDoctrineRegistry:
    """Test suite for DoctrineRegistry orchestration and scoring aggregation."""

    def test_registry_holds_all_sixteen_detectors(self) -> None:
        """DoctrineRegistry instantiates and registers all 16 military combat doctrine detectors."""
        registry = DoctrineRegistry()
        detectors = registry.all_detectors()
        assert len(detectors) == 16

        names = {d.name for d in detectors}
        expected = {
            "pursuit_curve", "lead_pursuit", "lag_pursuit", "defensive_break",
            "energy_management", "pincer_maneuver", "threat_prioritization",
            "terrain_cover", "mutual_support", "engagement_range_discipline",
            "standoff_engagement", "screen_formation", "evasive_maneuver",
            "air_ground_coordination", "sead_support", "maritime_patrol",
        }
        assert names == expected

    def test_detect_all_runs_on_episode(self, synthetic_air_episode: EpisodeData) -> None:
        """detect_all executes all 16 detectors on an episode dataset."""
        registry = DoctrineRegistry()
        results = registry.detect_all(synthetic_air_episode)
        assert len(results) == 16
        for r in results:
            assert isinstance(r, DoctrineResult)
            assert 0.0 <= r.confidence <= 1.0

    def test_aggregate_empty_results_fails(self) -> None:
        """aggregate returns 0.0 detection rate and FAIL for empty detector output."""
        registry = DoctrineRegistry()
        summary = registry.aggregate([])
        assert summary["total_doctrines"] == 16
        assert summary["detected_count"] == 0
        assert summary["detection_rate"] == 0.0
        assert summary["acceptance_verdict"] == "FAIL"

    def test_aggregate_all_detected_passes(self) -> None:
        """aggregate returns 1.0 detection rate and PASS when all doctrines detected."""
        registry = DoctrineRegistry()
        results = [
            DoctrineResult(d.name, detected=True, confidence=0.85)
            for d in registry.all_detectors()
        ]
        summary = registry.aggregate(results)
        assert summary["total_doctrines"] == 16
        assert summary["detected_count"] == 16
        assert summary["detection_rate"] == 1.0
        assert summary["acceptance_verdict"] == "PASS"
        for dom, rate in summary["by_domain"].items():
            assert rate == 1.0

    def test_aggregate_partial_detection_threshold(self) -> None:
        """Verifies threshold boundary (>= 60% passes, < 60% fails)."""
        registry = DoctrineRegistry()
        dets = registry.all_detectors()

        # 9 of 16 detected = 56.25% -> FAIL (confidence 0.8 >= 0.5 for 9, 0.2 < 0.5 for 7)
        res_fail = [
            DoctrineResult(dets[i].name, detected=(i < 9), confidence=0.8 if i < 9 else 0.2)
            for i in range(16)
        ]
        assert registry.aggregate(res_fail)["acceptance_verdict"] == "FAIL"

        # 10 of 16 detected = 62.5% -> PASS
        res_pass = [
            DoctrineResult(dets[i].name, detected=(i < 10), confidence=0.8 if i < 10 else 0.2)
            for i in range(16)
        ]
        assert registry.aggregate(res_pass)["acceptance_verdict"] == "PASS"

    def test_aggregate_prevalence_threshold_and_mean_confidence(self) -> None:
        """Verifies cross-episode prevalence threshold (20%) and confidence calculations."""
        registry = DoctrineRegistry()
        target_name = registry.all_detectors()[0].name

        # Case 1: 2 out of 20 episodes present (10% < 20%) -> detected=False
        results_below: list[DoctrineResult] = []
        for ep in range(20):
            conf = 0.85 if ep < 2 else 0.10
            results_below.append(DoctrineResult(target_name, detected=(conf >= 0.5), confidence=conf))

        summary_below = registry.aggregate(results_below, num_episodes=20)
        doc_below = summary_below["per_doctrine"][target_name]
        assert doc_below["episodes_present"] == 2
        assert doc_below["total_episodes"] == 20
        assert pytest.approx(doc_below["prevalence"], abs=1e-5) == 0.10
        assert doc_below["detected"] is False
        assert pytest.approx(doc_below["mean_confidence_when_present"], abs=1e-5) == 0.85
        assert len(doc_below["per_episode_confidence"]) == 20

        # Case 2: 5 out of 20 episodes present (25% >= 20%) -> detected=True
        results_above: list[DoctrineResult] = []
        for ep in range(20):
            conf = 0.70 if ep < 5 else 0.15
            results_above.append(DoctrineResult(target_name, detected=(conf >= 0.5), confidence=conf))

        summary_above = registry.aggregate(results_above, num_episodes=20)
        doc_above = summary_above["per_doctrine"][target_name]
        assert doc_above["episodes_present"] == 5
        assert doc_above["total_episodes"] == 20
        assert pytest.approx(doc_above["prevalence"], abs=1e-5) == 0.25
        assert doc_above["detected"] is True
        assert pytest.approx(doc_above["mean_confidence_when_present"], abs=1e-5) == 0.70
        assert len(doc_above["per_episode_confidence"]) == 20
