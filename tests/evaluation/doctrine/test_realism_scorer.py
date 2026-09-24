"""Tests for RealismScorer orchestrator."""

import os
import tempfile
import numpy as np
import pytest

from src.evaluation.doctrine.base_detector import EpisodeData
from src.evaluation.realism_scorer import RealismScorer
from src.marl.policies.air_fight import AirFightPolicy
from src.marl.policies.ground_engage import GroundEngagePolicy
from src.marl.policies.sea_engage import SeaEngagePolicy
from src.simulator.scenarios import generate_scenario


@pytest.fixture
def scorer() -> RealismScorer:
    policies = {
        "air_fight": AirFightPolicy(),
        "ground_engage": GroundEngagePolicy(),
        "sea_engage": SeaEngagePolicy(),
    }
    return RealismScorer(policies=policies)


class TestRealismScorer:
    """Test suite for RealismScorer simulation rollouts and multi-episode scoring."""

    def test_run_episode_returns_valid_episode_data(self, scorer: RealismScorer) -> None:
        """run_episode executes an episode in TacticalEnv and returns populated EpisodeData."""
        cfg = generate_scenario(level=2, rng=np.random.default_rng(42))
        ep_data = scorer.run_episode(cfg, seed=42, max_steps=10)

        assert isinstance(ep_data, EpisodeData)
        assert ep_data.episode_length > 0
        assert len(ep_data.entity_trajectories) > 0
        assert ep_data.outcome in ("blue_win", "red_win", "draw")

    def test_collect_episodes_returns_n_episodes(self, scorer: RealismScorer) -> None:
        """collect_episodes gathers specified number of simulation episodes."""
        eps = scorer.collect_episodes(num_episodes=3, scenario_level=1)
        assert len(eps) == 3
        for ep in eps:
            assert isinstance(ep, EpisodeData)

    def test_score_all_aggregates_across_episodes(self, scorer: RealismScorer, synthetic_air_episode: EpisodeData) -> None:
        """score_all computes overall metrics and per-doctrine stats across episode list."""
        summary = scorer.score_all([synthetic_air_episode])
        assert "total_doctrines" in summary
        assert "detected_count" in summary
        assert "detection_rate" in summary
        assert "by_domain" in summary
        assert "per_doctrine" in summary
        assert "acceptance_verdict" in summary
        assert summary["total_doctrines"] == 16
        assert 0.0 <= summary["detection_rate"] <= 1.0

    def test_run_full_validation_produces_acceptance_keys(self, scorer: RealismScorer) -> None:
        """run_full_validation returns all keys required for DRDO acceptance."""
        res = scorer.run_full_validation(num_episodes=5, scenario_level=3)
        assert "acceptance_verdict" in res
        assert res["acceptance_verdict"] in ("PASS", "FAIL")
        assert res["detection_rate"] >= 0.0
        assert "by_domain" in res
        assert "air" in res["by_domain"]
        assert "ground" in res["by_domain"]
        assert "sea" in res["by_domain"]
        assert "cross" in res["by_domain"]

    def test_save_report_persists_json_and_markdown(self, scorer: RealismScorer, synthetic_air_episode: EpisodeData) -> None:
        """save_report writes both JSON and Markdown documents."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = scorer.score_all([synthetic_air_episode])
            paths = scorer.save_report(summary, output_dir=tmp_dir, formats=["json", "markdown"])

            assert "json" in paths
            assert "markdown" in paths
            assert os.path.isfile(paths["json"])
            assert os.path.isfile(paths["markdown"])
            assert os.path.getsize(paths["json"]) > 100
            assert os.path.getsize(paths["markdown"]) > 500
