"""Tests for NonDeterminismVerifier orchestrator."""

import os
import tempfile
import pytest

from src.evaluation.verifier import NonDeterminismVerifier
from src.marl.policies.air_fight import AirFightPolicy
from src.marl.policies.ground_engage import GroundEngagePolicy
from src.marl.policies.sea_engage import SeaEngagePolicy


@pytest.fixture
def verifier() -> NonDeterminismVerifier:
    policies = {
        "air_fight": AirFightPolicy(),
        "ground_engage": GroundEngagePolicy(),
        "sea_engage": SeaEngagePolicy(),
    }
    return NonDeterminismVerifier(policies=policies)


class TestVerifier:
    """Test suite for the main NonDeterminismVerifier."""

    def test_verify_reproducibility_same_seed_bit_identical(self, verifier: NonDeterminismVerifier) -> None:
        """Same seed yields bit-identical results with max_diff == 0.0."""
        res = verifier.verify_reproducibility(seed=42)
        assert res["bit_identical"] is True
        assert res["max_diff"] == 0.0
        assert res["seed"] == 42
        assert res["steps_evaluated"] > 0

    def test_verify_same_seed_returns_pass(self, verifier: NonDeterminismVerifier) -> None:
        """Same-seed evaluation returns zero variance, chi2_p ≈ 1.0, and PASS."""
        res = verifier.verify_same_seed(num_runs=10, seed=42)
        assert res["verdict"] == "PASS"
        assert res["outcome_variance"] == pytest.approx(0.0, abs=1e-5)
        assert res["chi2_p_value"] == pytest.approx(1.0, abs=1e-5)
        assert res["levene_p_value"] == pytest.approx(1.0, abs=1e-5)
        assert res["trajectory_diversity"] == 1.0
        assert res["action_entropy"] >= 0.0

    def test_verify_different_seeds_returns_pass(self, verifier: NonDeterminismVerifier) -> None:
        """Different-seeds evaluation yields positive variance, chi2 p < 0.05, and PASS."""
        res = verifier.verify_different_seeds(num_runs=25)
        assert res["verdict"] == "PASS"
        assert res["chi2_p_value"] < 0.05
        assert res["levene_p_value"] < 0.05
        assert res["trajectory_diversity"]["distinct_clusters"] >= 3
        assert res["action_entropy"]["normalized"] > 0.50
        assert res["action_entropy"] >= 0.0

    def test_verify_action_stochasticity_exceeds_threshold(self, verifier: NonDeterminismVerifier) -> None:
        """Action stochasticity test returns normalized entropy > 0.50."""
        res = verifier.verify_action_stochasticity(num_samples=100)
        assert "entropy" in res
        assert "unique_actions" in res
        assert "most_common_action_freq" in res
        assert res["entropy"] > 0.50
        assert res["unique_actions"] >= 2
        assert 0.0 < res["most_common_action_freq"] < 1.0

    def test_run_full_verification_returns_all_keys(self, verifier: NonDeterminismVerifier) -> None:
        """run_full_verification aggregates all 5 evaluation dimensions."""
        res = verifier.run_full_verification(num_runs=15)
        assert "verdict" in res
        assert res["verdict"] in ("PASS", "FAIL")
        assert "reproducibility" in res
        assert "same_seed" in res
        assert "different_seeds" in res
        assert "action_stochasticity" in res
        assert "behavioral_entropy" in res
        assert "statistical_tests" in res

    def test_save_report_writes_json_and_markdown(self, verifier: NonDeterminismVerifier) -> None:
        """save_report produces both report.json and report.md in the target directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = verifier.run_full_verification(num_runs=10)
            paths = verifier.save_report(res, output_dir=tmp_dir, formats=["json", "markdown"])

            assert "json" in paths
            assert "markdown" in paths
            assert os.path.isfile(paths["json"])
            assert os.path.isfile(paths["markdown"])
            assert os.path.getsize(paths["json"]) > 100
            assert os.path.getsize(paths["markdown"]) > 500
