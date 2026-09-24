"""Tests for report generation, JSON serialization, and Markdown formatting."""

import json
import os
import tempfile
import pytest

from src.evaluation.report import (
    compare_reports,
    format_as_json,
    format_as_markdown,
    load_report,
)


@pytest.fixture
def sample_results() -> dict:
    return {
        "verdict": "PASS",
        "reproducibility": {
            "bit_identical": True,
            "max_diff": 0.0,
            "seed": 42,
            "steps_evaluated": 25,
        },
        "same_seed": {
            "outcome_distribution": {"blue_wins": 0, "red_wins": 0, "draws": 100},
            "outcome_variance": 0.0,
            "chi2_p_value": 1.0,
            "levene_p_value": 1.0,
            "trajectory_diversity": {"distinct_clusters": 1},
            "action_entropy": {"normalized": 0.0},
            "verdict": "PASS",
        },
        "different_seeds": {
            "outcome_distribution": {"blue_wins": 25, "red_wins": 35, "draws": 40},
            "outcome_variance": 0.65,
            "chi2_p_value": 1.2e-5,
            "levene_p_value": 3.4e-6,
            "chi2": {"chi2": 22.4, "df": 2, "p_value": 1.2e-5, "reject_null": True},
            "levene": {"W": 18.2, "p_value": 3.4e-6, "reject_null": True},
            "trajectory_diversity": {
                "diversity_score": 0.8,
                "n_clusters": 5,
                "distinct_clusters": 4,
                "silhouette_score": 0.55,
                "inertia": 120.5,
            },
            "action_entropy": {"mean_entropy": 2.1, "normalized": 0.82},
            "coefficient_of_variation": 0.28,
            "verdict": "PASS",
        },
        "action_stochasticity": {
            "entropy": 0.82,
            "unique_actions": 11,
            "most_common_action_freq": 0.15,
        },
        "behavioral_entropy": {
            "mean_entropy": 1.85,
            "std_entropy": 0.12,
            "normalized_entropy": 0.72,
            "per_timestep": [1.8, 1.9, 1.85],
        },
        "statistical_tests": {
            "chi2": {"chi2": 22.4, "df": 2, "p_value": 1.2e-5, "reject_null": True},
            "levene": {"W": 18.2, "p_value": 3.4e-6, "reject_null": True},
            "cv": 0.28,
            "ks": {"statistic": 0.75, "p_value": 2.1e-10, "reject_null": True},
        },
    }


class TestReport:
    """Test suite for report formatting and persistence."""

    def test_format_as_json_is_valid_json(self, sample_results: dict) -> None:
        """format_as_json produces syntactically valid JSON string."""
        json_str = format_as_json(sample_results)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["verdict"] == "PASS"
        assert parsed["reproducibility"]["bit_identical"] is True

    def test_format_as_markdown_contains_all_eight_sections(self, sample_results: dict) -> None:
        """format_as_markdown contains all 8 required sections."""
        md = format_as_markdown(sample_results)
        assert "## 1. Executive Summary" in md
        assert "## 2. Same-Seed Reproducibility" in md
        assert "## 3. Different-Seed Non-Determinism" in md
        assert "## 4. Action Stochasticity" in md
        assert "## 5. Trajectory Diversity" in md
        assert "## 6. Behavioral Entropy" in md
        assert "## 7. Statistical Test Results" in md
        assert "## 8. Conclusion & DRDO Acceptance Verdict" in md

    def test_load_report_round_trip(self, sample_results: dict) -> None:
        """JSON report saved to file loads back identically."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, "test_report.json")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(format_as_json(sample_results))

            loaded = load_report(file_path)
            assert loaded["verdict"] == sample_results["verdict"]
            assert loaded["same_seed"]["chi2_p_value"] == sample_results["same_seed"]["chi2_p_value"]

    def test_compare_reports_computes_deltas(self, sample_results: dict) -> None:
        """compare_reports detects metric differences between runs."""
        modified_results = dict(sample_results)
        modified_results["action_stochasticity"] = {"entropy": 0.90}

        cmp = compare_reports(sample_results, modified_results)
        assert cmp["verdict_match"] is True
        assert cmp["entropy_delta"] == pytest.approx(0.08, abs=1e-5)
