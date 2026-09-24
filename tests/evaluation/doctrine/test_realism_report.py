"""Tests for realism report formatting, JSON serialization, and Markdown generation."""

import json
import os
import tempfile
import pytest

from src.evaluation.realism_report import (
    format_realism_json,
    format_realism_markdown,
    load_realism_report,
)


@pytest.fixture
def sample_realism_summary() -> dict:
    return {
        "total_doctrines": 16,
        "detected_count": 12,
        "detection_rate": 0.75,
        "average_prevalence_detected": 0.65,
        "episodes_analyzed": 50,
        "acceptance_verdict": "PASS",
        "by_domain": {
            "air": 0.85,
            "ground": 0.70,
            "sea": 0.65,
            "cross": 0.72,
        },
        "per_doctrine": {
            "pursuit_curve": {
                "detected": True,
                "confidence": 0.88,
                "episodes_present": 40,
                "total_episodes": 50,
                "prevalence": 0.80,
                "mean_confidence_when_present": 0.88,
                "domain": "air",
                "description": "Tail chase alignment",
                "evidence_snippets": [{"mean_angular_error_rad": 0.25}],
            },
            "terrain_cover": {
                "detected": True,
                "confidence": 0.78,
                "episodes_present": 25,
                "total_episodes": 50,
                "prevalence": 0.50,
                "mean_confidence_when_present": 0.78,
                "domain": "ground",
                "description": "High terrain vantage",
                "evidence_snippets": [{"cover_ratio": 0.78}],
            },
            "standoff_engagement": {
                "detected": False,
                "confidence": 0.42,
                "episodes_present": 5,
                "total_episodes": 50,
                "prevalence": 0.10,
                "mean_confidence_when_present": 0.55,
                "domain": "sea",
                "description": "Naval missile standoff",
                "evidence_snippets": [],
            },
        },
    }


class TestRealismReport:
    """Test suite for realism JSON and Markdown reporting."""

    def test_format_realism_json_valid(self, sample_realism_summary: dict) -> None:
        """format_realism_json produces valid, parseable JSON."""
        json_str = format_realism_json(sample_realism_summary)
        parsed = json.loads(json_str)
        assert parsed["total_doctrines"] == 16
        assert parsed["acceptance_verdict"] == "PASS"

    def test_format_realism_markdown_contains_all_six_sections(self, sample_realism_summary: dict) -> None:
        """format_realism_markdown contains all 6 required DRDO sections."""
        md = format_realism_markdown(sample_realism_summary)
        assert "## 1. Executive Summary" in md
        assert "## 2. Per-Domain Detection Rates" in md
        assert "## 3. Per-Doctrine Detection Table" in md
        assert "## 4. Evidence Snippets" in md
        assert "## 5. Doctrinal Gaps" in md
        assert "## 6. Conclusion & DRDO Acceptance Verdict" in md

    def test_format_realism_markdown_table_and_prevalence(self, sample_realism_summary: dict) -> None:
        """format_realism_markdown includes the 5-column table and prevalence metrics."""
        md = format_realism_markdown(sample_realism_summary)
        assert "| Doctrine | Episodes Present | Prevalence | Mean Conf (Present) | Detected |" in md
        assert "Overall Doctrine Detection Rate:" in md
        assert "Average Prevalence Across Detected Doctrines:" in md
        assert "65.0%" in md
        assert "| `pursuit_curve` | 40/50 | 80.0% | 0.880 | **YES** |" in md
        assert "| `standoff_engagement` | 5/50 | 10.0% | 0.550 | NO |" in md

    def test_load_realism_report_round_trip(self, sample_realism_summary: dict) -> None:
        """JSON report saved to disk loads back with identical values."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "realism.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write(format_realism_json(sample_realism_summary))

            loaded = load_realism_report(path)
            assert loaded["total_doctrines"] == sample_realism_summary["total_doctrines"]
            assert loaded["detection_rate"] == sample_realism_summary["detection_rate"]
            assert loaded["acceptance_verdict"] == sample_realism_summary["acceptance_verdict"]
