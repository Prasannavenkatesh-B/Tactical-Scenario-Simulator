"""Unit tests for MetricsLogger.

Tests CSV file generation, header structure, metric appending,
and resource cleanup upon closing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import csv
import tempfile
import pytest

from src.training.metrics import MetricsLogger


class TestMetricsLogger:
    """Tests for MetricsLogger."""

    def test_creates_csv_with_headers(self) -> None:
        """MetricsLogger initializes CSV file with expected headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "metrics.csv"
            logger = MetricsLogger(log_dir=tmpdir, csv_path=csv_path)

            assert csv_path.exists()
            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = next(reader)
                assert headers == MetricsLogger.CSV_HEADERS
            logger.close()

    def test_log_iteration_appends_row(self) -> None:
        """log_iteration() appends properly formatted data rows to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "metrics.csv"
            logger = MetricsLogger(log_dir=tmpdir, csv_path=csv_path)

            losses = {"total_loss": 0.5, "policy_loss": 0.3, "value_loss": 0.1, "entropy_loss": 0.05}
            stats = {"blue_wins": 3, "red_wins": 1, "draws": 1, "mean_reward": 45.0, "mean_episode_length": 150}

            logger.log_iteration(iteration=1, level=2, loss_dict=losses, episode_stats=stats)

            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                _ = next(reader)  # Header
                row = next(reader)
                assert row[0] == "1"  # iteration
                assert row[1] == "2"  # level
                assert row[6] == "3"  # blue_wins
                assert float(row[9]) == pytest.approx(3 / 5)  # win_rate

            logger.close()

    def test_close_flushes_and_cleans_up(self) -> None:
        """close() gracefully flushes handles without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir, csv_path=Path(tmpdir) / "metrics.csv")
            logger.close()
            assert logger.tb_writer is None
