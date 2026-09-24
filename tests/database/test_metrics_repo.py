"""Tests for MetricsRepository logging, batch insertion, and CSV export."""

import csv
from pathlib import Path

from src.database.db import Database
from src.database.metrics_repo import MetricsRepository
from src.database.run_repo import RunRepository


def test_insert_single(db: Database) -> None:
    """Verify single metric insertion and column retrieval."""
    run_repo = RunRepository(db)
    metrics_repo = MetricsRepository(db)
    run_id = run_repo.start_run(config={})

    m_id = metrics_repo.insert(
        run_id=run_id,
        iteration=10,
        level=2,
        policy_name="AirFightPolicy",
        loss=0.123,
        win_rate=0.65,
        kill_death_ratio=1.8,
        action_entropy=0.45,
        episode_length=150.5,
    )
    assert m_id > 0

    metrics = metrics_repo.get_run_metrics(run_id)
    assert len(metrics) == 1
    m = metrics[0]
    assert m["iteration"] == 10
    assert m["level"] == 2
    assert m["policy_name"] == "AirFightPolicy"
    assert abs(m["loss"] - 0.123) < 1e-6
    assert abs(m["win_rate"] - 0.65) < 1e-6


def test_insert_batch_chunking(db: Database) -> None:
    """Verify batch insertion handles batches larger than BATCH_INSERT_SIZE (500)."""
    run_repo = RunRepository(db)
    metrics_repo = MetricsRepository(db)
    run_id = run_repo.start_run(config={})

    total_rows = 1250
    rows = [
        {
            "run_id": run_id,
            "iteration": i,
            "level": (i % 5) + 1,
            "policy_name": "AirFightPolicy",
            "loss": 0.5 / (i + 1),
            "win_rate": min(1.0, i / 1000.0),
            "kill_death_ratio": 1.0 + (i / 500.0),
            "action_entropy": 0.3,
            "episode_length": 200.0,
        }
        for i in range(total_rows)
    ]

    metrics_repo.insert_batch(rows)
    retrieved = metrics_repo.get_run_metrics(run_id)
    assert len(retrieved) == total_rows
    assert retrieved[0]["iteration"] == 0
    assert retrieved[-1]["iteration"] == total_rows - 1


def test_insert_batch_empty(db: Database) -> None:
    """Verify inserting an empty batch is a safe no-op."""
    metrics_repo = MetricsRepository(db)
    metrics_repo.insert_batch([])


def test_get_run_metrics_ordering(db: Database) -> None:
    """Verify metrics are returned in ascending iteration order."""
    run_repo = RunRepository(db)
    metrics_repo = MetricsRepository(db)
    run_id = run_repo.start_run(config={})

    metrics_repo.insert(run_id=run_id, iteration=30, loss=0.1)
    metrics_repo.insert(run_id=run_id, iteration=10, loss=0.3)
    metrics_repo.insert(run_id=run_id, iteration=20, loss=0.2)

    retrieved = metrics_repo.get_run_metrics(run_id)
    assert [r["iteration"] for r in retrieved] == [10, 20, 30]


def test_get_latest(db: Database) -> None:
    """Verify retrieving the latest metric for a policy."""
    run_repo = RunRepository(db)
    metrics_repo = MetricsRepository(db)
    run_id = run_repo.start_run(config={})

    metrics_repo.insert(run_id=run_id, iteration=10, policy_name="AirFight", loss=0.5)
    metrics_repo.insert(run_id=run_id, iteration=20, policy_name="AirFight", loss=0.3)
    metrics_repo.insert(run_id=run_id, iteration=15, policy_name="GroundEngage", loss=0.4)

    latest_air = metrics_repo.get_latest(run_id, "AirFight")
    assert latest_air is not None
    assert latest_air["iteration"] == 20
    assert abs(latest_air["loss"] - 0.3) < 1e-6

    latest_sea = metrics_repo.get_latest(run_id, "SeaEngage")
    assert latest_sea is None


def test_export_csv(db: Database, tmp_path: Path) -> None:
    """Verify exporting metrics to a CSV file."""
    run_repo = RunRepository(db)
    metrics_repo = MetricsRepository(db)
    run_id = run_repo.start_run(config={})

    metrics_repo.insert(run_id=run_id, iteration=1, level=1, loss=0.45, win_rate=0.5)
    metrics_repo.insert(run_id=run_id, iteration=2, level=1, loss=0.35, win_rate=0.7)

    out_csv = tmp_path / "exports" / "metrics_test.csv"
    metrics_repo.export_csv(run_id, str(out_csv))

    assert out_csv.exists()
    with open(out_csv, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["iteration"] == "1"
        assert abs(float(reader[0]["loss"]) - 0.45) < 1e-5
        assert reader[1]["iteration"] == "2"
        assert abs(float(reader[1]["win_rate"]) - 0.7) < 1e-5
