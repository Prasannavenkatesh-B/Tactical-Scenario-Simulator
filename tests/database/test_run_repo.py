"""Tests for RunRepository and training lifecycle management."""

from src.database.db import Database
from src.database.metrics_repo import MetricsRepository
from src.database.run_repo import RunRepository


def test_start_run_and_get(db: Database) -> None:
    """Verify starting a run creates a running record with parsed JSON config."""
    repo = RunRepository(db)
    config = {"lr": 3e-4, "curriculum": "L1-L5", "batch_size": 256}
    run_id = repo.start_run(config=config, git_commit="abc1234", notes="Run test")
    assert run_id > 0

    run = repo.get(run_id)
    assert run is not None
    assert run["run_id"] == run_id
    assert run["status"] == "running"
    assert run["git_commit"] == "abc1234"
    assert run["config"]["lr"] == 3e-4
    assert run["notes"] == "Run test"
    assert run["started_at"] is not None
    assert run["completed_at"] is None


def test_complete_run(db: Database) -> None:
    """Verify completing a run updates metrics, timestamp, and status."""
    repo = RunRepository(db)
    run_id = repo.start_run(config={"env": "tactical"})
    repo.complete_run(
        run_id=run_id,
        total_iterations=500,
        final_win_rate=0.82,
        final_kd_ratio=3.1,
    )

    run = repo.get(run_id)
    assert run is not None
    assert run["status"] == "completed"
    assert run["total_iterations"] == 500
    assert run["final_win_rate"] == 0.82
    assert run["final_kd_ratio"] == 3.1
    assert run["completed_at"] is not None


def test_fail_run(db: Database) -> None:
    """Verify marking a run as failed updates status and records error notes."""
    repo = RunRepository(db)
    run_id = repo.start_run(config={})
    repo.fail_run(run_id=run_id, error="CUDA out of memory error")

    run = repo.get(run_id)
    assert run is not None
    assert run["status"] == "failed"
    assert "CUDA out of memory error" in (run["notes"] or "")
    assert run["completed_at"] is not None


def test_list_recent(db: Database) -> None:
    """Verify listing recent runs."""
    repo = RunRepository(db)
    r1 = repo.start_run(config={"id": 1})
    r2 = repo.start_run(config={"id": 2})
    r3 = repo.start_run(config={"id": 3})

    recent = repo.list_recent(limit=2)
    assert len(recent) == 2
    assert recent[0]["run_id"] == r3
    assert recent[1]["run_id"] == r2


def test_cascade_delete_metrics(db: Database) -> None:
    """Verify deleting a training run cascades to delete its metric history."""
    run_repo = RunRepository(db)
    metrics_repo = MetricsRepository(db)

    run_id = run_repo.start_run(config={})
    metrics_repo.insert(run_id=run_id, iteration=1, loss=0.5)
    metrics_repo.insert(run_id=run_id, iteration=2, loss=0.4)

    assert len(metrics_repo.get_run_metrics(run_id)) == 2

    # Delete run directly
    db.execute("DELETE FROM training_runs WHERE run_id = ?;", (run_id,))

    # Metrics should be automatically deleted via ON DELETE CASCADE
    assert len(metrics_repo.get_run_metrics(run_id)) == 0
