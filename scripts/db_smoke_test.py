"""Runtime smoke test for SQLite database layer (Prompt 5 verification).

Validates schema creation, migrations, seeding, repository operations,
checkpoint registration, training run lifecycle, and CSV export against
the real persistent data/tactical_marl.db database file.
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database.agent_repo import AgentRepository
from src.database.checkpoint_repo import CheckpointRepository
from src.database.config import DB_PATH
from src.database.db import Database
from src.database.metrics_repo import MetricsRepository
from src.database.migrations import SchemaManager
from src.database.run_repo import RunRepository
from src.database.scenario_repo import ScenarioRepository
from src.database.seed import seed_all
from src.simulator.scenarios import ScenarioConfig


def run_smoke_test() -> None:
    print(f"[*] Starting Database Smoke Test on {DB_PATH}...")
    db = Database(DB_PATH)

    # 1. Apply Schema Migrations
    print("[1/6] Applying schema migrations...")
    mgr = SchemaManager(db)
    mgr.apply_migrations()
    ver = mgr.current_version
    print(f"      Schema version verified: {ver}")
    assert ver == 1, f"Expected version 1, got {ver}"

    # 2. Seed Default Data
    print("[2/6] Seeding default scenarios and agent parameters...")
    seed_all(db)

    sc_repo = ScenarioRepository(db)
    ag_repo = AgentRepository(db)
    for lvl in range(1, 6):
        sc_list = sc_repo.list_by_level(lvl)
        assert len(sc_list) >= 1, f"Level {lvl} scenario missing"
        cfg = sc_repo.to_scenario_config(sc_list[0]["scenario_id"])
        assert isinstance(cfg, ScenarioConfig), "Conversion to ScenarioConfig failed"
    print("      Seeded 5 curriculum scenarios successfully verified.")

    agents = ag_repo.list_all()
    assert len(agents) >= 4, "Expected at least 4 agent parameter sets"
    print(f"      Seeded {len(agents)} agent parameter profiles successfully.")

    # 3. Training Run Lifecycle
    print("[3/6] Starting simulated training run...")
    run_repo = RunRepository(db)
    run_id = run_repo.start_run(
        config={"curriculum": "L1-L5", "smoke_test": True, "lr": 3e-4},
        git_commit="smoke_test_sha",
        notes="Automated runtime smoke test run",
    )
    assert run_id > 0
    active_run = run_repo.get(run_id)
    assert active_run is not None
    assert active_run["status"] == "running"
    print(f"      Created training run ID: {run_id}")

    # 4. Metrics Logging & Batch Insert
    print("[4/6] Logging training metrics...")
    metrics_repo = MetricsRepository(db)
    sample_metrics = [
        {
            "run_id": run_id,
            "iteration": i,
            "level": 1,
            "policy_name": "AirFightPolicy",
            "loss": 0.5 - (0.04 * i),
            "win_rate": 0.5 + (0.04 * i),
            "kill_death_ratio": 1.0 + (0.2 * i),
            "action_entropy": 0.35,
            "episode_length": 180.0,
        }
        for i in range(1, 11)
    ]
    metrics_repo.insert_batch(sample_metrics)
    logged = metrics_repo.get_run_metrics(run_id)
    assert len(logged) == 10, f"Expected 10 metrics, got {len(logged)}"
    latest_metric = metrics_repo.get_latest(run_id, "AirFightPolicy")
    assert latest_metric is not None
    assert latest_metric["iteration"] == 10
    print(f"      Logged and queried {len(logged)} metric steps. Latest win rate: {latest_metric['win_rate']:.2f}")

    # 5. Checkpoint Registration & Run Completion
    print("[5/6] Registering model checkpoint and completing run...")
    ckpt_repo = CheckpointRepository(db)
    ckpt_path = f"checkpoints/smoke_test_run_{run_id}_iter_00010.pt"
    ckpt_id = ckpt_repo.register(
        path=ckpt_path,
        policy_name="AirFightPolicy",
        stage="stage_a",
        iteration=10,
        win_rate=0.90,
        kill_death_ratio=3.0,
        param_count=235456,
        file_size_bytes=980000,
        notes="Smoke test checkpoint",
    )
    assert ckpt_id > 0
    fetched_ckpt = ckpt_repo.get_by_path(ckpt_path)
    assert fetched_ckpt is not None
    assert fetched_ckpt["checkpoint_id"] == ckpt_id

    run_repo.complete_run(
        run_id=run_id,
        total_iterations=10,
        final_win_rate=0.90,
        final_kd_ratio=3.0,
        notes="Smoke test run completed successfully",
    )
    completed_run = run_repo.get(run_id)
    assert completed_run is not None
    assert completed_run["status"] == "completed"
    print(f"      Checkpoint {ckpt_id} registered and run {run_id} marked completed.")

    # 6. CSV Export
    print("[6/6] Exporting run metrics to CSV...")
    csv_out = "data/smoke_test_metrics.csv"
    metrics_repo.export_csv(run_id, csv_out)
    csv_path = Path(csv_out)
    assert csv_path.exists()
    assert csv_path.stat().st_size > 0
    print(f"      CSV exported successfully to {csv_out} ({csv_path.stat().st_size} bytes).")

    db.close()
    print("\n========================================================")
    print("  DATABASE SMOKE TEST COMPLETED SUCCESSFULLY (PASSED)   ")
    print("========================================================")


if __name__ == "__main__":
    try:
        run_smoke_test()
    except Exception as exc:
        print(f"\n[!] Smoke test FAILED: {exc}", file=sys.stderr)
        raise
