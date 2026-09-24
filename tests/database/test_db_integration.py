"""End-to-end integration tests for database layer and training hooks."""

from pathlib import Path
import pytest
import torch

from src.database.agent_repo import AgentRepository
from src.database.checkpoint_repo import CheckpointRepository
from src.database.db import Database
from src.database.metrics_repo import MetricsRepository
from src.database.run_repo import RunRepository
from src.database.scenario_repo import ScenarioRepository
from src.database.seed import seed_all
from src.simulator.scenarios import ScenarioConfig
from src.training.checkpoint import CheckpointManager
from src.training.metrics import MetricsLogger


def test_seed_all_idempotent(db: Database) -> None:
    """Verify seed_all creates 5 scenarios and 4 agent parameter sets idempotently."""
    seed_all(db)

    sc_repo = ScenarioRepository(db)
    agent_repo = AgentRepository(db)

    # 5 curriculum levels
    assert len(sc_repo.list_by_level(1)) == 1
    assert len(sc_repo.list_by_level(2)) == 1
    assert len(sc_repo.list_by_level(3)) == 1
    assert len(sc_repo.list_by_level(4)) == 1
    assert len(sc_repo.list_by_level(5)) == 1

    # 4 agent variants
    agents = agent_repo.list_all()
    assert len(agents) == 4
    agent_keys = {(a["domain"], a["variant"]) for a in agents}
    assert ("air", "AC1") in agent_keys
    assert ("air", "AC2") in agent_keys
    assert ("ground", "DEFAULT") in agent_keys
    assert ("sea", "DEFAULT") in agent_keys

    # Call seed_all again — should be idempotent
    seed_all(db)
    assert len(sc_repo.list_by_level(1)) == 1
    assert len(agent_repo.list_all()) == 4


def test_seed_scenarios_can_be_converted_to_config(db: Database) -> None:
    """Verify all seeded scenarios convert cleanly to simulator ScenarioConfigs."""
    seed_all(db)
    sc_repo = ScenarioRepository(db)

    for level in range(1, 6):
        sc_list = sc_repo.list_by_level(level)
        assert len(sc_list) == 1
        sc_id = sc_list[0]["scenario_id"]
        cfg = sc_repo.to_scenario_config(sc_id)
        assert isinstance(cfg, ScenarioConfig)
        assert len(cfg.blue_entities) > 0
        assert len(cfg.red_entities) > 0


def test_transaction_rollback(db: Database) -> None:
    """Verify unhandled exceptions trigger transaction rollback."""
    with pytest.raises(RuntimeError):
        with db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO scenarios (name, level, map_size_km, episode_horizon, blue_entities, red_entities)
                VALUES ('Rollback_Me', 1, 30.0, 200, '[]', '[]');
                """
            )
            raise RuntimeError("Simulated transaction failure")

    # The scenario should NOT exist in the database
    sc_repo = ScenarioRepository(db)
    assert sc_repo.get_by_name("Rollback_Me") is None


def test_training_pipeline_db_hooks_integration(db: Database, tmp_path: Path) -> None:
    """Verify CheckpointManager and MetricsLogger integrate with Database."""
    run_repo = RunRepository(db)
    ckpt_repo = CheckpointRepository(db)
    metrics_repo = MetricsRepository(db)

    run_id = run_repo.start_run(config={"seed": 42, "lr": 1e-4})

    # Test MetricsLogger hook
    logger = MetricsLogger(
        log_dir=tmp_path / "logs",
        csv_path=tmp_path / "logs" / "metrics.csv",
        db=db,
        run_id=run_id,
    )

    for i in range(1, 4):
        logger.log_iteration(
            iteration=i,
            level=1,
            loss_dict={"total_loss": 0.5 / i, "entropy_loss": 0.2},
            episode_stats={"blue_wins": i, "red_wins": 0, "draws": 0, "mean_episode_length": 100.0},
        )
    logger.close()

    db_metrics = metrics_repo.get_run_metrics(run_id)
    assert len(db_metrics) == 3
    assert db_metrics[-1]["iteration"] == 3

    # Test CheckpointManager hook
    dummy_policy = torch.nn.Linear(4, 2)
    ckpt_mgr = CheckpointManager(save_dir=tmp_path / "ckpts", db=db)
    saved_path = ckpt_mgr.save_policies(
        policies={"AirFight": dummy_policy},
        iteration=3,
        stage="stage_a",
        metadata={"win_rate": 1.0, "kd_ratio": 3.0},
    )

    db_ckpt = ckpt_repo.get_by_path(saved_path)
    assert db_ckpt is not None
    assert db_ckpt["policy_name"] == "AirFight"
    assert db_ckpt["iteration"] == 3
    assert db_ckpt["win_rate"] == 1.0

    # Complete run
    run_repo.complete_run(run_id=run_id, total_iterations=3, final_win_rate=1.0)
    final_run = run_repo.get(run_id)
    assert final_run is not None
    assert final_run["status"] == "completed"
