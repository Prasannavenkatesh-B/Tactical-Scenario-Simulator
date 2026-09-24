"""CLI script to train low-level domain policies (Stage A).

Usage:
    uv run --with torch --with numpy python scripts/train_low_level.py --dry-run
    uv run --with torch --with numpy python scripts/train_low_level.py --iterations 1000 --config configs/training_default.yaml
"""

import argparse
from pathlib import Path
import sys
from typing import Any

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from src.marl.policies.air_escape import AirEscapePolicy
from src.marl.policies.air_fight import AirFightPolicy
from src.marl.policies.ground_defend import GroundDefendPolicy
from src.marl.policies.ground_engage import GroundEngagePolicy
from src.marl.policies.sea_defend import SeaDefendPolicy
from src.marl.policies.sea_engage import SeaEngagePolicy
from src.simulator.env import TacticalEnv
from src.training.checkpoint import CheckpointManager
from src.training.config import (
    LEAGUE_DIR,
    LOG_DIR,
    METRICS_CSV,
    NUM_PARALLEL_ENVS,
    STAGE_A_DIR,
    get_default_config,
)
from src.training.curriculum import CurriculumScheduler
from src.training.league import League
from src.training.metrics import MetricsLogger
from src.training.rollout_collector import RolloutCollector
from src.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stage A: Train 6 Low-Level Domain Policies")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--dry-run", action="store_true", help="Run 5 verification iterations only")
    return parser.parse_args()


def load_config(config_path: str | None) -> dict[str, Any]:
    """Load configuration from YAML file or return defaults."""
    cfg = get_default_config()
    if config_path is not None:
        p = Path(config_path)
        if p.exists() and yaml is not None:
            with open(p, "r", encoding="utf-8") as f:
                custom = yaml.safe_load(f)
                if isinstance(custom, dict):
                    # Flatten top-level sections if present
                    for k, v in custom.items():
                        if isinstance(v, dict):
                            cfg.update(v)
                        else:
                            cfg[k] = v
    return cfg


def main() -> None:
    """CLI execution entrypoint for Stage A training."""
    args = parse_args()
    config = load_config(args.config)
    iterations = 5 if args.dry_run else args.iterations
    rng = np.random.default_rng(42)

    print("=" * 60)
    print("STAGE A: LOW-LEVEL POLICIES TRAINING")
    print(f"Iterations: {iterations} {'(DRY RUN)' if args.dry_run else ''}")
    print("=" * 60)

    # 1. Instantiate 6 domain policies
    policies = {
        "air_fight": AirFightPolicy(variant="AC1", config=config),
        "air_escape": AirEscapePolicy(variant="AC1", config=config),
        "ground_engage": GroundEngagePolicy(config=config),
        "ground_defend": GroundDefendPolicy(config=config),
        "sea_engage": SeaEngagePolicy(config=config),
        "sea_defend": SeaDefendPolicy(config=config),
    }

    ckpt_mgr = CheckpointManager(save_dir=config.get("stage_a_dir", STAGE_A_DIR))

    # Optional resume
    if args.resume is not None:
        print(f"Resuming policies from: {args.resume}")
        ckpt_mgr.load_policies(args.resume, policies)

    # 2. Environments & Curriculum
    curriculum = CurriculumScheduler(config=config, rng=rng)
    num_envs = 2 if args.dry_run else int(config.get("num_parallel_envs", NUM_PARALLEL_ENVS))
    initial_scenario = curriculum.get_scenario_config()

    envs = [
        TacticalEnv(scenario_config=initial_scenario, seed=int(rng.integers(1, 100_000)))
        for _ in range(num_envs)
    ]

    league = League(save_dir=config.get("league_dir", LEAGUE_DIR))
    logger = MetricsLogger(
        log_dir=config.get("log_dir", LOG_DIR),
        csv_path=config.get("metrics_csv", METRICS_CSV),
    )

    collector = RolloutCollector(
        policies=policies,
        envs=envs,
        config=config,
        rng=rng,
    )

    trainer = Trainer(
        policies=policies,
        optimizer=None,
        rollout_collector=collector,
        config=config,
        metrics_logger=logger,
        checkpoint_manager=ckpt_mgr,
        rng=rng,
    )

    try:
        results = trainer.train_low_level(
            curriculum=curriculum,
            league=league,
            num_iterations=iterations,
        )
        saved_path = ckpt_mgr.save_policies(
            policies=policies,
            iteration=iterations,
            stage="stage_a",
            metadata={"final_level": results.get("final_level", 1)},
        )
        print(f"\nStage A completed successfully! Checkpoint saved to: {saved_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving emergency checkpoint...")
        saved_path = ckpt_mgr.save_policies(
            policies=policies,
            iteration=0,
            stage="stage_a_interrupted",
            metadata={"reason": "KeyboardInterrupt"},
        )
        print(f"Emergency checkpoint saved to: {saved_path}")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
