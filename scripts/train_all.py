"""CLI script for full staged training pipeline: Stage A → freeze → Stage B.

Usage:
    uv run --with torch --with numpy python scripts/train_all.py --dry-run
    uv run --with torch --with numpy python scripts/train_all.py --iterations 5000 --config configs/training_default.yaml
"""

import argparse
from pathlib import Path
import sys
from typing import Any

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from src.training.config import get_default_config
from src.training.staged_trainer import StagedTrainer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Full Multi-Domain MARL Staged Training (Stage A → Stage B)")
    parser.add_argument("--iterations", type=int, default=None, help="Number of training iterations per stage")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--dry-run", action="store_true", help="Run 5 verification iterations per stage")
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
                    for k, v in custom.items():
                        if isinstance(v, dict):
                            cfg.update(v)
                        else:
                            cfg[k] = v
    return cfg


def main() -> None:
    """CLI execution entrypoint for full staged training."""
    args = parse_args()
    config = load_config(args.config)

    if args.dry_run:
        iterations = 5
        config["num_parallel_envs"] = 2
        config["rollout_steps"] = 16
        config["eval_interval"] = 5
        config["log_interval"] = 1
        config["checkpoint_interval"] = 5
    else:
        iterations = args.iterations if args.iterations is not None else config.get("total_iterations", 5000)

    print("=" * 60)
    print("MULTI-DOMAIN MARL FULL STAGED TRAINING PIPELINE")
    print(f"Iterations per stage: {iterations} {'(DRY RUN)' if args.dry_run else ''}")
    print("=" * 60)

    staged_trainer = StagedTrainer(config=config)

    try:
        results = staged_trainer.run(
            iterations_stage_a=iterations,
            iterations_stage_b=iterations,
        )
        print("\n" + "=" * 60)
        print("STAGED TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Stage A checkpoint: {results['checkpoints']['stage_a']}")
        print(f"Stage B checkpoint: {results['checkpoints']['stage_b']}")
        print(f"Final checkpoint:   {results['checkpoints']['final']}")
        print(f"Final Win Rate:     {results['final_eval']['win_rate']:.2%}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nFull training interrupted by user.")
        emergency_path = staged_trainer.ckpt_manager.save_policies(
            policies=staged_trainer.policies,
            iteration=0,
            stage="pipeline_interrupted",
            metadata={"reason": "KeyboardInterrupt"},
        )
        print(f"Emergency checkpoint saved to: {emergency_path}")

    finally:
        staged_trainer.logger.close()


if __name__ == "__main__":
    main()
