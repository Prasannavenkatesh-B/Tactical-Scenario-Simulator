"""Staged training orchestrator: Stage A (low-level) → freeze → Stage B (commander).

Orchestrates the entire multi-domain reinforcement learning pipeline
from ground zero up to fully trained, freeze-validated tactical policy checkpoints.
"""

from pathlib import Path
from typing import Any
import numpy as np
import torch

from src.marl.commander import CommanderPolicy
from src.marl.policies.air_escape import AirEscapePolicy
from src.marl.policies.air_fight import AirFightPolicy
from src.marl.policies.ground_defend import GroundDefendPolicy
from src.marl.policies.ground_engage import GroundEngagePolicy
from src.marl.policies.sea_defend import SeaDefendPolicy
from src.marl.policies.sea_engage import SeaEngagePolicy
from src.simulator.env import TacticalEnv
from src.training.checkpoint import CheckpointManager
from src.training.config import (
    CHECKPOINT_DIR,
    FINAL_DIR,
    LEAGUE_DIR,
    LOG_DIR,
    METRICS_CSV,
    NUM_PARALLEL_ENVS,
    STAGE_A_DIR,
    STAGE_B_DIR,
    TOTAL_ITERATIONS,
    get_default_config,
)
from src.training.curriculum import CurriculumScheduler
from src.training.evaluator import Evaluator
from src.training.league import League
from src.training.metrics import MetricsLogger
from src.training.rollout_collector import RolloutCollector
from src.training.trainer import Trainer


class StagedTrainer:
    """Orchestrates Stage A (low-level) → freeze → Stage B (commander)."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = get_default_config()
        if config is not None:
            self.config.update(config)

        self.rng = np.random.default_rng(42)

        # Paths
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", CHECKPOINT_DIR))
        self.stage_a_dir = Path(self.config.get("stage_a_dir", STAGE_A_DIR))
        self.stage_b_dir = Path(self.config.get("stage_b_dir", STAGE_B_DIR))
        self.final_dir = Path(self.config.get("final_dir", FINAL_DIR))
        self.league_dir = Path(self.config.get("league_dir", LEAGUE_DIR))
        self.log_dir = Path(self.config.get("log_dir", LOG_DIR))
        self.csv_path = Path(self.config.get("metrics_csv", METRICS_CSV))

        # Checkpoint & metric managers
        self.ckpt_manager = CheckpointManager(save_dir=self.checkpoint_dir)
        self.logger = MetricsLogger(log_dir=self.log_dir, csv_path=self.csv_path)

        # Policies
        self.policies = self._instantiate_policies()

        # Parallel environments
        num_envs = int(self.config.get("num_parallel_envs", NUM_PARALLEL_ENVS))
        self.curriculum = CurriculumScheduler(config=self.config, rng=self.rng)
        initial_scenario = self.curriculum.get_scenario_config()

        self.envs = [
            TacticalEnv(scenario_config=initial_scenario, seed=int(self.rng.integers(1, 100_000)))
            for _ in range(num_envs)
        ]

        # League & Rollout collector
        self.league = League(save_dir=str(self.league_dir))
        self.collector = RolloutCollector(
            policies=self.policies,
            envs=self.envs,
            config=self.config,
            rng=self.rng,
        )

        # Trainer instance
        self.trainer = Trainer(
            policies=self.policies,
            optimizer=None,
            rollout_collector=self.collector,
            config=self.config,
            metrics_logger=self.logger,
            checkpoint_manager=self.ckpt_manager,
            rng=self.rng,
        )

        self.evaluator = Evaluator(policies=self.policies, config=self.config, rng=self.rng)

    def _instantiate_policies(self) -> dict[str, Any]:
        """Instantiate all 9 policies (6 low-level variants + commander)."""
        return {
            "air_fight": AirFightPolicy(variant="AC1", config=self.config),
            "air_fight_ac2": AirFightPolicy(variant="AC2", config=self.config),
            "air_escape": AirEscapePolicy(variant="AC1", config=self.config),
            "air_escape_ac2": AirEscapePolicy(variant="AC2", config=self.config),
            "ground_engage": GroundEngagePolicy(config=self.config),
            "ground_defend": GroundDefendPolicy(config=self.config),
            "sea_engage": SeaEngagePolicy(config=self.config),
            "sea_defend": SeaDefendPolicy(config=self.config),
            "commander": CommanderPolicy(config=self.config),
        }

    def run(
        self,
        iterations_stage_a: int | None = None,
        iterations_stage_b: int | None = None,
    ) -> dict[str, Any]:
        """Execute full training pipeline from Stage A to Stage B.

        Returns:
            Dictionary with Stage A, Stage B, and final evaluation results.
        """
        iter_a = iterations_stage_a if iterations_stage_a is not None else int(self.config.get("total_iterations", TOTAL_ITERATIONS))
        iter_b = iterations_stage_b if iterations_stage_b is not None else int(self.config.get("total_iterations", TOTAL_ITERATIONS))

        print("=" * 60)
        print("STARTING FULL STAGED MARL TRAINING")
        print(f"Stage A iterations: {iter_a} | Stage B iterations: {iter_b}")
        print("=" * 60)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STAGE A: Low-Level Policy Training
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        print("\n--- STAGE A: Training 6 Low-Level Domain Policies ---")
        stage_a_res = self.trainer.train_low_level(
            curriculum=self.curriculum,
            league=self.league,
            num_iterations=iter_a,
        )

        # Freeze low-level policies
        print("\nFreezing low-level policies...")
        for name, policy in self.policies.items():
            if name != "commander":
                policy.eval()
                for param in policy.parameters():
                    param.requires_grad = False

        # Save Stage A checkpoints
        stage_a_mgr = CheckpointManager(save_dir=self.stage_a_dir)
        stage_a_path = stage_a_mgr.save_policies(
            policies=self.policies,
            iteration=iter_a,
            stage="stage_a",
            metadata={"description": "Frozen Stage A low-level policies"},
        )
        print(f"Stage A checkpoints saved: {stage_a_path}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STAGE B: High-Level Commander Training
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        print("\n--- STAGE B: Training Commander Policy ---")
        stage_b_res = self.trainer.train_commander(
            curriculum=self.curriculum,
            num_iterations=iter_b,
        )

        # Save Stage B and final checkpoints
        stage_b_mgr = CheckpointManager(save_dir=self.stage_b_dir)
        stage_b_path = stage_b_mgr.save_policies(
            policies=self.policies,
            iteration=iter_b,
            stage="stage_b",
            metadata={"description": "Stage B commander policy"},
        )
        print(f"Stage B checkpoints saved: {stage_b_path}")

        final_mgr = CheckpointManager(save_dir=self.final_dir)
        final_path = final_mgr.save_policies(
            policies=self.policies,
            iteration=iter_a + iter_b,
            stage="final",
            metadata={"description": "Final integrated multi-domain tactical policies"},
        )
        print(f"Final integrated checkpoints saved: {final_path}")

        # Final evaluation
        print("\n--- Running Final Evaluation ---")
        final_eval = self.evaluator.evaluate(num_episodes=20)
        print(f"Final Win Rate: {final_eval['win_rate']:.2%}")
        print(f"Final Kill/Death Ratio: {final_eval['kill_death_ratio']:.2f}")

        self.logger.close()

        return {
            "stage_a": stage_a_res,
            "stage_b": stage_b_res,
            "final_eval": final_eval,
            "checkpoints": {
                "stage_a": stage_a_path,
                "stage_b": stage_b_path,
                "final": final_path,
            },
        }
