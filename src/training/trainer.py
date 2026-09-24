"""Core training loop for Stage A (low-level policies) and Stage B (commander).

Handles rollout collection, GAE advantage calculation, PPO/HHAPPO updates,
curriculum progression, and league snapshot additions.
"""

from typing import Any
import numpy as np
import torch

from src.marl.config import GAE_LAMBDA, GAMMA
from src.training.checkpoint import CheckpointManager
from src.training.config import (
    CHECKPOINT_INTERVAL,
    EVAL_INTERVAL,
    LEAGUE_SAVE_INTERVAL,
    LOG_INTERVAL,
    ROLLOUT_STEPS,
)
from src.training.curriculum import CurriculumScheduler
from src.training.evaluator import Evaluator
from src.training.league import League
from src.training.metrics import MetricsLogger
from src.training.rollout_collector import RolloutCollector


class Trainer:
    """Coordinates iterative MARL training, curriculum progression, and league snapshots."""

    def __init__(
        self,
        policies: dict[str, Any],
        optimizer: torch.optim.Optimizer | dict[str, torch.optim.Optimizer] | None,
        rollout_collector: RolloutCollector,
        config: dict[str, Any] | None = None,
        metrics_logger: MetricsLogger | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.policies = policies
        self.optimizer = optimizer
        self.collector = rollout_collector
        self.config = config or {}
        self.logger = metrics_logger
        self.checkpoint_manager = checkpoint_manager
        self.rng = rng if rng is not None else np.random.default_rng(42)

        self.evaluator = Evaluator(policies=self.policies, config=self.config, rng=self.rng)

        self.rollout_steps = int(self.config.get("rollout_steps", ROLLOUT_STEPS))
        self.checkpoint_interval = int(self.config.get("checkpoint_interval", CHECKPOINT_INTERVAL))
        self.log_interval = int(self.config.get("log_interval", LOG_INTERVAL))
        self.eval_interval = int(self.config.get("eval_interval", EVAL_INTERVAL))
        self.league_save_interval = int(self.config.get("league_save_interval", LEAGUE_SAVE_INTERVAL))

    def train_iteration(self, target_policies: list[str] | None = None) -> dict[str, Any]:
        """Execute one full training iteration across environments.

        Steps:
          1. Collect rollouts across environments.
          2. Compute GAE and returns for active policy buffers.
          3. Perform PPO/HHAPPO updates on policies.
          4. Clear buffers.
          5. Return metrics dictionary.
        """
        # 1. Collect rollouts
        buffers = self.collector.collect(steps=self.rollout_steps)
        stats = self.collector.get_episode_stats()

        # Determine which policies to update
        active_names = target_policies if target_policies is not None else list(self.policies.keys())

        iteration_losses: dict[str, float] = {}
        total_loss_accum = 0.0
        policy_loss_accum = 0.0
        value_loss_accum = 0.0
        entropy_loss_accum = 0.0
        updates_performed = 0

        # 2. Update each active policy
        for name in active_names:
            policy = self.policies.get(name)
            buffer = buffers.get(name)

            if policy is None or buffer is None or len(buffer) < 10:
                continue

            # Compute GAE advantages
            buffer.compute_advantages_and_returns(
                last_value=0.0,
                gamma=GAMMA,
                gae_lambda=GAE_LAMBDA,
            )

            # Update policy
            loss_dict = policy.update(buffer)
            for k, v in loss_dict.items():
                iteration_losses[f"{name}_{k}"] = v

            total_loss_accum += loss_dict.get("total_loss", 0.0)
            policy_loss_accum += loss_dict.get("policy_loss", 0.0)
            value_loss_accum += loss_dict.get("value_loss", 0.0)
            entropy_loss_accum += loss_dict.get("entropy_loss", 0.0)
            updates_performed += 1

            # Clear buffer after on-policy update
            buffer.clear()

        # Mean loss summary
        divisor = max(1, updates_performed)
        summary_losses = {
            "total_loss": total_loss_accum / divisor,
            "policy_loss": policy_loss_accum / divisor,
            "value_loss": value_loss_accum / divisor,
            "entropy_loss": entropy_loss_accum / divisor,
        }
        summary_losses.update(iteration_losses)

        return {
            "losses": summary_losses,
            "episode_stats": stats,
        }

    def train_low_level(
        self,
        curriculum: CurriculumScheduler,
        league: League,
        num_iterations: int,
    ) -> dict[str, Any]:
        """Stage A: Train all 6 low-level domain policies with curriculum & league snapshots."""
        low_level_names = [
            "air_fight", "air_escape",
            "ground_engage", "ground_defend",
            "sea_engage", "sea_defend",
        ]

        print(f"Starting Stage A training for {num_iterations} iterations...")
        latest_eval: dict[str, Any] = {}

        for iteration in range(1, num_iterations + 1):
            # Update opponent policy for L4 / L5
            cur_level = curriculum.current_level
            if cur_level == 4:
                # Use recent snapshot if available
                sample_weights = league.sample(rng=self.rng)
                if sample_weights is not None:
                    opp = self.policies.get("air_fight")
                    self.collector.set_opponent_policy(opp)
            elif cur_level >= 5:
                # Sample from league pool with scripted fallback
                src, weights = league.sample_with_scripted_fallback(rng=self.rng)
                if src == "league" and weights is not None:
                    opp = self.policies.get("air_fight")
                    self.collector.set_opponent_policy(opp)
                else:
                    self.collector.set_opponent_policy(None)
            else:
                self.collector.set_opponent_policy(None)

            # Perform iteration
            metrics = self.train_iteration(target_policies=low_level_names)
            losses = metrics["losses"]
            stats = metrics["episode_stats"]

            # Record episode outcomes to curriculum
            blue_wins = stats.get("blue_wins", 0)
            red_wins = stats.get("red_wins", 0)
            total = blue_wins + red_wins
            if total > 0:
                curriculum.record_episode(win=(blue_wins >= red_wins))

            # Check curriculum progression
            if curriculum.should_advance():
                advanced = curriculum.advance()
                if advanced:
                    print(f"[Iter {iteration}] Advanced curriculum to Level {curriculum.current_level}!")
                    # Update scenario in environments
                    new_scenario = curriculum.get_scenario_config()
                    for env in self.collector.envs:
                        env.reset(scenario_config=new_scenario)

            # Add snapshot to league pool at L4+
            if cur_level >= 4 and iteration % self.league_save_interval == 0:
                policy_snapshots = {
                    name: self.policies[name].state_dict()
                    for name in low_level_names if name in self.policies
                }
                league.add(policy_snapshots, iteration=iteration)

            # Log metrics
            if self.logger is not None and iteration % self.log_interval == 0:
                self.logger.log_iteration(
                    iteration=iteration,
                    level=curriculum.current_level,
                    loss_dict=losses,
                    episode_stats=stats,
                )

            # Save periodic checkpoints
            if self.checkpoint_manager is not None and iteration % self.checkpoint_interval == 0:
                self.checkpoint_manager.save_policies(
                    policies=self.policies,
                    iteration=iteration,
                    stage="stage_a",
                    metadata={"level": curriculum.current_level, "win_rate": stats.get("win_rate", 0.0)},
                )

            # Evaluation
            if iteration % self.eval_interval == 0:
                latest_eval = self.evaluate(num_episodes=20)
                print(f"[Iter {iteration}] Win rate: {latest_eval.get('win_rate', 0.0):.2f}, Level: {curriculum.current_level}")
                if self.logger is not None:
                    self.logger.log_evaluation(iteration=iteration, eval_dict=latest_eval)

        return {
            "stage": "stage_a",
            "iterations_completed": num_iterations,
            "final_level": curriculum.current_level,
            "latest_eval": latest_eval,
        }

    def train_commander(
        self,
        curriculum: CurriculumScheduler,
        num_iterations: int,
    ) -> dict[str, Any]:
        """Stage B: Train high-level Commander policy with frozen low-level policies."""
        # 1. Freeze low-level policies
        for name, policy in self.policies.items():
            if name != "commander":
                policy.eval()
                for param in policy.parameters():
                    param.requires_grad = False

        # Ensure Commander policy is in training mode
        cmd_policy = self.policies.get("commander")
        if cmd_policy is not None:
            cmd_policy.train()
            for param in cmd_policy.parameters():
                param.requires_grad = True

        print(f"Starting Stage B (Commander) training for {num_iterations} iterations...")
        latest_eval: dict[str, Any] = {}

        for iteration in range(1, num_iterations + 1):
            metrics = self.train_iteration(target_policies=["commander"])
            losses = metrics["losses"]
            stats = metrics["episode_stats"]

            # Log metrics
            if self.logger is not None and iteration % self.log_interval == 0:
                self.logger.log_iteration(
                    iteration=iteration,
                    level=curriculum.current_level,
                    loss_dict=losses,
                    episode_stats=stats,
                )

            # Save periodic checkpoints
            if self.checkpoint_manager is not None and iteration % self.checkpoint_interval == 0:
                self.checkpoint_manager.save_policies(
                    policies=self.policies,
                    iteration=iteration,
                    stage="stage_b",
                    metadata={"level": curriculum.current_level, "stage": "stage_b"},
                )

            # Evaluation
            if iteration % self.eval_interval == 0:
                latest_eval = self.evaluate(num_episodes=20)
                print(f"[Stage B Iter {iteration}] Win rate: {latest_eval.get('win_rate', 0.0):.2f}")
                if self.logger is not None:
                    self.logger.log_evaluation(iteration=iteration, eval_dict=latest_eval)

        return {
            "stage": "stage_b",
            "iterations_completed": num_iterations,
            "latest_eval": latest_eval,
        }

    def evaluate(self, num_episodes: int = 50) -> dict[str, Any]:
        """Run policy evaluation against scripted baselines."""
        return self.evaluator.evaluate(num_episodes=num_episodes)
