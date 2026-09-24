"""Metrics logging to CSV and TensorBoard for tactical scenario MARL.

Logs training loss curves, curriculum level, win rates, episode stats,
and non-determinism benchmarks. Supports graceful fallback if TensorBoard
is not present in the runtime environment.
"""

import csv
from pathlib import Path
from typing import Any

from src.training.config import LOG_DIR, METRICS_CSV

try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    SummaryWriter = None  # type: ignore
    _TENSORBOARD_AVAILABLE = False


class MetricsLogger:
    """Logs iteration losses, episode outcomes, and evaluations to CSV and TensorBoard."""

    CSV_HEADERS: list[str] = [
        "iteration",
        "level",
        "total_loss",
        "policy_loss",
        "value_loss",
        "entropy_loss",
        "blue_wins",
        "red_wins",
        "draws",
        "win_rate",
        "mean_reward",
        "mean_episode_length",
    ]

    def __init__(
        self,
        log_dir: str | Path = LOG_DIR,
        csv_path: str | Path = METRICS_CSV,
        db: Any = None,
        run_id: int | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.csv_path = Path(csv_path)
        self.db = db
        self.run_id = run_id

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        self.tb_writer: Any = None
        if _TENSORBOARD_AVAILABLE and SummaryWriter is not None:
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir))

        # Initialize CSV file with headers if it does not already exist
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADERS)

    def log_iteration(
        self,
        iteration: int,
        level: int,
        loss_dict: dict[str, float],
        episode_stats: dict[str, Any],
    ) -> None:
        """Record one training iteration to CSV and TensorBoard."""
        blue_wins = int(episode_stats.get("blue_wins", 0))
        red_wins = int(episode_stats.get("red_wins", 0))
        draws = int(episode_stats.get("draws", 0))
        total_eps = blue_wins + red_wins + draws
        win_rate = (blue_wins / total_eps) if total_eps > 0 else 0.0

        row = [
            iteration,
            level,
            float(loss_dict.get("total_loss", 0.0)),
            float(loss_dict.get("policy_loss", 0.0)),
            float(loss_dict.get("value_loss", 0.0)),
            float(loss_dict.get("entropy_loss", 0.0)),
            blue_wins,
            red_wins,
            draws,
            win_rate,
            float(episode_stats.get("mean_reward", 0.0)),
            float(episode_stats.get("mean_episode_length", 0.0)),
        ]

        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        if self.tb_writer is not None:
            self.tb_writer.add_scalar("Curriculum/Level", level, iteration)
            self.tb_writer.add_scalar("Train/WinRate", win_rate, iteration)
            self.tb_writer.add_scalar("Train/MeanReward", float(episode_stats.get("mean_reward", 0.0)), iteration)
            for k, v in loss_dict.items():
                self.tb_writer.add_scalar(f"Loss/{k}", float(v), iteration)

        if self.db is not None and self.run_id is not None:
            try:
                from src.database.metrics_repo import MetricsRepository
                MetricsRepository(self.db).insert(
                    run_id=self.run_id,
                    iteration=iteration,
                    level=level,
                    policy_name="all",
                    loss=float(loss_dict.get("total_loss", 0.0)),
                    win_rate=win_rate,
                    kill_death_ratio=float(episode_stats.get("kill_death_ratio", 0.0)) if "kill_death_ratio" in episode_stats else None,
                    action_entropy=float(loss_dict.get("entropy_loss", 0.0)),
                    episode_length=float(episode_stats.get("mean_episode_length", 0.0)),
                )
            except Exception:
                pass

    def log_evaluation(self, iteration: int, eval_dict: dict[str, Any]) -> None:
        """Record evaluation metrics to TensorBoard."""
        if self.tb_writer is not None:
            for k, v in eval_dict.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(f"Eval/{k}", float(v), iteration)

    def log_non_determinism(self, iteration: int, nd_dict: dict[str, Any]) -> None:
        """Record non-determinism metrics to TensorBoard."""
        if self.tb_writer is not None:
            for k, v in nd_dict.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(f"NonDeterminism/{k}", float(v), iteration)

    def close(self) -> None:
        """Flush and close all open logging handles."""
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
            self.tb_writer = None
