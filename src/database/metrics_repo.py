"""Metrics repository for logging, querying, and exporting training metrics."""

import csv
from pathlib import Path
from typing import Any

from src.database.config import BATCH_INSERT_SIZE
from src.database.db import Database


class MetricsRepository:
    """Repository for managing training run metrics history."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def insert(
        self,
        run_id: int,
        iteration: int,
        level: int | None = None,
        policy_name: str | None = None,
        loss: float | None = None,
        win_rate: float | None = None,
        kill_death_ratio: float | None = None,
        action_entropy: float | None = None,
        episode_length: float | None = None,
    ) -> int:
        """Insert a single metric record.

        Returns:
            The metric_id of the inserted record.
        """
        query = """
        INSERT INTO metrics_history (
            run_id, iteration, level, policy_name,
            loss, win_rate, kill_death_ratio, action_entropy, episode_length
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            run_id,
            iteration,
            level,
            policy_name,
            loss,
            win_rate,
            kill_death_ratio,
            action_entropy,
            episode_length,
        )
        return self.db.execute(query, params)

    def insert_batch(self, rows: list[dict[str, Any]]) -> None:
        """Insert a batch of metric records in chunks of BATCH_INSERT_SIZE."""
        if not rows:
            return

        query = """
        INSERT INTO metrics_history (
            run_id, iteration, level, policy_name,
            loss, win_rate, kill_death_ratio, action_entropy, episode_length
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        for i in range(0, len(rows), BATCH_INSERT_SIZE):
            chunk = rows[i : i + BATCH_INSERT_SIZE]
            params_list = [
                (
                    r.get("run_id"),
                    r.get("iteration"),
                    r.get("level"),
                    r.get("policy_name"),
                    r.get("loss"),
                    r.get("win_rate"),
                    r.get("kill_death_ratio"),
                    r.get("action_entropy"),
                    r.get("episode_length"),
                )
                for r in chunk
            ]
            self.db.execute_many(query, params_list)

    def get_run_metrics(self, run_id: int) -> list[dict[str, Any]]:
        """Retrieve all metrics for a given run_id ordered by iteration."""
        query = """
        SELECT metric_id, run_id, iteration, level, policy_name,
               loss, win_rate, kill_death_ratio, action_entropy,
               episode_length, recorded_at
        FROM metrics_history
        WHERE run_id = ?
        ORDER BY iteration ASC, metric_id ASC;
        """
        return self.db.fetch_all(query, (run_id,))

    def get_latest(self, run_id: int, policy_name: str) -> dict[str, Any] | None:
        """Retrieve the most recent metric for a specific run and policy."""
        query = """
        SELECT metric_id, run_id, iteration, level, policy_name,
               loss, win_rate, kill_death_ratio, action_entropy,
               episode_length, recorded_at
        FROM metrics_history
        WHERE run_id = ? AND policy_name = ?
        ORDER BY iteration DESC, metric_id DESC
        LIMIT 1;
        """
        return self.db.fetch_one(query, (run_id, policy_name))

    def export_csv(self, run_id: int, path: str) -> None:
        """Export all metrics for a given run to a CSV file."""
        metrics = self.get_run_metrics(run_id)
        fieldnames = [
            "metric_id",
            "run_id",
            "iteration",
            "level",
            "policy_name",
            "loss",
            "win_rate",
            "kill_death_ratio",
            "action_entropy",
            "episode_length",
            "recorded_at",
        ]

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            if metrics:
                writer.writerows(metrics)
