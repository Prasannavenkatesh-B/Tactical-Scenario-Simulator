"""Model checkpoint repository for tracking neural network artifacts and evaluation metrics."""

from typing import Any

from src.database.db import Database


class CheckpointRepository:
    """Repository for registering and querying trained model checkpoints."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def register(
        self,
        path: str,
        policy_name: str,
        stage: str,
        iteration: int,
        win_rate: float | None = None,
        kill_death_ratio: float | None = None,
        param_count: int | None = None,
        file_size_bytes: int | None = None,
        notes: str = "",
    ) -> int:
        """Register a new policy checkpoint artifact.

        Returns:
            The newly created checkpoint_id.
        """
        query = """
        INSERT INTO model_checkpoints (
            path, policy_name, stage, iteration, win_rate,
            kill_death_ratio, param_count, file_size_bytes, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            str(path),
            str(policy_name),
            str(stage),
            int(iteration),
            float(win_rate) if win_rate is not None else None,
            float(kill_death_ratio) if kill_death_ratio is not None else None,
            int(param_count) if param_count is not None else None,
            int(file_size_bytes) if file_size_bytes is not None else None,
            str(notes),
        )
        return self.db.execute(query, params)

    def get(self, checkpoint_id: int) -> dict[str, Any] | None:
        """Retrieve a checkpoint record by checkpoint_id."""
        query = "SELECT * FROM model_checkpoints WHERE checkpoint_id = ?;"
        return self.db.fetch_one(query, (checkpoint_id,))

    def get_by_path(self, path: str) -> dict[str, Any] | None:
        """Retrieve a checkpoint record by unique file path."""
        query = "SELECT * FROM model_checkpoints WHERE path = ?;"
        return self.db.fetch_one(query, (str(path),))

    def list_by_policy(self, policy_name: str) -> list[dict[str, Any]]:
        """List all checkpoint records for a given policy name (newest first)."""
        query = """
        SELECT * FROM model_checkpoints
        WHERE policy_name = ?
        ORDER BY iteration DESC, checkpoint_id DESC;
        """
        return self.db.fetch_all(query, (policy_name,))

    def list_latest(self, n: int = 10) -> list[dict[str, Any]]:
        """List the latest checkpoint per policy."""
        query = """
        SELECT m.* FROM model_checkpoints m
        INNER JOIN (
            SELECT policy_name, MAX(iteration) AS max_iter
            FROM model_checkpoints
            GROUP BY policy_name
        ) r ON m.policy_name = r.policy_name AND m.iteration = r.max_iter
        ORDER BY m.policy_name ASC
        LIMIT ?;
        """
        return self.db.fetch_all(query, (n,))

    def delete(self, checkpoint_id: int) -> None:
        """Delete a checkpoint record by checkpoint_id."""
        query = "DELETE FROM model_checkpoints WHERE checkpoint_id = ?;"
        self.db.execute(query, (checkpoint_id,))
