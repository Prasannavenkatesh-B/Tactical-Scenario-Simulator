"""Training run repository for managing experiment lifecycles and metadata."""

import json
from typing import Any

from src.database.config import JSON_SEPARATORS
from src.database.db import Database


class RunRepository:
    """Repository for managing training run records and lifecycles."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def _parse_row(self, row: dict[str, Any] | None) -> dict[str, Any] | None:
        """Parse config_json from string to dict and provide config alias."""
        if row is None:
            return None
        res = dict(row)
        if isinstance(res.get("config_json"), str):
            parsed = json.loads(res["config_json"])
            res["config_json"] = parsed
            res["config"] = parsed
        return res

    def start_run(
        self,
        config_json: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        git_commit: str = "",
        notes: str = "",
    ) -> int:
        """Create a new training run record marked with status 'running'."""
        cfg = config if config is not None else (config_json or {})
        cfg_str = json.dumps(cfg, separators=JSON_SEPARATORS, default=str)
        query = """
        INSERT INTO training_runs (
            config_json, git_commit, status, notes
        ) VALUES (?, ?, 'running', ?);
        """
        return self.db.execute(query, (cfg_str, git_commit, notes))

    def complete_run(
        self,
        run_id: int,
        total_iterations: int,
        final_win_rate: float,
        final_kd_ratio: float = 0.0,
        notes: str | None = None,
    ) -> None:
        """Mark a training run as 'completed' and record final metrics."""
        if notes is not None:
            query = """
            UPDATE training_runs
            SET status = 'completed',
                completed_at = CURRENT_TIMESTAMP,
                total_iterations = ?,
                final_win_rate = ?,
                final_kd_ratio = ?,
                notes = ?
            WHERE run_id = ?;
            """
            params = (int(total_iterations), float(final_win_rate), float(final_kd_ratio), str(notes), run_id)
        else:
            query = """
            UPDATE training_runs
            SET status = 'completed',
                completed_at = CURRENT_TIMESTAMP,
                total_iterations = ?,
                final_win_rate = ?,
                final_kd_ratio = ?
            WHERE run_id = ?;
            """
            params = (int(total_iterations), float(final_win_rate), float(final_kd_ratio), run_id)  # type: ignore

        self.db.execute(query, params)

    def fail_run(self, run_id: int, notes: str = "", error: str = "") -> None:
        """Mark a training run as 'failed' with error notes."""
        msg = error if error else notes
        query = """
        UPDATE training_runs
        SET status = 'failed',
            completed_at = CURRENT_TIMESTAMP,
            notes = ?
        WHERE run_id = ?;
        """
        self.db.execute(query, (msg, run_id))

    def get(self, run_id: int) -> dict[str, Any] | None:
        """Retrieve a training run record by run_id."""
        query = "SELECT * FROM training_runs WHERE run_id = ?;"
        row = self.db.fetch_one(query, (run_id,))
        return self._parse_row(row)

    def list_recent(self, limit: int = 10, n: int | None = None) -> list[dict[str, Any]]:
        """List the N most recent training runs."""
        count = n if n is not None else limit
        query = "SELECT * FROM training_runs ORDER BY started_at DESC, run_id DESC LIMIT ?;"
        rows = self.db.fetch_all(query, (count,))
        return [self._parse_row(r) for r in rows if r is not None]  # type: ignore
