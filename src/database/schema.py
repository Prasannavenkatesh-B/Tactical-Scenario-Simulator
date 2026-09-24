"""SQLite schema definitions and index definitions for tactical simulation database."""

SCHEMA_V1: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS scenarios (
        scenario_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        level INTEGER NOT NULL,
        map_size_km REAL NOT NULL,
        episode_horizon INTEGER NOT NULL,
        weather TEXT DEFAULT 'clear',
        blue_entities TEXT NOT NULL,
        red_entities TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        notes TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS agent_params (
        param_id INTEGER PRIMARY KEY AUTOINCREMENT,
        domain TEXT NOT NULL,
        variant TEXT NOT NULL,
        max_speed REAL NOT NULL,
        min_speed REAL NOT NULL,
        turn_rate_max REAL NOT NULL,
        sensor_range_km REAL NOT NULL,
        weapon_range_km REAL NOT NULL,
        hit_probability REAL NOT NULL,
        ammo_cannon INTEGER NOT NULL,
        ammo_rocket INTEGER NOT NULL,
        UNIQUE(domain, variant)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS model_checkpoints (
        checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL UNIQUE,
        policy_name TEXT NOT NULL,
        stage TEXT NOT NULL,
        iteration INTEGER NOT NULL,
        win_rate REAL,
        kill_death_ratio REAL,
        param_count INTEGER,
        file_size_bytes INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        notes TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS training_runs (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        config_json TEXT NOT NULL,
        git_commit TEXT,
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP,
        total_iterations INTEGER,
        final_win_rate REAL,
        final_kd_ratio REAL,
        status TEXT NOT NULL,
        notes TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS metrics_history (
        metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        iteration INTEGER NOT NULL,
        level INTEGER,
        policy_name TEXT,
        loss REAL,
        win_rate REAL,
        kill_death_ratio REAL,
        action_entropy REAL,
        episode_length REAL,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_metrics_run_iter ON metrics_history(run_id, iteration);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_checkpoints_policy ON model_checkpoints(policy_name);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_scenarios_level ON scenarios(level);
    """,
]
