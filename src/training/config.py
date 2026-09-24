"""Training hyperparameters and configuration constants.

Defines all constants for curriculum levels, league self-play,
training iterations, staged training, and file paths.
"""

from typing import Any

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CURRICULUM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEVELS: list[int] = [1, 2, 3, 4, 5]
EPISODES_PER_LEVEL: list[int] = [500, 500, 800, 1000, 2000]
HORIZONS: dict[int, int] = {1: 200, 2: 200, 3: 250, 4: 300, 5: 350}
ADVANCE_THRESHOLD: float = 0.60  # Win rate to advance to next curriculum level
EVALUATION_WINDOW: int = 50      # Number of recent episodes to evaluate win rate

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LEAGUE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEAGUE_SIZE: int = 20           # Max checkpoints retained in pool
SCRIPTED_PROB: float = 0.20     # Probability of using scripted baseline at L5
LEAGUE_SAVE_INTERVAL: int = 50  # Episodes between league additions

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAINING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL_ITERATIONS: int = 5000    # Total iterations per stage
ROLLOUT_STEPS: int = 256        # Steps collected per env per iteration
NUM_PARALLEL_ENVS: int = 8      # Parallel TacticalEnv instances
PPO_EPOCHS: int = 10            # Update epochs per training iteration
EVAL_INTERVAL: int = 100        # Iterations between evaluations
EVAL_EPISODES: int = 1000       # Evaluation episodes from base paper
CHECKPOINT_INTERVAL: int = 100  # Iterations between checkpoint saves
LOG_INTERVAL: int = 10          # Iterations between metrics logs

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGED
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE_A_FREEZE_AT: str = "L5"   # Freeze low-level policies after reaching L5
STAGE_B_START_AT: int = 0       # Commander starts after Stage A completes

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATHS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHECKPOINT_DIR: str = "checkpoints/"
LOG_DIR: str = "logs/"
METRICS_CSV: str = "logs/metrics.csv"
LEAGUE_DIR: str = "checkpoints/league/"
STAGE_A_DIR: str = "checkpoints/stage_a/"
STAGE_B_DIR: str = "checkpoints/stage_b/"
FINAL_DIR: str = "checkpoints/final/"


def get_default_config() -> dict[str, Any]:
    """Return dictionary representation of default training parameters."""
    return {
        "levels": list(LEVELS),
        "episodes_per_level": list(EPISODES_PER_LEVEL),
        "horizons": dict(HORIZONS),
        "advance_threshold": ADVANCE_THRESHOLD,
        "evaluation_window": EVALUATION_WINDOW,
        "league_size": LEAGUE_SIZE,
        "scripted_prob": SCRIPTED_PROB,
        "league_save_interval": LEAGUE_SAVE_INTERVAL,
        "total_iterations": TOTAL_ITERATIONS,
        "rollout_steps": ROLLOUT_STEPS,
        "num_parallel_envs": NUM_PARALLEL_ENVS,
        "ppo_epochs": PPO_EPOCHS,
        "eval_interval": EVAL_INTERVAL,
        "eval_episodes": EVAL_EPISODES,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "log_interval": LOG_INTERVAL,
        "stage_a_freeze_at": STAGE_A_FREEZE_AT,
        "stage_b_start_at": STAGE_B_START_AT,
        "checkpoint_dir": CHECKPOINT_DIR,
        "log_dir": LOG_DIR,
        "metrics_csv": METRICS_CSV,
        "league_dir": LEAGUE_DIR,
        "stage_a_dir": STAGE_A_DIR,
        "stage_b_dir": STAGE_B_DIR,
        "final_dir": FINAL_DIR,
    }
