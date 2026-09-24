"""Training pipeline module for multi-domain tactical MARL.

Exports:
  - Trainer: Core iterative training loop
  - StagedTrainer: Orchestrator for Stage A → freeze → Stage B
  - CurriculumScheduler: 5-level progressive curriculum scheduler
  - League: Opponent snapshot pool for self-play
  - RolloutCollector: Multi-environment transition collector
  - Evaluator: Win rates, non-determinism, and ablations
  - CheckpointManager: Atomic checkpoint save/load
  - MetricsLogger: CSV and TensorBoard logger
"""

from src.training.checkpoint import CheckpointManager
from src.training.config import (
    ADVANCE_THRESHOLD,
    CHECKPOINT_DIR,
    CHECKPOINT_INTERVAL,
    EPISODES_PER_LEVEL,
    EVAL_EPISODES,
    EVAL_INTERVAL,
    FINAL_DIR,
    HORIZONS,
    LEAGUE_DIR,
    LEAGUE_SAVE_INTERVAL,
    LEAGUE_SIZE,
    LEVELS,
    LOG_DIR,
    LOG_INTERVAL,
    METRICS_CSV,
    NUM_PARALLEL_ENVS,
    PPO_EPOCHS,
    ROLLOUT_STEPS,
    SCRIPTED_PROB,
    STAGE_A_DIR,
    STAGE_A_FREEZE_AT,
    STAGE_B_DIR,
    STAGE_B_START_AT,
    TOTAL_ITERATIONS,
    get_default_config,
)
from src.training.curriculum import CurriculumScheduler
from src.training.evaluator import Evaluator
from src.training.league import League
from src.training.metrics import MetricsLogger
from src.training.rollout_collector import RolloutCollector
from src.training.staged_trainer import StagedTrainer
from src.training.trainer import Trainer

__all__ = [
    # Classes
    "Trainer",
    "StagedTrainer",
    "CurriculumScheduler",
    "League",
    "RolloutCollector",
    "Evaluator",
    "CheckpointManager",
    "MetricsLogger",
    # Config
    "LEVELS",
    "EPISODES_PER_LEVEL",
    "HORIZONS",
    "ADVANCE_THRESHOLD",
    "LEAGUE_SIZE",
    "SCRIPTED_PROB",
    "LEAGUE_SAVE_INTERVAL",
    "TOTAL_ITERATIONS",
    "ROLLOUT_STEPS",
    "NUM_PARALLEL_ENVS",
    "PPO_EPOCHS",
    "EVAL_INTERVAL",
    "EVAL_EPISODES",
    "CHECKPOINT_INTERVAL",
    "LOG_INTERVAL",
    "STAGE_A_FREEZE_AT",
    "STAGE_B_START_AT",
    "CHECKPOINT_DIR",
    "LOG_DIR",
    "METRICS_CSV",
    "LEAGUE_DIR",
    "STAGE_A_DIR",
    "STAGE_B_DIR",
    "FINAL_DIR",
    "get_default_config",
]
