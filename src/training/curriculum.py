"""Curriculum learning progression and scheduler across 5 tactical levels.

Manages progression from basic kinematic learning against static targets
up to league-based self-play across joint multi-domain scenarios.
"""

from collections import deque
from typing import Any
import numpy as np

from src.simulator.scenarios import ScenarioConfig, generate_scenario
from src.training.config import (
    ADVANCE_THRESHOLD,
    EVALUATION_WINDOW,
    HORIZONS,
    LEVELS,
)


class CurriculumScheduler:
    """Manages progression through L1 → L5 based on win rate threshold.

    Curriculum Levels (per base paper design):
      L1: Static opponents (no movement, no fire) — learn basic kinematics
      L2: Random opponents (random walk + random fire)
      L3: Scripted opponents (pursuit + fire at closest)
      L4: Previous policy (uses previous checkpoint as opponent)
      L5: League self-play (mix of L1-L4 policies + scripted baselines)
    """

    OPPONENT_MAPPING: dict[int, str] = {
        1: "static",
        2: "random",
        3: "scripted",
        4: "previous",
        5: "league",
    }

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        rng: np.random.Generator | None = None,
        initial_level: int = 1,
    ) -> None:
        self.config = config or {}
        self.rng = rng if rng is not None else np.random.default_rng(42)

        self._level = int(np.clip(initial_level, 1, 5))
        self.advance_threshold = float(self.config.get("advance_threshold", ADVANCE_THRESHOLD))
        self.evaluation_window = int(self.config.get("evaluation_window", EVALUATION_WINDOW))
        self.horizons: dict[int, int] = self.config.get("horizons", dict(HORIZONS))

        self.history: deque[bool] = deque(maxlen=self.evaluation_window)
        self.total_episodes_at_level: int = 0
        self.level_progression_history: list[dict[str, Any]] = []

    @property
    def current_level(self) -> int:
        """Return the current curriculum level (1 to 5)."""
        return self._level

    def get_scenario_config(self) -> ScenarioConfig:
        """Generate and return ScenarioConfig for current curriculum level."""
        scenario = generate_scenario(level=self._level, rng=self.rng)
        # Ensure horizon matches the curriculum schedule
        scenario.episode_horizon = self.get_horizon()
        return scenario

    def get_horizon(self) -> int:
        """Return the episode horizon for the current curriculum level."""
        return self.horizons.get(self._level, 200)

    def record_episode(self, win: bool) -> None:
        """Record the win/loss outcome of one training episode."""
        self.history.append(bool(win))
        self.total_episodes_at_level += 1

    def get_win_rate(self) -> float:
        """Compute the win rate over the current evaluation window."""
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    def should_advance(self) -> bool:
        """Check if agent performance meets the threshold to advance.

        Requires at least min(10, evaluation_window) episodes at the current level
        and a win rate >= advance_threshold.
        """
        if self._level >= 5:
            return False
        min_required = min(10, self.evaluation_window)
        if len(self.history) < min_required:
            return False
        return self.get_win_rate() >= self.advance_threshold

    def advance(self) -> bool:
        """Advance to the next curriculum level if eligible.

        Returns:
            True if level advanced, False if already at L5 or threshold not met.
        """
        if self._level >= 5:
            return False

        old_level = self._level
        self._level += 1
        self.level_progression_history.append({
            "from_level": old_level,
            "to_level": self._level,
            "episodes": self.total_episodes_at_level,
            "final_win_rate": self.get_win_rate(),
        })
        self.history.clear()
        self.total_episodes_at_level = 0
        return True

    def get_opponent_type(self) -> str:
        """Return opponent type description for the current level."""
        return self.OPPONENT_MAPPING.get(self._level, "scripted")

    def state_dict(self) -> dict[str, Any]:
        """Serialize scheduler state for checkpoint resumption."""
        return {
            "level": self._level,
            "history": list(self.history),
            "total_episodes_at_level": self.total_episodes_at_level,
            "level_progression_history": list(self.level_progression_history),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scheduler state from serialized dictionary."""
        self._level = int(state.get("level", 1))
        self.history = deque(state.get("history", []), maxlen=self.evaluation_window)
        self.total_episodes_at_level = int(state.get("total_episodes_at_level", 0))
        self.level_progression_history = list(state.get("level_progression_history", []))
