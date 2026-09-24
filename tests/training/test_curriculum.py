"""Unit tests for CurriculumScheduler.

Tests progression through levels L1 to L5 based on win rate thresholds,
opponent type mappings, horizon scaling, and state serialization.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pytest

from src.training.curriculum import CurriculumScheduler


class TestCurriculumScheduler:
    """Tests for CurriculumScheduler."""

    def test_starts_at_level_1(self) -> None:
        """Curriculum begins at Level 1 by default."""
        scheduler = CurriculumScheduler()
        assert scheduler.current_level == 1
        assert scheduler.get_opponent_type() == "static"
        assert scheduler.get_horizon() == 200

    def test_advances_after_exceeding_threshold(self) -> None:
        """Scheduler advances to next level when win rate meets threshold."""
        scheduler = CurriculumScheduler(config={"advance_threshold": 0.60, "evaluation_window": 10})
        # Record 8 wins out of 10 episodes (80% win rate >= 60%)
        for _ in range(8):
            scheduler.record_episode(win=True)
        for _ in range(2):
            scheduler.record_episode(win=False)

        assert scheduler.should_advance() is True
        advanced = scheduler.advance()
        assert advanced is True
        assert scheduler.current_level == 2
        assert scheduler.get_opponent_type() == "random"

    def test_does_not_advance_below_threshold(self) -> None:
        """Scheduler does not advance when win rate is below threshold."""
        scheduler = CurriculumScheduler(config={"advance_threshold": 0.60, "evaluation_window": 10})
        # Record 4 wins out of 10 episodes (40% win rate < 60%)
        for _ in range(4):
            scheduler.record_episode(win=True)
        for _ in range(6):
            scheduler.record_episode(win=False)

        assert scheduler.should_advance() is False
        assert scheduler.current_level == 1

    def test_horizon_scales_with_level(self) -> None:
        """Episode horizon scales from 200 at L1 up to 350 at L5."""
        scheduler = CurriculumScheduler()
        horizons = []
        for _ in range(5):
            horizons.append(scheduler.get_horizon())
            scheduler.advance()
        assert horizons == [200, 200, 250, 300, 350]

    def test_get_opponent_type_per_level(self) -> None:
        """Opponent mapping matches curriculum design specification."""
        scheduler = CurriculumScheduler()
        expected = ["static", "random", "scripted", "previous", "league"]
        actual = []
        for _ in range(5):
            actual.append(scheduler.get_opponent_type())
            scheduler.advance()
        assert actual == expected

    def test_cannot_advance_past_level_5(self) -> None:
        """Curriculum terminates at Level 5 and cannot advance further."""
        scheduler = CurriculumScheduler(initial_level=5)
        assert scheduler.current_level == 5
        # Even with 100% wins
        for _ in range(20):
            scheduler.record_episode(win=True)
        assert scheduler.should_advance() is False
        assert scheduler.advance() is False
        assert scheduler.current_level == 5

    def test_state_dict_roundtrip(self) -> None:
        """State serialization preserves current level and progression history."""
        scheduler = CurriculumScheduler()
        scheduler.advance()
        scheduler.record_episode(win=True)
        scheduler.record_episode(win=False)

        state = scheduler.state_dict()
        new_scheduler = CurriculumScheduler()
        new_scheduler.load_state_dict(state)

        assert new_scheduler.current_level == 2
        assert len(new_scheduler.history) == 2
