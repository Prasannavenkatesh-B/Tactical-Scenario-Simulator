"""Unit tests for League self-play pool.

Tests snapshot additions, FIFO capacity evictions, uniform sampling,
scripted fallback probabilities, and disk save/load persistence.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import tempfile
import numpy as np
import pytest
import torch

from src.training.league import League


class TestLeague:
    """Tests for League opponent checkpoint pool."""

    def test_add_snapshot_grows_size(self) -> None:
        """Adding policy snapshots increments the league pool size."""
        league = League(max_size=5)
        assert league.size() == 0

        weights = {"linear.weight": torch.randn(10, 10)}
        league.add(weights, iteration=10)
        assert league.size() == 1

    def test_eviction_when_over_max_size(self) -> None:
        """Oldest snapshots are evicted when max capacity is reached (FIFO)."""
        league = League(max_size=3)
        for i in range(5):
            league.add({"val": torch.tensor([float(i)])}, iteration=i)

        assert league.size() == 3
        # Should contain snapshots 2, 3, 4 (0 and 1 evicted)
        stored_iters = [s["iteration"] for s in league.snapshots]
        assert stored_iters == [2, 3, 4]

    def test_sample_empty_returns_none(self) -> None:
        """Sampling an empty league pool returns None."""
        league = League(max_size=10)
        assert league.sample() is None

    def test_sample_with_scripted_fallback(self) -> None:
        """Fallback sampling returns 'scripted' if empty or by probability."""
        league = League(max_size=5)
        rng = np.random.default_rng(42)

        # Empty pool must always fallback to scripted
        src, w = league.sample_with_scripted_fallback(rng=rng, scripted_prob=0.0)
        assert src == "scripted"
        assert w is None

        # Add a snapshot
        league.add({"dummy": torch.tensor([1.0])}, iteration=1)

        # With scripted_prob=1.0, must always return scripted
        src, w = league.sample_with_scripted_fallback(rng=rng, scripted_prob=1.0)
        assert src == "scripted"
        assert w is None

        # With scripted_prob=0.0, must return league snapshot
        src, w = league.sample_with_scripted_fallback(rng=rng, scripted_prob=0.0)
        assert src == "league"
        assert w is not None

    def test_save_load_roundtrip(self) -> None:
        """Saving and loading league pool preserves all snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            league = League(max_size=5, save_dir=tmpdir)
            for i in range(3):
                league.add({"w": torch.tensor([float(i)])}, iteration=i * 10)

            save_file = Path(tmpdir) / "league.pt"
            league.save(save_file)

            new_league = League(save_dir=tmpdir)
            new_league.load(save_file)

            assert new_league.size() == 3
            assert [s["iteration"] for s in new_league.snapshots] == [0, 10, 20]
