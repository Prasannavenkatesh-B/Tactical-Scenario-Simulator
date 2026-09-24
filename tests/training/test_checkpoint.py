"""Unit tests for CheckpointManager.

Tests atomic saving, weight preservation on reload, checkpoint listing,
and full state serialization.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import tempfile
import pytest
import torch

from src.marl.policies.air_fight import AirFightPolicy
from src.training.checkpoint import CheckpointManager


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_save_load_preserves_weights(self) -> None:
        """Saving and reloading reproduces identical network weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(save_dir=tmpdir)
            policy = AirFightPolicy(variant="AC1")
            policies = {"air_fight": policy}

            path = mgr.save_policies(policies, iteration=100, stage="stage_a", metadata={"acc": 0.95})
            assert Path(path).exists()

            # Create new policy and reload
            new_policy = AirFightPolicy(variant="AC1")
            meta = mgr.load_policies(path, {"air_fight": new_policy})

            assert meta.get("acc") == 0.95
            for p1, p2 in zip(policy.parameters(), new_policy.parameters()):
                assert torch.allclose(p1, p2)

    def test_list_checkpoints_order(self) -> None:
        """list_checkpoints() returns files sorted by modification time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(save_dir=tmpdir)
            policy = AirFightPolicy(variant="AC1")

            mgr.save_policies({"air_fight": policy}, iteration=10, stage="stage_a")
            mgr.save_policies({"air_fight": policy}, iteration=20, stage="stage_a")

            ckpts = mgr.list_checkpoints()
            assert len(ckpts) == 2
            assert "00010" in ckpts[0]
            assert "00020" in ckpts[1]

    def test_latest_checkpoint(self) -> None:
        """latest() returns the most recently written checkpoint path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(save_dir=tmpdir)
            assert mgr.latest() is None

            policy = AirFightPolicy(variant="AC1")
            mgr.save_policies({"air_fight": policy}, iteration=1, stage="test")
            path2 = mgr.save_policies({"air_fight": policy}, iteration=2, stage="test")

            assert mgr.latest() == path2

    def test_full_state_save_and_load(self) -> None:
        """Full state dictionary roundtrip preserves all components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(save_dir=tmpdir)
            state = {
                "iteration": 500,
                "curriculum_level": 3,
                "weights": [1.0, 2.0, 3.0],
            }
            path = mgr.save_full_state(state, filename="test_full_state.pt")
            loaded = mgr.load_full_state(path)

            assert loaded["iteration"] == 500
            assert loaded["curriculum_level"] == 3
            assert loaded["weights"] == [1.0, 2.0, 3.0]
