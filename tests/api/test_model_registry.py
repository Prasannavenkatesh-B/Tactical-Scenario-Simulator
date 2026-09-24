"""Tests for ModelRegistry lifecycle, checkpoint loading, and hot-swapping."""

from pathlib import Path
import pytest
import torch

from src.api.config import CANONICAL_POLICY_NAMES
from src.api.model_registry import ModelRegistry, build_policy
from src.marl.commander import CommanderPolicy
from src.marl.policies.air_fight import AirFightPolicy


def test_build_policy_all_canonical() -> None:
    """Every canonical policy name should construct properly in eval mode."""
    for name in CANONICAL_POLICY_NAMES:
        policy = build_policy(name)
        assert policy is not None
        assert not policy.training  # Must be in eval mode


def test_build_policy_unknown_name() -> None:
    """Unknown policy name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown canonical policy name"):
        build_policy("submarine_torpedo")


def test_empty_registry() -> None:
    """Freshly created registry without initialized policies."""
    reg = ModelRegistry(checkpoint_dir="non_existent_dir")
    assert not reg.is_ready
    assert reg.list_policies() == []
    with pytest.raises(KeyError, match="is not loaded in registry"):
        reg.get("air_fight_AC1")


def test_init_random_policies() -> None:
    """init_random_policies populates all canonical policies."""
    reg = ModelRegistry(checkpoint_dir="dummy_dir")
    reg.init_random_policies()
    assert reg.is_ready
    policies = reg.list_policies()
    assert len(policies) == len(CANONICAL_POLICY_NAMES)

    for p in policies:
        assert p["name"] in CANONICAL_POLICY_NAMES
        assert p["param_count"] > 0
        assert p["loaded"] is True


def test_load_checkpoint_from_pt_file(tmp_path: Path) -> None:
    """ModelRegistry can load weights from a .pt checkpoint file on disk."""
    reg = ModelRegistry(checkpoint_dir=str(tmp_path))
    reg.init_random_policies()

    # Create dummy checkpoint
    source_policy = AirFightPolicy(variant="AC1")
    ckpt_path = tmp_path / "air_fight_AC1.pt"
    torch.save({"state_dict": source_policy.state_dict()}, ckpt_path)

    # Hot-swap checkpoint
    param_count = reg.load_checkpoint("air_fight_AC1", str(ckpt_path))
    assert param_count > 0

    # Ensure non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        reg.load_checkpoint("air_fight_AC1", str(tmp_path / "missing.pt"))


def test_load_all_from_dir(tmp_path: Path) -> None:
    """load_all_from_dir automatically matches .pt files to canonical policy names."""
    # Save a couple checkpoints
    p1 = AirFightPolicy(variant="AC1")
    torch.save(p1.state_dict(), tmp_path / "air_fight_AC1.pt")

    reg = ModelRegistry(checkpoint_dir=str(tmp_path))
    loaded = reg.load_all_from_dir()

    assert "air_fight_AC1" in loaded
    assert reg.is_ready
    assert reg.get("air_fight_AC1") is not None


def test_commander_hidden_resets() -> None:
    """Verify reset_commander_hiddens and reset_agent_hidden."""
    reg = ModelRegistry()
    cmd = build_policy("commander")
    assert isinstance(cmd, CommanderPolicy)

    # Simulate an agent hidden state
    cmd.agent_hiddens["agent_blue_1"] = torch.zeros((1, 1, 128))
    cmd.agent_hiddens["agent_blue_2"] = torch.zeros((1, 1, 128))
    reg._policies["commander"] = cmd

    # Reset single agent
    reg.reset_agent_hidden("agent_blue_1")
    assert "agent_blue_1" not in cmd.agent_hiddens
    assert "agent_blue_2" in cmd.agent_hiddens

    # Reset all agents
    reg.reset_commander_hiddens()
    assert len(cmd.agent_hiddens) == 0
