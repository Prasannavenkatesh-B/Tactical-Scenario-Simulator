"""Atomic checkpoint management for policy weights, optimizers, and curriculum state.

Implements atomic saves (write to temp file, then atomic rename) to prevent
corrupted checkpoint states during sudden interruptions.
"""

import os
from pathlib import Path
import tempfile
from typing import Any
import torch

from src.training.config import CHECKPOINT_DIR


class CheckpointManager:
    """Manages saving and loading of policy checkpoints and full training state.

    Ensures atomic disk writes via temporary file renaming to maintain state
    integrity against power loss or process interrupts.
    """

    def __init__(
        self,
        save_dir: str | Path = CHECKPOINT_DIR,
        db: Any = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.db = db

    def _atomic_torch_save(self, obj: Any, target_path: Path) -> None:
        """Atomically serialize an object using a temporary file and rename."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        # Create temp file in the same directory to guarantee atomic rename across filesystems
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target_path.parent,
            prefix=".tmp_ckpt_",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            try:
                torch.save(obj, tmp_file)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            except Exception:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise

        # Atomic replacement
        os.replace(tmp_path, target_path)

    def save_policies(
        self,
        policies: dict[str, Any],
        iteration: int,
        stage: str = "stage_a",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save state dicts of all policy models with metadata.

        Args:
            policies: Mapping of policy name -> policy instance.
            iteration: Current training iteration.
            stage: "stage_a", "stage_b", or "final".
            metadata: Optional additional metadata dict.

        Returns:
            Path string to the saved checkpoint file.
        """
        policy_states: dict[str, Any] = {}
        for name, policy in policies.items():
            if hasattr(policy, "state_dict"):
                policy_states[name] = {
                    k: v.cpu().clone() for k, v in policy.state_dict().items()
                }

        ckpt_data = {
            "iteration": iteration,
            "stage": stage,
            "metadata": metadata or {},
            "policies": policy_states,
        }

        filename = f"checkpoint_{stage}_iter_{iteration:05d}.pt"
        target_path = self.save_dir / filename
        self._atomic_torch_save(ckpt_data, target_path)

        if self.db is not None:
            try:
                from src.database.checkpoint_repo import CheckpointRepository
                repo = CheckpointRepository(self.db)
                file_size = target_path.stat().st_size if target_path.exists() else 0
                win_rate = (metadata or {}).get("win_rate")
                kd_ratio = (metadata or {}).get("kd_ratio") or (metadata or {}).get("kill_death_ratio")
                if len(policies) == 1:
                    pol_name = list(policies.keys())[0]
                    p_inst = list(policies.values())[0]
                    param_count = sum(p.numel() for p in p_inst.parameters()) if hasattr(p_inst, "parameters") else None
                    repo.register(
                        path=str(target_path),
                        policy_name=pol_name,
                        stage=stage,
                        iteration=iteration,
                        win_rate=win_rate,
                        kill_death_ratio=kd_ratio,
                        param_count=param_count,
                        file_size_bytes=file_size,
                    )
                else:
                    for name, policy in policies.items():
                        pol_path = f"{target_path}#{name}"
                        param_count = sum(p.numel() for p in policy.parameters()) if hasattr(policy, "parameters") else None
                        repo.register(
                            path=pol_path,
                            policy_name=name,
                            stage=stage,
                            iteration=iteration,
                            win_rate=win_rate,
                            kill_death_ratio=kd_ratio,
                            param_count=param_count,
                            file_size_bytes=file_size,
                        )
            except Exception:
                pass

        return str(target_path)

    def load_policies(
        self,
        path: str | Path,
        policies: dict[str, Any],
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load state dicts from checkpoint into existing policy instances.

        Args:
            path: Path to checkpoint file.
            policies: Mapping of policy name -> policy instance to load weights into.
            strict: Strict weight matching.

        Returns:
            Checkpoint metadata dictionary.
        """
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        ckpt_data = torch.load(ckpt_path, weights_only=False)
        saved_policies = ckpt_data.get("policies", {})

        for name, policy in policies.items():
            if name in saved_policies and hasattr(policy, "load_state_dict"):
                policy.load_state_dict(saved_policies[name], strict=strict)

        return ckpt_data.get("metadata", {})

    def save_full_state(
        self,
        state: dict[str, Any],
        filename: str = "full_training_state.pt",
    ) -> str:
        """Save complete training state for clean resumption.

        Args:
            state: Dictionary containing policies, optimizers, curriculum,
                   league, iteration, and stage info.
            filename: Target checkpoint filename.

        Returns:
            Path string to saved state file.
        """
        target_path = self.save_dir / filename
        self._atomic_torch_save(state, target_path)
        return str(target_path)

    def load_full_state(self, path: str | Path) -> dict[str, Any]:
        """Load complete training state from checkpoint file.

        Args:
            path: Path to full state checkpoint.

        Returns:
            State dictionary.
        """
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"Full state checkpoint not found: {target_path}")
        return torch.load(target_path, weights_only=False)

    def list_checkpoints(self) -> list[str]:
        """Return sorted list of all checkpoint paths in save_dir."""
        files = list(self.save_dir.glob("checkpoint_*.pt"))
        files.sort(key=lambda p: os.path.getmtime(p))
        return [str(p) for p in files]

    def latest(self) -> str | None:
        """Return the path to the most recent checkpoint, or None if none exist."""
        ckpts = self.list_checkpoints()
        return ckpts[-1] if ckpts else None
