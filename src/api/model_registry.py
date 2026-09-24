"""Thread-safe model registry managing lifecycle, checkpoint loading, and hot-swapping."""

from pathlib import Path
import threading
from typing import Any, Union
import torch

from src.api.config import CANONICAL_POLICY_NAMES, DEFAULT_CHECKPOINT_DIR
from src.marl.commander import CommanderPolicy
from src.marl.policies.air_escape import AirEscapePolicy
from src.marl.policies.air_fight import AirFightPolicy
from src.marl.policies.base_policy import BaseLowLevelPolicy
from src.marl.policies.ground_defend import GroundDefendPolicy
from src.marl.policies.ground_engage import GroundEngagePolicy
from src.marl.policies.sea_defend import SeaDefendPolicy
from src.marl.policies.sea_engage import SeaEngagePolicy

AnyPolicy = Union[BaseLowLevelPolicy, CommanderPolicy]


def build_policy(policy_name: str) -> AnyPolicy:
    """Factory creating an uninitialized/randomly-initialized policy instance by canonical name."""
    name = policy_name.strip()
    policy: AnyPolicy

    if name == "air_fight_AC1":
        policy = AirFightPolicy(variant="AC1")
    elif name == "air_fight_AC2":
        policy = AirFightPolicy(variant="AC2")
    elif name == "air_escape_AC1":
        policy = AirEscapePolicy(variant="AC1")
    elif name == "air_escape_AC2":
        policy = AirEscapePolicy(variant="AC2")
    elif name == "ground_engage":
        policy = GroundEngagePolicy()
    elif name == "ground_defend":
        policy = GroundDefendPolicy()
    elif name == "sea_engage":
        policy = SeaEngagePolicy()
    elif name == "sea_defend":
        policy = SeaDefendPolicy()
    elif name == "commander":
        policy = CommanderPolicy()
    else:
        raise ValueError(f"Unknown canonical policy name: '{policy_name}'")

    policy.eval()
    return policy


class ModelRegistry:
    """Thread-safe registry holding in-memory policy instances with hot-swap capabilities."""

    def __init__(self, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self._lock = threading.RLock()
        self._policies: dict[str, AnyPolicy] = {}

    def init_random_policies(self) -> None:
        """Initialize all canonical policies with default weights in eval mode."""
        with self._lock:
            for name in CANONICAL_POLICY_NAMES:
                if name not in self._policies:
                    self._policies[name] = build_policy(name)

    def load_all_from_dir(self) -> dict[str, str]:
        """Scan checkpoint_dir for *.pt files and load weights into matching policy classes."""
        loaded: dict[str, str] = {}
        if not self.checkpoint_dir.exists():
            return loaded

        with self._lock:
            for pt_file in self.checkpoint_dir.glob("*.pt"):
                stem = pt_file.stem
                matched_name: str | None = None
                for canon in CANONICAL_POLICY_NAMES:
                    if canon.lower() in stem.lower() or stem.lower() in canon.lower():
                        matched_name = canon
                        break

                if matched_name is not None:
                    try:
                        self.load_checkpoint(matched_name, str(pt_file))
                        loaded[matched_name] = str(pt_file)
                    except Exception:
                        pass

        return loaded

    def get(self, policy_name: str) -> AnyPolicy:
        """Retrieve policy instance by canonical name.

        Raises:
            KeyError: If policy has not been registered or loaded.
        """
        if policy_name not in self._policies:
            raise KeyError(f"Policy '{policy_name}' is not loaded in registry.")
        return self._policies[policy_name]

    def load_checkpoint(self, policy_name: str, path: str) -> int:
        """Hot-swap model weights from checkpoint file into policy instance.

        Returns:
            Total neural network parameter count of loaded model.
        """
        with self._lock:
            if policy_name not in self._policies:
                self._policies[policy_name] = build_policy(policy_name)

            policy = self._policies[policy_name]
            ckpt_path = Path(path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found at {path}")

            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    policy.load_state_dict(checkpoint["state_dict"])
                elif "policies" in checkpoint and policy_name in checkpoint["policies"]:
                    policy.load_state_dict(checkpoint["policies"][policy_name])
                else:
                    try:
                        policy.load(str(ckpt_path))
                    except Exception:
                        policy.load_state_dict(checkpoint)
            else:
                policy.load(str(ckpt_path))

            policy.eval()
            return sum(p.numel() for p in policy.parameters())

    def list_policies(self) -> list[dict[str, Any]]:
        """Return descriptors for all currently registered policies."""
        with self._lock:
            res: list[dict[str, Any]] = []
            for name, policy in self._policies.items():
                variant = (
                    "AC1" if "AC1" in name
                    else ("AC2" if "AC2" in name
                    else ("GROUND" if "ground" in name
                    else ("SEA" if "sea" in name else "commander")))
                )
                res.append({
                    "name": name,
                    "variant": variant,
                    "param_count": sum(p.numel() for p in policy.parameters()),
                    "loaded": True,
                })
            return res

    def reset_commander_hiddens(self) -> None:
        """Clear all agent recurrent memory states in Commander policy."""
        with self._lock:
            if "commander" in self._policies:
                cmd_policy = self._policies["commander"]
                if isinstance(cmd_policy, CommanderPolicy):
                    cmd_policy.reset_all_hiddens()

    def reset_agent_hidden(self, agent_id: str) -> None:
        """Clear recurrent memory state for a specific agent in Commander policy."""
        with self._lock:
            if "commander" in self._policies:
                cmd_policy = self._policies["commander"]
                if isinstance(cmd_policy, CommanderPolicy):
                    cmd_policy.reset_agent_hidden(agent_id)

    @property
    def is_ready(self) -> bool:
        """Check whether any policies are resident in memory."""
        return len(self._policies) > 0
