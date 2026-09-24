"""Direct in-process TSS integration wrapper (Mode A) for maximum throughput."""

from pathlib import Path
from typing import Any
import numpy as np
import torch

from src.api.action_codec import ActionCodec
from src.api.config import resolve_policy_name
from src.api.model_registry import ModelRegistry
from src.core.actions import AirAction, CommanderAction, GroundAction, SeaAction
from src.api.config import DEFAULT_CHECKPOINT_DIR
from src.integration.tss_action_mapper import TSSActionMapper
from src.integration.tss_observation_mapper import TSSObservationMapper
from src.marl.commander import CommanderPolicy
from src.marl.policies.base_policy import BaseLowLevelPolicy


class InProcessTSSWrapper:
    """Mode A: In-process direct wrapper around MARL policies.

    Designed for high-speed deployment where DRDO TSS runs in Python on the same host,
    bypassing HTTP serialization overhead to achieve sub-5ms decision cycles.
    """

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        checkpoint_dir: str = "checkpoints/final",
        protocol_path: str = "src/integration/protocol.yaml",
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        self.obs_mapper = TSSObservationMapper(protocol_path=protocol_path)
        self.action_mapper = TSSActionMapper(protocol_path=protocol_path)

        if registry is not None:
            self.registry = registry
        else:
            self.registry = ModelRegistry(checkpoint_dir=checkpoint_dir)
            loaded = self.registry.load_all_from_dir()
            if not loaded or not self.registry.is_ready:
                self.registry.init_random_policies()

    def reset(self, episode_id: str | None = None) -> None:
        """Reset per-agent recurrent memory states at scenario/episode boundaries."""
        self.registry.reset_commander_hiddens()

    def get_action(
        self,
        agent_id: str,
        domain: str,
        variant: str = "default",
        tss_observation: dict[str, Any] | None = None,
        deterministic: bool = False,
    ) -> dict[str, Any]:
        """Execute single-agent decision cycle: TSS obs -> Policy inference -> TSS command.

        Args:
            agent_id: Unique agent identifier.
            domain: Operational domain ("air", "ground", "sea", "commander").
            variant: Platform variant ("AC1", "AC2", "default").
            tss_observation: Unnormalized telemetry dictionary from TSS.
            deterministic: Whether to take argmax/mode or sample action.

        Returns:
            DRDO TSS command dictionary.
        """
        obs_dict = tss_observation or {}
        dom = domain.lower().strip()

        if dom == "commander":
            return self.get_commander_action(agent_id, obs_dict, deterministic=deterministic)

        # 1. Map TSS telemetry to normalized numpy vector
        raw_obs: np.ndarray
        if dom == "air":
            raw_obs = self.obs_mapper.to_air_observation(obs_dict)
        elif dom == "ground":
            raw_obs = self.obs_mapper.to_ground_observation(obs_dict)
        elif dom == "sea":
            raw_obs = self.obs_mapper.to_sea_observation(obs_dict)
        else:
            raise ValueError(f"Unknown domain: '{domain}'")

        # 2. Retrieve policy
        pol_name = resolve_policy_name(dom, variant)
        policy = self.registry.get(pol_name)
        assert isinstance(policy, BaseLowLevelPolicy)

        # 3. Policy forward inference
        with torch.no_grad():
            raw_act, _log_prob, _val = policy.act(raw_obs, deterministic=deterministic)

        # 4. Decode to strongly typed Action dataclass via ActionCodec
        encoded_act = ActionCodec.encode(dom, raw_act)
        action = ActionCodec.decode(dom, variant, encoded_act)

        # 5. Map to TSS command format
        if isinstance(action, AirAction):
            return self.action_mapper.from_air_action(action, obs_dict)
        elif isinstance(action, GroundAction):
            return self.action_mapper.from_ground_action(action, obs_dict)
        elif isinstance(action, SeaAction):
            return self.action_mapper.from_sea_action(action, obs_dict)

        raise TypeError(f"Unexpected action type produced for domain '{domain}': {type(action)}")

    def get_actions_batch(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Batch execution of multiple vehicle decision requests for a TSS simulation tick."""
        results: list[dict[str, Any]] = []
        for req in requests:
            cmd = self.get_action(
                agent_id=req["agent_id"],
                domain=req["domain"],
                variant=req.get("variant", "default"),
                tss_observation=req.get("tss_observation", {}),
                deterministic=req.get("deterministic", False),
            )
            results.append(cmd)
        return results

    def get_commander_action(
        self,
        agent_id: str,
        tss_observation: dict[str, Any],
        deterministic: bool = False,
    ) -> dict[str, Any]:
        """Execute high-level commander tactical decision with per-agent recurrent hidden memory."""
        # 1. Map to canonical 53-dim vector
        padded_obs = self.obs_mapper.to_commander_padded_vector(tss_observation)

        # 2. Retrieve Commander policy
        policy = self.registry.get("commander")
        assert isinstance(policy, CommanderPolicy)

        # 3. Policy forward inference tracking hidden state internally
        with torch.no_grad():
            action_idx, _log_prob, _val, _new_hidden = policy.act(
                padded_obs,
                agent_id=agent_id,
                deterministic=deterministic,
            )

        # 4. Map to TSS command
        cmd_action = CommanderAction(action=int(action_idx))
        return self.action_mapper.from_commander_action(cmd_action, tss_observation)

    @property
    def is_ready(self) -> bool:
        """Indicate whether underlying neural policies are loaded and ready."""
        return self.registry.is_ready

    def shutdown(self) -> None:
        """Release allocated neural network resources."""
        self.reset()
