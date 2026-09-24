"""Gymnasium single-agent adapter wrapping a designated TSS tactical agent."""

from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.integration.config import INTEGRATION_MODE
from src.integration.parallel_env import TSSParallelEnv


class TSSSingleAgentEnv(gym.Env):
    """Gymnasium single-agent wrapper around a specific tactical entity.

    Enables standard single-agent algorithms (e.g. SB3 PPO, SAC, DQN) to fine-tune
    or control a specific agent while multi-agent coordination remains driven by TSS.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        agent_id: str = "blue_air_1",
        domain: str = "air",
        variant: str = "AC1",
        mode: str = INTEGRATION_MODE,
        registry_or_url: Any = None,
        protocol_path: str = "src/integration/protocol.yaml",
    ) -> None:
        super().__init__()
        self.agent_id = agent_id
        self.domain = domain.lower().strip()
        self.variant = variant
        self.mode = mode

        # ParallelEnv backend
        self.parallel_env = TSSParallelEnv(
            mode=mode,
            registry_or_url=registry_or_url,
            protocol_path=protocol_path,
            agent_ids=[agent_id],
            agent_domains={agent_id: self.domain},
        )

        self.observation_space = self.parallel_env.observation_space(self.agent_id)
        self.action_space = self.parallel_env.action_space(self.agent_id)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset agent environment state, returning observation vector and info dict."""
        super().reset(seed=seed)
        obs_dict, info_dict = self.parallel_env.reset(seed=seed, options=options)
        return obs_dict[self.agent_id], info_dict[self.agent_id]

    def step(
        self,
        action: Any,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step the agent, returning (obs, reward, terminated, truncated, info)."""
        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = self.parallel_env.step(
            {self.agent_id: action}
        )

        shape = self.observation_space.shape or (13,)
        obs = obs_dict.get(
            self.agent_id,
            np.zeros(shape, dtype=np.float32),
        )
        reward = float(rew_dict.get(self.agent_id, 0.0))
        terminated = bool(term_dict.get(self.agent_id, False))
        truncated = bool(trunc_dict.get(self.agent_id, False))
        info = info_dict.get(self.agent_id, {})

        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Render frame."""
        self.parallel_env.render()

    def close(self) -> None:
        """Cleanly release wrapper and connection handles."""
        self.parallel_env.close()
