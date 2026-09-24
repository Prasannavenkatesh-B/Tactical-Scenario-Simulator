"""PettingZoo ParallelEnv multi-agent integration adapter for DRDO TSS."""

from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo import ParallelEnv  # type: ignore[import-untyped]

from src.integration.config import INTEGRATION_MODE
from src.integration.http_wrapper import HTTPTSSWrapper
from src.integration.in_process_wrapper import InProcessTSSWrapper


class TSSParallelEnv(ParallelEnv):
    """PettingZoo ParallelEnv-compliant adapter for multi-agent tactical simulations.

    Allows standardized reinforcement learning frameworks (PettingZoo, RLlib, Tianshou)
    to interface with TSS and the policy wrappers using identical multi-agent semantics.
    Can be configured in either 'in_process' (Mode A) or 'http' (Mode B) mode.
    """

    metadata = {"render_modes": ["human"], "name": "tss_parallel_v1"}

    def __init__(
        self,
        mode: str = INTEGRATION_MODE,
        registry_or_url: Any = None,
        protocol_path: str = "src/integration/protocol.yaml",
        agent_ids: list[str] | None = None,
        agent_domains: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.mode = mode.lower().strip()
        self.protocol_path = protocol_path

        # 1. Initialize underlying wrapper
        if self.mode == "in_process":
            from src.api.model_registry import ModelRegistry
            reg = registry_or_url if isinstance(registry_or_url, ModelRegistry) else None
            self.wrapper: InProcessTSSWrapper | HTTPTSSWrapper = InProcessTSSWrapper(
                registry=reg, protocol_path=protocol_path
            )
        elif self.mode == "http":
            url = str(registry_or_url) if isinstance(registry_or_url, str) else "http://localhost:8000"
            self.wrapper = HTTPTSSWrapper(base_url=url, protocol_path=protocol_path)
        else:
            raise ValueError(f"Invalid integration mode: '{mode}'. Expected 'in_process' or 'http'.")

        # 2. Agent definitions
        self.possible_agents = (
            list(agent_ids)
            if agent_ids is not None
            else ["blue_air_1", "blue_air_2", "red_air_1", "red_air_2"]
        )
        self.agents = list(self.possible_agents)

        self._domains = agent_domains or {
            aid: ("air" if "air" in aid else ("ground" if "ground" in aid else ("sea" if "sea" in aid else "commander")))
            for aid in self.possible_agents
        }

        # 3. Define observation and action spaces per agent
        self._obs_spaces: dict[str, gym.Space] = {}
        self._action_spaces: dict[str, gym.Space] = {}

        for aid in self.possible_agents:
            dom = self._domains.get(aid, "air")
            if dom == "air":
                self._obs_spaces[aid] = spaces.Box(
                    low=0.0, high=1.0, shape=(13,), dtype=np.float32
                )
                # Factored multi-discrete: heading (13), velocity (9), cannon (2), rocket (2)
                self._action_spaces[aid] = spaces.MultiDiscrete([13, 9, 2, 2])
            elif dom == "ground":
                self._obs_spaces[aid] = spaces.Box(
                    low=0.0, high=1.0, shape=(9,), dtype=np.float32
                )
                self._action_spaces[aid] = spaces.Box(
                    low=np.array([-45.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    high=np.array([45.0, 5.0, 2.0, 1.0], dtype=np.float32),
                    dtype=np.float32,
                )
            elif dom == "sea":
                self._obs_spaces[aid] = spaces.Box(
                    low=0.0, high=1.0, shape=(9,), dtype=np.float32
                )
                self._action_spaces[aid] = spaces.Box(
                    low=np.array([-30.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    high=np.array([30.0, 5.0, 1.0, 1.0], dtype=np.float32),
                    dtype=np.float32,
                )
            elif dom == "commander":
                self._obs_spaces[aid] = spaces.Box(
                    low=0.0, high=1.0, shape=(53,), dtype=np.float32
                )
                self._action_spaces[aid] = spaces.Discrete(4)
            else:
                self._obs_spaces[aid] = spaces.Box(
                    low=0.0, high=1.0, shape=(13,), dtype=np.float32
                )
                self._action_spaces[aid] = spaces.Discrete(4)

        self._step_count = 0
        self._max_steps = 100

    def observation_space(self, agent: str) -> gym.Space:
        """Return the observation space for the specified agent."""
        return self._obs_spaces[agent]

    def action_space(self, agent: str) -> gym.Space:
        """Return the action space for the specified agent."""
        return self._action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """Reset scenario state and return initial observation and info dicts."""
        self._step_count = 0
        self.agents = list(self.possible_agents)
        self.wrapper.reset()

        obs: dict[str, np.ndarray] = {}
        infos: dict[str, dict[str, Any]] = {}

        for aid in self.agents:
            space = self.observation_space(aid)
            assert isinstance(space, spaces.Box)
            # Default zero or low observation
            obs[aid] = np.zeros(space.shape, dtype=np.float32)
            infos[aid] = {"domain": self._domains.get(aid, "air")}

        return obs, infos

    def step(
        self,
        actions: dict[str, Any],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Step environment given actions, returning 5-tuple of per-agent dictionaries."""
        self._step_count += 1
        is_truncated = self._step_count >= self._max_steps

        obs: dict[str, np.ndarray] = {}
        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict[str, Any]] = {}

        for aid in list(self.agents):
            space = self.observation_space(aid)
            assert isinstance(space, spaces.Box)
            obs[aid] = np.zeros(space.shape, dtype=np.float32)
            rewards[aid] = 0.0
            terminations[aid] = False
            truncations[aid] = is_truncated
            infos[aid] = {"step": self._step_count}

        if is_truncated:
            self.agents.clear()

        return obs, rewards, terminations, truncations, infos

    def render(self) -> None:
        """Render hook for PettingZoo visualization."""
        pass

    def close(self) -> None:
        """Cleanly terminate environment and wrapper resources."""
        self.wrapper.shutdown()
