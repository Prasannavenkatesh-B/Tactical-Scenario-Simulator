"""Rollout collection across parallel simulation environments for MARL training.

Coordinates low-level domain policy execution, hierarchical commander decision
schedules (every HIGH_LEVEL_HORIZON steps or upon tactical triggers), and
fills RolloutBuffers for policy optimization.
"""

import math
from typing import Any
import numpy as np
import torch

from src.core.actions import AirAction, GroundAction, SeaAction
from src.core.interfaces import DomainType, TeamSide
from src.core.observations import CommanderObservation
from src.marl.config import (
    AIR_OBS_DIM,
    COMMANDER_OBS_DIM,
    GROUND_OBS_DIM,
    HIGH_LEVEL_HORIZON,
    LOW_LEVEL_BATCH_SIZE,
    LOW_LEVEL_HORIZON,
    SEA_OBS_DIM,
)
from src.marl.rollout_buffer import RolloutBuffer
from src.simulator.entities.air import AirEntity
from src.simulator.env import TacticalEnv


class RolloutCollector:
    """Collects transitions from parallel TacticalEnv instances into RolloutBuffers.

    Coordinates hierarchical action dispatch:
      1. Evaluates tactical events or step horizons to invoke Commander.
      2. Commander selects operational policy (Fight/Engage vs Escape/Defend).
      3. Low-level domain policies generate physical flight/drive/sail commands.
      4. Fills on-policy RolloutBuffers for each policy.
    """

    def __init__(
        self,
        policies: dict[str, Any],
        envs: list[TacticalEnv],
        config: dict[str, Any] | None = None,
        rng: np.random.Generator | None = None,
        opponent_policy: Any | None = None,
    ) -> None:
        self.policies = policies
        self.envs = envs
        self.config = config or {}
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.opponent_policy = opponent_policy

        self.high_level_horizon = int(self.config.get("high_level_horizon", HIGH_LEVEL_HORIZON))
        self.low_level_horizon = int(self.config.get("low_level_horizon", LOW_LEVEL_HORIZON))
        self.buffer_size = int(self.config.get("buffer_size", LOW_LEVEL_BATCH_SIZE))

        # Per-policy RolloutBuffers
        self.buffers: dict[str, RolloutBuffer] = self._init_buffers()

        # Parallel environment state tracking
        self.num_envs = len(envs)
        self.env_obs: list[dict[str, np.ndarray]] = [e.reset() for e in self.envs]
        self.env_steps: list[int] = [0] * self.num_envs
        self.last_commander_step: list[dict[str, int]] = [{} for _ in range(self.num_envs)]
        self.assigned_policy_mode: list[dict[str, str]] = [{} for _ in range(self.num_envs)]

        # Episode performance metrics
        self.blue_wins: int = 0
        self.red_wins: int = 0
        self.draws: int = 0
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self._current_episode_rewards: list[float] = [0.0] * self.num_envs
        self._current_episode_lengths: list[int] = [0] * self.num_envs

    def _init_buffers(self) -> dict[str, RolloutBuffer]:
        """Instantiate preallocated RolloutBuffers for all policies."""
        buf: dict[str, RolloutBuffer] = {}

        # Air policies (multi-discrete actions)
        buf["air_fight"] = RolloutBuffer(
            batch_size=self.buffer_size,
            obs_dim=AIR_OBS_DIM,
            action_dim=4,
            action_type="discrete",
        )
        buf["air_escape"] = RolloutBuffer(
            batch_size=self.buffer_size,
            obs_dim=AIR_OBS_DIM,
            action_dim=4,
            action_type="discrete",
        )

        # Ground policies (hybrid actions: 2 cont + 2 discrete)
        buf["ground_engage"] = RolloutBuffer(
            batch_size=self.buffer_size,
            obs_dim=GROUND_OBS_DIM,
            action_dim=4,
            action_type="hybrid",
        )
        buf["ground_defend"] = RolloutBuffer(
            batch_size=self.buffer_size,
            obs_dim=GROUND_OBS_DIM,
            action_dim=4,
            action_type="hybrid",
        )

        # Sea policies (hybrid actions: 2 cont + 2 discrete)
        buf["sea_engage"] = RolloutBuffer(
            batch_size=self.buffer_size,
            obs_dim=SEA_OBS_DIM,
            action_dim=4,
            action_type="hybrid",
        )
        buf["sea_defend"] = RolloutBuffer(
            batch_size=self.buffer_size,
            obs_dim=SEA_OBS_DIM,
            action_dim=4,
            action_type="hybrid",
        )

        # Commander policy
        buf["commander"] = RolloutBuffer(
            batch_size=self.buffer_size,
            obs_dim=COMMANDER_OBS_DIM,
            action_dim=1,
            action_type="discrete",
        )

        return buf

    def set_opponent_policy(self, opponent_policy: Any | None) -> None:
        """Set or update the opponent policy used for Red forces."""
        self.opponent_policy = opponent_policy

    def _should_invoke_commander(self, env: TacticalEnv, env_idx: int, entity_id: str) -> bool:
        """Evaluate if Commander should be triggered for an agent.

        Triggers:
          - Periodic: Every HIGH_LEVEL_HORIZON (40) steps
          - Event 1: Any entity destroyed in the environment
          - Event 2: Agent within 6 km of map boundary
          - Event 3: Agent in favorable tactical position (d < 5km, ATA < 30 deg, AA < 50 deg)
          - Event 4: Multiple opponents close (< 5km) and facing agent
        """
        entity = env.entities.get(entity_id)
        if entity is None or not entity.is_alive():
            return False

        last_step = self.last_commander_step[env_idx].get(entity_id, -999)
        step_diff = self.env_steps[env_idx] - last_step

        # Enforce minimum low-level horizon unless urgent boundary or death event
        if step_diff < self.low_level_horizon:
            # Check boundary emergency (< 3 km)
            pos = entity.position[:2]
            map_lim = env.map.size_km
            if pos[0] < 3.0 or pos[0] > (map_lim - 3.0) or pos[1] < 3.0 or pos[1] > (map_lim - 3.0):
                return True
            return False

        # Periodic trigger
        if step_diff >= self.high_level_horizon:
            return True

        # Event: boundary proximity (< 6 km)
        pos = entity.position[:2]
        map_lim = env.map.size_km
        if pos[0] < 6.0 or pos[0] > (map_lim - 6.0) or pos[1] < 6.0 or pos[1] > (map_lim - 6.0):
            return True

        # Event: check active opponent positions
        opponents = env.red_entities if entity.team == TeamSide.BLUE else env.blue_entities
        alive_opps = [o for o in opponents if o.is_alive()]
        close_count = 0
        for opp in alive_opps:
            d = math.hypot(entity.position[0] - opp.position[0], entity.position[1] - opp.position[1])
            if d < 5.0:
                close_count += 1
                if close_count >= 2:
                    return True

        return False

    def _build_commander_observation(self, env: TacticalEnv, entity_id: str) -> np.ndarray:
        """Construct normalized 53-dimension CommanderObservation vector for an entity."""
        entity = env.entities.get(entity_id)
        if entity is None:
            return np.zeros(COMMANDER_OBS_DIM, dtype=np.float32)

        # Own state: [x, y, z, speed, heading] normalized to [0, 1]
        map_size = env.map.size_km * 1000.0
        own_pos = entity.position
        own_state = np.array([
            own_pos[0] / max(1.0, map_size),
            own_pos[1] / max(1.0, map_size),
            own_pos[2] / 15000.0,
            float(np.linalg.norm(entity.velocity)) / 600.0 if isinstance(entity.velocity, (list, tuple, np.ndarray)) else float(entity.velocity) / 600.0,
            entity.heading / (2.0 * math.pi),
        ], dtype=np.float32)

        opponents = env.red_entities if entity.team == TeamSide.BLUE else env.blue_entities
        alive_opps = [o for o in opponents if o.is_alive()]
        alive_opps.sort(key=lambda o: math.hypot(own_pos[0] - o.position[0], own_pos[1] - o.position[1]))

        opp_states: list[np.ndarray] = []
        for opp in alive_opps[:3]:
            dom_id = 0.0 if opp.domain == DomainType.AIR else (1.0 if opp.domain == DomainType.GROUND else 2.0)
            opp_states.append(np.array([
                opp.position[0] / max(1.0, map_size),
                opp.position[1] / max(1.0, map_size),
                opp.position[2] / 15000.0,
                0.5,
                opp.heading / (2.0 * math.pi),
                dom_id,
            ], dtype=np.float32))

        friendlies = env.blue_entities if entity.team == TeamSide.BLUE else env.red_entities
        alive_frs = [f for f in friendlies if f.is_alive() and f.entity_id != entity_id]
        alive_frs.sort(key=lambda f: math.hypot(own_pos[0] - f.position[0], own_pos[1] - f.position[1]))

        fr_states: list[np.ndarray] = []
        for fr in alive_frs[:2]:
            dom_id = 0.0 if fr.domain == DomainType.AIR else (1.0 if fr.domain == DomainType.GROUND else 2.0)
            fr_states.append(np.array([
                fr.position[0] / max(1.0, map_size),
                fr.position[1] / max(1.0, map_size),
                fr.position[2] / 15000.0,
                0.5,
                fr.heading / (2.0 * math.pi),
                dom_id,
            ], dtype=np.float32))

        cmd_obs = CommanderObservation(
            own_state=own_state,
            opponent_states=opp_states,
            friendly_states=fr_states,
        )
        return cmd_obs.to_padded_array(state_dim=5)

    def _select_policy_for_agent(self, domain: DomainType, cmd_action: int) -> str:
        """Map Commander discrete command {0..3} to domain policy key."""
        if domain == DomainType.AIR:
            return "air_escape" if cmd_action == 0 else "air_fight"
        elif domain == DomainType.GROUND:
            return "ground_defend" if cmd_action == 0 else "ground_engage"
        elif domain == DomainType.SEA:
            return "sea_defend" if cmd_action == 0 else "sea_engage"
        return "air_fight"

    def collect(self, steps: int) -> dict[str, RolloutBuffer]:
        """Collect simulation transitions across all parallel environments.

        Args:
            steps: Number of steps to step each environment forward.

        Returns:
            Dictionary of policy_name -> filled RolloutBuffer.
        """
        for _ in range(steps):
            for env_idx, env in enumerate(self.envs):
                obs_dict = self.env_obs[env_idx]
                action_dict: dict[str, Any] = {}

                # 1. Process Blue friendly agents
                for eid, obs in obs_dict.items():
                    entity = env.entities.get(eid)
                    if entity is None or not entity.is_alive() or entity.team != TeamSide.BLUE:
                        continue

                    # Commander decision step
                    commander_policy = self.policies.get("commander")
                    if commander_policy is not None and self._should_invoke_commander(env, env_idx, eid):
                        cmd_obs = self._build_commander_observation(env, eid)
                        cmd_act, cmd_lp, cmd_val, _ = commander_policy.act(cmd_obs, agent_id=eid)

                        # Store in commander buffer
                        if len(self.buffers["commander"]) < self.buffer_size:
                            self.buffers["commander"].add(
                                obs=cmd_obs,
                                action=np.array([float(cmd_act)], dtype=np.float32),
                                reward=0.0,
                                done=False,
                                value=cmd_val,
                                log_prob=cmd_lp,
                            )

                        mode = self._select_policy_for_agent(entity.domain, cmd_act)
                        self.assigned_policy_mode[env_idx][eid] = mode
                        self.last_commander_step[env_idx][eid] = self.env_steps[env_idx]

                    # Get assigned or default operational mode
                    policy_mode = self.assigned_policy_mode[env_idx].get(
                        eid,
                        "air_fight" if entity.domain == DomainType.AIR else (
                            "ground_engage" if entity.domain == DomainType.GROUND else "sea_engage"
                        ),
                    )

                    policy = self.policies.get(policy_mode)
                    if policy is None:
                        continue

                    # Generate domain action
                    if entity.domain == DomainType.AIR:
                        act_arr, lp, val = policy.act(obs[:AIR_OBS_DIM])
                        sub = act_arr.astype(int)
                        h_step = int(sub[0]) - 6 if len(sub) > 0 else 0
                        v_cmd = int(sub[1]) if len(sub) > 1 else 4
                        f_c = int(sub[2]) if len(sub) > 2 else 0
                        f_r = int(sub[3]) if len(sub) > 3 else 0
                        action_dict[eid] = AirAction.from_discrete(h_step, v_cmd, f_c, f_r)

                        if len(self.buffers[policy_mode]) < self.buffer_size:
                            padded_act = act_arr[:4] if len(act_arr) >= 4 else np.pad(act_arr, (0, 4 - len(act_arr)))
                            self.buffers[policy_mode].add(
                                obs=obs[:AIR_OBS_DIM],
                                action=padded_act,
                                reward=0.0,
                                done=False,
                                value=val,
                                log_prob=lp,
                            )

                    elif entity.domain == DomainType.GROUND:
                        act_arr, lp, val = policy.act(obs[:GROUND_OBS_DIM])
                        h_delta = float(act_arr[0]) if len(act_arr) > 0 else 0.0
                        v_cmd = int(round(float(act_arr[1]))) if len(act_arr) > 1 else 0
                        w_sel = int(round(float(act_arr[2]))) if len(act_arr) > 2 else 0
                        f_flag = int(round(float(act_arr[3]))) if len(act_arr) > 3 else 0
                        action_dict[eid] = GroundAction(
                            heading_delta=h_delta, velocity_cmd=v_cmd,
                            weapon_select=w_sel, fire=f_flag,
                        )

                        if len(self.buffers[policy_mode]) < self.buffer_size:
                            padded_act = act_arr[:4] if len(act_arr) >= 4 else np.pad(act_arr, (0, 4 - len(act_arr)))
                            self.buffers[policy_mode].add(
                                obs=obs[:GROUND_OBS_DIM],
                                action=padded_act,
                                reward=0.0,
                                done=False,
                                value=val,
                                log_prob=lp,
                            )

                    elif entity.domain == DomainType.SEA:
                        act_arr, lp, val = policy.act(obs[:SEA_OBS_DIM])
                        h_delta = float(act_arr[0]) if len(act_arr) > 0 else 0.0
                        v_cmd = int(round(float(act_arr[1]))) if len(act_arr) > 1 else 0
                        w_sel = int(round(float(act_arr[2]))) if len(act_arr) > 2 else 0
                        f_flag = int(round(float(act_arr[3]))) if len(act_arr) > 3 else 0
                        action_dict[eid] = SeaAction(
                            heading_delta=h_delta, velocity_cmd=v_cmd,
                            weapon_select=w_sel, fire=f_flag,
                        )

                        if len(self.buffers[policy_mode]) < self.buffer_size:
                            padded_act = act_arr[:4] if len(act_arr) >= 4 else np.pad(act_arr, (0, 4 - len(act_arr)))
                            self.buffers[policy_mode].add(
                                obs=obs[:SEA_OBS_DIM],
                                action=padded_act,
                                reward=0.0,
                                done=False,
                                value=val,
                                log_prob=lp,
                            )

                # 2. Process Red opponent actions if non-scripted opponent policy is loaded
                if self.opponent_policy is not None:
                    for eid, obs in obs_dict.items():
                        entity = env.entities.get(eid)
                        if entity is None or not entity.is_alive() or entity.team != TeamSide.RED:
                            continue
                        if entity.domain == DomainType.AIR and hasattr(self.opponent_policy, "act"):
                            act_arr, _, _ = self.opponent_policy.act(obs[:AIR_OBS_DIM])
                            sub = act_arr.astype(int)
                            h_step = int(sub[0]) - 6 if len(sub) > 0 else 0
                            v_cmd = int(sub[1]) if len(sub) > 1 else 4
                            f_c = int(sub[2]) if len(sub) > 2 else 0
                            f_r = int(sub[3]) if len(sub) > 3 else 0
                            action_dict[eid] = AirAction.from_discrete(h_step, v_cmd, f_c, f_r)

                # 3. Environment Step
                next_obs, rewards, dones, info = env.step(action_dict)
                self.env_obs[env_idx] = next_obs
                self.env_steps[env_idx] += 1
                self._current_episode_lengths[env_idx] += 1

                step_reward = sum(r for eid, r in rewards.items() if eid.startswith("blue"))
                self._current_episode_rewards[env_idx] += step_reward

                # 4. Handle Episode Termination
                is_done = bool(info.get("global_done", False) or all(dones.values()) or dones.get("__all__", False))
                if is_done:
                    winner = info.get("winner")
                    if str(winner) == "BLUE" or winner == TeamSide.BLUE:
                        self.blue_wins += 1
                    elif str(winner) == "RED" or winner == TeamSide.RED:
                        self.red_wins += 1
                    else:
                        self.draws += 1

                    self.episode_rewards.append(self._current_episode_rewards[env_idx])
                    self.episode_lengths.append(self._current_episode_lengths[env_idx])
                    self._current_episode_rewards[env_idx] = 0.0
                    self._current_episode_lengths[env_idx] = 0

                    # Reset env
                    self.env_obs[env_idx] = env.reset()
                    self.env_steps[env_idx] = 0
                    self.last_commander_step[env_idx].clear()
                    self.assigned_policy_mode[env_idx].clear()

        return self.buffers

    def get_episode_stats(self) -> dict[str, Any]:
        """Return cumulative statistics across completed episodes."""
        total = self.blue_wins + self.red_wins + self.draws
        mean_rew = float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
        mean_len = float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0
        return {
            "blue_wins": self.blue_wins,
            "red_wins": self.red_wins,
            "draws": self.draws,
            "total_episodes": total,
            "win_rate": (self.blue_wins / total) if total > 0 else 0.0,
            "mean_reward": mean_rew,
            "mean_episode_length": mean_len,
        }

    def reset_stats(self) -> None:
        """Reset episode outcome counters and statistics."""
        self.blue_wins = 0
        self.red_wins = 0
        self.draws = 0
        self.episode_rewards.clear()
        self.episode_lengths.clear()
