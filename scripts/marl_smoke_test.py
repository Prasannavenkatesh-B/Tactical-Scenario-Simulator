"""Integration smoke test for the hierarchical MARL engine.

Creates a Level 3 scenario, instantiates all policies + commander,
runs simulation steps, collects rewards, verifies shapes,
and confirms PPO update succeeds.

Usage:
    uv run --with torch --with numpy python scripts/marl_smoke_test.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import torch

from src.core.interfaces import DomainType
from src.core.actions import AirAction
from src.simulator.scenarios import generate_scenario
from src.simulator.env import TacticalEnv
from src.marl.policies.air_fight import AirFightPolicy
from src.marl.policies.air_escape import AirEscapePolicy
from src.marl.commander import CommanderPolicy
from src.marl.rollout_buffer import RolloutBuffer
from src.marl.config import (
    AIR_OBS_DIM,
    COMMANDER_OBS_DIM,
    GAMMA,
    GAE_LAMBDA,
)


def main() -> None:
    """Run integration smoke test."""
    print("=" * 60)
    print("MARL ENGINE INTEGRATION SMOKE TEST")
    print("=" * 60)
    passed = True

    try:
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1. Create scenario and environment
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        print("\n[1/6] Creating Level 3 scenario...")
        rng = np.random.default_rng(42)
        scenario = generate_scenario(level=3, rng=rng)
        env = TacticalEnv(scenario_config=scenario, seed=42)
        obs_dict = env.reset()
        print(f"  Entities: {list(obs_dict.keys())}")
        print(f"  Observation shapes: {[(k, v.shape) for k, v in obs_dict.items()]}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 2. Instantiate policies
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        print("\n[2/6] Instantiating policies...")
        fight_policy = AirFightPolicy(variant="AC1")
        escape_policy = AirEscapePolicy(variant="AC1")
        commander = CommanderPolicy()
        print(f"  AirFightPolicy params: {sum(p.numel() for p in fight_policy.parameters()):,}")
        print(f"  AirEscapePolicy params: {sum(p.numel() for p in escape_policy.parameters()):,}")
        print(f"  CommanderPolicy params: {sum(p.numel() for p in commander.parameters()):,}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 3. Run simulation steps
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        print("\n[3/6] Running 100 simulation steps...")
        buffer = RolloutBuffer(
            batch_size=500, obs_dim=AIR_OBS_DIM, action_dim=4,
        )
        total_reward = 0.0
        steps_run = 0

        for step in range(100):
            action_dict = {}
            blue_obs_list = []

            for eid, obs in obs_dict.items():
                entity = env.entities.get(eid)
                if entity is None or not entity.is_alive():
                    continue
                if entity.team.value != "BLUE":
                    continue
                if entity.domain != DomainType.AIR:
                    continue

                # Use fight policy for blue agents
                obs_array = obs.astype(np.float32) if not isinstance(obs, np.ndarray) else obs
                if obs_array.shape[0] < AIR_OBS_DIM:
                    obs_array = np.zeros(AIR_OBS_DIM, dtype=np.float32)
                    obs_array[:obs.shape[0]] = obs

                action_arr, log_prob, value = fight_policy.act(obs_array[:AIR_OBS_DIM])

                # Convert to AirAction
                sub_actions = action_arr.astype(int)
                heading_step = int(sub_actions[0]) - 6  # Map [0,12] -> [-6,6]
                velocity_cmd = int(sub_actions[1])
                fire_cannon = int(sub_actions[2]) if len(sub_actions) > 2 else 0
                fire_rocket = int(sub_actions[3]) if len(sub_actions) > 3 else 0

                air_action = AirAction.from_discrete(
                    heading_step=heading_step,
                    velocity_cmd=velocity_cmd,
                    fire_cannon=fire_cannon,
                    fire_rocket=fire_rocket,
                )
                action_dict[eid] = air_action

                # Store in buffer
                buffer.add(
                    obs=obs_array[:AIR_OBS_DIM],
                    action=action_arr[:4] if len(action_arr) >= 4 else np.pad(action_arr, (0, 4 - len(action_arr))),
                    reward=0.0,
                    done=False,
                    value=value,
                    log_prob=log_prob,
                )
                blue_obs_list.append(obs_array)

            # Step environment
            obs_dict, reward_dict, done_dict, info_dict = env.step(action_dict)
            total_reward += sum(reward_dict.values())
            steps_run += 1

            if done_dict.get("__all__", False):
                print(f"  Episode ended at step {step + 1}")
                break

        print(f"  Steps run: {steps_run}")
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Buffer size: {len(buffer)}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 4. Verify shapes
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        print("\n[4/6] Verifying shapes...")
        assert buffer.observations.shape[1] == AIR_OBS_DIM, f"obs_dim mismatch: {buffer.observations.shape[1]}"
        assert buffer.actions.shape[1] == 4, f"action_dim mismatch: {buffer.actions.shape[1]}"
        print("  Shapes OK")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 5. Test PPO update
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        print("\n[5/6] Testing PPO update...")
        if len(buffer) > 10:
            buffer.compute_advantages_and_returns(
                last_value=0.0, gamma=GAMMA, gae_lambda=GAE_LAMBDA,
            )
            losses = fight_policy.update(buffer)
            print(f"  Losses: {losses}")
            assert "total_loss" in losses
            assert np.isfinite(losses["total_loss"]), "Loss is not finite!"
            print("  PPO update OK")
        else:
            print("  Skipped (not enough data)")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 6. Commander test
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        print("\n[6/6] Testing commander policy...")
        cmd_obs = np.random.rand(COMMANDER_OBS_DIM).astype(np.float32)
        cmd_action, cmd_lp, cmd_val, _ = commander.act(cmd_obs, agent_id="test_agent")
        assert cmd_action in {0, 1, 2, 3}, f"Invalid commander action: {cmd_action}"
        print(f"  Commander action: {cmd_action}, log_prob: {cmd_lp:.4f}, value: {cmd_val:.4f}")
        print("  Commander OK")

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        passed = False

    print("\n" + "=" * 60)
    if passed:
        print("INTEGRATION SMOKE TEST PASSED")
    else:
        print("INTEGRATION SMOKE TEST FAILED")
    print("=" * 60)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
