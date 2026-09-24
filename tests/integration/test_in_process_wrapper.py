"""Unit tests and latency benchmarks for Mode A (InProcessTSSWrapper)."""

import time
import numpy as np
import pytest

from src.integration.in_process_wrapper import InProcessTSSWrapper


@pytest.fixture
def sample_air_obs() -> dict:
    return {
        "pos_x": 30.0, "pos_y": 40.0, "pos_z": 6.0, "speed": 600.0,
        "heading": 180.0, "heading_off": 30.0, "aspect_angle": 60.0,
        "antenna_train_angle": 15.0, "distance_to_opponent": 20.0,
        "cannon_ammo": 300.0, "rocket_ammo": 4.0, "rocket_ready": 1.0,
        "is_shooting": 0.0,
    }


def test_in_process_get_action(in_process_wrapper: InProcessTSSWrapper, sample_air_obs: dict) -> None:
    """In-process wrapper returns valid TSS command dictionary."""
    cmd = in_process_wrapper.get_action(
        agent_id="blue_air_1",
        domain="air",
        variant="AC1",
        tss_observation=sample_air_obs,
        deterministic=True,
    )
    assert isinstance(cmd, dict)
    assert "turn" in cmd
    assert "set_speed" in cmd
    assert "fire_cannon" in cmd
    assert "fire_rocket" in cmd


def test_in_process_commander_action(in_process_wrapper: InProcessTSSWrapper) -> None:
    """In-process wrapper returns tactical directive for commander."""
    cmd_obs = {
        "own_state": {"pos_x": 50.0, "pos_y": 50.0, "pos_z": 5.0, "speed": 500.0, "heading": 90.0},
        "opponents": [{"pos_x": 30.0, "pos_y": 30.0, "pos_z": 5.0, "speed": 400.0, "heading": 0.0, "domain": "air"}],
    }
    directive = in_process_wrapper.get_commander_action(
        agent_id="hq_blue",
        tss_observation=cmd_obs,
        deterministic=True,
    )
    assert "activate_policy" in directive
    assert "target_index" in directive


def test_in_process_batch_actions(in_process_wrapper: InProcessTSSWrapper, sample_air_obs: dict) -> None:
    """Batch execution handles 10 concurrent vehicle requests."""
    batch_reqs = [
        {
            "agent_id": f"agent_{i}",
            "domain": "air",
            "variant": "AC1",
            "tss_observation": sample_air_obs,
        }
        for i in range(10)
    ]
    cmds = in_process_wrapper.get_actions_batch(batch_reqs)
    assert len(cmds) == 10
    for c in cmds:
        assert "turn" in c
        assert "set_speed" in c


def test_in_process_reset_clears_hiddens(in_process_wrapper: InProcessTSSWrapper) -> None:
    """Calling reset() clears all recurrent memory states in Commander policy."""
    from src.marl.commander import CommanderPolicy
    cmd_policy = in_process_wrapper.registry.get("commander")
    assert isinstance(cmd_policy, CommanderPolicy)
    import torch
    cmd_policy.agent_hiddens["agent_1"] = torch.zeros(1, 1, 128)
    assert len(cmd_policy.agent_hiddens) > 0

    in_process_wrapper.reset()
    assert len(cmd_policy.agent_hiddens) == 0


def test_in_process_latency_sub_5ms(in_process_wrapper: InProcessTSSWrapper, sample_air_obs: dict) -> None:
    """Verify in-process inference achieves strict < 5.0ms mean latency."""
    # Warmup
    for _ in range(10):
        in_process_wrapper.get_action("bench_agent", "air", "AC1", sample_air_obs)

    latencies: list[float] = []
    for _ in range(100):
        t0 = time.perf_counter()
        in_process_wrapper.get_action("bench_agent", "air", "AC1", sample_air_obs)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(latencies))
    print(f"\n[IN-PROCESS LATENCY] Mean: {mean_ms:.3f} ms (Target: < 5.0 ms)")
    assert mean_ms < 5.0, f"In-process latency {mean_ms:.2f}ms exceeds 5.0ms target"
