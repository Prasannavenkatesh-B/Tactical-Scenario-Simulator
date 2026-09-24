"""CLI demo executing the TSS Integration Wrapper in in-process or HTTP mode."""

import argparse
from pathlib import Path
import random
import sys
import time
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.config import DEFAULT_CHECKPOINT_DIR
from src.integration.http_wrapper import HTTPTSSWrapper
from src.integration.in_process_wrapper import InProcessTSSWrapper


def generate_synthetic_tss_obs(domain: str) -> dict:
    """Generate mock unnormalized TSS telemetry dictionary conforming to protocol.yaml."""
    if domain == "air":
        return {
            "pos_x": random.uniform(10.0, 90.0),
            "pos_y": random.uniform(10.0, 90.0),
            "pos_z": random.uniform(2.0, 12.0),
            "speed": random.uniform(300.0, 900.0),
            "heading": random.uniform(0.0, 360.0),
            "heading_off": random.uniform(0.0, 180.0),
            "aspect_angle": random.uniform(0.0, 180.0),
            "antenna_train_angle": random.uniform(0.0, 180.0),
            "distance_to_opponent": random.uniform(5.0, 80.0),
            "cannon_ammo": float(random.randint(50, 500)),
            "rocket_ammo": float(random.randint(1, 8)),
            "rocket_ready": 1.0,
            "is_shooting": 0.0,
        }
    elif domain == "ground":
        return {
            "pos_x": random.uniform(10.0, 90.0),
            "pos_y": random.uniform(10.0, 90.0),
            "speed": random.uniform(5.0, 45.0),
            "heading": random.uniform(0.0, 360.0),
            "elevation": random.uniform(0.1, 3.5),
            "antenna_train_angle": random.uniform(0.0, 180.0),
            "distance_to_opponent": random.uniform(2.0, 50.0),
            "ammo": float(random.randint(10, 100)),
            "sensor_range": random.uniform(10.0, 45.0),
        }
    elif domain == "sea":
        return {
            "pos_x": random.uniform(10.0, 90.0),
            "pos_y": random.uniform(10.0, 90.0),
            "speed": random.uniform(5.0, 35.0),
            "heading": random.uniform(0.0, 360.0),
            "sea_state": float(random.randint(1, 6)),
            "antenna_train_angle": random.uniform(0.0, 180.0),
            "distance_to_opponent": random.uniform(5.0, 80.0),
            "ammo": float(random.randint(20, 200)),
            "radar_range": random.uniform(20.0, 90.0),
        }
    elif domain == "commander":
        return {
            "own_state": {
                "pos_x": random.uniform(10.0, 90.0),
                "pos_y": random.uniform(10.0, 90.0),
                "pos_z": 5.0,
                "speed": 500.0,
                "heading": 90.0,
            },
            "opponents": [
                {"pos_x": 40.0, "pos_y": 50.0, "pos_z": 5.0, "speed": 400.0, "heading": 180.0, "domain": "air", "team": "red"}
            ],
            "friendlies": [
                {"pos_x": 20.0, "pos_y": 30.0, "pos_z": 5.0, "speed": 450.0, "heading": 90.0, "domain": "air", "team": "blue"}
            ],
        }
    raise ValueError(f"Unknown domain: {domain}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TSS Integration Wrapper Demonstration")
    parser.add_argument("--mode", type=str, choices=["in_process", "http"], default="in_process",
                        help="Integration mode: in_process (Mode A) or http (Mode B)")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of benchmark iterations")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Inference service URL for HTTP mode")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/final", help="Path to checkpoint directory")

    args = parser.parse_args()

    print(f"============================================================")
    print(f"DRDO TSS AI Integration Wrapper Benchmark")
    print(f"  Mode:       {args.mode.upper()}")
    print(f"  Iterations: {args.iterations}")
    if args.mode == "http":
        print(f"  API URL:    {args.url}")
    print(f"============================================================")

    wrapper: InProcessTSSWrapper | HTTPTSSWrapper
    if args.mode == "in_process":
        wrapper = InProcessTSSWrapper(checkpoint_dir=args.checkpoint_dir)
    else:
        wrapper = HTTPTSSWrapper(base_url=args.url)
        print("[*] Checking service health...")
        health = wrapper.health()
        print(f"[+] Service online: {health}")

    wrapper.reset()

    domains = [
        ("air", "AC1", "B_AIR_1"),
        ("air", "AC2", "B_AIR_2"),
        ("ground", "default", "B_GND_1"),
        ("sea", "default", "B_SEA_1"),
        ("commander", "default", "HQ_BLUE"),
    ]

    latencies_ms: list[float] = []

    print(f"[*] Executing {args.iterations} simulation decision ticks...")
    t_start = time.perf_counter()

    for i in range(args.iterations):
        dom, var, aid = random.choice(domains)
        obs = generate_synthetic_tss_obs(dom)

        t0 = time.perf_counter()
        if dom == "commander":
            cmd = wrapper.get_commander_action(agent_id=aid, tss_observation=obs, deterministic=True)
            assert "activate_policy" in cmd
        else:
            cmd = wrapper.get_action(agent_id=aid, domain=dom, variant=var, tss_observation=obs, deterministic=True)
            assert "turn" in cmd
            assert "set_speed" in cmd

        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

        if (i + 1) % (max(1, args.iterations // 5)) == 0:
            print(f"    Completed {i + 1}/{args.iterations} ticks (Mean so far: {np.mean(latencies_ms):.2f} ms)...")

    total_wall_s = time.perf_counter() - t_start
    mean_lat = float(np.mean(latencies_ms))
    p50_lat = float(np.percentile(latencies_ms, 50))
    p95_lat = float(np.percentile(latencies_ms, 95))
    min_lat = float(np.min(latencies_ms))
    max_lat = float(np.max(latencies_ms))

    print(f"------------------------------------------------------------")
    print(f"BENCHMARK RESULTS ({args.mode.upper()} Mode):")
    print(f"  Total Requests:  {args.iterations}")
    print(f"  Total Wall Time: {total_wall_s:.2f} s")
    print(f"  Throughput:      {args.iterations / total_wall_s:.1f} calls/sec")
    print(f"  Latency (Mean):  {mean_lat:.2f} ms")
    print(f"  Latency (p50):   {p50_lat:.2f} ms")
    print(f"  Latency (p95):   {p95_lat:.2f} ms")
    print(f"  Latency (Min):   {min_lat:.2f} ms")
    print(f"  Latency (Max):   {max_lat:.2f} ms")
    print(f"============================================================")

    wrapper.shutdown()


if __name__ == "__main__":
    main()
