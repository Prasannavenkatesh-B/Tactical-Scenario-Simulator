"""Performance and latency benchmark tests for the FastAPI inference API."""

import time
from fastapi.testclient import TestClient
import numpy as np

from src.api.config import LATENCY_TARGET_MS


def test_inference_latency_benchmark(client: TestClient) -> None:
    """Benchmark 100 sequential inference calls to ensure strict latency compliance (< 50ms)."""
    payload = {
        "domain": "air",
        "variant": "AC1",
        "agent_id": "latency_bench_agent",
        "observation": [0.05] * 13,
        "deterministic": True,
    }

    # 1. Warm-up calls (10 iterations)
    for _ in range(10):
        resp = client.post("/act", json=payload)
        assert resp.status_code == 200

    # 2. Benchmark 100 sequential requests
    num_requests = 100
    latencies_server_ms: list[float] = []
    latencies_roundtrip_ms: list[float] = []

    for _ in range(num_requests):
        t0 = time.perf_counter()
        resp = client.post("/act", json=payload)
        roundtrip_ms = (time.perf_counter() - t0) * 1000.0

        assert resp.status_code == 200
        data = resp.json()
        latencies_server_ms.append(data["latency_ms"])
        latencies_roundtrip_ms.append(roundtrip_ms)

    mean_server = float(np.mean(latencies_server_ms))
    p50_server = float(np.percentile(latencies_server_ms, 50))
    p95_server = float(np.percentile(latencies_server_ms, 95))
    max_server = float(np.max(latencies_server_ms))

    print(
        f"\n[LATENCY BENCHMARK - 100 Requests]\n"
        f"  Server Mean: {mean_server:.2f} ms (Target: < {LATENCY_TARGET_MS} ms)\n"
        f"  Server p50:  {p50_server:.2f} ms\n"
        f"  Server p95:  {p95_server:.2f} ms\n"
        f"  Server Max:  {max_server:.2f} ms\n"
        f"  Roundtrip Mean: {np.mean(latencies_roundtrip_ms):.2f} ms\n"
        f"  Roundtrip p95:  {np.percentile(latencies_roundtrip_ms, 95):.2f} ms"
    )

    # Assert strict requirements
    assert mean_server < LATENCY_TARGET_MS, (
        f"Mean server inference latency {mean_server:.2f}ms exceeds target {LATENCY_TARGET_MS}ms"
    )
    assert p95_server < 100.0, (
        f"95th percentile latency {p95_server:.2f}ms exceeds 100ms bound"
    )


def test_batch_inference_latency(client: TestClient) -> None:
    """Benchmark batch inference with 10 heterogeneous agents."""
    batch_payload = {
        "requests": [
            {"domain": "air", "variant": "AC1", "agent_id": f"air_{i}", "observation": [0.0] * 13}
            for i in range(4)
        ] + [
            {"domain": "ground", "agent_id": f"ground_{i}", "observation": [0.0] * 9}
            for i in range(3)
        ] + [
            {"domain": "sea", "agent_id": f"sea_{i}", "observation": [0.0] * 9}
            for i in range(2)
        ] + [
            {"domain": "commander", "agent_id": "cmd_0", "observation": [0.0] * 53}
        ]
    }

    # Warmup
    for _ in range(3):
        resp = client.post("/act/batch", json=batch_payload)
        assert resp.status_code == 200

    # Benchmark 20 batch requests
    latencies: list[float] = []
    for _ in range(20):
        t0 = time.perf_counter()
        resp = client.post("/act/batch", json=batch_payload)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        assert resp.status_code == 200

    mean_batch = float(np.mean(latencies))
    print(f"\n[BATCH LATENCY - 10 Agents] Mean: {mean_batch:.2f} ms across 20 batches")
    assert mean_batch < 250.0  # 10 agents at < 25ms each
