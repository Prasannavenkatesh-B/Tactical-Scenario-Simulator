"""Concurrency and thread safety tests for the inference API."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from fastapi.testclient import TestClient


def _send_worker_requests(client: TestClient, thread_id: int, num_requests: int) -> list[int]:
    """Worker function executing inference requests across diverse domains."""
    status_codes: list[int] = []
    domains = [
        ("air", "AC1", [0.0] * 13),
        ("air", "AC2", [0.0] * 13),
        ("ground", None, [0.0] * 9),
        ("sea", None, [0.0] * 9),
        ("commander", None, [0.0] * 53),
    ]

    for req_idx in range(num_requests):
        dom, var, obs = random.choice(domains)
        payload = {
            "domain": dom,
            "agent_id": f"th_{thread_id}_agent_{req_idx}",
            "observation": obs,
            "deterministic": True,
        }
        if var is not None:
            payload["variant"] = var

        resp = client.post("/act", json=payload)
        status_codes.append(resp.status_code)

    return status_codes


def test_concurrent_inference_requests(client: TestClient) -> None:
    """Test 8 concurrent threads sending 20 requests each (160 total requests)."""
    num_threads = 8
    requests_per_thread = 20
    all_statuses: list[int] = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(_send_worker_requests, client, thread_id, requests_per_thread)
            for thread_id in range(num_threads)
        ]

        for fut in as_completed(futures):
            all_statuses.extend(fut.result())

    assert len(all_statuses) == num_threads * requests_per_thread
    assert all(code == 200 for code in all_statuses), (
        f"Non-200 responses observed during concurrent execution: "
        f"{[code for code in all_statuses if code != 200]}"
    )


def test_concurrent_read_and_reset(client: TestClient) -> None:
    """Test concurrent acts while episode resets occur simultaneously."""
    num_workers = 6
    iterations = 15

    def worker_act(tid: int) -> bool:
        for _ in range(iterations):
            resp = client.post(
                "/act",
                json={
                    "domain": "commander",
                    "agent_id": f"cmd_worker_{tid}",
                    "observation": [0.0] * 53,
                },
            )
            if resp.status_code != 200:
                return False
        return True

    def worker_reset() -> bool:
        for _ in range(iterations):
            resp = client.post("/reset", json={})
            if resp.status_code != 200:
                return False
        return True

    with ThreadPoolExecutor(max_workers=num_workers + 2) as executor:
        act_futures = [executor.submit(worker_act, i) for i in range(num_workers)]
        reset_futures = [executor.submit(worker_reset) for _ in range(2)]

        for fut in as_completed(act_futures + reset_futures):
            assert fut.result() is True
