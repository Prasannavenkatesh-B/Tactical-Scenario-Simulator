"""Integration tests for all FastAPI route handlers."""

from pathlib import Path
from fastapi.testclient import TestClient
import torch

from src.marl.policies.air_fight import AirFightPolicy


def test_get_health(client: TestClient) -> None:
    """GET /health should return 200 with status ok and loaded policies."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"
    assert len(data["loaded_policies"]) == 9
    assert "air_fight_AC1" in data["loaded_policies"]


def test_post_act_air(client: TestClient) -> None:
    """POST /act with valid air domain request."""
    payload = {
        "domain": "air",
        "variant": "AC1",
        "agent_id": "fighter_1",
        "observation": [0.1] * 13,
        "deterministic": True,
    }
    response = client.post("/act", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["action"], list)
    assert len(data["action"]) == 4
    assert isinstance(data["log_prob"], float)
    assert isinstance(data["value"], float)
    assert data["hidden_state"] is None
    assert data["latency_ms"] >= 0.0


def test_post_act_ground(client: TestClient) -> None:
    """POST /act with valid ground domain request."""
    payload = {
        "domain": "ground",
        "agent_id": "tank_1",
        "observation": [0.0] * 9,
        "deterministic": True,
    }
    response = client.post("/act", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["action"], list)
    assert len(data["action"]) == 4


def test_post_act_sea(client: TestClient) -> None:
    """POST /act with valid sea domain request."""
    payload = {
        "domain": "sea",
        "agent_id": "frigate_1",
        "observation": [0.0] * 9,
        "deterministic": True,
    }
    response = client.post("/act", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["action"], list)
    assert len(data["action"]) == 4


def test_post_act_commander_recurrent(client: TestClient) -> None:
    """POST /act with commander domain, verifying hidden state tracking."""
    payload_1 = {
        "domain": "commander",
        "agent_id": "hq_blue",
        "observation": [0.0] * 53,
        "deterministic": True,
    }
    response_1 = client.post("/act", json=payload_1)
    assert response_1.status_code == 200
    data_1 = response_1.json()
    assert len(data_1["action"]) == 1
    assert data_1["hidden_state"] is not None
    assert len(data_1["hidden_state"]) == 128

    # Step 2 passing back external hidden state
    payload_2 = {
        "domain": "commander",
        "agent_id": "hq_blue",
        "observation": [0.05] * 53,
        "hidden_state": data_1["hidden_state"],
        "deterministic": True,
    }
    response_2 = client.post("/act", json=payload_2)
    assert response_2.status_code == 200
    data_2 = response_2.json()
    assert len(data_2["hidden_state"]) == 128


def test_post_act_invalid_observation_dim(client: TestClient) -> None:
    """POST /act with wrong observation dimension returns 422."""
    payload = {
        "domain": "air",
        "variant": "AC1",
        "agent_id": "fighter_1",
        "observation": [0.1] * 5,  # Needs 13
    }
    response = client.post("/act", json=payload)
    assert response.status_code == 422
    assert "Expected 13 floats" in response.json()["detail"]


def test_post_act_unknown_domain(client: TestClient) -> None:
    """POST /act with unsupported domain returns 422 due to schema validation."""
    payload = {
        "domain": "space",
        "agent_id": "satellite_1",
        "observation": [0.0] * 10,
    }
    response = client.post("/act", json=payload)
    assert response.status_code == 422


def test_post_act_batch(client: TestClient) -> None:
    """POST /act/batch executes multiple heterogeneous requests sequentially."""
    batch_payload = {
        "requests": [
            {
                "domain": "air",
                "variant": "AC1",
                "agent_id": "fighter_1",
                "observation": [0.0] * 13,
            },
            {
                "domain": "air",
                "variant": "AC2",
                "agent_id": "fighter_2",
                "observation": [0.0] * 13,
            },
            {
                "domain": "ground",
                "agent_id": "tank_1",
                "observation": [0.0] * 9,
            },
            {
                "domain": "sea",
                "agent_id": "ship_1",
                "observation": [0.0] * 9,
            },
            {
                "domain": "commander",
                "agent_id": "cmd_1",
                "observation": [0.0] * 53,
            },
        ]
    }
    response = client.post("/act/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["responses"]) == 5
    assert data["total_latency_ms"] >= 0.0


def test_post_reset(client: TestClient) -> None:
    """POST /reset clears commander hidden states."""
    # Reset all
    resp_all = client.post("/reset", json={})
    assert resp_all.status_code == 200
    assert resp_all.json()["cleared_agents"] == ["all"]

    # Reset specific agents
    resp_spec = client.post("/reset", json={"agent_ids": ["cmd_1", "cmd_2"]})
    assert resp_spec.status_code == 200
    assert resp_spec.json()["cleared_agents"] == ["cmd_1", "cmd_2"]


def test_post_load_checkpoint(client: TestClient, tmp_path: Path) -> None:
    """POST /load_checkpoint hot-swaps weights dynamically."""
    # Create checkpoint file
    p = AirFightPolicy(variant="AC1")
    ckpt_file = tmp_path / "air_fight_AC1.pt"
    torch.save(p.state_dict(), ckpt_file)

    payload = {
        "policy_name": "air_fight_AC1",
        "checkpoint_path": str(ckpt_file),
    }
    response = client.post("/load_checkpoint", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "loaded"
    assert data["policy_name"] == "air_fight_AC1"
    assert data["param_count"] > 0

    # Non-existent checkpoint returns 404
    missing_payload = {
        "policy_name": "air_fight_AC1",
        "checkpoint_path": str(tmp_path / "non_existent.pt"),
    }
    err_resp = client.post("/load_checkpoint", json=missing_payload)
    assert err_resp.status_code == 404


def test_get_policies(client: TestClient) -> None:
    """GET /policies returns metadata for all loaded models."""
    response = client.get("/policies")
    assert response.status_code == 200
    data = response.json()
    assert len(data["policies"]) == 9
    names = [p["name"] for p in data["policies"]]
    assert "commander" in names
    assert "air_fight_AC1" in names
