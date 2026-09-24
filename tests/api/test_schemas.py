"""Tests for Pydantic v2 API data models and schema validation."""

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    ActRequest,
    ActResponse,
    BatchActRequest,
    BatchActResponse,
    HealthResponse,
    LoadCheckpointRequest,
    LoadCheckpointResponse,
    PoliciesResponse,
    PolicyInfo,
    ResetRequest,
    ResetResponse,
)


def test_act_request_valid() -> None:
    """Verify ActRequest constructs with valid domain and field values."""
    req = ActRequest(
        agent_id="B_AIR_1",
        domain="air",
        variant="AC1",
        observation=[0.5] * 13,
        deterministic=True,
    )
    assert req.agent_id == "B_AIR_1"
    assert req.domain == "air"
    assert req.variant == "AC1"
    assert len(req.observation) == 13
    assert req.deterministic is True


def test_act_request_rejects_invalid_domain() -> None:
    """Verify ActRequest rejects domain values not in literal set."""
    with pytest.raises(ValidationError):
        ActRequest(
            agent_id="B_SPACE_1",
            domain="space",  # type: ignore[arg-type]
            variant="SATELLITE",
            observation=[0.1] * 10,
        )


def test_act_response_json_serialization() -> None:
    """Verify ActResponse serializes cleanly to JSON dictionary."""
    resp = ActResponse(
        action=[2, 4, 1, 0],
        log_prob=-0.42,
        value=1.85,
        hidden_state=None,
        latency_ms=2.15,
    )
    data = resp.model_dump()
    assert data["action"] == [2, 4, 1, 0]
    assert abs(data["log_prob"] - (-0.42)) < 1e-5
    assert abs(data["value"] - 1.85) < 1e-5
    assert abs(data["latency_ms"] - 2.15) < 1e-5

    # Check JSON string dump
    json_str = resp.model_dump_json()
    assert "latency_ms" in json_str


def test_batch_models_validation() -> None:
    """Verify BatchActRequest and BatchActResponse models."""
    req1 = ActRequest(agent_id="A1", domain="air", variant="AC1", observation=[0.1] * 13)
    batch_req = BatchActRequest(requests=[req1])
    assert len(batch_req.requests) == 1

    resp1 = ActResponse(action=[0], log_prob=0.0, value=0.0, latency_ms=1.0)
    batch_resp = BatchActResponse(responses=[resp1], total_latency_ms=1.5)
    assert len(batch_resp.responses) == 1
    assert batch_resp.total_latency_ms == 1.5


def test_lifecycle_and_management_schemas() -> None:
    """Verify Reset, Health, LoadCheckpoint, and Policy schemas."""
    reset_req = ResetRequest(episode_id="ep_001", agent_ids=["A1", "A2"])
    assert reset_req.episode_id == "ep_001"
    assert reset_req.agent_ids == ["A1", "A2"]

    reset_resp = ResetResponse(status="reset", cleared_agents=["A1", "A2"])
    assert reset_resp.cleared_agents == ["A1", "A2"]

    health_resp = HealthResponse(status="ok", loaded_policies=["air_fight_AC1"], version="1.0.0")
    assert health_resp.version == "1.0.0"

    load_req = LoadCheckpointRequest(policy_name="air_fight_AC1", checkpoint_path="test.pt")
    assert load_req.policy_name == "air_fight_AC1"

    load_resp = LoadCheckpointResponse(status="loaded", policy_name="air_fight_AC1", param_count=235000)
    assert load_resp.param_count == 235000

    pol_info = PolicyInfo(name="air_fight_AC1", variant="AC1", param_count=235000, loaded=True)
    policies_resp = PoliciesResponse(policies=[pol_info])
    assert len(policies_resp.policies) == 1
