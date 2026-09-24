"""Unit tests for Mode B (HTTPTSSWrapper) including retries and error handling."""

from unittest.mock import MagicMock
import httpx
import pytest

from src.integration.http_wrapper import HTTPTSSWrapper, TSSConnectionError
from src.integration.tss_observation_mapper import TSSMappingError


@pytest.fixture
def sample_air_obs() -> dict:
    return {
        "pos_x": 30.0, "pos_y": 40.0, "pos_z": 6.0, "speed": 600.0,
        "heading": 180.0, "heading_off": 30.0, "aspect_angle": 60.0,
        "antenna_train_angle": 15.0, "distance_to_opponent": 20.0,
        "cannon_ammo": 300.0, "rocket_ammo": 4.0, "rocket_ready": 1.0,
        "is_shooting": 0.0,
    }


def test_http_health_check(http_wrapper: HTTPTSSWrapper) -> None:
    """HTTP wrapper can query health endpoint successfully."""
    status = http_wrapper.health()
    assert status["status"] == "ok"
    assert "version" in status


def test_http_get_action(http_wrapper: HTTPTSSWrapper, sample_air_obs: dict) -> None:
    """HTTP wrapper round-trips single-agent act request."""
    cmd = http_wrapper.get_action(
        agent_id="blue_air_1",
        domain="air",
        variant="AC1",
        tss_observation=sample_air_obs,
        deterministic=True,
    )
    assert isinstance(cmd, dict)
    assert "turn" in cmd
    assert "set_speed" in cmd


def test_http_commander_action(http_wrapper: HTTPTSSWrapper) -> None:
    """HTTP wrapper round-trips commander act request and tracks hidden state."""
    cmd_obs = {
        "own_state": {"pos_x": 50.0, "pos_y": 50.0, "pos_z": 5.0, "speed": 500.0, "heading": 90.0},
        "opponents": [{"pos_x": 30.0, "pos_y": 30.0, "pos_z": 5.0, "speed": 400.0, "heading": 0.0, "domain": "air"}],
    }
    cmd1 = http_wrapper.get_commander_action("hq_blue", cmd_obs, deterministic=True)
    assert "activate_policy" in cmd1
    assert "hq_blue" in http_wrapper._hidden_states

    # Step 2 maintains recurrence
    cmd2 = http_wrapper.get_commander_action("hq_blue", cmd_obs, deterministic=True)
    assert "activate_policy" in cmd2


def test_http_batch_actions(http_wrapper: HTTPTSSWrapper, sample_air_obs: dict) -> None:
    """HTTP wrapper round-trips batch act request."""
    requests = [
        {"agent_id": f"air_{i}", "domain": "air", "variant": "AC1", "tss_observation": sample_air_obs}
        for i in range(5)
    ]
    cmds = http_wrapper.get_actions_batch(requests)
    assert len(cmds) == 5
    for c in cmds:
        assert "turn" in c


def test_http_retry_on_simulated_500(protocol_path: str) -> None:
    """HTTP wrapper retries on transient 500 errors before succeeding."""
    mock_client = MagicMock(spec=httpx.Client)

    # First call returns 500, second call returns 200
    mock_resp_500 = MagicMock()
    mock_resp_500.status_code = 500
    mock_resp_500.text = "Internal Server Error"

    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.json.return_value = {
        "action": [0, 4, 1, 0],
        "log_prob": 0.0,
        "value": 0.0,
        "hidden_state": None,
        "latency_ms": 1.0,
    }

    mock_client.post.side_effect = [mock_resp_500, mock_resp_200]

    wrapper = HTTPTSSWrapper(
        base_url="http://mockserver",
        client=mock_client,
        max_retries=3,
        protocol_path=protocol_path,
    )

    sample_obs = {
        "pos_x": 30.0, "pos_y": 40.0, "pos_z": 6.0, "speed": 600.0,
        "heading": 180.0, "heading_off": 30.0, "aspect_angle": 60.0,
        "antenna_train_angle": 15.0, "distance_to_opponent": 20.0,
        "cannon_ammo": 300.0, "rocket_ammo": 4.0, "rocket_ready": 1.0,
        "is_shooting": 0.0,
    }
    cmd = wrapper.get_action("agent_1", "air", "AC1", sample_obs)
    assert "turn" in cmd
    assert mock_client.post.call_count == 2


def test_http_timeout_raises_connection_error(protocol_path: str) -> None:
    """HTTP wrapper raises TSSConnectionError when all retries timeout."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.side_effect = httpx.TimeoutException("Connection timed out")

    wrapper = HTTPTSSWrapper(
        base_url="http://mockserver",
        client=mock_client,
        max_retries=2,
        protocol_path=protocol_path,
    )

    sample_obs = {
        "pos_x": 30.0, "pos_y": 40.0, "pos_z": 6.0, "speed": 600.0,
        "heading": 180.0, "heading_off": 30.0, "aspect_angle": 60.0,
        "antenna_train_angle": 15.0, "distance_to_opponent": 20.0,
        "cannon_ammo": 300.0, "rocket_ammo": 4.0, "rocket_ready": 1.0,
        "is_shooting": 0.0,
    }
    with pytest.raises(TSSConnectionError, match="Failed communicating with inference API"):
        wrapper.get_action("agent_1", "air", "AC1", sample_obs)
