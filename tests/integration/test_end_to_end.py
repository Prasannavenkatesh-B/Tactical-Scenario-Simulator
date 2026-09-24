"""End-to-end integration tests verifying the full TSS <-> Wrapper <-> Model pipeline."""

import pytest

from src.integration.http_wrapper import HTTPTSSWrapper
from src.integration.in_process_wrapper import InProcessTSSWrapper


@pytest.fixture
def synthetic_observations() -> dict[str, dict]:
    return {
        "air": {
            "pos_x": 45.0, "pos_y": 55.0, "pos_z": 8.0, "speed": 650.0,
            "heading": 210.0, "heading_off": 25.0, "aspect_angle": 45.0,
            "antenna_train_angle": 10.0, "distance_to_opponent": 18.0,
            "cannon_ammo": 400.0, "rocket_ammo": 6.0, "rocket_ready": 1.0,
            "is_shooting": 0.0,
        },
        "ground": {
            "pos_x": 35.0, "pos_y": 65.0, "speed": 22.0, "heading": 135.0,
            "elevation": 1.2, "antenna_train_angle": 20.0, "distance_to_opponent": 12.0,
            "ammo": 80.0, "sensor_range": 30.0,
        },
        "sea": {
            "pos_x": 75.0, "pos_y": 25.0, "speed": 18.0, "heading": 315.0,
            "sea_state": 3.0, "antenna_train_angle": 15.0, "distance_to_opponent": 35.0,
            "ammo": 150.0, "radar_range": 60.0,
        },
        "commander": {
            "own_state": {"pos_x": 50.0, "pos_y": 50.0, "pos_z": 5.0, "speed": 500.0, "heading": 90.0},
            "opponents": [{"pos_x": 40.0, "pos_y": 50.0, "pos_z": 5.0, "speed": 400.0, "heading": 180.0, "domain": "air"}],
            "friendlies": [{"pos_x": 20.0, "pos_y": 30.0, "pos_z": 5.0, "speed": 450.0, "heading": 90.0, "domain": "air"}],
        },
    }


def test_end_to_end_all_domains_in_process(
    in_process_wrapper: InProcessTSSWrapper, synthetic_observations: dict[str, dict]
) -> None:
    """Execute end-to-end telemetry -> policy -> command loop across all domains in Mode A."""
    in_process_wrapper.reset()

    # Air
    air_cmd = in_process_wrapper.get_action(
        agent_id="blue_air_1", domain="air", variant="AC1",
        tss_observation=synthetic_observations["air"], deterministic=True
    )
    assert isinstance(air_cmd["turn"], float)
    assert isinstance(air_cmd["set_speed"], float)
    assert isinstance(air_cmd["fire_cannon"], bool)

    # Ground
    gnd_cmd = in_process_wrapper.get_action(
        agent_id="blue_gnd_1", domain="ground", variant="default",
        tss_observation=synthetic_observations["ground"], deterministic=True
    )
    assert isinstance(gnd_cmd["turn"], float)
    assert isinstance(gnd_cmd["set_speed"], float)
    assert isinstance(gnd_cmd["fire"], bool)

    # Sea
    sea_cmd = in_process_wrapper.get_action(
        agent_id="blue_sea_1", domain="sea", variant="default",
        tss_observation=synthetic_observations["sea"], deterministic=True
    )
    assert isinstance(sea_cmd["turn"], float)
    assert isinstance(sea_cmd["set_speed"], float)

    # Commander
    cmd_dir = in_process_wrapper.get_commander_action(
        agent_id="hq_blue", tss_observation=synthetic_observations["commander"], deterministic=True
    )
    assert cmd_dir["activate_policy"] in ("fight", "escape", "engage", "defend")
    assert isinstance(cmd_dir["target_index"], int)


def test_in_process_vs_http_parity(
    in_process_wrapper: InProcessTSSWrapper,
    http_wrapper: HTTPTSSWrapper,
    synthetic_observations: dict[str, dict],
) -> None:
    """Deterministic mode must yield identical TSS commands between In-Process and HTTP modes."""
    in_process_wrapper.reset()
    http_wrapper.reset()

    # Air
    air_in_proc = in_process_wrapper.get_action(
        agent_id="test_air", domain="air", variant="AC1",
        tss_observation=synthetic_observations["air"], deterministic=True
    )
    air_http = http_wrapper.get_action(
        agent_id="test_air", domain="air", variant="AC1",
        tss_observation=synthetic_observations["air"], deterministic=True
    )
    assert air_in_proc == air_http

    # Ground
    gnd_in_proc = in_process_wrapper.get_action(
        agent_id="test_gnd", domain="ground", variant="default",
        tss_observation=synthetic_observations["ground"], deterministic=True
    )
    gnd_http = http_wrapper.get_action(
        agent_id="test_gnd", domain="ground", variant="default",
        tss_observation=synthetic_observations["ground"], deterministic=True
    )
    assert gnd_in_proc == gnd_http

    # Sea
    sea_in_proc = in_process_wrapper.get_action(
        agent_id="test_sea", domain="sea", variant="default",
        tss_observation=synthetic_observations["sea"], deterministic=True
    )
    sea_http = http_wrapper.get_action(
        agent_id="test_sea", domain="sea", variant="default",
        tss_observation=synthetic_observations["sea"], deterministic=True
    )
    assert sea_in_proc == sea_http
