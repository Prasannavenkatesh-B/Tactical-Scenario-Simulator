"""Tests for observation and action encoding and decoding codecs."""

import math
import numpy as np
import pytest

from src.api.action_codec import ActionCodec, ObservationCodec
from src.core.actions import AirAction, CommanderAction, GroundAction, SeaAction


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ObservationCodec Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_observation_codec_validate_valid() -> None:
    """Valid observation vectors should pass validation without error."""
    ObservationCodec.validate("air", [0.0] * 13)
    ObservationCodec.validate("ground", [0.0] * 9)
    ObservationCodec.validate("sea", [0.0] * 9)
    ObservationCodec.validate("commander", [0.0] * 53)


def test_observation_codec_validate_rejects_wrong_dims() -> None:
    """Invalid observation vector lengths should raise ValueError."""
    with pytest.raises(ValueError, match="Expected 13 floats for domain 'air'"):
        ObservationCodec.validate("air", [0.0] * 12)

    with pytest.raises(ValueError, match="Expected 13 floats for domain 'air'"):
        ObservationCodec.validate("air", [0.0] * 14)

    with pytest.raises(ValueError, match="Expected 9 floats for domain 'ground'"):
        ObservationCodec.validate("ground", [0.0] * 8)

    with pytest.raises(ValueError, match="Expected 9 floats for domain 'sea'"):
        ObservationCodec.validate("sea", [0.0] * 10)

    with pytest.raises(ValueError, match="Expected 53 floats for domain 'commander'"):
        ObservationCodec.validate("commander", [0.0] * 52)


def test_observation_codec_encode_sanitizes_nan_inf() -> None:
    """ObservationCodec.encode should sanitize NaNs and infinities to 0.0."""
    raw = [float("nan"), float("inf"), float("-inf"), 1.5] + [0.0] * 9
    encoded = ObservationCodec.encode("air", "AC1", raw)

    assert encoded.shape == (1, 13)
    assert encoded.dtype == np.float32
    assert encoded[0, 0] == 0.0
    assert encoded[0, 1] == 0.0
    assert encoded[0, 2] == 0.0
    assert encoded[0, 3] == 1.5


def test_observation_codec_encode_pads_and_truncates() -> None:
    """ObservationCodec.encode handles under-length and over-length gracefully."""
    # Under-length gets padded
    short_obs = [1.0, 2.0]
    encoded_padded = ObservationCodec.encode("ground", "default", short_obs)
    assert encoded_padded.shape == (1, 9)
    assert encoded_padded[0, 0] == 1.0
    assert encoded_padded[0, 1] == 2.0
    assert (encoded_padded[0, 2:] == 0.0).all()

    # Over-length gets truncated
    long_obs = [float(i) for i in range(20)]
    encoded_truncated = ObservationCodec.encode("sea", "default", long_obs)
    assert encoded_truncated.shape == (1, 9)
    assert list(encoded_truncated[0]) == [float(i) for i in range(9)]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ActionCodec Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_action_codec_encode_air_action() -> None:
    """AirAction encodes into a list of 4 integers."""
    action = AirAction(heading_delta=3.0, velocity_cmd=6, fire_cannon=1, fire_rocket=0)
    encoded = ActionCodec.encode("air", action)
    assert isinstance(encoded, list)
    assert len(encoded) == 4
    assert encoded[1] == 6
    assert encoded[2] == 1
    assert encoded[3] == 0


def test_action_codec_encode_ground_action() -> None:
    """GroundAction encodes into a list of 4 integers."""
    action = GroundAction(heading_delta=15.0, velocity_cmd=4, weapon_select=1, fire=1)
    encoded = ActionCodec.encode("ground", action)
    assert encoded == [15, 4, 1, 1]


def test_action_codec_encode_sea_action() -> None:
    """SeaAction encodes into a list of 4 integers."""
    action = SeaAction(heading_delta=-10.0, velocity_cmd=3, weapon_select=0, fire=0)
    encoded = ActionCodec.encode("sea", action)
    assert encoded == [-10, 3, 0, 0]


def test_action_codec_encode_commander_action() -> None:
    """CommanderAction encodes into a 1-element list containing the action index."""
    action = CommanderAction(action=2)
    encoded = ActionCodec.encode("commander", action)
    assert encoded == [2]


def test_action_codec_encode_raw_array() -> None:
    """Raw numpy arrays or numeric lists encode cleanly."""
    encoded_arr = ActionCodec.encode("air", np.array([2, 5, 0, 1]))
    assert encoded_arr == [2, 5, 0, 1]

    encoded_cmd = ActionCodec.encode("commander", np.array([3]))
    assert encoded_cmd == [3]

    encoded_int = ActionCodec.encode("commander", 1)
    assert encoded_int == [1]


def test_action_codec_encode_unsupported_type() -> None:
    """Unsupported action type raises TypeError."""
    with pytest.raises(TypeError, match="Unsupported action type"):
        ActionCodec.encode("air", "invalid_action_string")


def test_action_codec_decode_roundtrip_air() -> None:
    """Decode JSON action back into AirAction dataclass."""
    decoded_list = ActionCodec.decode("air", "AC1", [2, 5, 1, 0])
    assert isinstance(decoded_list, AirAction)
    assert decoded_list.velocity_cmd == 5
    assert decoded_list.fire_cannon == 1
    assert decoded_list.fire_rocket == 0

    # From dict
    decoded_dict = ActionCodec.decode("air", "AC1", {"heading_delta": 4.5, "velocity_cmd": 7, "fire_cannon": 0, "fire_rocket": 1})
    assert isinstance(decoded_dict, AirAction)
    assert decoded_dict.heading_delta == 4.5
    assert decoded_dict.fire_rocket == 1


def test_action_codec_decode_roundtrip_ground() -> None:
    """Decode JSON action back into GroundAction dataclass."""
    decoded_list = ActionCodec.decode("ground", "default", [10, 3, 2, 1])
    assert isinstance(decoded_list, GroundAction)
    assert decoded_list.heading_delta == 10.0
    assert decoded_list.velocity_cmd == 3
    assert decoded_list.weapon_select == 2
    assert decoded_list.fire == 1

    decoded_dict = ActionCodec.decode("ground", "default", {"heading_delta": -5.0, "velocity_cmd": 2, "weapon_select": 1, "fire": 0})
    assert isinstance(decoded_dict, GroundAction)
    assert decoded_dict.heading_delta == -5.0


def test_action_codec_decode_roundtrip_sea() -> None:
    """Decode JSON action back into SeaAction dataclass."""
    decoded_list = ActionCodec.decode("sea", "default", [-15, 4, 1, 0])
    assert isinstance(decoded_list, SeaAction)
    assert decoded_list.heading_delta == -15.0
    assert decoded_list.velocity_cmd == 4
    assert decoded_list.weapon_select == 1
    assert decoded_list.fire == 0


def test_action_codec_decode_roundtrip_commander() -> None:
    """Decode JSON action back into CommanderAction dataclass."""
    decoded_list = ActionCodec.decode("commander", "default", [2])
    assert isinstance(decoded_list, CommanderAction)
    assert decoded_list.action == 2

    decoded_dict = ActionCodec.decode("commander", "default", {"action": 3})
    assert isinstance(decoded_dict, CommanderAction)
    assert decoded_dict.action == 3


def test_action_codec_decode_invalid_inputs() -> None:
    """Invalid actions or unknown domains raise appropriate errors."""
    with pytest.raises(ValueError, match="Unknown domain"):
        ActionCodec.decode("space", "default", [1, 2, 3])

    with pytest.raises(ValueError, match="Invalid Air action list length"):
        ActionCodec.decode("air", "AC1", [1])

    with pytest.raises(ValueError, match="Invalid Ground action list length"):
        ActionCodec.decode("ground", "default", [1, 2])

    with pytest.raises(ValueError, match="Invalid Sea action list length"):
        ActionCodec.decode("sea", "default", [1, 2])
