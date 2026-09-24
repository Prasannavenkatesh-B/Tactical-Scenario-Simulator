"""Unit tests for TSSObservationMapper conversions, normalization, and validations."""

import math
import numpy as np
import pytest

from src.integration.tss_observation_mapper import TSSMappingError, TSSObservationMapper


@pytest.fixture
def mapper(protocol_path: str) -> TSSObservationMapper:
    return TSSObservationMapper(protocol_path=protocol_path)


def test_air_observation_mapping(mapper: TSSObservationMapper) -> None:
    """Air observation dictionary correctly maps to (13,) normalized float32 array."""
    sample_air = {
        "pos_x": 50.0,
        "pos_y": 50.0,
        "pos_z": 7.5,
        "speed": 583.0,
        "heading": 180.0,
        "heading_off": 45.0,
        "aspect_angle": 90.0,
        "antenna_train_angle": 30.0,
        "distance_to_opponent": 25.0,
        "cannon_ammo": 250.0,
        "rocket_ammo": 4.0,
        "rocket_ready": 1.0,
        "is_shooting": 0.0,
    }
    vec = mapper.to_air_observation(sample_air)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (13,)
    assert vec.dtype == np.float32
    assert np.all(vec >= 0.0) and np.all(vec <= 1.0)
    assert np.isclose(vec[0], 0.5)  # 50km / 100km
    assert np.isclose(vec[2], 0.5)  # 7.5km / 15km
    assert np.isclose(vec[4], 0.5)  # 180deg / 360deg


def test_ground_observation_mapping(mapper: TSSObservationMapper) -> None:
    """Ground observation dictionary correctly maps to (9,) normalized float32 array."""
    sample_gnd = {
        "pos_x": 20.0,
        "pos_y": 30.0,
        "speed": 29.15,
        "heading": 90.0,
        "elevation": 2.5,
        "antenna_train_angle": 45.0,
        "distance_to_opponent": 10.0,
        "ammo": 50.0,
        "sensor_range": 25.0,
    }
    vec = mapper.to_ground_observation(sample_gnd)
    assert vec.shape == (9,)
    assert vec.dtype == np.float32
    assert np.all(vec >= 0.0) and np.all(vec <= 1.0)
    assert np.isclose(vec[0], 0.2)
    assert np.isclose(vec[4], 0.5)  # 2.5km / 5km


def test_sea_observation_mapping(mapper: TSSObservationMapper) -> None:
    """Sea observation dictionary correctly maps to (9,) normalized float32 array."""
    sample_sea = {
        "pos_x": 80.0,
        "pos_y": 70.0,
        "speed": 24.3,
        "heading": 270.0,
        "sea_state": 4.5,
        "antenna_train_angle": 0.0,
        "distance_to_opponent": 50.0,
        "ammo": 100.0,
        "radar_range": 50.0,
    }
    vec = mapper.to_sea_observation(sample_sea)
    assert vec.shape == (9,)
    assert vec.dtype == np.float32
    assert np.all(vec >= 0.0) and np.all(vec <= 1.0)
    assert np.isclose(vec[4], 0.5)  # 4.5 / 9.0


def test_commander_observation_mapping(mapper: TSSObservationMapper) -> None:
    """Commander observation extracts own_state and variable other entities."""
    sample_cmd = {
        "own_state": {"pos_x": 50.0, "pos_y": 50.0, "pos_z": 5.0, "speed": 600.0, "heading": 0.0},
        "other_entities": [
            {"pos_x": 20.0, "pos_y": 30.0, "pos_z": 2.0, "speed": 20.0, "heading": 180.0, "domain": "ground"},
            {"pos_x": 80.0, "pos_y": 80.0, "pos_z": 0.0, "speed": 15.0, "heading": 270.0, "domain": "sea"},
        ],
    }
    own_state, others = mapper.to_commander_observation(sample_cmd)
    assert own_state.shape == (5,)
    assert others.shape == (2, 6)
    assert np.isclose(others[0, 5], 1.0)  # Ground domain index
    assert np.isclose(others[1, 5], 2.0)  # Sea domain index


def test_commander_padded_vector(mapper: TSSObservationMapper) -> None:
    """Commander padded vector produces canonical 53-dim float32 vector."""
    sample_cmd = {
        "own_state": {"pos_x": 50.0, "pos_y": 50.0, "pos_z": 5.0, "speed": 600.0, "heading": 0.0},
        "opponents": [
            {"pos_x": 20.0, "pos_y": 30.0, "pos_z": 2.0, "speed": 20.0, "heading": 180.0, "domain": "ground"},
        ],
        "friendlies": [
            {"pos_x": 45.0, "pos_y": 55.0, "pos_z": 5.0, "speed": 550.0, "heading": 10.0, "domain": "air"},
        ],
    }
    vec = mapper.to_commander_padded_vector(sample_cmd)
    assert vec.shape == (53,)
    assert vec.dtype == np.float32
    assert np.all(vec >= 0.0) and np.all(vec <= 1.0)


def test_missing_field_raises_mapping_error(mapper: TSSObservationMapper) -> None:
    """Missing required field must raise actionable TSSMappingError."""
    incomplete_air = {
        "pos_x": 50.0,
        "pos_y": 50.0,
        # missing pos_z and others
    }
    with pytest.raises(TSSMappingError, match="Missing required TSS observation field"):
        mapper.to_air_observation(incomplete_air)


def test_nan_value_raises_mapping_error(mapper: TSSObservationMapper) -> None:
    """Non-finite value (NaN/Inf) must raise TSSMappingError."""
    bad_air = {
        "pos_x": float("nan"),
        "pos_y": 50.0,
        "pos_z": 5.0,
        "speed": 500.0,
        "heading": 180.0,
        "heading_off": 0.0,
        "aspect_angle": 0.0,
        "antenna_train_angle": 0.0,
        "distance_to_opponent": 10.0,
        "cannon_ammo": 200.0,
        "rocket_ammo": 2.0,
        "rocket_ready": 1.0,
        "is_shooting": 0.0,
    }
    with pytest.raises(TSSMappingError, match="contains non-finite value"):
        mapper.to_air_observation(bad_air)


def test_out_of_range_value_warns_and_clamps(mapper: TSSObservationMapper) -> None:
    """Values out of expected range by > 5% trigger a warning and clamp."""
    out_of_range_air = {
        "pos_x": 150.0,  # Range is [0, 100], 150 is > 105 (5% margin)
        "pos_y": 50.0,
        "pos_z": 5.0,
        "speed": 500.0,
        "heading": 180.0,
        "heading_off": 0.0,
        "aspect_angle": 0.0,
        "antenna_train_angle": 0.0,
        "distance_to_opponent": 10.0,
        "cannon_ammo": 200.0,
        "rocket_ammo": 2.0,
        "rocket_ready": 1.0,
        "is_shooting": 0.0,
    }
    with pytest.warns(UserWarning, match="is outside expected range"):
        vec = mapper.to_air_observation(out_of_range_air)
    assert vec[0] == 1.0  # Clamped to maximum 1.0


def test_unit_conversions() -> None:
    """Verify angular, linear, and speed unit conversion helpers."""
    # Angles
    assert math.isclose(TSSObservationMapper.convert_angle(180.0, "degrees", "radians"), math.pi)
    assert math.isclose(TSSObservationMapper.convert_angle(math.pi, "radians", "degrees"), 180.0)

    # Distances
    assert math.isclose(TSSObservationMapper.convert_distance(1.0, "km", "m"), 1000.0)
    assert math.isclose(TSSObservationMapper.convert_distance(1852.0, "m", "nm"), 1.0)
    assert math.isclose(TSSObservationMapper.convert_distance(1.0, "nm", "km"), 1.852)

    # Speed
    # 1 knot = 0.514444 m/s
    assert math.isclose(TSSObservationMapper.convert_speed(100.0, "knots", "m/s"), 51.4444444, rel_tol=1e-4)
    # 100 km/h = 100 / 3.6 m/s
    assert math.isclose(TSSObservationMapper.convert_speed(100.0, "km/h", "m/s"), 100.0 / 3.6)
