"""Unit tests for observation dataclasses across air, ground, sea, and commander domains."""

import numpy as np
import pytest

from src.core.observations import (
    AirObservation,
    CommanderObservation,
    FullObservation,
    GroundObservation,
    NORMALIZATION_CONFIG,
    SeaObservation,
)


class TestAirObservation:
    """Test suite for AirObservation dataclass."""

    def test_dimension(self) -> None:
        """Verify AirObservation produces an array of dimension 13."""
        obs = AirObservation(
            x=0.1, y=0.2, z=0.3, v=0.4,
            alpha_h=0.5, alpha_off=0.6, alpha_AA=0.7, alpha_ATA=0.8,
            d_o=0.9, c1=1.0, c2=0.5, w=1.0, s_r=0.0,
        )
        arr = obs.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (13,)
        assert arr.dtype == np.float32

    def test_zero_values(self) -> None:
        """Verify all zero values are preserved as 0.0."""
        obs = AirObservation(
            x=0.0, y=0.0, z=0.0, v=0.0,
            alpha_h=0.0, alpha_off=0.0, alpha_AA=0.0, alpha_ATA=0.0,
            d_o=0.0, c1=0.0, c2=0.0, w=0.0, s_r=0.0,
        )
        arr = obs.to_array()
        assert np.all(arr == 0.0)

    def test_max_values(self) -> None:
        """Verify all maximum values (1.0) are preserved as 1.0."""
        obs = AirObservation(
            x=1.0, y=1.0, z=1.0, v=1.0,
            alpha_h=1.0, alpha_off=1.0, alpha_AA=1.0, alpha_ATA=1.0,
            d_o=1.0, c1=1.0, c2=1.0, w=1.0, s_r=1.0,
        )
        arr = obs.to_array()
        assert np.all(arr == 1.0)

    def test_out_of_range_clamped(self) -> None:
        """Verify out-of-range negative and positive values are clamped to [0, 1]."""
        obs = AirObservation(
            x=-10.0, y=5.5, z=-0.01, v=1.05,
            alpha_h=-3.14, alpha_off=2.0, alpha_AA=-1.0, alpha_ATA=99.0,
            d_o=-0.5, c1=1.5, c2=-100.0, w=2.0, s_r=-1.0,
        )
        assert obs.x == 0.0
        assert obs.y == 1.0
        assert obs.z == 0.0
        assert obs.v == 1.0
        assert obs.alpha_h == 0.0
        assert obs.alpha_off == 1.0
        assert obs.alpha_AA == 0.0
        assert obs.alpha_ATA == 1.0
        assert obs.d_o == 0.0
        assert obs.c1 == 1.0
        assert obs.c2 == 0.0
        assert obs.w == 1.0
        assert obs.s_r == 0.0

    def test_from_array_round_trip(self) -> None:
        """Verify conversion to and from numpy array preserves normalized values."""
        raw_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.5, 1.0, 0.0]
        obs = AirObservation.from_array(raw_vals)
        arr = obs.to_array()
        np.testing.assert_allclose(arr, raw_vals, atol=1e-5)

    def test_from_raw(self) -> None:
        """Verify unnormalized physical inputs are correctly normalized and clamped."""
        obs = AirObservation.from_raw(
            x=50_000.0,           # 50 km / 100 km = 0.5
            y=25_000.0,           # 25 km / 100 km = 0.25
            z=7_500.0,            # 7.5 km / 15 km = 0.5
            v=300.0,              # 300 m/s / 600 m/s = 0.5
            heading_rad=np.pi,    # pi / 2pi = 0.5
            heading_off_rad=np.pi / 2,  # (pi/2) / pi = 0.5
            aspect_angle_rad=np.pi,     # pi / pi = 1.0
            antenna_train_angle_rad=0.0,# 0 / pi = 0.0
            distance_to_opponent=50_000.0, # 50 km / 100 km = 0.5
            cannon_ammo=250.0,    # 250 / 500 = 0.5
            rocket_ammo=4.0,      # 4 / 8 = 0.5
            rocket_ready=True,    # 1.0
            is_shooting=False,    # 0.0
        )
        assert pytest.approx(obs.x, 1e-4) == 0.5
        assert pytest.approx(obs.y, 1e-4) == 0.25
        assert pytest.approx(obs.z, 1e-4) == 0.5
        assert pytest.approx(obs.v, 1e-4) == 0.5
        assert pytest.approx(obs.alpha_h, 1e-4) == 0.5
        assert pytest.approx(obs.alpha_off, 1e-4) == 0.5
        assert pytest.approx(obs.alpha_AA, 1e-4) == 1.0
        assert pytest.approx(obs.alpha_ATA, 1e-4) == 0.0
        assert pytest.approx(obs.d_o, 1e-4) == 0.5
        assert pytest.approx(obs.c1, 1e-4) == 0.5
        assert pytest.approx(obs.c2, 1e-4) == 0.5
        assert obs.w == 1.0
        assert obs.s_r == 0.0


class TestGroundObservation:
    """Test suite for GroundObservation dataclass."""

    def test_dimension(self) -> None:
        """Verify GroundObservation produces an array of dimension 9."""
        obs = GroundObservation(
            x=0.2, y=0.3, v=0.5, theta=0.4, elev=0.1,
            alpha_ATA=0.6, d_o=0.7, c1=0.8, sensor_range=0.9,
        )
        arr = obs.to_array()
        assert arr.shape == (9,)
        assert arr.dtype == np.float32

    def test_zero_and_max_values(self) -> None:
        """Verify zero and max bounds are respected."""
        obs_zero = GroundObservation(
            x=0.0, y=0.0, v=0.0, theta=0.0, elev=0.0,
            alpha_ATA=0.0, d_o=0.0, c1=0.0, sensor_range=0.0,
        )
        assert np.all(obs_zero.to_array() == 0.0)

        obs_max = GroundObservation(
            x=1.0, y=1.0, v=1.0, theta=1.0, elev=1.0,
            alpha_ATA=1.0, d_o=1.0, c1=1.0, sensor_range=1.0,
        )
        assert np.all(obs_max.to_array() == 1.0)

    def test_out_of_range_clamped(self) -> None:
        """Verify values outside [0, 1] are clamped."""
        obs = GroundObservation(
            x=-5.0, y=2.0, v=-1.0, theta=10.0, elev=-0.2,
            alpha_ATA=3.0, d_o=-0.01, c1=100.0, sensor_range=-1.0,
        )
        arr = obs.to_array()
        assert arr[0] == 0.0  # x clamped to 0
        assert arr[1] == 1.0  # y clamped to 1
        assert arr[2] == 0.0  # v clamped to 0
        assert arr[3] == 1.0  # theta clamped to 1
        assert arr[4] == 0.0  # elev clamped to 0
        assert arr[5] == 1.0  # alpha_ATA clamped to 1
        assert arr[6] == 0.0  # d_o clamped to 0
        assert arr[7] == 1.0  # c1 clamped to 1
        assert arr[8] == 0.0  # sensor_range clamped to 0

    def test_from_raw(self) -> None:
        """Verify physical scaling for ground entity."""
        obs = GroundObservation.from_raw(
            x=50_000.0,
            y=100_000.0,
            v=15.0,              # 15 / 30 = 0.5
            heading_rad=np.pi,   # pi / 2pi = 0.5
            elevation=2500.0,    # 2500 / 5000 = 0.5
            antenna_train_angle_rad=np.pi / 2, # (pi/2) / pi = 0.5
            distance_to_opponent=20_000.0,     # 20 / 100 = 0.2
            ammo=50.0,           # 50 / 100 = 0.5
            sensor_range=25_000.0, # 25 / 50 = 0.5
        )
        assert pytest.approx(obs.x, 1e-4) == 0.5
        assert pytest.approx(obs.y, 1e-4) == 1.0
        assert pytest.approx(obs.v, 1e-4) == 0.5
        assert pytest.approx(obs.elev, 1e-4) == 0.5
        assert pytest.approx(obs.c1, 1e-4) == 0.5


class TestSeaObservation:
    """Test suite for SeaObservation dataclass."""

    def test_dimension(self) -> None:
        """Verify SeaObservation produces an array of dimension 9."""
        obs = SeaObservation(
            x=0.1, y=0.2, v=0.3, theta=0.4, sea_state=0.5,
            alpha_ATA=0.6, d_o=0.7, c1=0.8, radar_range=0.9,
        )
        arr = obs.to_array()
        assert arr.shape == (9,)
        assert arr.dtype == np.float32

    def test_clamping_and_beaufort(self) -> None:
        """Verify clamping and sea state normalization (Beaufort 0-9)."""
        obs_raw = SeaObservation.from_raw(
            x=10_000.0,
            y=20_000.0,
            v=12.5,
            heading_rad=0.0,
            sea_state_beaufort=4.5, # 4.5 / 9 = 0.5
            antenna_train_angle_rad=0.0,
            distance_to_opponent=10_000.0,
            ammo=100.0,
            radar_range=50_000.0,
        )
        assert pytest.approx(obs_raw.sea_state, 1e-4) == 0.5

        # Out-of-bounds Beaufort scale clamped
        obs_clamped = SeaObservation(
            x=0.5, y=0.5, v=0.5, theta=0.5, sea_state=12.0,
            alpha_ATA=0.5, d_o=0.5, c1=0.5, radar_range=-1.0,
        )
        assert obs_clamped.sea_state == 1.0
        assert obs_clamped.radar_range == 0.0


class TestCommanderObservation:
    """Test suite for CommanderObservation dataclass."""

    def test_variable_dimension_handling(self) -> None:
        """Verify variable opponent/friendly lists are concatenated properly."""
        own_state = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        opp_states = [
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.0], dtype=np.float32),
            np.array([0.6, 0.7, 0.8, 0.9, 1.0, 0.5], dtype=np.float32),
        ]
        friendly_states = [
            np.array([0.2, 0.3, 0.4, 0.5, 0.6, 1.0], dtype=np.float32),
        ]
        domain_ids = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        cmd_obs = CommanderObservation(
            own_state=own_state,
            opponent_states=opp_states,
            friendly_states=friendly_states,
            domain_ids=domain_ids,
        )

        flat_arr = cmd_obs.to_array()
        # 5 (own) + 2*6 (opp) + 1*6 (fr) + 9 (domain_ids) = 32
        expected_len = 5 + (2 * 6) + (1 * 6) + 9
        assert flat_arr.shape == (expected_len,)
        assert np.all((flat_arr >= 0.0) & (flat_arr <= 1.0))

    def test_to_padded_array(self) -> None:
        """Verify fixed-length zero-padding format."""
        own_state = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        cmd_obs = CommanderObservation(own_state=own_state)

        padded = cmd_obs.to_padded_array(state_dim=5)
        # own (5) + 3 opp (3 * 6 = 18) + 2 fr (2 * 6 = 12) + 6 entities * 3 domains (18) = 53
        assert padded.shape == (53,)
        np.testing.assert_allclose(padded[:5], [0.5, 0.5, 0.5, 0.5, 0.5])
        assert np.all(padded[5:] == 0.0)


class TestFullObservation:
    """Test suite for FullObservation container."""

    def test_full_observation_container(self) -> None:
        """Verify aggregation of observations across diverse agents."""
        air_obs = AirObservation(
            x=0.1, y=0.2, z=0.3, v=0.4, alpha_h=0.5, alpha_off=0.6,
            alpha_AA=0.7, alpha_ATA=0.8, d_o=0.9, c1=1.0, c2=0.5, w=1.0, s_r=0.0
        )
        ground_obs = GroundObservation(
            x=0.1, y=0.2, v=0.3, theta=0.4, elev=0.5,
            alpha_ATA=0.6, d_o=0.7, c1=0.8, sensor_range=0.9
        )
        full = FullObservation(
            air_observations={"blue_air_1": air_obs},
            ground_observations={"blue_ground_1": ground_obs},
        )
        obs_dict = full.to_dict()
        assert "blue_air_1" in obs_dict
        assert "blue_ground_1" in obs_dict
        assert obs_dict["blue_air_1"].shape == (13,)
        assert obs_dict["blue_ground_1"].shape == (9,)

        arr = full.to_array()
        assert arr.shape == (13 + 9,)
