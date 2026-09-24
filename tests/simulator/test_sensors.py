"""Unit tests for stochastic sensor models and detection non-determinism."""

import math
import numpy as np
import pytest

from src.core.interfaces import TeamSide
from src.simulator.config import SENSOR_FALSE_ALARM_RATE
from src.simulator.entities.air import AirEntity
from src.simulator.sensors import Sensor


class TestSensors:
    """Test suite for sensor perception and stochastic properties."""

    def test_pd_decreases_with_distance(self) -> None:
        """Verify probability of detection decreases monotonically with distance."""
        rng = np.random.default_rng(42)
        sensor = Sensor("RADAR", max_range_km=30.0, rng=rng)

        observer = AirEntity("obs", TeamSide.BLUE, "AC1", position=(0.0, 0.0, 5.0), heading=0.0)

        # Targets at increasing distances along observer heading
        target_near = AirEntity("tgt_near", TeamSide.RED, "AC1", position=(5.0, 0.0, 5.0), heading=0.0)
        target_mid = AirEntity("tgt_mid", TeamSide.RED, "AC1", position=(15.0, 0.0, 5.0), heading=0.0)
        target_far = AirEntity("tgt_far", TeamSide.RED, "AC1", position=(28.0, 0.0, 5.0), heading=0.0)
        target_oob = AirEntity("tgt_oob", TeamSide.RED, "AC1", position=(35.0, 0.0, 5.0), heading=0.0)

        pd_near = sensor.compute_pd(target_near, observer)
        pd_mid = sensor.compute_pd(target_mid, observer)
        pd_far = sensor.compute_pd(target_far, observer)
        pd_oob = sensor.compute_pd(target_oob, observer)

        assert pd_near > pd_mid > pd_far > 0.0
        assert pd_oob == 0.0

    def test_sensor_seed_reproducibility(self) -> None:
        """Verify identical seed produces exact same detection result."""
        obs = AirEntity("obs", TeamSide.BLUE, "AC1", position=(0.0, 0.0, 5.0), heading=0.0)
        tgt = AirEntity("tgt", TeamSide.RED, "AC1", position=(15.0, 0.0, 5.0), heading=0.0)

        s1 = Sensor("RADAR", max_range_km=30.0, rng=np.random.default_rng(12345))
        s2 = Sensor("RADAR", max_range_km=30.0, rng=np.random.default_rng(12345))

        res1 = [s1.detect(tgt, obs, current_time=i * 0.1) is not None for i in range(20)]
        res2 = [s2.detect(tgt, obs, current_time=i * 0.1) is not None for i in range(20)]

        assert res1 == res2

    def test_sensor_seed_nondeterminism(self) -> None:
        """Verify different seeds produce differing detection sequences (NON-DETERMINISM)."""
        obs = AirEntity("obs", TeamSide.BLUE, "AC1", position=(0.0, 0.0, 5.0), heading=0.0)
        tgt = AirEntity("tgt", TeamSide.RED, "AC1", position=(15.0, 0.0, 5.0), heading=0.0)

        s1 = Sensor("RADAR", max_range_km=30.0, rng=np.random.default_rng(111))
        s2 = Sensor("RADAR", max_range_km=30.0, rng=np.random.default_rng(999))

        res1 = [s1.detect(tgt, obs, current_time=i * 0.1) is not None for i in range(50)]
        res2 = [s2.detect(tgt, obs, current_time=i * 0.1) is not None for i in range(50)]

        # Must differ in stochastic rolls
        assert res1 != res2

    def test_false_alarm_rate(self) -> None:
        """Verify false alarm generation rate approximately matches SENSOR_FALSE_ALARM_RATE over 10,000 trials."""
        rng = np.random.default_rng(42)
        sensor = Sensor("RADAR", max_range_km=30.0, rng=rng, false_alarm_rate=0.01)
        obs = AirEntity("obs", TeamSide.BLUE, "AC1", position=(10.0, 10.0, 5.0), heading=0.0)

        trials = 10_000
        ghost_count = sum(1 for i in range(trials) if sensor.sample_false_alarm(obs, current_time=i) is not None)
        empirical_rate = ghost_count / trials

        # Should be within +/- 30% of nominal rate (0.007 to 0.013)
        assert 0.007 <= empirical_rate <= 0.013

    def test_contact_position_noise(self) -> None:
        """Verify reported contact coordinates exhibit non-zero variance around true position."""
        rng = np.random.default_rng(100)
        sensor = Sensor("RADAR", max_range_km=50.0, rng=rng)
        obs = AirEntity("obs", TeamSide.BLUE, "AC1", position=(0.0, 0.0, 5.0), heading=0.0)
        tgt = AirEntity("tgt", TeamSide.RED, "AC1", position=(5.0, 0.0, 5.0), heading=0.0)

        positions = []
        for t in range(200):
            contact = sensor.detect(tgt, obs, current_time=t * 0.1)
            if contact is not None:
                positions.append(contact.estimated_position)

        assert len(positions) > 50
        pos_array = np.array(positions)
        # Standard deviation along X and Y must be positive
        std_x = float(np.std(pos_array[:, 0]))
        std_y = float(np.std(pos_array[:, 1]))
        assert std_x > 0.1
        assert std_y > 0.05
