"""Shared test fixtures for doctrine detector and realism validation tests."""

import math
from typing import Any
import numpy as np
import pytest

from src.evaluation.doctrine.base_detector import EpisodeData
from src.simulator.scenarios import generate_scenario


@pytest.fixture
def empty_episode() -> EpisodeData:
    """Fixture producing an empty episode dataset to verify robust error-free handling."""
    cfg = generate_scenario(level=1, rng=np.random.default_rng(42))
    return EpisodeData(
        entity_trajectories={},
        engagements=[],
        observations={},
        commander_decisions=[],
        scenario_config=cfg,
        episode_length=0,
        outcome="draw",
    )


@pytest.fixture
def synthetic_air_episode() -> EpisodeData:
    """Episode data specifically exhibiting authentic Air-to-Air combat maneuvers."""
    cfg = generate_scenario(level=1, rng=np.random.default_rng(42))

    # Attacker (blue_air_1) tail-chasing defender (red_air_1) with lead heading and speed-altitude exchange
    b_traj = []
    r_traj = []

    # Defender flying eastward at constant heading 0.0 rad, constant speed 600 km/h, altitude 5.0 km
    # Attacker trailing by 3 km, matching heading, diving from 6.0 to 4.5 km while accelerating from 550 to 700 km/h
    for t in range(20):
        r_x = 20.0 + t * 0.2
        r_y = 50.0
        r_z = 5.0
        r_heading = 0.0 if t < 15 else 1.2  # Breaks hard at step 15
        r_traj.append({
            "t": t,
            "position": np.array([r_x, r_y, r_z]),
            "heading": r_heading,
            "speed": 600.0,
            "status": "ALIVE",
        })

        b_x = 17.0 + t * 0.22  # Closing
        b_y = 50.0
        b_z = 6.0 - t * 0.08   # Diving
        b_spd = 550.0 + t * 8.0  # Speed trade
        b_traj.append({
            "t": t,
            "position": np.array([b_x, b_y, b_z]),
            "heading": 0.0,  # Exact tail chase
            "speed": b_spd,
            "status": "ALIVE",
        })

    # Second friendly flanking from the east/opposing side for pincer maneuver
    b2_traj = []
    for t in range(20):
        b2_x = 26.0 - t * 0.25  # Converging from east towards target
        b2_y = 50.0
        b2_traj.append({
            "t": t,
            "position": np.array([b2_x, b2_y, 5.0]),
            "heading": math.pi,
            "speed": 600.0,
            "status": "ALIVE",
        })

    engagements = [
        {
            "t": 18,
            "shooter_id": "blue_air_1",
            "target_id": "red_air_1",
            "weapon": "cannon",
            "hit": True,
            "distance_km": 1.5,
        }
    ]

    commander_decisions = [
        {"t": t, "agent_id": "blue_air_1", "action": 1, "target_index": 1}
        for t in range(0, 20, 5)
    ]

    return EpisodeData(
        entity_trajectories={
            "blue_air_1": b_traj,
            "blue_air_2": b2_traj,
            "red_air_1": r_traj,
        },
        engagements=engagements,
        observations={},
        commander_decisions=commander_decisions,
        scenario_config=cfg,
        episode_length=20,
        outcome="blue_win",
    )


@pytest.fixture
def synthetic_ground_episode() -> EpisodeData:
    """Episode data exhibiting ground combat tactics (terrain cover, mutual support, discipline)."""
    cfg = generate_scenario(level=3, rng=np.random.default_rng(42))

    g1_traj = []
    g2_traj = []
    r_traj = []

    for t in range(20):
        # Two friendly ground units advancing in tight mutual support (8 km separation)
        g1_pos = np.array([30.0 + t * 0.1, 40.0, 0.6])  # High elevated ground 0.6
        g2_pos = np.array([30.0 + t * 0.1, 46.0, 0.55]) # 6 km offset
        r_pos = np.array([45.0, 42.0, 0.2])            # Hostile position

        g1_traj.append({"t": t, "position": g1_pos, "heading": 0.0, "speed": 30.0, "status": "ALIVE"})
        g2_traj.append({"t": t, "position": g2_pos, "heading": 0.0, "speed": 30.0, "status": "ALIVE"})
        r_traj.append({"t": t, "position": r_pos, "heading": math.pi, "speed": 0.0, "status": "ALIVE"})

    engagements = [
        {
            "t": 15,
            "shooter_id": "blue_ground_1",
            "target_id": "red_ground_1",
            "weapon": "cannon",
            "hit": True,
            "distance_km": 10.0,  # Disciplined range (< 12 km)
        }
    ]

    return EpisodeData(
        entity_trajectories={
            "blue_ground_1": g1_traj,
            "blue_ground_2": g2_traj,
            "red_ground_1": r_traj,
        },
        engagements=engagements,
        observations={},
        commander_decisions=[],
        scenario_config=cfg,
        episode_length=20,
        outcome="blue_win",
    )


@pytest.fixture
def synthetic_sea_episode() -> EpisodeData:
    """Episode data exhibiting naval combat tactics (standoff, screen formation, evasive turns)."""
    cfg = generate_scenario(level=4, rng=np.random.default_rng(42))

    s1_traj = []
    s2_traj = []
    r_traj = []

    for t in range(20):
        # Two surface vessels maintaining parallel line-of-bearing formation
        s1_heading = 0.0 if t < 12 else 0.8  # Evasive turn at step 12
        s2_heading = 0.0 if t < 12 else 0.8

        s1_pos = np.array([20.0 + t * 0.05, 30.0, 0.0])
        s2_pos = np.array([20.0 + t * 0.05, 40.0, 0.0])
        r_pos = np.array([42.0, 35.0, 0.0])  # Standoff distance ~22 km (> 18 km)

        s1_traj.append({"t": t, "position": s1_pos, "heading": s1_heading, "speed": 25.0, "status": "ALIVE"})
        s2_traj.append({"t": t, "position": s2_pos, "heading": s2_heading, "speed": 25.0, "status": "ALIVE"})
        r_traj.append({"t": t, "position": r_pos, "heading": math.pi, "speed": 20.0, "status": "ALIVE"})

    engagements = [
        {
            "t": 10,
            "shooter_id": "blue_sea_1",
            "target_id": "red_sea_1",
            "weapon": "missile",
            "hit": True,
            "distance_km": 21.0,  # 70% of 30km range (standoff)
        }
    ]

    return EpisodeData(
        entity_trajectories={
            "blue_sea_1": s1_traj,
            "blue_sea_2": s2_traj,
            "red_sea_1": r_traj,
        },
        engagements=engagements,
        observations={},
        commander_decisions=[],
        scenario_config=cfg,
        episode_length=20,
        outcome="blue_win",
    )


@pytest.fixture
def synthetic_joint_episode() -> EpisodeData:
    """Episode data exhibiting cross-domain joint coordination (CAS, SEAD, maritime patrol)."""
    cfg = generate_scenario(level=5, rng=np.random.default_rng(42))
    cfg.map_size_km = 100.0

    air_traj = []
    gnd_traj = []
    sea_traj = []
    r_gnd_traj = []
    r_sam_traj = []

    # Map is 100x100. Center line is Y=50 or X=50
    for t in range(20):
        # Air flying forward sweep (X > 50) and CAS proximity (< 4 km to red ground)
        air_pos = np.array([48.0 + t * 0.5, 41.0, 4.0])
        gnd_pos = np.array([40.0 + t * 0.2, 40.0, 0.4])
        sea_pos = np.array([20.0, 52.0, 0.0])  # Near contested boundary Y=50 (< 10 km)
        r_gnd = np.array([50.0, 40.0, 0.2])
        r_sam = np.array([60.0, 40.0, 0.2])

        air_traj.append({"t": t, "position": air_pos, "heading": 0.0, "speed": 600.0, "status": "ALIVE"})
        gnd_traj.append({"t": t, "position": gnd_pos, "heading": 0.0, "speed": 30.0, "status": "ALIVE"})
        sea_traj.append({"t": t, "position": sea_pos, "heading": math.pi / 2.0, "speed": 20.0, "status": "ALIVE"})
        r_gnd_traj.append({"t": t, "position": r_gnd, "heading": math.pi, "speed": 0.0, "status": "ALIVE"})
        r_sam_traj.append({"t": t, "position": r_sam, "heading": math.pi, "speed": 0.0, "status": "ALIVE"})

    engagements = [
        # SEAD strike: Air destroys enemy SAM early (t=5)
        {"t": 5, "shooter_id": "blue_air_1", "target_id": "red_sam_1", "weapon": "missile", "hit": True, "distance_km": 12.0},
        # Ground engages enemy infantry (t=12)
        {"t": 12, "shooter_id": "blue_ground_1", "target_id": "red_ground_1", "weapon": "cannon", "hit": True, "distance_km": 7.0},
    ]

    return EpisodeData(
        entity_trajectories={
            "blue_air_1": air_traj,
            "blue_ground_1": gnd_traj,
            "blue_sea_1": sea_traj,
            "red_ground_1": r_gnd_traj,
            "red_sam_1": r_sam_traj,
        },
        engagements=engagements,
        observations={},
        commander_decisions=[],
        scenario_config=cfg,
        episode_length=20,
        outcome="blue_win",
    )
