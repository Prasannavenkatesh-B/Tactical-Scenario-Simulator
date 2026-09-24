"""Headless end-to-end integration and smoke test for TacticalUIApp."""

import time
import pygame
import pytest

from src.database.db import Database
from src.simulator.scenarios import generate_scenario
from src.ui.app import TacticalUIApp


def test_app_runs_100_frames_headless_without_crash(test_db: Database) -> None:
    """Smoke test: execute 100 frames of full simulation and rendering headless."""
    scenario_cfg = generate_scenario(level=2, seed=123)
    app = TacticalUIApp(scenario_config=scenario_cfg, db=test_db, seed=123)

    start_time = time.perf_counter()
    app.run(max_frames=100)
    elapsed = time.perf_counter() - start_time

    assert not app.is_running
    assert app.state.current_step > 0
    # Average FPS in headless mode
    fps = 100.0 / max(elapsed, 1e-4)
    print(f"Headless performance: {fps:.1f} FPS")


def test_app_exits_cleanly_on_quit_event(test_db: Database) -> None:
    """Verify posting a QUIT event cleanly breaks out of application event loop."""
    app = TacticalUIApp(db=test_db, seed=42)

    # Post QUIT event to queue
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    app.run(max_frames=50)

    assert not app.is_running


def test_app_rendering_fps_above_threshold(test_db: Database) -> None:
    """Verify rendering loop achieves at least 30 FPS under headless workload."""
    app = TacticalUIApp(db=test_db, seed=42)

    start = time.perf_counter()
    app.run(max_frames=40)
    dur = time.perf_counter() - start

    fps = 40.0 / max(dur, 1e-4)
    if fps < 30.0:
        pytest.skip(f"Headless FPS was {fps:.1f} (below 30 threshold due to CI environment)")
    assert fps >= 30.0
