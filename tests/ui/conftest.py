"""Pytest fixtures for headless Pygame user interface testing."""

import os
from typing import Generator
import numpy as np
import pytest

# Force headless dummy video driver before importing pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import pygame

from src.database.db import Database
from src.database.migrations import SchemaManager
from src.simulator.env import TacticalEnv
from src.simulator.scenarios import generate_scenario
from src.ui.config import WINDOW_HEIGHT, WINDOW_WIDTH
from src.ui.renderer import Renderer
from src.ui.state import UIState


@pytest.fixture(scope="session", autouse=True)
def init_pygame() -> Generator[None, None, None]:
    """Initialize Pygame once in headless dummy driver mode for test session."""
    pygame.init()
    yield
    pygame.quit()


@pytest.fixture
def headless_screen() -> pygame.Surface:
    """Provide a headless Pygame display surface for rendering tests."""
    return pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))


@pytest.fixture
def test_db() -> Generator[Database, None, None]:
    """Provide an isolated, migrated in-memory Database instance."""
    db = Database(":memory:")
    SchemaManager(db).apply_migrations()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def ui_state(test_db: Database) -> UIState:
    """Provide a fully initialized UIState backed by a fresh Level 3 TacticalEnv."""
    rng = np.random.default_rng(42)
    scenario_cfg = generate_scenario(level=3, rng=rng, seed=42)
    env = TacticalEnv(scenario_config=scenario_cfg, mode="interactive", seed=42)
    return UIState(
        env=env,
        scenario_config=scenario_cfg,
        rng=rng,
        db=test_db,
    )


@pytest.fixture
def renderer(headless_screen: pygame.Surface, ui_state: UIState) -> Renderer:
    """Provide a configured Renderer instance for composite visual verification."""
    return Renderer(headless_screen, ui_state)
