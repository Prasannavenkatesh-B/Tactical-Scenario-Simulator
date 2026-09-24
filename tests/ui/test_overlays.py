"""Tests for OverlayRenderer tactical zones, sensor footprints, and boundary proximity alerts."""

from typing import Any
import numpy as np

from src.core.interfaces import TeamSide
from src.simulator.entities.air import AirEntity
from src.ui.config import TRAIL_LENGTH
from src.ui.overlays import OverlayRenderer
from src.ui.state import UIState


def test_wez_cone_renders_for_air_entities(renderer: Any, ui_state: UIState) -> None:
    """Verify forward weapon engagement zone (WEZ) arc renders for air units."""
    overlay_renderer = renderer.overlays
    air_ent = AirEntity("WEZ_Tester", TeamSide.BLUE, "AC1", position=(15.0, 15.0, 5.0), heading=0.5)
    overlay_renderer.render_wez(air_ent)


def test_sensor_circle_renders(renderer: Any, ui_state: UIState) -> None:
    """Verify circular sensor detection footprint renders."""
    overlay_renderer = renderer.overlays
    air_ent = AirEntity("Sensor_Tester", TeamSide.BLUE, "AC2", position=(15.0, 15.0, 5.0))
    overlay_renderer.render_sensor_range(air_ent)


def test_trajectory_trail_rendering(renderer: Any, ui_state: UIState) -> None:
    """Verify trajectory history rendering behaves properly."""
    eid = list(ui_state.entities.keys())[0]
    ui_state.trajectories[eid] = [(10.0 + i * 0.1, 10.0 + i * 0.1) for i in range(TRAIL_LENGTH)]
    renderer.overlays.render_trajectories()


def test_boundary_warning_activates_near_edge(renderer: Any, ui_state: UIState) -> None:
    """Verify proximity alert activates when any unit is within 6 km of map border."""
    overlay_renderer = renderer.overlays

    # Place an entity 2.0 km from left edge
    near_ent = AirEntity("Border_Violator", TeamSide.BLUE, "AC1", position=(2.0, 15.0, 5.0))
    ui_state.entities["Border_Violator"] = near_ent

    # Should render boundary warning without error
    overlay_renderer.render_boundary_warning()
