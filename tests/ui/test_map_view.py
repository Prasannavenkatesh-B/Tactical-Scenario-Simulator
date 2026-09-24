"""Tests for MapView coordinate transformations, border selection, and viewport interactions."""

import math
from typing import Any
import pygame

from src.ui.map_view import MapView
from src.ui.state import UIState


def test_coordinate_transforms_are_inverses(renderer: Any, ui_state: UIState) -> None:
    """Verify world_to_screen and screen_to_world are reciprocal coordinate projections."""
    map_view = renderer.map_view
    map_size = ui_state.env.map.size_km

    test_points = [
        (0.0, 0.0),
        (map_size / 2.0, map_size / 2.0),
        (map_size, map_size),
        (12.3, 18.7),
    ]

    for wx, wy in test_points:
        sx, sy = map_view.world_to_screen(wx, wy)
        re_wx, re_wy = map_view.screen_to_world(sx, sy)
        # Tolerance within one pixel's world dimension
        pixel_world_res = map_size / map_view.rect.width
        assert abs(wx - re_wx) <= pixel_world_res + 1e-4
        assert abs(wy - re_wy) <= pixel_world_res + 1e-4


def test_screen_to_world_clamps_out_of_bounds(renderer: Any, ui_state: UIState) -> None:
    """Verify clicking outside active map boundaries clamps safely within [0, map_size]."""
    map_view = renderer.map_view
    map_size = ui_state.env.map.size_km

    # Far left/top
    wx, wy = map_view.screen_to_world(-500, -500)
    assert 0.0 <= wx <= map_size
    assert 0.0 <= wy <= map_size

    # Far right/bottom
    wx2, wy2 = map_view.screen_to_world(5000, 5000)
    assert 0.0 <= wx2 <= map_size
    assert 0.0 <= wy2 <= map_size


def test_border_selection_overlay_renders_without_error(renderer: Any) -> None:
    """Verify border drag selection overlay draws smoothly."""
    map_view = renderer.map_view
    map_view.render_border_selection_overlay((100, 100), (300, 300))


def test_map_view_click_handling(renderer: Any, ui_state: UIState) -> None:
    """Verify clicking inside map viewport is handled."""
    map_view = renderer.map_view
    # Outside map rect
    consumed_out = map_view.handle_click(1000, 50, 1)
    assert not consumed_out

    # Inside map rect
    consumed_in = map_view.handle_click(400, 400, 1)
    assert consumed_in
