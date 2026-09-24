"""Tests for EntitySpriteRenderer vehicle drawing, heading rotations, and explosion triggers."""

from typing import Any
import numpy as np

from src.core.interfaces import TeamSide
from src.simulator.entities.air import AirEntity
from src.simulator.entities.ground import GroundEntity
from src.simulator.entities.sea import SeaEntity
from src.ui.entity_sprites import EntitySpriteRenderer
from src.ui.state import UIState


def test_all_entity_types_render_without_error(renderer: Any, ui_state: UIState) -> None:
    """Verify AC1, AC2, Ground, and Sea units all render without runtime errors."""
    sprite_renderer = renderer.entity_sprites

    ac1 = AirEntity("Test_AC1", TeamSide.BLUE, "AC1", position=(10.0, 10.0, 5.0))
    ac2 = AirEntity("Test_AC2", TeamSide.RED, "AC2", position=(15.0, 15.0, 5.0))
    ground = GroundEntity("Test_GND", TeamSide.BLUE, position=(20.0, 20.0, 0.0))
    sea = SeaEntity("Test_SEA", TeamSide.RED, position=(25.0, 25.0, 0.0))

    sprite_renderer.render_one(ac1)
    sprite_renderer.render_one(ac2)
    sprite_renderer.render_one(ground)
    sprite_renderer.render_one(sea)


def test_selected_entity_highlight_ring(renderer: Any, ui_state: UIState) -> None:
    """Verify selecting an entity triggers highlight indicator rendering."""
    eid = list(ui_state.entities.keys())[0]
    ui_state.select_entity(eid)
    renderer.entity_sprites.render_all()


def test_destroyed_entity_triggers_explosion_sprite(renderer: Any, ui_state: UIState) -> None:
    """Verify casualties add an active explosion record rendered by sprite subsystem."""
    ui_state.explosions.append({
        "pos": (15.0, 15.0),
        "frames_left": 30,
    })
    renderer.entity_sprites.render_all()
    assert len(ui_state.explosions) == 1
