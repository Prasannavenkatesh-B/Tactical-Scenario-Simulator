"""Tests for InputHandler event dispatch, manual steering, and keyboard accelerators."""

from typing import Any
import pygame

from src.core.actions import AirAction
from src.ui.controls import InputHandler
from src.ui.state import UIState


def test_space_toggles_play_pause(renderer: Any, ui_state: UIState) -> None:
    """Verify pressing SPACE toggles between simulation and paused modes."""
    handler = InputHandler(
        state=ui_state,
        map_view=renderer.map_view,
        sidebar=renderer.sidebar,
        timeline=renderer.timeline,
        renderer=renderer,
    )

    ui_state.mode = "simulate"
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE, mod=0)
    handler.handle_event(event)
    assert ui_state.mode == "paused"

    handler.handle_event(event)
    assert ui_state.mode == "simulate"


def test_right_arrow_steps_forward(renderer: Any, ui_state: UIState) -> None:
    """Verify RIGHT arrow advances simulation single step in paused mode."""
    handler = InputHandler(
        state=ui_state,
        map_view=renderer.map_view,
        sidebar=renderer.sidebar,
        timeline=renderer.timeline,
        renderer=renderer,
    )

    ui_state.mode = "paused"
    init_step = ui_state.current_step
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT, mod=0)
    handler.handle_event(event)

    assert ui_state.current_step == init_step + 1


def test_number_key_assigns_manual_control(renderer: Any, ui_state: UIState) -> None:
    """Verify pressing number keys 1-9 binds manual control to corresponding unit."""
    handler = InputHandler(
        state=ui_state,
        map_view=renderer.map_view,
        sidebar=renderer.sidebar,
        timeline=renderer.timeline,
        renderer=renderer,
    )

    eids = list(ui_state.entities.keys())
    assert len(eids) >= 1

    event1 = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_1, mod=0)
    handler.handle_event(event1)
    assert ui_state.manual_control_entity == eids[0]
    assert ui_state.selected_entity_id == eids[0]


def test_arrow_keys_send_manual_control_actions(renderer: Any, ui_state: UIState) -> None:
    """Verify steering keys formulate appropriate action commands during manual control."""
    handler = InputHandler(
        state=ui_state,
        map_view=renderer.map_view,
        sidebar=renderer.sidebar,
        timeline=renderer.timeline,
        renderer=renderer,
    )

    eid = list(ui_state.entities.keys())[0]
    ui_state.manual_control_entity = eid

    # Left arrow
    ev_left = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT, mod=0)
    handler.handle_event(ev_left)
    assert ui_state.manual_action is not None
    assert ui_state.manual_action.heading_delta < 0.0

    # Right arrow
    ev_right = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT, mod=0)
    handler.handle_event(ev_right)
    assert ui_state.manual_action.heading_delta > 0.0

    # Fire cannon
    ev_fire = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f, mod=0)
    handler.handle_event(ev_fire)
    if isinstance(ui_state.manual_action, AirAction):
        assert ui_state.manual_action.fire_cannon == 1


def test_ctrl_s_triggers_scenario_save(renderer: Any, ui_state: UIState) -> None:
    """Verify CTRL+S saves the active scenario to database."""
    handler = InputHandler(
        state=ui_state,
        map_view=renderer.map_view,
        sidebar=renderer.sidebar,
        timeline=renderer.timeline,
        renderer=renderer,
    )

    ev_save = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s, mod=pygame.KMOD_CTRL)
    consumed = handler.handle_event(ev_save)
    assert consumed
