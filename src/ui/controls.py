"""Input dispatch: keyboard and mouse event routing to simulation state, map, sidebar, and timeline."""

from typing import Any
import pygame

from src.core.actions import AirAction, GroundAction, SeaAction
from src.core.interfaces import DomainType
from src.ui.map_view import MapView
from src.ui.sidebar import Sidebar
from src.ui.state import UIState
from src.ui.timeline import Timeline


class InputHandler:
    """Routes keyboard shortcuts, mouse clicks, and manual steering commands."""

    def __init__(
        self,
        state: UIState,
        map_view: MapView,
        sidebar: Sidebar,
        timeline: Timeline,
        renderer: Any = None,
    ) -> None:
        self.state = state
        self.map_view = map_view
        self.sidebar = sidebar
        self.timeline = timeline
        self.renderer = renderer

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Route Pygame event. Returns True if event was consumed."""
        # 1. Mouse Button Down
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            # Sidebar check
            if self.sidebar.handle_click(mx, my):
                return True
            # Timeline check
            if self.timeline.handle_click(mx, my):
                return True
            # Map View check
            if self.map_view.handle_click(mx, my, event.button):
                return True
            return False

        # 2. Keyboard Key Down
        elif event.type == pygame.KEYDOWN:
            mods = pygame.key.get_mods()
            ctrl_pressed = bool(mods & pygame.KMOD_CTRL or mods & pygame.KMOD_META)

            # --- CTRL SHORTCUTS ---
            if ctrl_pressed:
                if event.key == pygame.K_s:
                    self.state.save_scenario("User_Tactical_Scenario")
                    return True
                elif event.key == pygame.K_o:
                    scenarios = self.state.scenario_io.list_scenarios()
                    if scenarios:
                        self.state.load_scenario(scenarios[-1]["scenario_id"])
                    return True
                elif event.key == pygame.K_n:
                    self.state.mode = "edit"
                    self.state.scenario_config.blue_entities.clear()
                    self.state.scenario_config.red_entities.clear()
                    self.state.reset_simulation()
                    return True

            # --- MANUAL VEHICLE OVERRIDE (STEERING) ---
            if self.state.manual_control_entity is not None and self.state.manual_control_entity in self.state.entities:
                consumed = self._handle_manual_control_keys(event.key)
                if consumed:
                    return True

            # --- NUMERIC KEYS 1-9: SELECT/ASSIGN MANUAL CONTROL ---
            if pygame.K_1 <= event.key <= pygame.K_9:
                idx = event.key - pygame.K_1
                eids = list(self.state.entities.keys())
                if idx < len(eids):
                    target_eid = eids[idx]
                    self.state.select_entity(target_eid)
                    self.state.manual_control_entity = target_eid
                    return True

            # --- TAB: CYCLE MANUAL OVERRIDE TARGET ---
            if event.key == pygame.K_TAB:
                eids = [eid for eid, e in self.state.entities.items() if e.is_alive()]
                if eids:
                    if self.state.manual_control_entity in eids:
                        cur_idx = eids.index(self.state.manual_control_entity)
                        next_eid = eids[(cur_idx + 1) % len(eids)]
                    else:
                        next_eid = eids[0]
                    self.state.select_entity(next_eid)
                    self.state.manual_control_entity = next_eid
                return True

            # --- DISPLAY & MODE TOGGLES ---
            if event.key == pygame.K_g:
                self.state.show_grid = not self.state.show_grid
                return True
            elif event.key == pygame.K_w:
                self.state.show_wez = not self.state.show_wez
                return True
            elif event.key == pygame.K_s and not ctrl_pressed:
                self.state.show_sensor = not self.state.show_sensor
                return True
            elif event.key == pygame.K_t:
                self.state.show_trajectories = not self.state.show_trajectories
                return True
            elif event.key == pygame.K_l:
                self.state.show_labels = not self.state.show_labels
                return True
            elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                if self.state.selected_entity_id:
                    self.state.delete_entity(self.state.selected_entity_id)
                    return True
            elif event.key == pygame.K_ESCAPE:
                self.state.pending_placement = None
                self.state.select_entity(None)
                self.state.manual_control_entity = None
                self.state.mode = "simulate"
                return True

            # --- TIMELINE PLAYBACK KEYS (SPACE, ARROWS) ---
            if self.timeline.handle_key(event.key):
                return True

        return False

    def _handle_manual_control_keys(self, key: int) -> bool:
        """Construct domain-specific action command from keyboard steering inputs."""
        eid = self.state.manual_control_entity
        if eid is None or eid not in self.state.entities:
            return False

        entity = self.state.entities[eid]
        h_delta = 0.0
        v_cmd = 4
        fire_cannon = 0
        fire_rocket = 0

        # Current action defaults if available
        if isinstance(self.state.manual_action, (AirAction, GroundAction, SeaAction)):
            v_cmd = getattr(self.state.manual_action, "velocity_cmd", 4)

        if key == pygame.K_LEFT:
            h_delta = -5.0
        elif key == pygame.K_RIGHT:
            h_delta = 5.0
        elif key == pygame.K_UP:
            v_cmd = min(8, v_cmd + 1)
        elif key == pygame.K_DOWN:
            v_cmd = max(0, v_cmd - 1)
        elif key == pygame.K_f:
            fire_cannon = 1
        elif key == pygame.K_r:
            fire_rocket = 1
        else:
            return False

        if entity.domain == DomainType.AIR:
            self.state.manual_action = AirAction(
                heading_delta=h_delta,
                velocity_cmd=v_cmd,
                fire_cannon=fire_cannon,
                fire_rocket=fire_rocket,
            )
        elif entity.domain == DomainType.GROUND:
            self.state.manual_action = GroundAction(
                heading_delta=h_delta,
                velocity_cmd=v_cmd,
                weapon_select=0 if fire_cannon else (1 if fire_rocket else 0),
                fire=1 if (fire_cannon or fire_rocket) else 0,
            )
        elif entity.domain == DomainType.SEA:
            self.state.manual_action = SeaAction(
                heading_delta=h_delta,
                velocity_cmd=v_cmd,
                weapon_select=0 if fire_cannon else (1 if fire_rocket else 0),
                fire=1 if (fire_cannon or fire_rocket) else 0,
            )

        return True
