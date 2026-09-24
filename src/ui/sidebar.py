"""Sidebar control panel providing entity palette, entity inspection telemetry, scenario I/O, and view toggles."""

import math
from typing import Any
import pygame

from src.core.interfaces import AgentStatus, DomainType, TeamSide, WeaponType
from src.simulator.config import DEFAULT_EPISODE_HORIZON
from src.simulator.entities.air import AirEntity
from src.simulator.scenarios import generate_scenario
from src.ui.config import (
    ACCENT_COLOR,
    BORDER_COLOR,
    BUTTON_ACTIVE_COLOR,
    BUTTON_BG_COLOR,
    BUTTON_HOVER_COLOR,
    HIGHLIGHT_COLOR,
    PANEL_BG_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
)
from src.ui.state import UIState


class Sidebar:
    """Sidebar control panel containing the unit palette, live telemetry, scenario controls, and display filters."""

    def __init__(
        self,
        screen: pygame.Surface,
        state: UIState,
        rect: pygame.Rect,
    ) -> None:
        self.screen = screen
        self.state = state
        self.rect = rect

        self.title_font = pygame.font.SysFont("sans-serif", 13, bold=True)
        self.font = pygame.font.SysFont("monospace", 11)
        self.bold_font = pygame.font.SysFont("sans-serif", 11, bold=True)
        self.small_font = pygame.font.SysFont("monospace", 10)

        # Palette configuration
        self.selected_team: TeamSide = TeamSide.BLUE
        self.click_regions: list[tuple[pygame.Rect, str, Any]] = []

    def render(self) -> None:
        """Render all sections of the sidebar."""
        self.click_regions.clear()

        # Background
        pygame.draw.rect(self.screen, PANEL_BG_COLOR, self.rect)
        pygame.draw.line(self.screen, BORDER_COLOR, self.rect.topleft, self.rect.bottomleft, 2)

        cur_y = self.rect.top + 10
        cur_x = self.rect.left + 16
        content_w = self.rect.width - 32

        # =====================================================================
        # 1. ENTITY PALETTE
        # =====================================================================
        cur_y = self._render_section_header("► ENTITY PALETTE (Click then Place)", cur_x, cur_y, content_w)

        # Team Toggle Buttons
        team_w = (content_w - 8) // 2
        blue_rect = pygame.Rect(cur_x, cur_y, team_w, 24)
        red_rect = pygame.Rect(cur_x + team_w + 8, cur_y, team_w, 24)

        b_active = self.selected_team == TeamSide.BLUE
        pygame.draw.rect(self.screen, (30, 70, 140) if b_active else BUTTON_BG_COLOR, blue_rect, border_radius=3)
        if b_active:
            pygame.draw.rect(self.screen, HIGHLIGHT_COLOR, blue_rect, 1, border_radius=3)
        b_txt = self.bold_font.render("BLUE TEAM", True, TEXT_COLOR)
        self.screen.blit(b_txt, b_txt.get_rect(center=blue_rect.center))
        self.click_regions.append((blue_rect, "select_team", TeamSide.BLUE))

        r_active = self.selected_team == TeamSide.RED
        pygame.draw.rect(self.screen, (140, 40, 40) if r_active else BUTTON_BG_COLOR, red_rect, border_radius=3)
        if r_active:
            pygame.draw.rect(self.screen, HIGHLIGHT_COLOR, red_rect, 1, border_radius=3)
        r_txt = self.bold_font.render("RED TEAM", True, TEXT_COLOR)
        self.screen.blit(r_txt, r_txt.get_rect(center=red_rect.center))
        self.click_regions.append((red_rect, "select_team", TeamSide.RED))

        cur_y += 32

        # 4 Entity Palette Options
        palette_options = [
            ("Air AC1 (Dogfight)", "air", "AC1"),
            ("Air AC2 (Interceptor)", "air", "AC2"),
            ("Ground (SAM / Tank)", "ground", "DEFAULT"),
            ("Sea (Combatant)", "sea", "DEFAULT"),
        ]

        btn_w = (content_w - 8) // 2
        btn_h = 28
        for i, (label, domain, variant) in enumerate(palette_options):
            bx = cur_x if (i % 2 == 0) else (cur_x + btn_w + 8)
            by = cur_y + (i // 2) * (btn_h + 6)
            b_rect = pygame.Rect(bx, by, btn_w, btn_h)

            is_selected = (
                self.state.mode == "edit"
                and self.state.pending_placement is not None
                and self.state.pending_placement.get("domain") == domain
                and self.state.pending_placement.get("variant") == variant
                and self.state.pending_placement.get("team") == self.selected_team
            )

            bg_col = BUTTON_ACTIVE_COLOR if is_selected else BUTTON_BG_COLOR
            pygame.draw.rect(self.screen, bg_col, b_rect, border_radius=3)
            if is_selected:
                pygame.draw.rect(self.screen, HIGHLIGHT_COLOR, b_rect, 2, border_radius=3)
            else:
                pygame.draw.rect(self.screen, BORDER_COLOR, b_rect, 1, border_radius=3)

            txt = self.font.render(label, True, TEXT_COLOR)
            self.screen.blit(txt, txt.get_rect(center=b_rect.center))

            self.click_regions.append(
                (b_rect, "set_placement", {"domain": domain, "variant": variant, "team": self.selected_team})
            )

        cur_y += (len(palette_options) // 2) * (btn_h + 6) + 16

        # =====================================================================
        # 2. SELECTED ENTITY INFO
        # =====================================================================
        cur_y = self._render_section_header("► SELECTED ENTITY TELEMETRY", cur_x, cur_y, content_w)

        info_box = pygame.Rect(cur_x, cur_y, content_w, 140)
        pygame.draw.rect(self.screen, (16, 16, 22), info_box, border_radius=3)
        pygame.draw.rect(self.screen, BORDER_COLOR, info_box, 1, border_radius=3)

        selected_id = self.state.selected_entity_id
        if selected_id and selected_id in self.state.entities:
            ent = self.state.entities[selected_id]
            health_val = getattr(ent, "health", 100.0 if ent.is_alive() else 0.0)
            lines = [
                f"ID: {ent.entity_id} | Team: {ent.team.value} | Domain: {ent.domain.value}",
                f"Status: {ent.status.value} | Health: {float(health_val):.0f}%",
                f"Pos: X={ent.position[0]:.2f}, Y={ent.position[1]:.2f}, Z={ent.position[2]:.2f} km",
                f"Speed: {ent.speed:.1f} km/h | Heading: {math.degrees(ent.heading):.1f}°",
                f"Ammo: Cannon={ent.ammo.get(WeaponType.CANNON, 0)} Rocket={ent.ammo.get(WeaponType.ROCKET, 0)}",
            ]
            for li, line in enumerate(lines):
                line_surf = self.font.render(line, True, TEXT_COLOR)
                self.screen.blit(line_surf, (info_box.x + 8, info_box.y + 8 + li * 18))

            # Action Buttons: Manual Control & Delete
            mc_w = (content_w - 24) // 2
            mc_rect = pygame.Rect(info_box.x + 8, info_box.bottom - 32, mc_w, 24)
            is_manual = (self.state.manual_control_entity == selected_id)
            pygame.draw.rect(self.screen, (0, 140, 160) if is_manual else BUTTON_BG_COLOR, mc_rect, border_radius=2)
            mc_text = "RELEASE KEYBOARD" if is_manual else "MANUAL CONTROL"
            mc_surf = self.small_font.render(mc_text, True, TEXT_COLOR)
            self.screen.blit(mc_surf, mc_surf.get_rect(center=mc_rect.center))
            self.click_regions.append((mc_rect, "toggle_manual", selected_id))

            del_rect = pygame.Rect(info_box.x + 16 + mc_w, info_box.bottom - 32, mc_w, 24)
            pygame.draw.rect(self.screen, (160, 40, 40), del_rect, border_radius=2)
            del_surf = self.small_font.render("DELETE UNIT", True, TEXT_COLOR)
            self.screen.blit(del_surf, del_surf.get_rect(center=del_rect.center))
            self.click_regions.append((del_rect, "delete_entity", selected_id))
        else:
            empty_msg = self.font.render("No unit selected. Click a unit to inspect.", True, TEXT_MUTED_COLOR)
            self.screen.blit(empty_msg, (info_box.x + 10, info_box.y + 55))

        cur_y += 154

        # =====================================================================
        # 3. SCENARIO CONTROLS
        # =====================================================================
        cur_y = self._render_section_header("► SCENARIO MANAGEMENT", cur_x, cur_y, content_w)

        sc_btn_w = (content_w - 16) // 3
        sc_btn_h = 26
        labels = [("NEW", "new_scenario"), ("SAVE", "save_scenario"), ("LOAD", "load_scenario")]
        for i, (lbl, act) in enumerate(labels):
            sb_rect = pygame.Rect(cur_x + i * (sc_btn_w + 8), cur_y, sc_btn_w, sc_btn_h)
            pygame.draw.rect(self.screen, BUTTON_BG_COLOR, sb_rect, border_radius=3)
            pygame.draw.rect(self.screen, BORDER_COLOR, sb_rect, 1, border_radius=3)
            s_txt = self.bold_font.render(lbl, True, ACCENT_COLOR)
            self.screen.blit(s_txt, s_txt.get_rect(center=sb_rect.center))
            self.click_regions.append((sb_rect, act, None))

        cur_y += 34

        # Curriculum quick-select buttons
        cur_lbl = self.small_font.render("Curriculum Quick-Load:", True, TEXT_MUTED_COLOR)
        self.screen.blit(cur_lbl, (cur_x, cur_y))
        cur_y += 16
        lvl_w = (content_w - 16) // 5
        for lvl in range(1, 6):
            l_rect = pygame.Rect(cur_x + (lvl - 1) * (lvl_w + 4), cur_y, lvl_w, 22)
            pygame.draw.rect(self.screen, BUTTON_BG_COLOR, l_rect, border_radius=2)
            l_txt = self.small_font.render(f"L{lvl}", True, TEXT_COLOR)
            self.screen.blit(l_txt, l_txt.get_rect(center=l_rect.center))
            self.click_regions.append((l_rect, "load_level", lvl))

        cur_y += 34

        # =====================================================================
        # 4. VIEW TOGGLES (Checkboxes)
        # =====================================================================
        cur_y = self._render_section_header("► VIEW TOGGLES", cur_x, cur_y, content_w)

        toggles = [
            ("Show WEZ Firing Cones", "toggle_wez", self.state.show_wez),
            ("Show Sensor Footprints", "toggle_sensor", self.state.show_sensor),
            ("Show Flight Trajectories", "toggle_trajectories", self.state.show_trajectories),
            ("Show Coordinate Grid", "toggle_grid", self.state.show_grid),
            ("Show Entity Tactical Labels", "toggle_labels", self.state.show_labels),
        ]

        cb_size = 14
        for lbl, act, val in toggles:
            cb_rect = pygame.Rect(cur_x, cur_y + 2, cb_size, cb_size)
            pygame.draw.rect(self.screen, (30, 30, 40), cb_rect, border_radius=2)
            pygame.draw.rect(self.screen, BORDER_COLOR, cb_rect, 1, border_radius=2)

            if val:
                # Checkmark fill
                inner = pygame.Rect(cb_rect.x + 3, cb_rect.y + 3, cb_size - 6, cb_size - 6)
                pygame.draw.rect(self.screen, ACCENT_COLOR, inner, border_radius=1)

            t_surf = self.font.render(lbl, True, TEXT_COLOR)
            self.screen.blit(t_surf, (cur_x + cb_size + 10, cur_y + 2))

            # Full row is clickable
            row_rect = pygame.Rect(cur_x, cur_y, content_w, 20)
            self.click_regions.append((row_rect, act, None))
            cur_y += 22

    def _render_section_header(self, text: str, x: int, y: int, width: int) -> int:
        """Render consistent sub-header styling."""
        head_surf = self.title_font.render(text, True, ACCENT_COLOR)
        self.screen.blit(head_surf, (x, y))
        line_y = y + head_surf.get_height() + 3
        pygame.draw.line(self.screen, BORDER_COLOR, (x, line_y), (x + width, line_y), 1)
        return line_y + 8

    def handle_click(self, x: int, y: int) -> bool:
        """Process click in sidebar region. Returns True if event consumed."""
        if not self.rect.collidepoint(x, y):
            return False

        for rect, action, param in self.click_regions:
            if rect.collidepoint(x, y):
                if action == "select_team":
                    self.selected_team = param
                    if self.state.pending_placement:
                        self.state.pending_placement["team"] = param
                    return True

                elif action == "set_placement":
                    self.state.mode = "edit"
                    self.state.pending_placement = dict(param)
                    return True

                elif action == "toggle_manual":
                    if self.state.manual_control_entity == param:
                        self.state.manual_control_entity = None
                    else:
                        self.state.manual_control_entity = param
                    return True

                elif action == "delete_entity":
                    self.state.delete_entity(param)
                    return True

                elif action == "new_scenario":
                    # Clear scenario
                    self.state.mode = "edit"
                    self.state.scenario_config.blue_entities.clear()
                    self.state.scenario_config.red_entities.clear()
                    self.state.reset_simulation()
                    return True

                elif action == "save_scenario":
                    self.state.save_scenario("User_Tactical_Scenario")
                    return True

                elif action == "load_scenario":
                    scenarios = self.state.scenario_io.list_scenarios()
                    if scenarios:
                        self.state.load_scenario(scenarios[-1]["scenario_id"])
                    return True

                elif action == "load_level":
                    self.state.scenario_config = generate_scenario(int(param))
                    self.state.reset_simulation()
                    return True

                elif action == "toggle_wez":
                    self.state.show_wez = not self.state.show_wez
                    return True

                elif action == "toggle_sensor":
                    self.state.show_sensor = not self.state.show_sensor
                    return True

                elif action == "toggle_trajectories":
                    self.state.show_trajectories = not self.state.show_trajectories
                    return True

                elif action == "toggle_grid":
                    self.state.show_grid = not self.state.show_grid
                    return True

                elif action == "toggle_labels":
                    self.state.show_labels = not self.state.show_labels
                    return True

        return True
