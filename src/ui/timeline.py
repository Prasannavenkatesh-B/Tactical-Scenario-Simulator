"""Timeline playback controls: play/pause, stepping, simulation speed scaling, and step progress."""

from typing import Any
import pygame

from src.ui.config import (
    ACCENT_COLOR,
    BORDER_COLOR,
    BUTTON_ACTIVE_COLOR,
    BUTTON_BG_COLOR,
    HIGHLIGHT_COLOR,
    PANEL_BG_COLOR,
    SPEED_MULTIPLIERS,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
)
from src.ui.state import UIState


class Timeline:
    """Timeline control bar for playback transport, stepping, speed modulation, and progress monitoring."""

    def __init__(
        self,
        screen: pygame.Surface,
        state: UIState,
        rect: pygame.Rect,
    ) -> None:
        self.screen = screen
        self.state = state
        self.rect = rect

        self.font = pygame.font.SysFont("monospace", 11, bold=True)
        self.badge_font = pygame.font.SysFont("sans-serif", 10, bold=True)
        self.click_regions: list[tuple[pygame.Rect, str, Any]] = []

    def render(self) -> None:
        """Render timeline panel, transport buttons, progress bar, and speed options."""
        self.click_regions.clear()

        # Background and top boundary line
        pygame.draw.rect(self.screen, PANEL_BG_COLOR, self.rect)
        pygame.draw.line(self.screen, BORDER_COLOR, self.rect.topleft, self.rect.topright, 2)

        center_y = self.rect.centery
        start_x = self.rect.left + 24

        # =====================================================================
        # 1. TRANSPORT BUTTONS: Rewind, Step Back, Play/Pause, Step Forward
        # =====================================================================
        btn_size = 32
        transport_btns = [
            ("⏮", "rewind", None),
            ("◀", "step_back", None),
            ("⏸" if self.state.mode == "simulate" else "▶", "toggle_play", None),
            ("▶|", "step_forward", None),
        ]

        for i, (symbol, action, param) in enumerate(transport_btns):
            b_rect = pygame.Rect(start_x + i * (btn_size + 8), center_y - btn_size // 2, btn_size, btn_size)
            is_play_btn = (action == "toggle_play")
            bg = (30, 90, 60) if (is_play_btn and self.state.mode == "simulate") else BUTTON_BG_COLOR
            pygame.draw.rect(self.screen, bg, b_rect, border_radius=4)
            pygame.draw.rect(self.screen, BORDER_COLOR, b_rect, 1, border_radius=4)

            txt = self.font.render(symbol, True, HIGHLIGHT_COLOR if is_play_btn else TEXT_COLOR)
            self.screen.blit(txt, txt.get_rect(center=b_rect.center))
            self.click_regions.append((b_rect, action, param))

        start_x += len(transport_btns) * (btn_size + 8) + 24

        # =====================================================================
        # 2. PROGRESS BAR & STEP COUNTER
        # =====================================================================
        bar_w = 400
        bar_h = 10
        bar_x = start_x
        bar_y = center_y - 12
        bar_rect = pygame.Rect(bar_x, bar_y, bar_w, bar_h)

        pygame.draw.rect(self.screen, (16, 16, 22), bar_rect, border_radius=5)
        pygame.draw.rect(self.screen, BORDER_COLOR, bar_rect, 1, border_radius=5)

        max_steps = max(1, self.state.max_steps)
        progress = max(0.0, min(1.0, self.state.current_step / max_steps))
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            fill_rect = pygame.Rect(bar_x, bar_y, fill_w, bar_h)
            pygame.draw.rect(self.screen, ACCENT_COLOR, fill_rect, border_radius=5)

        # Step & Time readout
        sim_time = getattr(self.state.env, "sim_time", self.state.current_step * 0.1)
        step_txt = f"Step: {self.state.current_step:3d} / {max_steps}  ({sim_time:5.1f}s)"
        step_surf = self.font.render(step_txt, True, TEXT_COLOR)
        self.screen.blit(step_surf, (bar_x, bar_y + bar_h + 6))

        # Mode Badge
        mode_str = self.state.mode.upper()
        mode_col = (50, 200, 100) if mode_str == "SIMULATE" else ((240, 180, 40) if mode_str == "PAUSED" else (80, 180, 255))
        badge_surf = self.badge_font.render(f"[{mode_str}]", True, mode_col)
        self.screen.blit(badge_surf, (bar_x + bar_w - badge_surf.get_width(), bar_y + bar_h + 6))

        start_x += bar_w + 40

        # =====================================================================
        # 3. SPEED MULTIPLIER BUTTONS: 0.5x, 1x, 2x, 4x, 8x
        # =====================================================================
        speed_lbl = self.font.render("Speed:", True, TEXT_MUTED_COLOR)
        self.screen.blit(speed_lbl, (start_x, center_y - speed_lbl.get_height() // 2))
        start_x += speed_lbl.get_width() + 10

        sp_w = 40
        sp_h = 24
        for sp in SPEED_MULTIPLIERS:
            sp_rect = pygame.Rect(start_x, center_y - sp_h // 2, sp_w, sp_h)
            is_active = abs(self.state.speed_multiplier - sp) < 1e-4

            bg = BUTTON_ACTIVE_COLOR if is_active else BUTTON_BG_COLOR
            pygame.draw.rect(self.screen, bg, sp_rect, border_radius=3)
            if is_active:
                pygame.draw.rect(self.screen, HIGHLIGHT_COLOR, sp_rect, 1, border_radius=3)
            else:
                pygame.draw.rect(self.screen, BORDER_COLOR, sp_rect, 1, border_radius=3)

            sp_txt = f"{sp:g}x"
            t_surf = self.badge_font.render(sp_txt, True, TEXT_COLOR)
            self.screen.blit(t_surf, t_surf.get_rect(center=sp_rect.center))

            self.click_regions.append((sp_rect, "set_speed", sp))
            start_x += sp_w + 6

    def handle_click(self, x: int, y: int) -> bool:
        """Process click within timeline area. Returns True if event consumed."""
        if not self.rect.collidepoint(x, y):
            return False

        for rect, action, param in self.click_regions:
            if rect.collidepoint(x, y):
                if action == "toggle_play":
                    if self.state.mode == "simulate":
                        self.state.mode = "paused"
                    else:
                        self.state.mode = "simulate"
                    return True

                elif action == "rewind":
                    self.state.reset_simulation()
                    self.state.mode = "paused"
                    return True

                elif action == "step_forward":
                    self.state.mode = "paused"
                    self.state.step_simulation()
                    return True

                elif action == "step_back":
                    # Simulators cannot step backwards physically; reset back to step 0
                    self.state.reset_simulation()
                    self.state.mode = "paused"
                    return True

                elif action == "set_speed":
                    self.state.speed_multiplier = float(param)
                    return True

        return True

    def handle_key(self, key: int) -> bool:
        """Process key press mapped to playback controls."""
        if key == pygame.K_SPACE:
            if self.state.mode == "simulate":
                self.state.mode = "paused"
            else:
                self.state.mode = "simulate"
            return True

        elif key == pygame.K_RIGHT:
            self.state.mode = "paused"
            self.state.step_simulation()
            return True

        elif key == pygame.K_LEFT:
            self.state.reset_simulation()
            self.state.mode = "paused"
            return True

        elif key == pygame.K_UP:
            # Step up speed multiplier
            cur_idx = SPEED_MULTIPLIERS.index(self.state.speed_multiplier) if self.state.speed_multiplier in SPEED_MULTIPLIERS else 1
            if cur_idx < len(SPEED_MULTIPLIERS) - 1:
                self.state.speed_multiplier = SPEED_MULTIPLIERS[cur_idx + 1]
            return True

        elif key == pygame.K_DOWN:
            # Step down speed multiplier
            cur_idx = SPEED_MULTIPLIERS.index(self.state.speed_multiplier) if self.state.speed_multiplier in SPEED_MULTIPLIERS else 1
            if cur_idx > 0:
                self.state.speed_multiplier = SPEED_MULTIPLIERS[cur_idx - 1]
            return True

        return False
