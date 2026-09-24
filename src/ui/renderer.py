"""Top-level visual compositor managing layer Z-ordering, HUD, and frame diagnostics."""

import pygame

from src.ui.config import (
    ACCENT_COLOR,
    BG_COLOR,
    BORDER_COLOR,
    HIGHLIGHT_COLOR,
    MENU_HEIGHT,
    SIDEBAR_HEIGHT,
    SIDEBAR_OFFSET_X,
    SIDEBAR_OFFSET_Y,
    SIDEBAR_WIDTH,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TIMELINE_HEIGHT,
    TIMELINE_OFFSET_X,
    TIMELINE_OFFSET_Y,
    WINDOW_WIDTH,
)
from src.ui.entity_sprites import EntitySpriteRenderer
from src.ui.map_view import MapView
from src.ui.overlays import OverlayRenderer
from src.ui.sidebar import Sidebar
from src.ui.state import UIState
from src.ui.timeline import Timeline


class Renderer:
    """Top-level compositor: integrates map viewport, entity sprites, overlays, sidebar, and transport timeline."""

    def __init__(self, screen: pygame.Surface, state: UIState) -> None:
        self.screen = screen
        self.state = state

        # Subsystems
        self.map_view = MapView(screen, state)
        self.entity_sprites = EntitySpriteRenderer(screen, state, self.map_view)
        self.overlays = OverlayRenderer(screen, state, self.map_view)

        sidebar_rect = pygame.Rect(SIDEBAR_OFFSET_X, SIDEBAR_OFFSET_Y, SIDEBAR_WIDTH, SIDEBAR_HEIGHT)
        self.sidebar = Sidebar(screen, state, sidebar_rect)

        timeline_rect = pygame.Rect(TIMELINE_OFFSET_X, TIMELINE_OFFSET_Y, WINDOW_WIDTH, TIMELINE_HEIGHT)
        self.timeline = Timeline(screen, state, timeline_rect)

        self.menu_font = pygame.font.SysFont("sans-serif", 11)
        self.title_font = pygame.font.SysFont("sans-serif", 12, bold=True)
        self.fps_font = pygame.font.SysFont("monospace", 10, bold=True)
        self.last_fps: float = 60.0

    def render(self) -> None:
        """Execute full frame composite across all visual layers in strict Z-order."""
        # 1. Background Fill
        self.screen.fill(BG_COLOR)

        # 2. Map (Terrain elevation, operational zones, grid, border)
        self.map_view.render()

        # 3. Trajectory flight trails (under vehicles)
        if self.state.show_trajectories:
            self.overlays.render_trajectories()

        # 4. Sensor footprints (under WEZ)
        if self.state.show_sensor:
            for entity in self.state.entities.values():
                if entity.is_alive():
                    self.overlays.render_sensor_range(entity)

        # 5. Weapon Engagement Zone (WEZ) arcs
        if self.state.show_wez:
            for entity in self.state.entities.values():
                if entity.is_alive():
                    self.overlays.render_wez(entity)

        # 6. Entity sprites, selections, and explosion bursts
        self.entity_sprites.render_all()

        # 7. Entity tactical labels
        if self.state.show_labels:
            self.overlays.render_labels()

        # 8. Boundary proximity alert
        self.overlays.render_boundary_warning()

        # 9. Sidebar control panel & palette
        self.sidebar.render()

        # 10. Timeline playback bar
        self.timeline.render()

        # 11. Application Menubar
        self._render_menubar()

        # 12. Frame diagnostics / FPS indicator
        self.render_fps()

    def _render_menubar(self) -> None:
        """Render top menu strip with branding and system status."""
        menu_rect = pygame.Rect(0, 0, WINDOW_WIDTH, MENU_HEIGHT)
        pygame.draw.rect(self.screen, (22, 22, 30), menu_rect)
        pygame.draw.line(self.screen, BORDER_COLOR, menu_rect.bottomleft, menu_rect.bottomright, 1)

        # Branding
        title_surf = self.title_font.render("TSS  TACTICAL SIMULATION SYSTEM  [DRDO]", True, ACCENT_COLOR)
        self.screen.blit(title_surf, (14, 7))

        # Menu options
        items = ["File", "Scenario", "View", "Help"]
        cur_x = 340
        for item in items:
            t_surf = self.menu_font.render(item, True, TEXT_COLOR)
            self.screen.blit(t_surf, (cur_x, 8))
            cur_x += t_surf.get_width() + 24

        # Mode hint
        if self.state.mode == "edit":
            hint = self.menu_font.render("EDIT MODE: Click map to place unit | ESC to cancel", True, HIGHLIGHT_COLOR)
            self.screen.blit(hint, (WINDOW_WIDTH - 520, 8))

    def render_fps(self, fps: float | None = None) -> None:
        """Render top-right real-time rendering FPS counter."""
        if fps is not None:
            self.last_fps = fps

        fps_text = f"FPS: {self.last_fps:4.1f}"
        fps_surf = self.fps_font.render(fps_text, True, (100, 220, 120))
        self.screen.blit(fps_surf, (WINDOW_WIDTH - fps_surf.get_width() - 14, 8))
