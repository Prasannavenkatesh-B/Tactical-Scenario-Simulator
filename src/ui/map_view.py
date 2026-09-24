"""Map viewport rendering, terrain elevation heatmap, operational zones, and border selection."""

import math
from typing import Any
import numpy as np
import pygame

from src.simulator.config import MAX_TERRAIN_ELEVATION_KM
from src.ui.config import (
    BLUE_ZONE_COLOR,
    BORDER_COLOR,
    GRID_COLOR,
    GRID_MAJOR_COLOR,
    HIGHLIGHT_COLOR,
    MAP_OFFSET_X,
    MAP_OFFSET_Y,
    MAP_VIEW_SIZE,
    RED_ZONE_COLOR,
    TERRAIN_COLS,
    TERRAIN_ROWS,
    TEXT_MUTED_COLOR,
)
from src.ui.state import UIState


class MapView:
    """Renders the 2D spatial tactical map, terrain elevation, and operational boundaries."""

    def __init__(self, screen: pygame.Surface, state: UIState) -> None:
        self.screen = screen
        self.state = state
        self.rect = pygame.Rect(MAP_OFFSET_X, MAP_OFFSET_Y, MAP_VIEW_SIZE, MAP_VIEW_SIZE)
        self.font = pygame.font.SysFont("monospace", 10)
        self.hud_font = pygame.font.SysFont("sans-serif", 12, bold=True)

        self._terrain_surface: pygame.Surface | None = None
        self._cached_terrain_seed: int | None = None
        self._cached_map_size: float = -1.0

        # Border drag selection state
        self.drag_start: tuple[int, int] | None = None
        self.drag_current: tuple[int, int] | None = None
        self.is_dragging_border: bool = False

    def world_to_screen(self, world_x: float, world_y: float) -> tuple[int, int]:
        """Convert world coordinates (in km) to screen pixel coordinates."""
        map_size = max(self.state.env.map.size_km, 1e-5)
        norm_x = world_x / map_size
        norm_y = world_y / map_size

        sx = MAP_OFFSET_X + int(round(norm_x * MAP_VIEW_SIZE))
        # Invert Y so world Y=0 is at bottom of map
        sy = MAP_OFFSET_Y + MAP_VIEW_SIZE - int(round(norm_y * MAP_VIEW_SIZE))
        return (sx, sy)

    def screen_to_world(self, screen_x: int, screen_y: int) -> tuple[float, float]:
        """Convert screen pixel coordinates to world coordinates in km with boundary clamping."""
        map_size = max(self.state.env.map.size_km, 1e-5)
        rel_x = screen_x - MAP_OFFSET_X
        rel_y = screen_y - MAP_OFFSET_Y

        norm_x = rel_x / MAP_VIEW_SIZE
        norm_y = (MAP_VIEW_SIZE - rel_y) / MAP_VIEW_SIZE

        world_x = float(np.clip(norm_x * map_size, 0.0, map_size))
        world_y = float(np.clip(norm_y * map_size, 0.0, map_size))
        return (world_x, world_y)

    def _build_terrain_surface(self) -> pygame.Surface:
        """Procedurally generate smooth colormapped terrain surface."""
        map_ref = self.state.env.map
        grid = map_ref.terrain_grid
        h, w = grid.shape

        surf = pygame.Surface((w, h))
        max_elev = max(MAX_TERRAIN_ELEVATION_KM, 1e-5)
        norm_grid = np.clip(grid / max_elev, 0.0, 1.0)

        # Vectorized color mapping
        # 0.0 -> dark valley (20, 32, 24)
        # 0.4 -> low hill (45, 60, 35)
        # 0.7 -> highland brown (85, 70, 50)
        # 1.0 -> snow-capped peak (180, 185, 195)
        r = np.zeros_like(norm_grid, dtype=np.uint8)
        g = np.zeros_like(norm_grid, dtype=np.uint8)
        b = np.zeros_like(norm_grid, dtype=np.uint8)

        # Interpolate RGB bands
        c0 = np.array([20, 32, 24])
        c1 = np.array([45, 60, 35])
        c2 = np.array([85, 70, 50])
        c3 = np.array([180, 185, 195])

        mask1 = norm_grid < 0.4
        t1 = norm_grid[mask1] / 0.4
        for idx in range(3):
            col = c0[idx] + (c1[idx] - c0[idx]) * t1
            if idx == 0: r[mask1] = col.astype(np.uint8)
            elif idx == 1: g[mask1] = col.astype(np.uint8)
            else: b[mask1] = col.astype(np.uint8)

        mask2 = (norm_grid >= 0.4) & (norm_grid < 0.7)
        t2 = (norm_grid[mask2] - 0.4) / 0.3
        for idx in range(3):
            col = c1[idx] + (c2[idx] - c1[idx]) * t2
            if idx == 0: r[mask2] = col.astype(np.uint8)
            elif idx == 1: g[mask2] = col.astype(np.uint8)
            else: b[mask2] = col.astype(np.uint8)

        mask3 = norm_grid >= 0.7
        t3 = (norm_grid[mask3] - 0.7) / 0.3
        for idx in range(3):
            col = c2[idx] + (c3[idx] - c2[idx]) * t3
            if idx == 0: r[mask3] = col.astype(np.uint8)
            elif idx == 1: g[mask3] = col.astype(np.uint8)
            else: b[mask3] = col.astype(np.uint8)

        rgb_array = np.dstack((r, g, b))
        pygame.surfarray.blit_array(surf, np.transpose(rgb_array, (1, 0, 2)))
        return pygame.transform.smoothscale(surf, (MAP_VIEW_SIZE, MAP_VIEW_SIZE))

    def render(self) -> None:
        """Render map terrain heatmap, operational zones, grid, and borders."""
        # 1. Terrain elevation heatmap
        map_size = self.state.env.map.size_km
        terrain_seed = self.state.env.map.terrain_seed
        if (
            self._terrain_surface is None
            or self._cached_map_size != map_size
            or self._cached_terrain_seed != terrain_seed
        ):
            self._terrain_surface = self._build_terrain_surface()
            self._cached_map_size = map_size
            self._cached_terrain_seed = terrain_seed

        if self._terrain_surface is not None:
            self.screen.blit(self._terrain_surface, self.rect.topleft)

        # 2. Blue and Red Team Zones
        self._render_team_zones()

        # 3. Grid lines and distance markings
        if self.state.show_grid:
            self._render_grid()

        # 4. Map Border
        pygame.draw.rect(self.screen, BORDER_COLOR, self.rect, 2)

        # 5. Active border drag overlay if dragging
        if self.is_dragging_border and self.drag_start and self.drag_current:
            self.render_border_selection_overlay(self.drag_start, self.drag_current)

    def _render_team_zones(self) -> None:
        """Render transparent overlays indicating team spawn/patrol zones."""
        blue_zone = self.state.env.map.blue_zone
        red_zone = self.state.env.map.red_zone

        # Zone format: (x_min, x_max, y_min, y_max)
        for zone, color, label in [(blue_zone, BLUE_ZONE_COLOR, "BLUE ZONE"), (red_zone, RED_ZONE_COLOR, "RED ZONE")]:
            if zone is None:
                continue
            x_min, x_max, y_min, y_max = zone
            sx1, sy1 = self.world_to_screen(x_min, y_max)  # Top-left in screen coords
            sx2, sy2 = self.world_to_screen(x_max, y_min)  # Bottom-right in screen coords

            w = max(1, abs(sx2 - sx1))
            h = max(1, abs(sy2 - sy1))
            zone_rect = pygame.Rect(min(sx1, sx2), min(sy1, sy2), w, h)

            overlay = pygame.Surface((w, h), pygame.SRCALPHA)
            overlay.fill(color)
            self.screen.blit(overlay, zone_rect.topleft)
            pygame.draw.rect(self.screen, (color[0], color[1], color[2], 120), zone_rect, 1)

            # Zone label
            lbl_surf = self.hud_font.render(label, True, (color[0], color[1], color[2]))
            self.screen.blit(lbl_surf, (zone_rect.x + 8, zone_rect.y + 8))

    def _render_grid(self) -> None:
        """Draw tactical coordinate grid with distance intervals."""
        map_size = self.state.env.map.size_km
        step_km = 5.0 if map_size <= 35.0 else 10.0

        for x in np.arange(0.0, map_size + 0.1, step_km):
            sx, _ = self.world_to_screen(x, 0.0)
            is_major = (abs(x % (step_km * 2)) < 1e-4) or (x == 0.0)
            col = GRID_MAJOR_COLOR if is_major else GRID_COLOR
            pygame.draw.line(self.screen, col, (sx, self.rect.top), (sx, self.rect.bottom), 1)

            # Draw X axis labels at bottom
            if is_major and sx < self.rect.right - 20:
                txt = self.font.render(f"{x:.0f}k", True, TEXT_MUTED_COLOR)
                self.screen.blit(txt, (sx + 2, self.rect.bottom - 14))

        for y in np.arange(0.0, map_size + 0.1, step_km):
            _, sy = self.world_to_screen(0.0, y)
            is_major = (abs(y % (step_km * 2)) < 1e-4) or (y == 0.0)
            col = GRID_MAJOR_COLOR if is_major else GRID_COLOR
            pygame.draw.line(self.screen, col, (self.rect.left, sy), (self.rect.right, sy), 1)

            # Draw Y axis labels at left
            if is_major and sy > self.rect.top + 10:
                txt = self.font.render(f"{y:.0f}k", True, TEXT_MUTED_COLOR)
                self.screen.blit(txt, (self.rect.left + 4, sy - 12))

    def render_border_selection_overlay(
        self,
        drag_start: tuple[int, int],
        drag_end: tuple[int, int],
    ) -> None:
        """Render selection rectangle when human user is redefining operational borders."""
        x1, y1 = drag_start
        x2, y2 = drag_end
        rx = min(x1, x2)
        ry = min(y1, y2)
        rw = max(abs(x2 - x1), 2)
        rh = max(abs(y2 - y1), 2)

        sel_rect = pygame.Rect(rx, ry, rw, rh)
        overlay = pygame.Surface((rw, rh), pygame.SRCALPHA)
        overlay.fill((255, 255, 100, 35))
        self.screen.blit(overlay, sel_rect.topleft)
        pygame.draw.rect(self.screen, HIGHLIGHT_COLOR, sel_rect, 2)

        # Dimension tooltip
        w_start, h_start = self.screen_to_world(rx, ry + rh)
        w_end, h_end = self.screen_to_world(rx + rw, ry)
        dw = abs(w_end - w_start)
        dh = abs(h_end - h_start)
        dim_text = f"{dw:.1f} x {dh:.1f} km"
        dim_surf = self.hud_font.render(dim_text, True, HIGHLIGHT_COLOR)
        self.screen.blit(dim_surf, (rx + 6, ry + 6))

    def handle_click(self, screen_x: int, screen_y: int, button: int) -> bool:
        """Process click within map viewport. Returns True if event consumed."""
        if not self.rect.collidepoint(screen_x, screen_y):
            return False

        if button == 1:  # Left click
            world_x, world_y = self.screen_to_world(screen_x, screen_y)

            # Edit mode placement
            if self.state.mode == "edit" and self.state.pending_placement:
                self.state.place_entity(world_x, world_y)
                return True

            # Entity selection: find closest entity within 2.0 km click radius
            closest_eid: str | None = None
            min_dist = float("inf")
            for eid, ent in self.state.entities.items():
                if not ent.is_alive():
                    continue
                d = math.hypot(ent.position[0] - world_x, ent.position[1] - world_y)
                if d < min_dist and d <= 2.5:  # 2.5 km tolerance
                    min_dist = d
                    closest_eid = eid

            self.state.select_entity(closest_eid)
            return True

        elif button == 3:  # Right click: delete entity under cursor
            world_x, world_y = self.screen_to_world(screen_x, screen_y)
            for eid, ent in list(self.state.entities.items()):
                if math.hypot(ent.position[0] - world_x, ent.position[1] - world_y) <= 2.0:
                    self.state.delete_entity(eid)
                    return True

        return True
