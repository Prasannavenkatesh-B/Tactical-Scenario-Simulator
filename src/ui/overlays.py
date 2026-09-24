"""Tactical visual overlays: weapon engagement zones (WEZ), sensor envelopes, trails, and boundary alerts."""

import math
from typing import Any
import numpy as np
import pygame

from src.core.interfaces import DomainType
from src.simulator.config import (
    AC1_WEZ_ANGLE_DEG,
    AC1_WEZ_RANGE_KM,
    AC2_WEZ_ANGLE_DEG,
    AC2_WEZ_RANGE_KM,
    GROUND_SENSOR_RANGE_KM,
    GROUND_WEZ_ANGLE_DEG,
    GROUND_WEZ_RANGE_KM,
    SEA_RADAR_RANGE_KM,
    SEA_WEZ_ANGLE_DEG,
    SEA_WEZ_RANGE_KM,
)
from src.simulator.entities.air import AirEntity
from src.simulator.entities.base import SimEntity
from src.ui.config import (
    HIGHLIGHT_COLOR,
    SENSOR_COLOR,
    TEXT_COLOR,
    TRAJECTORY_COLOR,
    WEZ_COLOR,
)
from src.ui.map_view import MapView
from src.ui.state import UIState


class OverlayRenderer:
    """Renders tactical sensor footprints, firing cones, flight trails, and safety warnings."""

    def __init__(
        self,
        screen: pygame.Surface,
        state: UIState,
        map_view: MapView,
    ) -> None:
        self.screen = screen
        self.state = state
        self.map_view = map_view
        self.font = pygame.font.SysFont("monospace", 10, bold=True)
        self.warn_font = pygame.font.SysFont("sans-serif", 12, bold=True)

    def render_all(self) -> None:
        """Render all active overlays in back-to-front layer ordering."""
        # 1. Trajectory historical flight trails
        if self.state.show_trajectories:
            self.render_trajectories()

        # 2. Sensor detection envelopes (under WEZ)
        if self.state.show_sensor:
            for entity in self.state.entities.values():
                if entity.is_alive():
                    self.render_sensor_range(entity)

        # 3. Weapon Engagement Zone (WEZ) firing cones
        if self.state.show_wez:
            for entity in self.state.entities.values():
                if entity.is_alive():
                    self.render_wez(entity)

        # 4. Entity identification labels
        if self.state.show_labels:
            self.render_labels()

        # 5. Boundary proximity alarm
        self.render_boundary_warning()

    def render_wez(self, entity: SimEntity) -> None:
        """Draw the forward Weapon Engagement Zone (WEZ) arc."""
        wez_range = 4.0
        wez_angle_deg = 20.0

        if entity.domain == DomainType.AIR:
            if isinstance(entity, AirEntity) and entity.aircraft_type.upper() == "AC2":
                wez_range = AC2_WEZ_RANGE_KM
                wez_angle_deg = AC2_WEZ_ANGLE_DEG
            else:
                wez_range = AC1_WEZ_RANGE_KM
                wez_angle_deg = AC1_WEZ_ANGLE_DEG
        elif entity.domain == DomainType.GROUND:
            wez_range = GROUND_WEZ_RANGE_KM
            wez_angle_deg = GROUND_WEZ_ANGLE_DEG
        elif entity.domain == DomainType.SEA:
            wez_range = SEA_WEZ_RANGE_KM
            wez_angle_deg = SEA_WEZ_ANGLE_DEG

        # Origin
        wx0, wy0 = entity.position[0], entity.position[1]
        sx0, sy0 = self.map_view.world_to_screen(wx0, wy0)

        # Calculate arc points
        half_angle_rad = math.radians(wez_angle_deg / 2.0)
        heading = entity.heading
        num_arc_pts = 12
        angles = np.linspace(heading - half_angle_rad, heading + half_angle_rad, num_arc_pts)

        poly_pts: list[tuple[int, int]] = [(sx0, sy0)]
        for ang in angles:
            wx = wx0 + wez_range * math.cos(ang)
            wy = wy0 + wez_range * math.sin(ang)
            poly_pts.append(self.map_view.world_to_screen(wx, wy))

        if len(poly_pts) >= 3:
            # Draw on transparent overlay
            rect = self.map_view.rect
            overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            # Adjust points relative to overlay surface
            local_pts = [(p[0] - rect.left, p[1] - rect.top) for p in poly_pts]
            pygame.draw.polygon(overlay, WEZ_COLOR, local_pts)
            pygame.draw.polygon(overlay, (WEZ_COLOR[0], WEZ_COLOR[1], WEZ_COLOR[2], 160), local_pts, 1)
            self.screen.blit(overlay, rect.topleft)

    def render_sensor_range(self, entity: SimEntity) -> None:
        """Draw translucent circular sensor coverage footprint."""
        sensor_range_km = 20.0
        eid = entity.entity_id
        if eid in self.state.env.sensors:
            sensor_range_km = self.state.env.sensors[eid].max_range_km
        elif entity.domain == DomainType.GROUND:
            sensor_range_km = GROUND_SENSOR_RANGE_KM
        elif entity.domain == DomainType.SEA:
            sensor_range_km = SEA_RADAR_RANGE_KM

        map_size = max(self.state.env.map.size_km, 1e-5)
        pixel_radius = int(round((sensor_range_km / map_size) * self.map_view.rect.width))
        if pixel_radius <= 2:
            return

        sx, sy = self.map_view.world_to_screen(entity.position[0], entity.position[1])

        # Clip surface to avoid massive allocations
        diameter = pixel_radius * 2
        surf = pygame.Surface((diameter + 2, diameter + 2), pygame.SRCALPHA)
        center = (pixel_radius + 1, pixel_radius + 1)
        pygame.draw.circle(surf, SENSOR_COLOR, center, pixel_radius)
        pygame.draw.circle(surf, (SENSOR_COLOR[0], SENSOR_COLOR[1], SENSOR_COLOR[2], 90), center, pixel_radius, 1)

        dest_x = sx - pixel_radius - 1
        dest_y = sy - pixel_radius - 1
        self.screen.blit(surf, (dest_x, dest_y))

    def render_trajectories(self) -> None:
        """Draw historical position breadcrumb trails."""
        rect = self.map_view.rect
        for eid, pts in self.state.trajectories.items():
            if len(pts) < 2:
                continue

            screen_pts = [self.map_view.world_to_screen(wx, wy) for wx, wy in pts]
            # Draw segment lines
            pygame.draw.lines(self.screen, (TRAJECTORY_COLOR[0], TRAJECTORY_COLOR[1], TRAJECTORY_COLOR[2]), False, screen_pts, 1)

    def render_labels(self) -> None:
        """Draw entity tactical identifiers above active units."""
        for eid, entity in self.state.entities.items():
            if not entity.is_alive():
                continue
            sx, sy = self.map_view.world_to_screen(entity.position[0], entity.position[1])
            if not self.map_view.rect.collidepoint(sx, sy):
                continue

            lbl = self.font.render(eid, True, TEXT_COLOR)
            self.screen.blit(lbl, (sx - lbl.get_width() // 2, sy - 22))

    def render_boundary_warning(self) -> None:
        """Trigger visual warning if any entity enters border proximity zone (within 6 km)."""
        map_size = self.state.env.map.size_km
        warn_dist_km = 6.0
        near_edge = False

        for entity in self.state.entities.values():
            if not entity.is_alive():
                continue
            x, y = entity.position[0], entity.position[1]
            if x < warn_dist_km or x > (map_size - warn_dist_km) or y < warn_dist_km or y > (map_size - warn_dist_km):
                near_edge = True
                break

        if near_edge:
            rect = self.map_view.rect
            pulse = int(127 + 127 * math.sin(pygame.time.get_ticks() * 0.008))
            warn_color = (255, 40, 40, pulse)

            warn_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            pygame.draw.rect(warn_surf, warn_color, warn_surf.get_rect(), 4)
            self.screen.blit(warn_surf, rect.topleft)

            # Warning banner
            badge = self.warn_font.render("! BOUNDARY PROXIMITY WARNING (< 6 KM) !", True, (255, 70, 70))
            badge_rect = badge.get_rect(center=(rect.centerx, rect.top + 20))
            bg = pygame.Surface((badge_rect.width + 16, badge_rect.height + 8))
            bg.fill((40, 15, 15))
            self.screen.blit(bg, (badge_rect.left - 8, badge_rect.top - 4))
            pygame.draw.rect(self.screen, (220, 50, 50), (badge_rect.left - 8, badge_rect.top - 4, badge_rect.width + 16, badge_rect.height + 8), 1)
            self.screen.blit(badge, badge_rect.topleft)
