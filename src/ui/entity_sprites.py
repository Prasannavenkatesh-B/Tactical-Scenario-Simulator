"""Entity sprite rendering with team tinting, rotational kinematics, health status, and destruction effects."""

import math
from pathlib import Path
from typing import Any
import pygame

from src.core.interfaces import AgentStatus, DomainType, TeamSide
from src.simulator.entities.air import AirEntity
from src.simulator.entities.base import SimEntity
from src.ui.assets import ensure_default_assets
from src.ui.config import (
    BLUE_ENTITY_COLOR,
    ENTITY_RADIUS,
    HIGHLIGHT_COLOR,
    RED_ENTITY_COLOR,
)
from src.ui.map_view import MapView
from src.ui.state import UIState


class EntitySpriteRenderer:
    """Renders entity visual representations, tactical rotation, health status, and selection rings."""

    def __init__(
        self,
        screen: pygame.Surface,
        state: UIState,
        map_view: MapView,
    ) -> None:
        self.screen = screen
        self.state = state
        self.map_view = map_view
        self.assets_dir = Path(__file__).resolve().parent / "assets"

        # Ensure base assets exist procedurally
        ensure_default_assets()

        # Cache of loaded base sprites: key -> Surface
        self._raw_sprites: dict[str, pygame.Surface] = {}
        # Cache of team-tinted sprites: (key, team) -> Surface
        self._tinted_sprites: dict[tuple[str, TeamSide], pygame.Surface] = {}

        self._load_sprites()

    def _load_sprites(self) -> None:
        """Load 32x32 sprite PNGs and build team-tinted variants."""
        sprite_files = {
            "air_ac1": self.assets_dir / "air_ac1.png",
            "air_ac2": self.assets_dir / "air_ac2.png",
            "ground": self.assets_dir / "ground.png",
            "sea": self.assets_dir / "sea.png",
            "explosion": self.assets_dir / "explosion.png",
        }

        for key, path in sprite_files.items():
            if path.exists():
                surf = pygame.image.load(str(path)).convert_alpha()
            else:
                surf = pygame.Surface((32, 32), pygame.SRCALPHA)
            self._raw_sprites[key] = surf

        # Build blue and red tinted versions for vehicle sprites
        for key in ["air_ac1", "air_ac2", "ground", "sea"]:
            raw = self._raw_sprites[key]

            # Blue variant
            blue_surf = raw.copy()
            tint_blue = pygame.Surface(raw.get_size(), pygame.SRCALPHA)
            tint_blue.fill((70, 130, 255, 110))
            blue_surf.blit(tint_blue, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
            self._tinted_sprites[(key, TeamSide.BLUE)] = blue_surf

            # Red variant
            red_surf = raw.copy()
            tint_red = pygame.Surface(raw.get_size(), pygame.SRCALPHA)
            tint_red.fill((255, 60, 60, 110))
            red_surf.blit(tint_red, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
            self._tinted_sprites[(key, TeamSide.RED)] = red_surf

    def _get_entity_sprite_key(self, entity: SimEntity) -> str:
        """Map entity domain and variant to asset key."""
        if entity.domain == DomainType.AIR:
            if isinstance(entity, AirEntity) and entity.aircraft_type.upper() == "AC2":
                return "air_ac2"
            return "air_ac1"
        elif entity.domain == DomainType.GROUND:
            return "ground"
        elif entity.domain == DomainType.SEA:
            return "sea"
        return "air_ac1"

    def render_all(self) -> None:
        """Render all alive entities, selection markers, and active explosion effects."""
        # 1. Alive entities
        for eid, entity in self.state.entities.items():
            if entity.is_alive():
                self.render_one(entity)

        # 2. Transient explosions from recently destroyed entities
        exp_surf = self._raw_sprites.get("explosion")
        if exp_surf is not None:
            for exp in self.state.explosions:
                frames_left = exp.get("frames_left", 30)
                wx, wy = exp["pos"]
                sx, sy = self.map_view.world_to_screen(wx, wy)

                # Pulsate scale slightly based on lifetime
                scale = max(0.4, frames_left / 30.0)
                size = int(32 * scale)
                if size > 4:
                    scaled_exp = pygame.transform.smoothscale(exp_surf, (size, size))
                    rect = scaled_exp.get_rect(center=(sx, sy))
                    self.screen.blit(scaled_exp, rect.topleft)

    def render_one(self, entity: SimEntity) -> None:
        """Render a single entity with heading rotation, health bar, and selection indicator."""
        sx, sy = self.map_view.world_to_screen(entity.position[0], entity.position[1])

        # Check visibility within map view
        if not self.map_view.rect.collidepoint(sx, sy):
            return

        key = self._get_entity_sprite_key(entity)
        base_surf = self._tinted_sprites.get((key, entity.team), self._raw_sprites[key])

        # Rotation: base sprite points UP (North, +Y). Heading in rad (0=East, pi/2=North)
        # Angle in degrees counter-clockwise from initial UP orientation:
        rot_angle_deg = math.degrees(entity.heading) - 90.0
        rotated_surf = pygame.transform.rotate(base_surf, rot_angle_deg)
        rect = rotated_surf.get_rect(center=(sx, sy))

        # Damaged red overlay (< 50% health)
        health_val = getattr(entity, "health", 100.0 if entity.is_alive() else 0.0)
        health_pct = max(0.0, min(1.0, float(health_val) / 100.0))
        if health_pct < 0.5:
            damage_overlay = pygame.Surface(rotated_surf.get_size(), pygame.SRCALPHA)
            damage_overlay.fill((255, 30, 30, 90))
            rotated_surf.blit(damage_overlay, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

        self.screen.blit(rotated_surf, rect.topleft)

        # Selection highlight ring
        if entity.entity_id == self.state.selected_entity_id:
            pygame.draw.circle(self.screen, HIGHLIGHT_COLOR, (sx, sy), ENTITY_RADIUS + 7, 2)

        # Manual control indicator (cyan pulsing ring)
        if entity.entity_id == self.state.manual_control_entity:
            pygame.draw.circle(self.screen, (0, 240, 255), (sx, sy), ENTITY_RADIUS + 11, 1)

        # Health bar (above entity)
        bar_w = 24
        bar_h = 3
        bar_x = sx - bar_w // 2
        bar_y = rect.top - 6
        pygame.draw.rect(self.screen, (30, 30, 40), (bar_x - 1, bar_y - 1, bar_w + 2, bar_h + 2))
        fill_w = int(bar_w * health_pct)
        health_col = (50, 220, 80) if health_pct >= 0.5 else ((240, 160, 40) if health_pct >= 0.25 else (240, 50, 50))
        if fill_w > 0:
            pygame.draw.rect(self.screen, health_col, (bar_x, bar_y, fill_w, bar_h))
