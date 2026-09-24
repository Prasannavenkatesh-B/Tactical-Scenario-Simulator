"""Procedural sprite asset generation and loading for tactical UI."""

from pathlib import Path
import pygame

ASSETS_DIR = Path(__file__).resolve().parent


def ensure_default_assets() -> None:
    """Generate default 32x32 pixel sprites if not present on disk."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. air_ac1.png (Agile fighter: sleek delta triangle pointing up (heading 0/north))
    ac1_path = ASSETS_DIR / "air_ac1.png"
    if not ac1_path.exists():
        surf = pygame.Surface((32, 32), pygame.SRCALPHA)
        # Main fuselage
        pygame.draw.polygon(
            surf,
            (220, 230, 255),
            [(16, 2), (24, 28), (16, 23), (8, 28)],
        )
        # Wings
        pygame.draw.polygon(
            surf,
            (100, 160, 255),
            [(16, 12), (30, 26), (16, 20), (2, 26)],
        )
        # Cockpit
        pygame.draw.polygon(
            surf,
            (255, 230, 80),
            [(16, 6), (18, 14), (16, 17), (14, 14)],
        )
        pygame.image.save(surf, str(ac1_path))

    # 2. air_ac2.png (Interceptor: longer fuselage, swept wings)
    ac2_path = ASSETS_DIR / "air_ac2.png"
    if not ac2_path.exists():
        surf = pygame.Surface((32, 32), pygame.SRCALPHA)
        # Fuselage
        pygame.draw.polygon(
            surf,
            (255, 220, 200),
            [(16, 1), (22, 30), (16, 26), (10, 30)],
        )
        # Forward canards / wings
        pygame.draw.polygon(
            surf,
            (255, 140, 80),
            [(16, 14), (31, 22), (16, 19), (1, 22)],
        )
        # Cockpit
        pygame.draw.polygon(
            surf,
            (80, 220, 255),
            [(16, 5), (18, 13), (16, 16), (14, 13)],
        )
        pygame.image.save(surf, str(ac2_path))

    # 3. ground.png (SAM/Tank: square chassis with treads and gun barrel)
    ground_path = ASSETS_DIR / "ground.png"
    if not ground_path.exists():
        surf = pygame.Surface((32, 32), pygame.SRCALPHA)
        # Treads
        pygame.draw.rect(surf, (60, 60, 60), (4, 4, 6, 24), border_radius=2)
        pygame.draw.rect(surf, (60, 60, 60), (22, 4, 6, 24), border_radius=2)
        # Main chassis
        pygame.draw.rect(surf, (140, 110, 70), (8, 6, 16, 20), border_radius=3)
        # Turret
        pygame.draw.circle(surf, (180, 150, 90), (16, 16), 6)
        # Barrel pointing up
        pygame.draw.line(surf, (40, 40, 40), (16, 16), (16, 2), 3)
        pygame.image.save(surf, str(ground_path))

    # 4. sea.png (Surface combatant: elongated hull, pointed bow)
    sea_path = ASSETS_DIR / "sea.png"
    if not sea_path.exists():
        surf = pygame.Surface((32, 32), pygame.SRCALPHA)
        # Hull pointing up
        pygame.draw.polygon(
            surf,
            (80, 120, 170),
            [(16, 2), (25, 12), (24, 28), (8, 28), (7, 12)],
        )
        # Deck
        pygame.draw.polygon(
            surf,
            (130, 165, 205),
            [(16, 5), (22, 13), (21, 26), (11, 26), (10, 13)],
        )
        # Superstructure / Bridge
        pygame.draw.rect(surf, (220, 230, 245), (13, 14, 6, 8), border_radius=1)
        # Radar mast
        pygame.draw.line(surf, (40, 40, 50), (16, 12), (16, 15), 2)
        pygame.image.save(surf, str(sea_path))

    # 5. explosion.png (Multi-layered burst pattern)
    exp_path = ASSETS_DIR / "explosion.png"
    if not exp_path.exists():
        surf = pygame.Surface((32, 32), pygame.SRCALPHA)
        # Outer orange spikes
        points_outer = [
            (16, 0), (20, 10), (31, 8), (24, 16),
            (30, 24), (20, 23), (16, 31), (12, 23),
            (2, 24), (8, 16), (1, 8), (12, 10)
        ]
        pygame.draw.polygon(surf, (255, 100, 20), points_outer)
        # Middle yellow burst
        points_inner = [
            (16, 5), (19, 11), (26, 11), (21, 16),
            (24, 22), (18, 20), (16, 26), (14, 20),
            (8, 22), (11, 16), (6, 11), (13, 11)
        ]
        pygame.draw.polygon(surf, (255, 220, 40), points_inner)
        # Core white flash
        pygame.draw.circle(surf, (255, 255, 240), (16, 16), 5)
        pygame.image.save(surf, str(exp_path))
