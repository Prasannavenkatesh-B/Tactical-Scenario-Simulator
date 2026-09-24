"""UI configuration constants: display dimensions, colors, simulation timing, and layout."""

from typing import Final

# =============================================================================
# COLOR DEFINITIONS
# =============================================================================
BG_COLOR: Final[tuple[int, int, int]] = (18, 18, 24)
GRID_COLOR: Final[tuple[int, int, int]] = (40, 40, 50)
GRID_MAJOR_COLOR: Final[tuple[int, int, int]] = (60, 60, 75)

BLUE_ZONE_COLOR: Final[tuple[int, int, int, int]] = (30, 60, 120, 60)      # RGBA
RED_ZONE_COLOR: Final[tuple[int, int, int, int]] = (120, 40, 40, 60)        # RGBA

BLUE_ENTITY_COLOR: Final[tuple[int, int, int]] = (80, 140, 255)
RED_ENTITY_COLOR: Final[tuple[int, int, int]] = (255, 80, 80)

GROUND_COLOR: Final[tuple[int, int, int]] = (140, 100, 60)
SEA_COLOR: Final[tuple[int, int, int]] = (40, 80, 140)

WEZ_COLOR: Final[tuple[int, int, int, int]] = (255, 200, 0, 80)            # RGBA
SENSOR_COLOR: Final[tuple[int, int, int, int]] = (100, 200, 100, 40)        # RGBA
TRAJECTORY_COLOR: Final[tuple[int, int, int, int]] = (200, 200, 220, 100)  # RGBA

TEXT_COLOR: Final[tuple[int, int, int]] = (230, 230, 240)
TEXT_MUTED_COLOR: Final[tuple[int, int, int]] = (150, 150, 165)
ACCENT_COLOR: Final[tuple[int, int, int]] = (255, 180, 40)
HIGHLIGHT_COLOR: Final[tuple[int, int, int]] = (255, 255, 80)
PANEL_BG_COLOR: Final[tuple[int, int, int]] = (24, 24, 32)
BUTTON_BG_COLOR: Final[tuple[int, int, int]] = (40, 44, 58)
BUTTON_HOVER_COLOR: Final[tuple[int, int, int]] = (55, 60, 78)
BUTTON_ACTIVE_COLOR: Final[tuple[int, int, int]] = (70, 110, 180)
BORDER_COLOR: Final[tuple[int, int, int]] = (50, 52, 66)

# =============================================================================
# WINDOW & COMPONENT DIMENSIONS
# =============================================================================
WINDOW_WIDTH: Final[int] = 1200
WINDOW_HEIGHT: Final[int] = 900
MENU_HEIGHT: Final[int] = 30
SIDEBAR_WIDTH: Final[int] = 400
TIMELINE_HEIGHT: Final[int] = 70
MAP_VIEW_SIZE: Final[int] = 800
ENTITY_RADIUS: Final[int] = 10
TRAIL_LENGTH: Final[int] = 50

# Map View offset on screen: (x=0, y=MENU_HEIGHT, width=MAP_VIEW_SIZE, height=MAP_VIEW_SIZE)
MAP_OFFSET_X: Final[int] = 0
MAP_OFFSET_Y: Final[int] = MENU_HEIGHT

# Sidebar offset on screen: (x=MAP_VIEW_SIZE, y=MENU_HEIGHT, width=SIDEBAR_WIDTH, height=WINDOW_HEIGHT - MENU_HEIGHT - TIMELINE_HEIGHT)
SIDEBAR_OFFSET_X: Final[int] = MAP_VIEW_SIZE
SIDEBAR_OFFSET_Y: Final[int] = MENU_HEIGHT
SIDEBAR_HEIGHT: Final[int] = WINDOW_HEIGHT - MENU_HEIGHT - TIMELINE_HEIGHT

# Timeline offset on screen: (x=0, y=WINDOW_HEIGHT - TIMELINE_HEIGHT, width=WINDOW_WIDTH, height=TIMELINE_HEIGHT)
TIMELINE_OFFSET_X: Final[int] = 0
TIMELINE_OFFSET_Y: Final[int] = WINDOW_HEIGHT - TIMELINE_HEIGHT

# =============================================================================
# SIMULATION TIMING & PERFORMANCE
# =============================================================================
TARGET_FPS: Final[int] = 60
SIM_STEPS_PER_FRAME: Final[int] = 1
SPEED_MULTIPLIERS: Final[list[float]] = [0.5, 1.0, 2.0, 4.0, 8.0]
MAX_SIM_STEPS_PER_FRAME: Final[int] = 8

# =============================================================================
# PROCEDURAL TERRAIN RENDERING
# =============================================================================
TERRAIN_COLS: Final[int] = 100
TERRAIN_ROWS: Final[int] = 100
TERRAIN_COLORMAP: Final[str] = "terrain"
