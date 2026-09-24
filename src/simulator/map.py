"""2.5D map with selectable borders, procedural terrain elevation, and team zones.

Supports batch training and interactive scenario construction with deterministic
procedural terrain generation via pure numpy.
"""

import math
import typing
import numpy as np

from src.core.interfaces import TeamSide
from src.simulator.config import (
    DEFAULT_MAP_SIZE_KM,
    MAX_ALTITUDE_KM,
    MAX_TERRAIN_ELEVATION_KM,
    MIN_ALTITUDE_KM,
    TERRAIN_GRID_RESOLUTION,
    TWO_PI,
)


class Map2D:
    """2.5D spatial map with selectable operational boundaries and terrain.

    Attributes:
        size_km: Nominal map dimension (width and height) in kilometers.
        x_bounds: Minimum and maximum X boundaries (km).
        y_bounds: Minimum and maximum Y boundaries (km).
        z_bounds: Minimum and maximum altitude boundaries (km).
        terrain_grid: 2D numpy array of shape (100, 100) containing elevations in km.
        blue_zone: (x_min, x_max, y_min, y_max) bounding box for Blue team.
        red_zone: (x_min, x_max, y_min, y_max) bounding box for Red team.
    """

    def __init__(
        self,
        size_km: float = DEFAULT_MAP_SIZE_KM,
        terrain_seed: int | None = None,
    ) -> None:
        """Initialize 2.5D map and procedural elevation grid.

        Args:
            size_km: Map dimension in kilometers.
            terrain_seed: Deterministic seed for procedural terrain generation.
        """
        self.size_km: float = float(size_km)
        self.x_bounds: tuple[float, float] = (0.0, self.size_km)
        self.y_bounds: tuple[float, float] = (0.0, self.size_km)
        self.z_bounds: tuple[float, float] = (MIN_ALTITUDE_KM, MAX_ALTITUDE_KM)

        self.terrain_seed: int | None = terrain_seed
        self.rng: np.random.Generator = np.random.default_rng(terrain_seed)
        self.terrain_grid: np.ndarray = self._generate_terrain(terrain_seed)

        # Default team zones (split vertically: Blue West, Red East)
        half_x = self.size_km / 2.0
        self.blue_zone: tuple[float, float, float, float] = (0.0, half_x, 0.0, self.size_km)
        self.red_zone: tuple[float, float, float, float] = (half_x, self.size_km, 0.0, self.size_km)

    def _generate_terrain(self, seed: int | None) -> np.ndarray:
        """Generate smooth 2D procedural terrain elevation using multi-octave noise.

        Args:
            seed: Deterministic integer seed or None for random.

        Returns:
            2D numpy array of shape (TERRAIN_GRID_RESOLUTION, TERRAIN_GRID_RESOLUTION) in km.
        """
        rng = self.rng
        grid_res = TERRAIN_GRID_RESOLUTION
        elevation = np.zeros((grid_res, grid_res), dtype=np.float32)

        # Multi-octave value noise synthesis
        octaves = [4, 8, 16]
        weights = [0.6, 0.3, 0.1]

        for oct_size, weight in zip(octaves, weights):
            coarse = rng.uniform(0.0, 1.0, size=(oct_size, oct_size)).astype(np.float32)

            # Bilinear interpolation of coarse grid to grid_res x grid_res
            x_indices = np.linspace(0, oct_size - 1, grid_res)
            y_indices = np.linspace(0, oct_size - 1, grid_res)

            x0 = np.floor(x_indices).astype(int)
            x1 = np.clip(x0 + 1, 0, oct_size - 1)
            y0 = np.floor(y_indices).astype(int)
            y1 = np.clip(y0 + 1, 0, oct_size - 1)

            wx = (x_indices - x0)[:, None]
            wy = (y_indices - y0)[None, :]

            # Bilinear blend
            c00 = coarse[np.ix_(x0, y0)]
            c10 = coarse[np.ix_(x1, y0)]
            c01 = coarse[np.ix_(x0, y1)]
            c11 = coarse[np.ix_(x1, y1)]

            layer = (
                (1.0 - wx) * (1.0 - wy) * c00
                + wx * (1.0 - wy) * c10
                + (1.0 - wx) * wy * c01
                + wx * wy * c11
            )
            elevation += weight * layer

        # Normalize and scale to MAX_TERRAIN_ELEVATION_KM
        min_val = float(np.min(elevation))
        max_val = float(np.max(elevation))
        if max_val > min_val:
            elevation = (elevation - min_val) / (max_val - min_val)
        return elevation * MAX_TERRAIN_ELEVATION_KM

    def is_in_bounds(
        self,
        position: tuple[float, float, float] | np.ndarray,
        check_z: bool = True,
    ) -> bool:
        """Check if a 3D coordinate resides inside active map boundaries.

        Args:
            position: Coordinate (x, y, z) in kilometers.
            check_z: Whether to check vertical altitude boundaries.

        Returns:
            True if position is within map boundaries.
        """
        x, y = float(position[0]), float(position[1])

        if not (self.x_bounds[0] <= x <= self.x_bounds[1]):
            return False
        if not (self.y_bounds[0] <= y <= self.y_bounds[1]):
            return False

        if check_z and len(position) > 2:
            z = float(position[2])
            if z < 0.0 or z > self.z_bounds[1]:
                return False

        return True

    def elevation_at(self, x: float, y: float) -> float:
        """Sample terrain elevation at coordinate (x, y) with bilinear interpolation.

        Args:
            x: X position in kilometers.
            y: Y position in kilometers.

        Returns:
            Terrain elevation in kilometers above sea level.
        """
        grid_res = TERRAIN_GRID_RESOLUTION
        norm_x = float(np.clip(x / max(self.size_km, 1e-6), 0.0, 1.0)) * (grid_res - 1)
        norm_y = float(np.clip(y / max(self.size_km, 1e-6), 0.0, 1.0)) * (grid_res - 1)

        ix0 = int(math.floor(norm_x))
        iy0 = int(math.floor(norm_y))
        ix1 = min(ix0 + 1, grid_res - 1)
        iy1 = min(iy0 + 1, grid_res - 1)

        fx = norm_x - ix0
        fy = norm_y - iy0

        h00 = float(self.terrain_grid[ix0, iy0])
        h10 = float(self.terrain_grid[ix1, iy0])
        h01 = float(self.terrain_grid[ix0, iy1])
        h11 = float(self.terrain_grid[ix1, iy1])

        elev = (
            (1.0 - fx) * (1.0 - fy) * h00
            + fx * (1.0 - fy) * h10
            + (1.0 - fx) * fy * h01
            + fx * fy * h11
        )
        return float(elev)

    def set_borders(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        """Update selectable operational borders.

        Args:
            x_min: Minimum X boundary in km.
            x_max: Maximum X boundary in km.
            y_min: Minimum Y boundary in km.
            y_max: Maximum Y boundary in km.
        """
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("Border maximums must be strictly greater than minimums")
        self.x_bounds = (float(x_min), float(x_max))
        self.y_bounds = (float(y_min), float(y_max))

    def assign_team_zones(
        self,
        randomize: bool,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Assign spatial zones for Blue and Red teams (half-map partitions).

        Args:
            randomize: Whether to randomly alternate split axis and sides.
            rng: Seeded numpy Generator.
        """
        x_min, x_max = self.x_bounds
        y_min, y_max = self.y_bounds
        mid_x = (x_min + x_max) / 2.0
        mid_y = (y_min + y_max) / 2.0

        if not randomize or rng is None:
            # Default: Blue on left (West), Red on right (East)
            self.blue_zone = (x_min, mid_x, y_min, y_max)
            self.red_zone = (mid_x, x_max, y_min, y_max)
            return

        split_axis = rng.choice(["horizontal", "vertical"])
        side_flip = rng.choice([False, True])

        if split_axis == "vertical":
            # Left / Right split
            if not side_flip:
                self.blue_zone = (x_min, mid_x, y_min, y_max)
                self.red_zone = (mid_x, x_max, y_min, y_max)
            else:
                self.blue_zone = (mid_x, x_max, y_min, y_max)
                self.red_zone = (x_min, mid_x, y_min, y_max)
        else:
            # Bottom / Top split
            if not side_flip:
                self.blue_zone = (x_min, x_max, y_min, mid_y)
                self.red_zone = (x_min, x_max, mid_y, y_max)
            else:
                self.blue_zone = (x_min, x_max, mid_y, y_max)
                self.red_zone = (x_min, x_max, y_min, mid_y)

    def sample_position(
        self,
        team: TeamSide,
        rng: np.random.Generator,
        altitude_km: float | None = None,
    ) -> tuple[float, float, float]:
        """Sample a valid coordinate within the designated team zone.

        Applies a 10% interior buffer margin to avoid spawning on borders.

        Args:
            team: Blue or Red team allegiance.
            rng: Seeded numpy Generator.
            altitude_km: Specific altitude in km, or None for random sampling.

        Returns:
            Tuple of (x, y, z) in kilometers.
        """
        zone = self.blue_zone if team == TeamSide.BLUE else self.red_zone
        zx_min, zx_max, zy_min, zy_max = zone

        margin_x = 0.1 * (zx_max - zx_min)
        margin_y = 0.1 * (zy_max - zy_min)

        x = rng.uniform(zx_min + margin_x, zx_max - margin_x)
        y = rng.uniform(zy_min + margin_y, zy_max - margin_y)

        if altitude_km is not None:
            z = float(altitude_km)
        else:
            z = rng.uniform(MIN_ALTITUDE_KM * 5.0, MAX_ALTITUDE_KM * 0.6)

        return (float(x), float(y), float(z))

    def sample_heading(
        self,
        team: TeamSide,
        rng: np.random.Generator,
    ) -> float:
        """Sample an initial heading angle pointing generally toward the enemy zone.

        Args:
            team: Blue or Red team allegiance.
            rng: Seeded numpy Generator.

        Returns:
            Heading angle in radians in [0, 2pi).
        """
        own_zone = self.blue_zone if team == TeamSide.BLUE else self.red_zone
        enemy_zone = self.red_zone if team == TeamSide.BLUE else self.blue_zone

        own_center_x = (own_zone[0] + own_zone[1]) / 2.0
        own_center_y = (own_zone[2] + own_zone[3]) / 2.0

        enemy_center_x = (enemy_zone[0] + enemy_zone[1]) / 2.0
        enemy_center_y = (enemy_zone[2] + enemy_zone[3]) / 2.0

        dx = enemy_center_x - own_center_x
        dy = enemy_center_y - own_center_y

        base_angle = math.atan2(dy, dx)
        # Add random angular variation (+/- 20 degrees)
        angular_noise = rng.uniform(-math.radians(20.0), math.radians(20.0))
        heading = (base_angle + angular_noise) % TWO_PI
        if heading < 0.0:
            heading += TWO_PI
        return float(heading)

    def to_dict(self) -> dict[str, typing.Any]:
        """Serialize map configuration and zone coordinates for UI/export."""
        return {
            "size_km": self.size_km,
            "x_bounds": list(self.x_bounds),
            "y_bounds": list(self.y_bounds),
            "z_bounds": list(self.z_bounds),
            "blue_zone": list(self.blue_zone),
            "red_zone": list(self.red_zone),
            "terrain_seed": self.terrain_seed,
        }
