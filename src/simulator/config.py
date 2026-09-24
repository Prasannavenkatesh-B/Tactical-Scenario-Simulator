"""Configuration constants and default parameters for the 2.5D tactical simulator.

All numerical parameters and operational limits are centralized here to eliminate
hardcoded magic numbers across the simulation engine.
"""

import math

# =============================================================================
# MAP BOUNDS & GEOMETRY
# =============================================================================
DEFAULT_MAP_SIZE_KM: float = 30.0        # For low-level domain training
HIGH_LEVEL_MAP_SIZE_KM: float = 50.0     # For commander-level scenario training
MIN_ALTITUDE_KM: float = 0.1             # Ground clearance floor
MAX_ALTITUDE_KM: float = 15.0            # Ceiling limit for air combat
TERRAIN_GRID_RESOLUTION: int = 100       # 100x100 elevation grid
MAX_TERRAIN_ELEVATION_KM: float = 3.0    # Peak mountain height

# =============================================================================
# SIMULATION TIMING & HORIZONS
# =============================================================================
DEFAULT_DT_SECONDS: float = 0.1          # 10 Hz physical integration step
DEFAULT_EPISODE_HORIZON: int = 200       # Steps for levels 1-4
DEFAULT_LEVEL5_HORIZON: int = 350        # Steps for multi-domain level 5

# =============================================================================
# AIR DOMAIN: AIRCRAFT TYPE 1 (AC1 - Agile Dogfighter with Rockets)
# =============================================================================
AC1_ANGULAR_VELOCITY_RANGE_DEG: tuple[float, float] = (0.0, 5.0)  # Max 5.0 deg/s
AC1_MAX_TURN_RATE_DEG: float = 5.0
AC1_SPEED_RANGE_KNOTS: tuple[float, float] = (100.0, 900.0)
AC1_WEZ_ANGLE_DEG: float = 10.0          # Angular width of firing cone
AC1_WEZ_RANGE_KM: float = 2.0            # Maximum cannon engagement range
AC1_HIT_PROB_CANNON: float = 0.70        # Base Pk for cannon
AC1_AMMO_CANNON: int = 200
AC1_AMMO_ROCKET: int = 5
AC1_ROCKET_WEZ_RANGE_KM: float = 6.0     # Rocket range
AC1_ROCKET_WEZ_ANGLE_DEG: float = 15.0   # Rocket cone
AC1_ROCKET_HIT_PROB: float = 0.65        # Rocket base Pk
AC1_ACCELERATION_KNOTS_S: float = 50.0   # Max linear acceleration
AC1_CLIMB_RATE_KMS: float = 0.09         # ~300 ft/s in km/s

# =============================================================================
# AIR DOMAIN: AIRCRAFT TYPE 2 (AC2 - Interceptor, Longer Cannon, No Rockets)
# =============================================================================
AC2_ANGULAR_VELOCITY_RANGE_DEG: tuple[float, float] = (0.0, 3.5)  # Max 3.5 deg/s
AC2_MAX_TURN_RATE_DEG: float = 3.5
AC2_SPEED_RANGE_KNOTS: tuple[float, float] = (100.0, 600.0)
AC2_WEZ_ANGLE_DEG: float = 7.0
AC2_WEZ_RANGE_KM: float = 4.5
AC2_HIT_PROB_CANNON: float = 0.85
AC2_AMMO_CANNON: int = 200
AC2_AMMO_ROCKET: int = 0                 # AC2 has no rockets
AC2_ACCELERATION_KNOTS_S: float = 40.0
AC2_CLIMB_RATE_KMS: float = 0.07

# =============================================================================
# GROUND DOMAIN: MOBILE UNITS & SAM / ARTILLERY SITES
# =============================================================================
GROUND_SPEED_RANGE_KMPH: tuple[float, float] = (0.0, 60.0)
GROUND_MAX_TURN_RATE_DEG: float = 15.0   # Fast pivot on ground
GROUND_WEZ_RANGE_KM: float = 8.0         # SAM engagement envelope
GROUND_WEZ_ANGLE_DEG: float = 60.0       # Forward firing sector
GROUND_HIT_PROB: float = 0.75
GROUND_AMMO: int = 100
GROUND_SENSOR_RANGE_KM: float = 25.0
GROUND_ACCELERATION_KMH_S: float = 20.0
GROUND_MAX_SLOPE: float = 0.35           # Max slope grade before blockage

# =============================================================================
# SEA DOMAIN: SURFACE COMBATANTS
# =============================================================================
SEA_SPEED_RANGE_KNOTS: tuple[float, float] = (0.0, 35.0)
SEA_MAX_TURN_RATE_DEG: float = 4.0       # Ship turning inertia
SEA_WEZ_RANGE_KM: float = 15.0           # Naval missile envelope
SEA_WEZ_ANGLE_DEG: float = 45.0
SEA_HIT_PROB: float = 0.80
SEA_AMMO: int = 80
SEA_RADAR_RANGE_KM: float = 40.0
SEA_ACCELERATION_KNOTS_S: float = 5.0

# =============================================================================
# SENSOR STOCHASTICITY (Perception Layer)
# =============================================================================
SENSOR_BASE_PD: float = 0.95             # Nominal peak probability of detection
SENSOR_RANGE_FALLOFF: float = 2.0        # Exponent for range degradation: (1 - d/R)^falloff
SENSOR_ASPECT_PENALTY: float = 0.30      # Detection reduction when target is beam-on
SENSOR_FALSE_ALARM_RATE: float = 0.01    # Probability of ghost contact per step
SENSOR_RANGE_NOISE_SIGMA_FRAC: float = 0.05  # 5% of max range
SENSOR_BEARING_NOISE_DEG: float = 2.0    # 2.0 degrees angular noise

# =============================================================================
# WEAPON STOCHASTICITY & COOLDOWNS (Effect Layer)
# =============================================================================
WEAPON_PK_SCALE: float = 1.0             # Global calibration multiplier
WEAPON_MISS_DISTANCE_SIGMA_KM: float = 0.15  # Gaussian miss dispersion

COOLDOWN_CANNON_SECONDS: float = 0.5
COOLDOWN_ROCKET_SECONDS: float = 5.0
COOLDOWN_MISSILE_SECONDS: float = 4.0
COOLDOWN_SAM_SECONDS: float = 3.0

# =============================================================================
# RANDOM NUMBER GENERATOR
# =============================================================================
DEFAULT_SEED: int | None = None          # None = nondeterministic, int = reproducible

# =============================================================================
# UNIT CONVERSION CONSTANTS
# =============================================================================
KNOTS_TO_KMH: float = 1.852
KMH_TO_KNOTS: float = 1.0 / 1.852
KMH_TO_MS: float = 1.0 / 3.6
MS_TO_KMH: float = 3.6
KNOTS_TO_KMS: float = 1.852 / 3600.0     # Nautical knots to km/s
KMH_TO_KMS: float = 1.0 / 3600.0         # km/h to km/s

DEG_TO_RAD: float = math.pi / 180.0
RAD_TO_DEG: float = 180.0 / math.pi
TWO_PI: float = 2.0 * math.pi
