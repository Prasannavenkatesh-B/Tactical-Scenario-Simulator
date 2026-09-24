"""Doctrine detection thresholds and configuration constants.

Defines operational thresholds across air, ground, sea, and cross-domain tactics
as specified in DRDO tactical realism guidelines and Shaw's 'Fighter Combat: Tactics and Maneuvering'.
"""

from typing import Final

# Air-to-Air Doctrine Thresholds
PURSUIT_CURVE_ANGULAR_ERROR_MAX: Final[float] = 0.4    # rad (max angular deviation for pursuit)
LEAD_PURSUIT_LEAD_DISTANCE_MIN: Final[float] = 0.5     # km (lead intercept distance)
LAG_PURSUIT_DISTANCE_MIN: Final[float] = 2.0           # km (rear hemisphere standoff)
DEFENSIVE_BREAK_TURN_RATE: Final[float] = 1.0          # rad/s (~57 deg/s break turn)
ENERGY_TRADE_CORRELATION_MIN: Final[float] = 0.5       # Pearson |r| for altitude-speed exchange
PINCER_BEARING_MIN: Final[float] = 2.09                # rad (~120 deg bearing separation)
PINCER_CLOSING_THRESHOLD: Final[float] = 0.1           # km/step closing rate

# Ground Doctrine Thresholds
TERRAIN_COVER_ELEVATION_PCT: Final[float] = 0.7        # Top 30% of terrain elevation (70th percentile)
MUTUAL_SUPPORT_DISTANCE_MAX: Final[float] = 15.0       # km (maximum separation for mutual ground support)
ENGAGEMENT_RANGE_DISCIPLINE: Final[float] = 0.8        # Fraction of maximum weapon range (effective range)

# Sea Doctrine Thresholds
STANDOFF_ENGAGEMENT_RANGE_PCT: Final[float] = 0.6      # Minimum 60% of weapon range during standoff
SCREEN_FORMATION_ANGLE_TOLERANCE: Final[float] = 0.3   # rad (formation geometry tolerance)
EVASIVE_MANEUVER_TURN_RATE: Final[float] = 0.5         # rad/s (evasive turn under missile threat)

# Cross-Domain Doctrine Thresholds
CAS_PROXIMITY_KM: Final[float] = 5.0                   # km (air support proximity to ground engagement)
SEAD_PRECEDENCE_WINDOW_S: Final[float] = 60.0          # seconds (SEAD strike preceding ground advance)
MARITIME_PATROL_RADIUS_KM: Final[float] = 30.0         # km (patrol corridor from contested boundary)

# Acceptance Standards
MIN_CONFIDENCE_TO_COUNT: Final[float] = 0.5            # Minimum per-episode detector confidence to mark detected
DOCTRINE_PREVALENCE_THRESHOLD: Final[float] = 0.20     # Minimum fraction of episodes (>=20%) where doctrine is present
REALISM_ACCEPTANCE_THRESHOLD: Final[float] = 0.60      # Fraction of 16 doctrines required to pass (>= 60%)
