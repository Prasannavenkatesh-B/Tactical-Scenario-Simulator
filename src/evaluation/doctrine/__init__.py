"""Tactical doctrine analysis and realism evaluation package.

Exports all 16 military combat doctrine detectors, base detector interfaces,
and the registry aggregator.
"""

from src.evaluation.doctrine.air_doctrine import (
    DefensiveBreakDetector,
    EnergyManagementDetector,
    LagPursuitDetector,
    LeadPursuitDetector,
    PincerManeuverDetector,
    PursuitCurveDetector,
    ThreatPrioritizationDetector,
    angular_error,
    compute_bearing,
    extract_engagement_trajectories,
)
from src.evaluation.doctrine.base_detector import (
    DoctrineDetector,
    DoctrineResult,
    EpisodeData,
)
from src.evaluation.doctrine.config import (
    CAS_PROXIMITY_KM,
    DEFENSIVE_BREAK_TURN_RATE,
    ENERGY_TRADE_CORRELATION_MIN,
    ENGAGEMENT_RANGE_DISCIPLINE,
    EVASIVE_MANEUVER_TURN_RATE,
    LAG_PURSUIT_DISTANCE_MIN,
    LEAD_PURSUIT_LEAD_DISTANCE_MIN,
    MARITIME_PATROL_RADIUS_KM,
    MIN_CONFIDENCE_TO_COUNT,
    MUTUAL_SUPPORT_DISTANCE_MAX,
    PINCER_BEARING_MIN,
    PINCER_CLOSING_THRESHOLD,
    PURSUIT_CURVE_ANGULAR_ERROR_MAX,
    REALISM_ACCEPTANCE_THRESHOLD,
    SCREEN_FORMATION_ANGLE_TOLERANCE,
    SEAD_PRECEDENCE_WINDOW_S,
    STANDOFF_ENGAGEMENT_RANGE_PCT,
    TERRAIN_COVER_ELEVATION_PCT,
)
from src.evaluation.doctrine.cross_domain_doctrine import (
    AirGroundCoordinationDetector,
    MaritimePatrolDetector,
    SEADSupportDetector,
)
from src.evaluation.doctrine.doctrine_registry import DoctrineRegistry
from src.evaluation.doctrine.ground_doctrine import (
    EngagementRangeDisciplineDetector,
    MutualSupportDetector,
    TerrainCoverDetector,
)
from src.evaluation.doctrine.sea_doctrine import (
    EvasiveManeuverDetector,
    ScreenFormationDetector,
    StandoffEngagementDetector,
)

__all__ = [
    # Base interfaces
    "DoctrineDetector",
    "DoctrineResult",
    "EpisodeData",
    "DoctrineRegistry",
    # Air detectors
    "PursuitCurveDetector",
    "LeadPursuitDetector",
    "LagPursuitDetector",
    "DefensiveBreakDetector",
    "EnergyManagementDetector",
    "PincerManeuverDetector",
    "ThreatPrioritizationDetector",
    # Ground detectors
    "TerrainCoverDetector",
    "MutualSupportDetector",
    "EngagementRangeDisciplineDetector",
    # Sea detectors
    "StandoffEngagementDetector",
    "ScreenFormationDetector",
    "EvasiveManeuverDetector",
    # Cross-domain detectors
    "AirGroundCoordinationDetector",
    "SEADSupportDetector",
    "MaritimePatrolDetector",
    # Math & helpers
    "compute_bearing",
    "angular_error",
    "extract_engagement_trajectories",
    # Configuration
    "PURSUIT_CURVE_ANGULAR_ERROR_MAX",
    "LEAD_PURSUIT_LEAD_DISTANCE_MIN",
    "LAG_PURSUIT_DISTANCE_MIN",
    "DEFENSIVE_BREAK_TURN_RATE",
    "ENERGY_TRADE_CORRELATION_MIN",
    "PINCER_BEARING_MIN",
    "PINCER_CLOSING_THRESHOLD",
    "TERRAIN_COVER_ELEVATION_PCT",
    "MUTUAL_SUPPORT_DISTANCE_MAX",
    "ENGAGEMENT_RANGE_DISCIPLINE",
    "STANDOFF_ENGAGEMENT_RANGE_PCT",
    "SCREEN_FORMATION_ANGLE_TOLERANCE",
    "EVASIVE_MANEUVER_TURN_RATE",
    "CAS_PROXIMITY_KM",
    "SEAD_PRECEDENCE_WINDOW_S",
    "MARITIME_PATROL_RADIUS_KM",
    "MIN_CONFIDENCE_TO_COUNT",
    "REALISM_ACCEPTANCE_THRESHOLD",
]
