"""Evaluation package for DRDO Tactical Scenario AI.

Provides:
- Non-Determinism Verification Module (statistical tests, entropy, clustering)
- Realism Validation Module (16 tactical doctrine detectors, realism scoring, reporting)
"""

from src.evaluation.behavioral_entropy import (
    compute_action_category_entropy,
    compute_policy_entropy_over_episode,
)
from src.evaluation.config import (
    BEHAVIORAL_ENTROPY_FACTOR,
    CV_THRESHOLD,
    ENTROPY_THRESHOLD,
    KMEANS_K,
    KMEANS_RANDOM_STATE,
    MIN_TRAJECTORY_CLUSTERS,
    PLOT_OUTPUT_DIR,
    REPORT_OUTPUT_DIR,
    SAMPLE_SIZE,
    SIGNIFICANCE_LEVEL,
)
from src.evaluation.doctrine import (
    AirGroundCoordinationDetector,
    DefensiveBreakDetector,
    DoctrineDetector,
    DoctrineRegistry,
    DoctrineResult,
    EnergyManagementDetector,
    EngagementRangeDisciplineDetector,
    EpisodeData,
    EvasiveManeuverDetector,
    LagPursuitDetector,
    LeadPursuitDetector,
    MaritimePatrolDetector,
    MutualSupportDetector,
    PincerManeuverDetector,
    PursuitCurveDetector,
    ScreenFormationDetector,
    SEADSupportDetector,
    StandoffEngagementDetector,
    TerrainCoverDetector,
    ThreatPrioritizationDetector,
)
from src.evaluation.entropy import (
    differential_entropy_gaussian,
    normalized_shannon_entropy,
    policy_entropy,
    shannon_entropy,
)
from src.evaluation.plots import (
    plot_all,
    plot_entropy_over_time,
    plot_outcome_distribution,
    plot_trajectory_clusters,
)
from src.evaluation.realism_report import (
    format_realism_json,
    format_realism_markdown,
    load_realism_report,
)
from src.evaluation.realism_scorer import RealismScorer
from src.evaluation.report import (
    compare_reports,
    format_as_json,
    format_as_markdown,
    load_report,
)
from src.evaluation.statistical_tests import (
    chi_square_goodness_of_fit,
    coefficient_of_variation,
    levene_variance_test,
    two_sample_ks_test,
)
from src.evaluation.trajectory_diversity import (
    cluster_trajectories,
    compute_diversity_score,
    extract_final_positions,
)
from src.evaluation.verifier import NonDeterminismVerifier

__all__ = [
    # Verifiers & Scorers
    "NonDeterminismVerifier",
    "RealismScorer",
    # Doctrine Interfaces & Registry
    "DoctrineRegistry",
    "DoctrineDetector",
    "DoctrineResult",
    "EpisodeData",
    # All 16 Doctrine Detectors
    "PursuitCurveDetector",
    "LeadPursuitDetector",
    "LagPursuitDetector",
    "DefensiveBreakDetector",
    "EnergyManagementDetector",
    "PincerManeuverDetector",
    "ThreatPrioritizationDetector",
    "TerrainCoverDetector",
    "MutualSupportDetector",
    "EngagementRangeDisciplineDetector",
    "StandoffEngagementDetector",
    "ScreenFormationDetector",
    "EvasiveManeuverDetector",
    "AirGroundCoordinationDetector",
    "SEADSupportDetector",
    "MaritimePatrolDetector",
    # Entropy
    "shannon_entropy",
    "normalized_shannon_entropy",
    "differential_entropy_gaussian",
    "policy_entropy",
    # Statistical tests
    "chi_square_goodness_of_fit",
    "levene_variance_test",
    "coefficient_of_variation",
    "two_sample_ks_test",
    # Trajectory diversity
    "extract_final_positions",
    "cluster_trajectories",
    "compute_diversity_score",
    # Behavioral entropy
    "compute_action_category_entropy",
    "compute_policy_entropy_over_episode",
    # Reporting & Plots
    "format_as_json",
    "format_as_markdown",
    "load_report",
    "compare_reports",
    "format_realism_json",
    "format_realism_markdown",
    "load_realism_report",
    "plot_all",
    "plot_outcome_distribution",
    "plot_trajectory_clusters",
    "plot_entropy_over_time",
    # Config
    "SAMPLE_SIZE",
    "SIGNIFICANCE_LEVEL",
    "ENTROPY_THRESHOLD",
    "CV_THRESHOLD",
    "MIN_TRAJECTORY_CLUSTERS",
    "BEHAVIORAL_ENTROPY_FACTOR",
    "KMEANS_K",
    "KMEANS_RANDOM_STATE",
    "REPORT_OUTPUT_DIR",
    "PLOT_OUTPUT_DIR",
]
