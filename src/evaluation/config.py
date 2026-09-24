"""Configuration constants and thresholds for the Non-Determinism Verification Module.

These thresholds define the statistical criteria mandated by DRDO for acceptance:
- Action entropy > 0.5 (normalized) ensures non-degenerate policy exploration.
- Outcome distribution χ² test p < 0.05 rejects deterministic simulation.
- Levene's variance test p < 0.05 proves dispersion across seed groups.
- Trajectory clustering requires >= 3 distinct final state clusters.
- Same-seed reproducibility requires bit-identical execution.
"""

from typing import Final

# Sample Sizes & Significance
SAMPLE_SIZE: Final[int] = 100
SIGNIFICANCE_LEVEL: Final[float] = 0.05

# Statistical Thresholds
ENTROPY_THRESHOLD: Final[float] = 0.5          # Normalized Shannon entropy [0, 1]
CV_THRESHOLD: Final[float] = 0.10               # Coefficient of Variation (σ / μ)
MIN_TRAJECTORY_CLUSTERS: Final[int] = 3        # Minimum distinct clusters (>= 2 members)
BEHAVIORAL_ENTROPY_FACTOR: Final[float] = 0.3   # Fraction of log(action_dim) for action sequence

# Clustering Configuration
KMEANS_K: Final[int] = 5
KMEANS_RANDOM_STATE: Final[int] = 42

# Output Paths
REPORT_OUTPUT_DIR: Final[str] = "reports/non_determinism"
PLOT_OUTPUT_DIR: Final[str] = "reports/non_determinism/plots"
