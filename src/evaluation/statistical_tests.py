"""Rigorous statistical testing functions for non-determinism verification.

Uses scipy.stats to calculate exact statistics and p-values:
- chi_square_goodness_of_fit: Outcome distribution uniformity test (χ²)
- levene_variance_test: Cross-seed variance equality test (Levene's W)
- coefficient_of_variation: Dispersion metric (CV = σ / μ)
- two_sample_ks_test: Two-sample Kolmogorov-Smirnov test for distribution equivalence
"""

from typing import Any
import numpy as np
import scipy.stats  # type: ignore[import-untyped]

from src.evaluation.config import SIGNIFICANCE_LEVEL


def chi_square_goodness_of_fit(
    observed_counts: np.ndarray | list[int] | list[float],
) -> dict[str, Any]:
    """Test whether observed outcome distribution differs significantly from uniform.

    Null hypothesis H0: Outcomes are uniformly distributed across categories.
    Alternative hypothesis H1: Outcomes deviate from uniform distribution.

    Args:
        observed_counts: Array or list of counts per category (e.g. [wins, losses, draws]).

    Returns:
        dict containing:
            - 'chi2': float (test statistic)
            - 'df': int (degrees of freedom, k - 1)
            - 'p_value': float (exact p-value from scipy.stats.chisquare)
            - 'reject_null': bool (True if p_value < SIGNIFICANCE_LEVEL)
    """
    counts = np.asarray(observed_counts, dtype=np.float64).flatten()
    k = len(counts)
    if k <= 1:
        return {
            "chi2": 0.0,
            "df": 0,
            "p_value": 1.0,
            "reject_null": False,
        }

    total = np.sum(counts)
    if total <= 0:
        return {
            "chi2": 0.0,
            "df": k - 1,
            "p_value": 1.0,
            "reject_null": False,
        }

    # If all categories have the exact same count, chi2 is 0.0 and p_value is 1.0
    if np.all(counts == counts[0]):
        return {
            "chi2": 0.0,
            "df": k - 1,
            "p_value": 1.0,
            "reject_null": False,
        }

    res = scipy.stats.chisquare(f_obs=counts)
    chi2_stat = float(res.statistic)
    p_val = float(res.pvalue)

    return {
        "chi2": chi2_stat,
        "df": k - 1,
        "p_value": p_val,
        "reject_null": bool(p_val < SIGNIFICANCE_LEVEL),
    }


def levene_variance_test(
    samples: list[np.ndarray | list[float]],
) -> dict[str, Any]:
    """Levene's test for equality of variances across groups (e.g., across seeds).

    Robust to non-normality (uses median centering).

    Args:
        samples: List of arrays containing metric values for each group/seed.

    Returns:
        dict containing:
            - 'W': float (Levene's W statistic)
            - 'p_value': float (exact p-value)
            - 'reject_null': bool (True if variances differ significantly)
    """
    if len(samples) < 2:
        return {
            "W": 0.0,
            "p_value": 1.0,
            "reject_null": False,
        }

    clean_samples = [np.asarray(s, dtype=np.float64).flatten() for s in samples]
    # Check if all samples have at least 2 points
    if any(len(s) < 2 for s in clean_samples):
        # Fall back if sample size per group is too small
        return {
            "W": 0.0,
            "p_value": 1.0,
            "reject_null": False,
        }

    # Check if all variances are zero (e.g. same seed produces identical results)
    variances = [float(np.var(s)) for s in clean_samples]
    if all(abs(v) < 1e-12 for v in variances):
        return {
            "W": 0.0,
            "p_value": 1.0,
            "reject_null": False,
        }

    try:
        stat, p_val = scipy.stats.levene(*clean_samples, center="median")
        if np.isnan(stat) or np.isnan(p_val):
            return {"W": 0.0, "p_value": 1.0, "reject_null": False}
        return {
            "W": float(stat),
            "p_value": float(p_val),
            "reject_null": bool(p_val < SIGNIFICANCE_LEVEL),
        }
    except Exception:
        return {
            "W": 0.0,
            "p_value": 1.0,
            "reject_null": False,
        }


def coefficient_of_variation(values: np.ndarray | list[float]) -> float:
    """Compute the Coefficient of Variation: CV = σ / μ.

    Args:
        values: Array or list of numerical values.

    Returns:
        CV value (float >= 0.0). Returns 0.0 if mean is 0 or array is empty.
    """
    arr = np.asarray(values, dtype=np.float64).flatten()
    if arr.size == 0:
        return 0.0

    mean_val = float(np.mean(arr))
    if abs(mean_val) < 1e-12:
        return 0.0

    std_val = float(np.std(arr))
    return float(std_val / abs(mean_val))


def two_sample_ks_test(
    sample1: np.ndarray | list[float],
    sample2: np.ndarray | list[float],
) -> dict[str, Any]:
    """Two-sample Kolmogorov-Smirnov test for whether two distributions differ.

    Null hypothesis H0: The two samples are drawn from the same continuous distribution.

    Args:
        sample1: First sample array.
        sample2: Second sample array.

    Returns:
        dict containing:
            - 'statistic': float (KS statistic D)
            - 'p_value': float (exact p-value)
            - 'reject_null': bool (True if distributions differ significantly)
    """
    s1 = np.asarray(sample1, dtype=np.float64).flatten()
    s2 = np.asarray(sample2, dtype=np.float64).flatten()

    if s1.size == 0 or s2.size == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "reject_null": False,
        }

    res = scipy.stats.ks_2samp(s1, s2)
    stat = float(res.statistic)
    p_val = float(res.pvalue)

    return {
        "statistic": stat,
        "p_value": p_val,
        "reject_null": bool(p_val < SIGNIFICANCE_LEVEL),
    }
