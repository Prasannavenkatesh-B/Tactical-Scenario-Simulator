"""Trajectory diversity evaluation using spatial clustering and silhouette analysis.

Provides:
- extract_final_positions: Extracts final (x, y, z) coordinates per episode
- cluster_trajectories: Clusters spatial endpoints using K-Means
- compute_diversity_score: Computes multi-modal trajectory diversity metric and silhouette score
"""

from typing import Any, Sequence
import numpy as np
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from sklearn.metrics import silhouette_score  # type: ignore[import-untyped]

from src.evaluation.config import KMEANS_K, KMEANS_RANDOM_STATE


def extract_final_positions(
    trajectories: list[Any],
) -> np.ndarray:
    """Extract final (x, y, z) 3D coordinates for each episode trajectory.

    Args:
        trajectories: List of episodes, where each episode is a sequence of coordinate arrays.

    Returns:
        np.ndarray of shape (num_episodes, 3) with dtype float64.
    """
    if not trajectories:
        return np.empty((0, 3), dtype=np.float64)

    finals: list[list[float]] = []
    for ep in trajectories:
        if not ep:
            finals.append([0.0, 0.0, 0.0])
            continue

        last_pos = np.asarray(ep[-1], dtype=np.float64).flatten()
        if len(last_pos) == 0:
            finals.append([0.0, 0.0, 0.0])
        elif len(last_pos) == 1:
            finals.append([float(last_pos[0]), 0.0, 0.0])
        elif len(last_pos) == 2:
            finals.append([float(last_pos[0]), float(last_pos[1]), 0.0])
        else:
            finals.append([float(last_pos[0]), float(last_pos[1]), float(last_pos[2])])

    return np.asarray(finals, dtype=np.float64)


def cluster_trajectories(
    final_positions: np.ndarray,
    k: int = KMEANS_K,
    random_state: int = KMEANS_RANDOM_STATE,
) -> dict[str, Any]:
    """Perform k-means clustering on final 3D positions.

    Args:
        final_positions: Array of shape (num_episodes, 3).
        k: Target number of clusters (capped at num_episodes).
        random_state: Random state for reproducible clustering.

    Returns:
        dict containing:
            - 'n_clusters': int (number of clusters used)
            - 'labels': np.ndarray of cluster assignments
            - 'inertia': float (sum of squared distances to centroids)
            - 'distinct_clusters': int (clusters with >= 2 members)
    """
    pos = np.asarray(final_positions, dtype=np.float64)
    n_samples = len(pos)

    if n_samples == 0:
        return {
            "n_clusters": 0,
            "labels": np.array([], dtype=int),
            "inertia": 0.0,
            "distinct_clusters": 0,
        }

    if n_samples == 1:
        return {
            "n_clusters": 1,
            "labels": np.array([0], dtype=int),
            "inertia": 0.0,
            "distinct_clusters": 0,
        }

    effective_k = min(k, n_samples)
    
    # If all positions are identical, k-means might fail or yield 0 inertia
    if np.all(pos == pos[0]):
        return {
            "n_clusters": effective_k,
            "labels": np.zeros(n_samples, dtype=int),
            "inertia": 0.0,
            "distinct_clusters": 1 if n_samples >= 2 else 0,
        }

    kmeans = KMeans(n_clusters=effective_k, random_state=random_state, n_init="auto")
    kmeans.fit(pos)

    labels = kmeans.labels_
    counts = np.bincount(labels, minlength=effective_k)
    distinct_clusters = int(sum(1 for c in counts if c >= 2))

    return {
        "n_clusters": effective_k,
        "labels": labels,
        "inertia": float(kmeans.inertia_),
        "distinct_clusters": distinct_clusters,
    }


def compute_diversity_score(
    trajectories: list[Any],
    k: int = KMEANS_K,
) -> dict[str, Any]:
    """Execute end-to-end trajectory diversity evaluation.

    Pipeline:
        1. Extract final positions.
        2. Cluster endpoints via K-Means.
        3. Count distinct clusters (clusters with >= 2 members).
        4. Calculate silhouette score and normalized diversity score in [0, 1].

    Args:
        trajectories: List of episode trajectory coordinate sequences.
        k: Target cluster count.

    Returns:
        dict containing:
            - 'diversity_score': float in [0.0, 1.0]
            - 'n_clusters': int
            - 'distinct_clusters': int
            - 'labels': np.ndarray
            - 'silhouette_score': float
    """
    n_traj = len(trajectories)
    if n_traj <= 1:
        return {
            "diversity_score": 0.0,
            "n_clusters": n_traj,
            "distinct_clusters": 0,
            "labels": np.zeros(n_traj, dtype=int),
            "silhouette_score": 0.0,
        }

    final_positions = extract_final_positions(trajectories)
    cluster_res = cluster_trajectories(final_positions, k=k)

    labels = cluster_res["labels"]
    distinct = cluster_res["distinct_clusters"]
    num_unique_labels = len(np.unique(labels))

    # Calculate silhouette score if more than 1 cluster and fewer than n_samples
    sil_score = 0.0
    if 1 < num_unique_labels < len(final_positions):
        try:
            sil_score = float(silhouette_score(final_positions, labels))
        except Exception:
            sil_score = 0.0

    diversity = float(np.clip(distinct / max(1, k), 0.0, 1.0))

    return {
        "diversity_score": diversity,
        "n_clusters": cluster_res["n_clusters"],
        "distinct_clusters": distinct,
        "labels": labels,
        "silhouette_score": sil_score,
    }
