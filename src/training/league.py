"""League-based self-play management for tactical agents.

Maintains a pool of opponent checkpoint snapshots. Used for self-play
at curriculum levels L4 and L5 to avoid strategy collapse and non-transitivity.
"""

from pathlib import Path
from typing import Any
import numpy as np
import torch

from src.training.config import LEAGUE_DIR, LEAGUE_SIZE, SCRIPTED_PROB


class League:
    """Manages pool of opponent checkpoints for self-play training at L4+.

    Stores snapshots of policy weights and metadata. When capacity is reached,
    evicts the oldest snapshot (FIFO). Supports sampling with scripted fallback.
    """

    def __init__(
        self,
        max_size: int = LEAGUE_SIZE,
        save_dir: str = LEAGUE_DIR,
    ) -> None:
        self.max_size = max(1, max_size)
        self.save_dir = Path(save_dir)
        self.snapshots: list[dict[str, Any]] = []

    def add(self, policy_weights: dict[str, Any], iteration: int) -> None:
        """Add a new opponent snapshot to the league. Evict oldest if full.

        Args:
            policy_weights: Dictionary of policy state dicts (cloned/CPU detached).
            iteration: Training iteration at which snapshot was captured.
        """
        # Detach and move weights to CPU to avoid GPU memory leaks
        detached_weights: dict[str, Any] = {}
        for k, v in policy_weights.items():
            if isinstance(v, dict):
                detached_weights[k] = {
                    param_name: param.detach().cpu().clone() if isinstance(param, torch.Tensor) else param
                    for param_name, param in v.items()
                }
            elif isinstance(v, torch.Tensor):
                detached_weights[k] = v.detach().cpu().clone()
            else:
                detached_weights[k] = v

        entry = {
            "iteration": iteration,
            "weights": detached_weights,
        }

        self.snapshots.append(entry)
        while len(self.snapshots) > self.max_size:
            self.snapshots.pop(0)  # FIFO eviction

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, Any] | None:
        """Return a random snapshot's policy weights. Returns None if empty."""
        if not self.snapshots:
            return None
        generator = rng if rng is not None else np.random.default_rng()
        idx = int(generator.integers(0, len(self.snapshots)))
        return self.snapshots[idx]["weights"]

    def sample_with_scripted_fallback(
        self,
        rng: np.random.Generator | None = None,
        scripted_prob: float = SCRIPTED_PROB,
    ) -> tuple[str, dict[str, Any] | None]:
        """Sample opponent source: either scripted baseline or league snapshot.

        Args:
            rng: NumPy random generator.
            scripted_prob: Probability of selecting scripted opponent.

        Returns:
            Tuple of (source, weights):
            source in {"scripted", "league"}.
            If "scripted", weights is None.
        """
        generator = rng if rng is not None else np.random.default_rng()

        if not self.snapshots or float(generator.random()) < scripted_prob:
            return "scripted", None

        weights = self.sample(rng=generator)
        return "league", weights

    def size(self) -> int:
        """Return number of snapshots currently stored in the league."""
        return len(self.snapshots)

    def clear(self) -> None:
        """Clear all snapshots in the league."""
        self.snapshots.clear()

    def save(self, path: str | Path | None = None) -> str:
        """Save league pool snapshots to disk."""
        target_path = Path(path) if path is not None else (self.save_dir / "league_pool.pt")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "max_size": self.max_size,
            "snapshots": self.snapshots,
        }, target_path)
        return str(target_path)

    def load(self, path: str | Path | None = None) -> None:
        """Load league pool snapshots from disk."""
        target_path = Path(path) if path is not None else (self.save_dir / "league_pool.pt")
        if not target_path.exists():
            raise FileNotFoundError(f"League checkpoint not found at: {target_path}")

        data = torch.load(target_path, weights_only=False)
        self.max_size = int(data.get("max_size", self.max_size))
        self.snapshots = list(data.get("snapshots", []))
