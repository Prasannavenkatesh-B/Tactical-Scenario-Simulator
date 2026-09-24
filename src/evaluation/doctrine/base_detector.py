"""Base abstract detector and data structures for tactical doctrine evaluation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.simulator.scenarios import ScenarioConfig


@dataclass
class DoctrineResult:
    """Detection evaluation result for a single tactical doctrine pattern."""

    doctrine_name: str
    detected: bool
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)
    prevalence: float = 0.0
    episodes_present: int = 0
    total_episodes: int = 1
    mean_confidence_when_present: float = 0.0
    mean_confidence_overall: float = 0.0
    per_episode_confidence: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        if not self.per_episode_confidence and self.total_episodes == 1:
            self.per_episode_confidence = [self.confidence]
            self.mean_confidence_overall = self.confidence
            if self.detected:
                self.episodes_present = 1
                self.prevalence = 1.0
                self.mean_confidence_when_present = self.confidence


@dataclass
class EpisodeData:
    """Preprocessed trajectory, engagement, and decision records for an episode."""

    entity_trajectories: dict[str, list[dict[str, Any]]]
    engagements: list[dict[str, Any]]
    observations: dict[str, list[Any]]
    commander_decisions: list[dict[str, Any]]
    scenario_config: ScenarioConfig
    episode_length: int
    outcome: str = "draw"


class DoctrineDetector(ABC):
    """Abstract base class for all tactical doctrine pattern detectors."""

    name: str
    domain: str  # "air" | "ground" | "sea" | "cross"
    description: str

    @abstractmethod
    def detect(self, episode_data: EpisodeData) -> DoctrineResult:
        """Analyze episode telemetry and return detection result with confidence in [0, 1]."""
        raise NotImplementedError

    def score(self, episode_data: EpisodeData) -> float:
        """Calculate and return confidence score in [0, 1]."""
        return self.detect(episode_data).confidence
