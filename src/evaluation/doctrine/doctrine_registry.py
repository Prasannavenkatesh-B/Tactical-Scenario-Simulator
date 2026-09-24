"""Registry managing all 16 tactical doctrine detectors and result aggregation."""

from typing import Any

import numpy as np

from src.evaluation.doctrine.air_doctrine import (
    DefensiveBreakDetector,
    EnergyManagementDetector,
    LagPursuitDetector,
    LeadPursuitDetector,
    PincerManeuverDetector,
    PursuitCurveDetector,
    ThreatPrioritizationDetector,
)
from src.evaluation.doctrine.base_detector import DoctrineDetector, DoctrineResult, EpisodeData
from src.evaluation.doctrine.config import (
    DOCTRINE_PREVALENCE_THRESHOLD,
    MIN_CONFIDENCE_TO_COUNT,
    REALISM_ACCEPTANCE_THRESHOLD,
)
from src.evaluation.doctrine.cross_domain_doctrine import (
    AirGroundCoordinationDetector,
    MaritimePatrolDetector,
    SEADSupportDetector,
)
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


class DoctrineRegistry:
    """Central repository and execution engine for all 16 tactical doctrine detectors."""

    def __init__(self) -> None:
        self._detectors: dict[str, DoctrineDetector] = {}

        # Register Air Doctrines (1-7)
        self.register(PursuitCurveDetector())
        self.register(LeadPursuitDetector())
        self.register(LagPursuitDetector())
        self.register(DefensiveBreakDetector())
        self.register(EnergyManagementDetector())
        self.register(PincerManeuverDetector())
        self.register(ThreatPrioritizationDetector())

        # Register Ground Doctrines (8-10)
        self.register(TerrainCoverDetector())
        self.register(MutualSupportDetector())
        self.register(EngagementRangeDisciplineDetector())

        # Register Sea Doctrines (11-13)
        self.register(StandoffEngagementDetector())
        self.register(ScreenFormationDetector())
        self.register(EvasiveManeuverDetector())

        # Register Cross-Domain Doctrines (14-16)
        self.register(AirGroundCoordinationDetector())
        self.register(SEADSupportDetector())
        self.register(MaritimePatrolDetector())

    def register(self, detector: DoctrineDetector) -> None:
        """Register a new doctrine detector."""
        self._detectors[detector.name] = detector

    def all_detectors(self) -> list[DoctrineDetector]:
        """Return list of all registered detectors."""
        return list(self._detectors.values())

    def detect_all(self, episode_data: EpisodeData) -> list[DoctrineResult]:
        """Run all registered detectors on the provided episode telemetry."""
        results: list[DoctrineResult] = []
        for detector in self._detectors.values():
            try:
                res = detector.detect(episode_data)
            except Exception as e:
                res = DoctrineResult(
                    doctrine_name=detector.name,
                    detected=False,
                    confidence=0.0,
                    evidence={"error": str(e)},
                )
            results.append(res)
        return results

    def aggregate(
        self,
        results: list[DoctrineResult] | list[list[DoctrineResult]],
        num_episodes: int | None = None,
    ) -> dict[str, Any]:
        """Aggregate detector outputs into overall metrics, domain breakdowns, and verdict.

        Implements the two-level doctrinal evaluation system:
        Level 1: Per-episode presence requires raw_confidence >= MIN_CONFIDENCE_TO_COUNT (0.5).
        Level 2: Cross-episode acceptance requires prevalence >= DOCTRINE_PREVALENCE_THRESHOLD (0.20).
        """
        flat_results: list[DoctrineResult] = []
        if results and isinstance(results[0], list):
            for sublist in results:  # type: ignore[union-attr]
                flat_results.extend(sublist)
        else:
            flat_results = results  # type: ignore[assignment]

        if not flat_results:
            return {
                "total_doctrines": len(self._detectors),
                "detected_count": 0,
                "detection_rate": 0.0,
                "average_prevalence_detected": 0.0,
                "by_domain": {"air": 0.0, "ground": 0.0, "sea": 0.0, "cross": 0.0},
                "per_doctrine": {},
                "acceptance_verdict": "FAIL",
                "episodes_analyzed": 0,
            }

        # Group results by doctrine name
        results_by_doc: dict[str, list[DoctrineResult]] = {det.name: [] for det in self._detectors.values()}
        for r in flat_results:
            if r.doctrine_name in results_by_doc:
                results_by_doc[r.doctrine_name].append(r)
            else:
                results_by_doc[r.doctrine_name] = [r]

        # Determine total episodes N
        if num_episodes is not None and num_episodes > 0:
            N = num_episodes
        else:
            N = max(len(res_list) for res_list in results_by_doc.values()) if results_by_doc else 1
        N = max(1, N)

        total_doctrines = len(self._detectors)
        detected_count = 0
        per_doctrine: dict[str, dict[str, Any]] = {}
        domain_counts: dict[str, int] = {"air": 0, "ground": 0, "sea": 0, "cross": 0}
        domain_detected: dict[str, int] = {"air": 0, "ground": 0, "sea": 0, "cross": 0}
        prevalences_detected: list[float] = []

        for name, det in self._detectors.items():
            doc_res = results_by_doc.get(name, [])
            dom = det.domain
            domain_counts[dom] = domain_counts.get(dom, 0) + 1

            per_ep_conf = [float(r.confidence) for r in doc_res]

            # Level 1: per-episode detection requires raw_confidence >= MIN_CONFIDENCE_TO_COUNT (0.5)
            episodes_present = sum(1 for c in per_ep_conf if c >= MIN_CONFIDENCE_TO_COUNT)
            prevalence = float(episodes_present / float(N))

            # Level 2: overall doctrine detection requires prevalence >= DOCTRINE_PREVALENCE_THRESHOLD (0.20)
            is_detected = bool(prevalence >= DOCTRINE_PREVALENCE_THRESHOLD)

            conf_present = [c for c in per_ep_conf if c >= MIN_CONFIDENCE_TO_COUNT]
            mean_conf_present = float(np.mean(conf_present)) if conf_present else 0.0
            mean_conf_overall = float(np.mean(per_ep_conf)) if per_ep_conf else 0.0

            evidence_snippets: list[Any] = []
            for r in doc_res:
                if r.evidence and r.confidence >= MIN_CONFIDENCE_TO_COUNT and len(evidence_snippets) < 2:
                    evidence_snippets.append(r.evidence)

            if is_detected:
                detected_count += 1
                domain_detected[dom] = domain_detected.get(dom, 0) + 1
                prevalences_detected.append(prevalence)

            per_doctrine[name] = {
                "detected": is_detected,
                "prevalence": prevalence,
                "episodes_present": episodes_present,
                "total_episodes": N,
                "mean_confidence_when_present": mean_conf_present,
                "mean_confidence_overall": mean_conf_overall,
                "confidence": mean_conf_overall,  # backward compatibility
                "per_episode_confidence": per_ep_conf,
                "domain": dom,
                "description": det.description,
                "evidence_snippets": evidence_snippets,
            }

        detection_rate = float(detected_count / max(1, total_doctrines))
        avg_prevalence = float(np.mean(prevalences_detected)) if prevalences_detected else 0.0

        by_domain = {
            dom: float(domain_detected[dom] / max(1, domain_counts[dom]))
            for dom in domain_counts
        }

        verdict = "PASS" if detection_rate >= REALISM_ACCEPTANCE_THRESHOLD else "FAIL"

        return {
            "total_doctrines": total_doctrines,
            "detected_count": detected_count,
            "detection_rate": detection_rate,
            "average_prevalence_detected": avg_prevalence,
            "by_domain": by_domain,
            "per_doctrine": per_doctrine,
            "acceptance_verdict": verdict,
            "episodes_analyzed": N,
        }
