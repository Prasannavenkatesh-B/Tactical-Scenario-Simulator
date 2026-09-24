"""Ground combat tactical doctrine detectors.

Detects:
8. TERRAIN_COVER: Ground units utilize elevated terrain contours for tactical masking.
9. MUTUAL_SUPPORT: Friendly ground units maintain cohesive spacing within supporting distance (< 15 km).
10. ENGAGEMENT_RANGE_DISCIPLINE: Weapon releases occur within effective tactical envelope (< 80% max range).
"""

from typing import Any
import numpy as np

from src.evaluation.doctrine.base_detector import DoctrineDetector, DoctrineResult, EpisodeData
from src.evaluation.doctrine.config import (
    ENGAGEMENT_RANGE_DISCIPLINE,
    MIN_CONFIDENCE_TO_COUNT,
    MUTUAL_SUPPORT_DISTANCE_MAX,
    TERRAIN_COVER_ELEVATION_PCT,
)


class TerrainCoverDetector(DoctrineDetector):
    """Detector for tactical terrain masking and elevated vantage utilization."""

    name = "terrain_cover"
    domain = "ground"
    description = "Ground units navigate to elevated or masked terrain features when within threat sensor range."

    def detect(self, episode_data: EpisodeData) -> DoctrineResult:
        cover_steps = 0
        threat_steps = 0

        blue_grounds = [eid for eid in episode_data.entity_trajectories if "ground" in eid and "blue" in eid]
        red_entities = [eid for eid in episode_data.entity_trajectories if "red" in eid]

        if not blue_grounds:
            return DoctrineResult(
                doctrine_name=self.name,
                detected=False,
                confidence=0.0,
                evidence={"reason": "No friendly ground units present"},
            )

        for bg_id in blue_grounds:
            bg_traj = episode_data.entity_trajectories[bg_id]
            for pt in bg_traj:
                bg_pos = np.asarray(pt["position"])
                t_idx = pt.get("t", 0)

                # Check if enemy is within sensor horizon (20 km)
                enemy_near = False
                for r_id in red_entities:
                    r_traj = episode_data.entity_trajectories[r_id]
                    if t_idx < len(r_traj):
                        r_pos = np.asarray(r_traj[t_idx]["position"])
                        if np.linalg.norm(bg_pos[:2] - r_pos[:2]) < 20.0:
                            enemy_near = True
                            break

                if enemy_near:
                    threat_steps += 1
                    # Ground elevation is stored in position[2] or terrain map
                    elevation = float(bg_pos[2]) if len(bg_pos) > 2 else 0.0
                    # Standard scenario elevation threshold for defensible high ground
                    if elevation >= TERRAIN_COVER_ELEVATION_PCT * 0.5 or elevation > 0.1:
                        cover_steps += 1

        if threat_steps == 0:
            return DoctrineResult(
                doctrine_name=self.name,
                detected=False,
                confidence=0.0,
                evidence={"reason": "No threat encounters observed for terrain cover evaluation"},
            )

        confidence = float(cover_steps / float(threat_steps))
        detected = bool(confidence >= MIN_CONFIDENCE_TO_COUNT)

        return DoctrineResult(
            doctrine_name=self.name,
            detected=detected,
            confidence=confidence,
            evidence={
                "cover_steps": cover_steps,
                "threat_steps": threat_steps,
                "cover_ratio": confidence,
            },
        )


class MutualSupportDetector(DoctrineDetector):
    """Detector for mutual supporting distance maintenance between friendly ground forces."""

    name = "mutual_support"
    domain = "ground"
    description = "Friendly ground units remain within mutual defense and support spacing (< 15 km)."

    def detect(self, episode_data: EpisodeData) -> DoctrineResult:
        blue_grounds = [eid for eid in episode_data.entity_trajectories if "ground" in eid and "blue" in eid]

        if len(blue_grounds) < 2:
            return DoctrineResult(
                doctrine_name=self.name,
                detected=False,
                confidence=0.0,
                evidence={"reason": "Fewer than 2 friendly ground units present for mutual support"},
            )

        supported_steps = 0
        total_steps = 0

        # Sample trajectories across timesteps
        traj1 = episode_data.entity_trajectories[blue_grounds[0]]
        traj2 = episode_data.entity_trajectories[blue_grounds[1]]
        t_max = min(len(traj1), len(traj2))

        for i in range(t_max):
            p1 = np.asarray(traj1[i]["position"])[:2]
            p2 = np.asarray(traj2[i]["position"])[:2]
            dist = float(np.linalg.norm(p1 - p2))
            total_steps += 1
            if dist <= MUTUAL_SUPPORT_DISTANCE_MAX:
                supported_steps += 1

        if total_steps == 0:
            return DoctrineResult(
                doctrine_name=self.name,
                detected=False,
                confidence=0.0,
                evidence={"reason": "No ground movement steps recorded"},
            )

        confidence = float(supported_steps / float(total_steps))
        detected = bool(confidence >= MIN_CONFIDENCE_TO_COUNT)

        return DoctrineResult(
            doctrine_name=self.name,
            detected=detected,
            confidence=confidence,
            evidence={
                "supported_steps": supported_steps,
                "total_steps": total_steps,
                "support_ratio": confidence,
            },
        )


class EngagementRangeDisciplineDetector(DoctrineDetector):
    """Detector for tactical firing range discipline (firing within effective envelope)."""

    name = "engagement_range_discipline"
    domain = "ground"
    description = "Ground weapons are discharged within effective engagement distance (< 80% maximum range)."

    def detect(self, episode_data: EpisodeData) -> DoctrineResult:
        disciplined_shots = 0
        ground_shots = 0

        # Max range for ground weapons in simulator config (SAM=20km, cannon=8km)
        nominal_max_range = 15.0

        for eng in episode_data.engagements:
            shooter = eng.get("shooter_id", "")
            if "ground" not in shooter:
                continue

            ground_shots += 1
            dist = float(eng.get("distance_km", 8.0))
            if dist <= nominal_max_range * ENGAGEMENT_RANGE_DISCIPLINE:
                disciplined_shots += 1

        if ground_shots == 0:
            return DoctrineResult(
                doctrine_name=self.name,
                detected=False,
                confidence=0.0,
                evidence={"reason": "No ground weapon discharges observed"},
            )

        confidence = float(disciplined_shots / float(ground_shots))
        detected = bool(confidence >= MIN_CONFIDENCE_TO_COUNT)

        return DoctrineResult(
            doctrine_name=self.name,
            detected=detected,
            confidence=confidence,
            evidence={
                "disciplined_shots": disciplined_shots,
                "total_ground_shots": ground_shots,
                "discipline_rate": confidence,
            },
        )
