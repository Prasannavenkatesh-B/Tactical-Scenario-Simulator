"""Cross-domain joint tactical doctrine detectors.

Detects:
14. AIR_GROUND_COORDINATION: Tactical air provides close air support (CAS) within 5 km of engaged ground forces.
15. SEAD_SUPPORT: Air strikes suppress enemy SAM/radar defenses preceding or enabling ground ingress.
16. MARITIME_PATROL: Naval combatants patrol contested maritime borders (< 30 km) to maintain radar picture.
"""

from typing import Any
import numpy as np

from src.evaluation.doctrine.base_detector import DoctrineDetector, DoctrineResult, EpisodeData
from src.evaluation.doctrine.config import (
    CAS_PROXIMITY_KM,
    MARITIME_PATROL_RADIUS_KM,
    MIN_CONFIDENCE_TO_COUNT,
    SEAD_PRECEDENCE_WINDOW_S,
)


class AirGroundCoordinationDetector(DoctrineDetector):
    """Detector for Air-to-Ground Close Air Support (CAS) and synchronized strike coordination."""

    name = "air_ground_coordination"
    domain = "cross"
    description = "Air agents fly close support profiles (< 5 km) near enemy targets actively engaged by ground units."

    def detect(self, episode_data: EpisodeData) -> DoctrineResult:
        cas_events = 0
        ground_engagements = 0

        blue_airs = [eid for eid in episode_data.entity_trajectories if "air" in eid and "blue" in eid]
        blue_grounds = [eid for eid in episode_data.entity_trajectories if "ground" in eid and "blue" in eid]

        if not blue_airs or not blue_grounds:
            return DoctrineResult(
                doctrine_name=self.name,
                detected=False,
                confidence=0.0,
                evidence={"reason": "Missing air or ground components for cross-domain CAS"},
            )

        # 1. Analyze explicit ground engagements
        for eng in episode_data.engagements:
            shooter = eng.get("shooter_id", "")
            target = eng.get("target_id", "")
            t_step = eng.get("t", 0)

            if "ground" in shooter:
                ground_engagements += 1
                t_traj = episode_data.entity_trajectories.get(target, [])
                if not t_traj:
                    continue
                t_pos = np.asarray(t_traj[min(t_step, len(t_traj) - 1)]["position"])[:2]

                # Check if any blue air was within CAS proximity
                for b_air in blue_airs:
                    a_traj = episode_data.entity_trajectories[b_air]
                    if t_step < len(a_traj):
                        a_pos = np.asarray(a_traj[t_step]["position"])[:2]
                        if np.linalg.norm(a_pos - t_pos) <= CAS_PROXIMITY_KM * 2.0:
                            cas_events += 1
                            break

        # 2. If no engagements yet, analyze positional proximity to common adversary
        if ground_engagements == 0:
            red_entities = [eid for eid in episode_data.entity_trajectories if "red" in eid]
            if red_entities:
                for r_id in red_entities:
                    r_traj = episode_data.entity_trajectories[r_id]
                    t_len = len(r_traj)
                    for step in range(0, t_len, 2):
                        r_pos = np.asarray(r_traj[step]["position"])[:2]
                        # Check distance from blue ground and blue air
                        g_near = any(
                            np.linalg.norm(np.asarray(episode_data.entity_trajectories[g][step]["position"])[:2] - r_pos) < 25.0
                            for g in blue_grounds if step < len(episode_data.entity_trajectories[g])
                        )
                        a_near = any(
                            np.linalg.norm(np.asarray(episode_data.entity_trajectories[a][step]["position"])[:2] - r_pos) < CAS_PROXIMITY_KM * 2.5
                            for a in blue_airs if step < len(episode_data.entity_trajectories[a])
                        )
                        if g_near and a_near:
                            cas_events += 1
                            ground_engagements += 1

        if ground_engagements == 0:
            return DoctrineResult(
                doctrine_name=self.name,
                detected=False,
                confidence=0.0,
                evidence={"reason": "No ground combat engagements to support"},
            )

        confidence = float(min(1.0, (cas_events / max(1, ground_engagements)) * 1.5))
        detected = bool(confidence >= MIN_CONFIDENCE_TO_COUNT)

        return DoctrineResult(
            doctrine_name=self.name,
            detected=detected,
            confidence=confidence,
            evidence={
                "cas_events": cas_events,
                "ground_engagements": ground_engagements,
                "coordination_rate": confidence,
            },
        )


class SEADSupportDetector(DoctrineDetector):
    """Detector for Suppression of Enemy Air Defenses (SEAD) preceding tactical advance."""

    name = "sead_support"
    domain = "cross"
    description = "Air agents strike or suppress adversary air defense nodes preceding ground border ingress."

    def detect(self, episode_data: EpisodeData) -> DoctrineResult:
        sead_strikes = 0
        total_air_engagements = 0

        # SAM or ground defense target names
        for eng in episode_data.engagements:
            shooter = eng.get("shooter_id", "")
            target = eng.get("target_id", "")
            t_time = eng.get("t", 0.0)

            if "air" in shooter:
                total_air_engagements += 1
                # Check if target is a ground defense/SAM
                if "ground" in target or "sam" in target.lower():
                    sead_strikes += 1

        if total_air_engagements == 0:
            # Trajectory analysis: check if air penetrated into enemy zone early in mission
            map_size = float(episode_data.scenario_config.map_size_km)
            midpoint = map_size / 2.0

            early_penetrations = 0
            for eid, traj in episode_data.entity_trajectories.items():
                if "air" in eid and "blue" in eid:
                    for pt in traj[:15]:  # First 15 steps
                        x_pos = float(pt["position"][0])
                        if x_pos > midpoint:
                            early_penetrations += 1
                            break

            confidence = 0.75 if early_penetrations > 0 else 0.40
            return DoctrineResult(
                doctrine_name=self.name,
                detected=confidence >= MIN_CONFIDENCE_TO_COUNT,
                confidence=confidence,
                evidence={"early_air_penetrations": early_penetrations, "mode": "offensive_air_sweep"},
            )

        confidence = float(sead_strikes / float(total_air_engagements)) if total_air_engagements > 0 else 0.0
        detected = bool(confidence >= MIN_CONFIDENCE_TO_COUNT)

        return DoctrineResult(
            doctrine_name=self.name,
            detected=detected,
            confidence=confidence,
            evidence={
                "sead_strikes": sead_strikes,
                "total_air_engagements": total_air_engagements,
            },
        )


class MaritimePatrolDetector(DoctrineDetector):
    """Detector for persistent maritime surveillance and radar barrier patrol."""

    name = "maritime_patrol"
    domain = "cross"
    description = "Naval assets hold station along the maritime contested zone boundary (< 30 km)."

    def detect(self, episode_data: EpisodeData) -> DoctrineResult:
        blue_ships = [eid for eid in episode_data.entity_trajectories if "sea" in eid and "blue" in eid]

        if not blue_ships:
            return DoctrineResult(
                doctrine_name=self.name,
                detected=False,
                confidence=0.0,
                evidence={"reason": "No naval units present in scenario"},
            )

        map_size = float(episode_data.scenario_config.map_size_km)
        contested_line_y = map_size / 2.0  # Centerline boundary

        patrol_steps = 0
        total_steps = 0

        for b_id in blue_ships:
            traj = episode_data.entity_trajectories[b_id]
            for pt in traj:
                total_steps += 1
                y_pos = float(pt["position"][1])
                dist_to_boundary = abs(y_pos - contested_line_y)
                if dist_to_boundary <= MARITIME_PATROL_RADIUS_KM:
                    patrol_steps += 1

        if total_steps == 0:
            return DoctrineResult(
                doctrine_name=self.name,
                detected=False,
                confidence=0.0,
                evidence={"reason": "No maritime movement recorded"},
            )

        confidence = float(patrol_steps / float(total_steps))
        detected = bool(confidence >= MIN_CONFIDENCE_TO_COUNT)

        return DoctrineResult(
            doctrine_name=self.name,
            detected=detected,
            confidence=confidence,
            evidence={
                "patrol_steps": patrol_steps,
                "total_steps": total_steps,
                "corridor_dwell_ratio": confidence,
            },
        )
