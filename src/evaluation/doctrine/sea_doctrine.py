"""Maritime / Naval tactical doctrine detectors.

Detects:
11. STANDOFF_ENGAGEMENT: Naval combatants engage at long ranges (> 60% weapon envelope) to preserve survivability.
12. SCREEN_FORMATION: Surface vessels maintain geometric defensive screen/line geometry.
13. EVASIVE_MANEUVER: Combatant ships initiate high-rate evasive rudder (> 0.5 rad/s) when under threat.
"""

from typing import Any
import numpy as np

from src.evaluation.doctrine.air_doctrine import angular_error, compute_bearing
from src.evaluation.doctrine.base_detector import DoctrineDetector, DoctrineResult, EpisodeData
from src.evaluation.doctrine.config import (
    EVASIVE_MANEUVER_TURN_RATE,
    MIN_CONFIDENCE_TO_COUNT,
    SCREEN_FORMATION_ANGLE_TOLERANCE,
    STANDOFF_ENGAGEMENT_RANGE_PCT,
)


class StandoffEngagementDetector(DoctrineDetector):
    """Detector for long-range naval standoff engagements."""

    name = "standoff_engagement"
    domain = "sea"
    description = "Naval vessels discharge anti-surface munitions while remaining at standoff distances (> 60% range)."

    def detect(self, episode_data: EpisodeData) -> DoctrineResult:
        standoff_shots = 0
        sea_shots = 0
        nominal_sea_range = 30.0  # Max anti-ship missile range in simulator

        for eng in episode_data.engagements:
            shooter = eng.get("shooter_id", "")
            if "sea" not in shooter:
                continue

            sea_shots += 1
            dist = float(eng.get("distance_km", 20.0))
            if dist >= nominal_sea_range * STANDOFF_ENGAGEMENT_RANGE_PCT:
                standoff_shots += 1

        if sea_shots == 0:
            # Fallback if no missile engagements: evaluate positioning relative to nearest threat
            blue_ships = [eid for eid in episode_data.entity_trajectories if "sea" in eid and "blue" in eid]
            red_ships = [eid for eid in episode_data.entity_trajectories if "sea" in eid and "red" in eid]

            if not blue_ships or not red_ships:
                return DoctrineResult(
                    doctrine_name=self.name,
                    detected=False,
                    confidence=0.0,
                    evidence={"reason": "No naval combatants present"},
                )

            dists = []
            for b in blue_ships:
                b_traj = episode_data.entity_trajectories[b]
                for r in red_ships:
                    r_traj = episode_data.entity_trajectories[r]
                    t_len = min(len(b_traj), len(r_traj))
                    for i in range(t_len):
                        d = np.linalg.norm(np.asarray(b_traj[i]["position"])[:2] - np.asarray(r_traj[i]["position"])[:2])
                        dists.append(d)

            mean_d = float(np.mean(dists)) if dists else nominal_sea_range * STANDOFF_ENGAGEMENT_RANGE_PCT
            conf = float(min(1.0, mean_d / (nominal_sea_range * STANDOFF_ENGAGEMENT_RANGE_PCT)))
            return DoctrineResult(
                doctrine_name=self.name,
                detected=conf >= MIN_CONFIDENCE_TO_COUNT,
                confidence=conf,
                evidence={"mean_separation_km": mean_d, "evaluated": "trajectory_proximity"},
            )

        confidence = float(standoff_shots / float(sea_shots))
        detected = bool(confidence >= MIN_CONFIDENCE_TO_COUNT)

        return DoctrineResult(
            doctrine_name=self.name,
            detected=detected,
            confidence=confidence,
            evidence={
                "standoff_shots": standoff_shots,
                "total_sea_shots": sea_shots,
                "standoff_ratio": confidence,
            },
        )


class ScreenFormationDetector(DoctrineDetector):
    """Detector for naval defensive screen and line-of-bearing formations."""

    name = "screen_formation"
    domain = "sea"
    description = "Surface ships position along a coordinated screening barrier or line formation."

    def detect(self, episode_data: EpisodeData) -> DoctrineResult:
        blue_ships = [eid for eid in episode_data.entity_trajectories if "sea" in eid and "blue" in eid]

        if len(blue_ships) < 2:
            return DoctrineResult(
                doctrine_name=self.name,
                detected=False,
                confidence=0.0,
                evidence={"reason": "Fewer than 2 friendly vessels present for screen formation"},
            )

        traj1 = episode_data.entity_trajectories[blue_ships[0]]
        traj2 = episode_data.entity_trajectories[blue_ships[1]]
        t_len = min(len(traj1), len(traj2))

        if t_len < 3:
            return DoctrineResult(
                doctrine_name=self.name,
                detected=False,
                confidence=0.0,
                evidence={"reason": "Insufficient maritime steps"},
            )

        bearing_diffs = []
        for i in range(t_len):
            h1 = traj1[i].get("heading", 0.0)
            h2 = traj2[i].get("heading", 0.0)
            diff = angular_error(h1, h2)
            bearing_diffs.append(diff)

        # In formation, ships maintain parallel headings or steady barrier geometry
        mean_diff = float(np.mean(bearing_diffs))
        std_diff = float(np.std(bearing_diffs))

        # Formation stability is characterized by low standard deviation of relative headings
        formation_stability = float(np.clip(1.0 - std_diff / 1.0, 0.0, 1.0))
        detected = bool(formation_stability >= MIN_CONFIDENCE_TO_COUNT)

        return DoctrineResult(
            doctrine_name=self.name,
            detected=detected,
            confidence=formation_stability,
            evidence={
                "relative_heading_std": std_diff,
                "formation_stability": formation_stability,
            },
        )


class EvasiveManeuverDetector(DoctrineDetector):
    """Detector for naval evasive maneuvers under inbound weapon/missile threat."""

    name = "evasive_maneuver"
    domain = "sea"
    description = "Naval vessels execute sharp rudder changes (> 0.5 rad/s) when in proximity to hostile assets."

    def detect(self, episode_data: EpisodeData) -> DoctrineResult:
        blue_ships = [eid for eid in episode_data.entity_trajectories if "sea" in eid and "blue" in eid]
        red_entities = [eid for eid in episode_data.entity_trajectories if "red" in eid]

        if not blue_ships:
            return DoctrineResult(
                doctrine_name=self.name,
                detected=False,
                confidence=0.0,
                evidence={"reason": "No friendly naval entities present"},
            )

        dt = float(getattr(episode_data.scenario_config, "dt", 0.1))
        dt = max(dt, 0.01)

        evasive_turns = 0
        threat_steps = 0
        max_rate = 0.0

        for b_id in blue_ships:
            b_traj = episode_data.entity_trajectories[b_id]
            for i in range(1, len(b_traj)):
                b_pos = np.asarray(b_traj[i]["position"])

                # Check if hostile within threat envelope (< 25 km)
                in_threat = False
                for r_id in red_entities:
                    r_traj = episode_data.entity_trajectories[r_id]
                    if i < len(r_traj):
                        r_pos = np.asarray(r_traj[i]["position"])
                        if np.linalg.norm(b_pos[:2] - r_pos[:2]) < 25.0:
                            in_threat = True
                            break

                if in_threat:
                    threat_steps += 1
                    h_curr = b_traj[i].get("heading", 0.0)
                    h_prev = b_traj[i - 1].get("heading", 0.0)
                    rate = angular_error(h_curr, h_prev) / dt
                    max_rate = max(max_rate, rate)
                    if rate >= EVASIVE_MANEUVER_TURN_RATE * 0.6:
                        evasive_turns += 1

        if threat_steps == 0:
            return DoctrineResult(
                doctrine_name=self.name,
                detected=False,
                confidence=0.0,
                evidence={"reason": "No active inbound threats encountered"},
            )

        confidence = float(min(1.0, max(evasive_turns / max(1, threat_steps) * 4.0, max_rate / EVASIVE_MANEUVER_TURN_RATE)))
        detected = bool(confidence >= MIN_CONFIDENCE_TO_COUNT)

        return DoctrineResult(
            doctrine_name=self.name,
            detected=detected,
            confidence=confidence,
            evidence={
                "evasive_turns": evasive_turns,
                "threat_steps": threat_steps,
                "max_turn_rate_rad_s": max_rate,
            },
        )
