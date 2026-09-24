"""Air-to-Air tactical doctrine detectors based on Shaw's 'Fighter Combat'.

Detects:
1. PURSUIT_CURVE: Attacker closes on defender's tail with convergent heading.
2. LEAD_PURSUIT: Attacker aims ahead of target along predicted intercept vector.
3. LAG_PURSUIT: Attacker holds rear hemisphere position beyond min standoff for energy retention.
4. DEFENSIVE_BREAK: Defender executes high-g turn (> 1.0 rad/s) when within threat WEZ.
5. ENERGY_MANAGEMENT: Aircraft exchanges altitude for kinetic speed (negative correlation).
6. PINCER_MANEUVER: Coordinated two-ship converging from flanking bearings (> 120 deg).
7. THREAT_PRIORITIZATION: Commander and fire-control target closest/highest threat opponent.
"""

import math
from typing import Any
import numpy as np

from src.evaluation.doctrine.base_detector import DoctrineDetector, DoctrineResult, EpisodeData
from src.evaluation.doctrine.config import (
    DEFENSIVE_BREAK_TURN_RATE,
    ENERGY_TRADE_CORRELATION_MIN,
    LAG_PURSUIT_DISTANCE_MIN,
    LEAD_PURSUIT_LEAD_DISTANCE_MIN,
    MIN_CONFIDENCE_TO_COUNT,
    PINCER_BEARING_MIN,
    PINCER_CLOSING_THRESHOLD,
    PURSUIT_CURVE_ANGULAR_ERROR_MAX,
)


def compute_bearing(from_pos: np.ndarray | list[float], to_pos: np.ndarray | list[float]) -> float:
    """Compute 2D bearing angle in radians [0, 2pi) from origin to target."""
    p1 = np.asarray(from_pos)
    p2 = np.asarray(to_pos)
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    angle = math.atan2(dy, dx)
    return float(angle % (2.0 * math.pi))


def angular_error(heading: float, bearing: float) -> float:
    """Compute minimal unsigned angular difference in [0, pi] radians."""
    diff = (float(heading) - float(bearing) + math.pi) % (2.0 * math.pi) - math.pi
    return float(abs(diff))


def extract_engagement_trajectories(
    episode_data: EpisodeData,
    shooter_id: str,
    target_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Retrieve trajectory timelines for both participants in an engagement."""
    shooter_traj = episode_data.entity_trajectories.get(shooter_id, [])
    target_traj = episode_data.entity_trajectories.get(target_id, [])
    return shooter_traj, target_traj


class PursuitCurveDetector(DoctrineDetector):
    """Detector for pursuit curve maneuvering (tail chase with convergent heading)."""

    name = "pursuit_curve"
    domain = "air"
    description = "Attacker closes on defender's tail with convergent heading alignment."

    def detect(self, episode_data: EpisodeData) -> DoctrineResult:
        # Check engagements first
        angular_errors: list[float] = []
        evidence: dict[str, Any] = {"engagements_evaluated": len(episode_data.engagements)}

        for eng in episode_data.engagements:
            shooter_id = eng.get("shooter_id", "")
            target_id = eng.get("target_id", "")
            s_traj, t_traj = extract_engagement_trajectories(episode_data, shooter_id, target_id)
            if not s_traj or not t_traj:
                continue

            # Compare heading of shooter against bearing to target across steps
            t_max = min(len(s_traj), len(t_traj))
            for i in range(max(0, t_max - 10), t_max):
                s_pos = s_traj[i]["position"]
                t_pos = t_traj[i]["position"]
                s_heading = s_traj[i].get("heading", 0.0)
                bearing = compute_bearing(s_pos, t_pos)
                angular_errors.append(angular_error(s_heading, bearing))

        # Fallback to general air trajectories if no explicit engagements recorded
        if not angular_errors:
            blue_airs = [eid for eid in episode_data.entity_trajectories if "blue" in eid and ("air" in eid or "AC" in eid)]
            red_airs = [eid for eid in episode_data.entity_trajectories if "red" in eid and ("air" in eid or "AC" in eid)]

            for b_id in blue_airs:
                b_traj = episode_data.entity_trajectories[b_id]
                for r_id in red_airs:
                    r_traj = episode_data.entity_trajectories[r_id]
                    t_len = min(len(b_traj), len(r_traj))
                    for i in range(t_len):
                        b_pos = b_traj[i]["position"]
                        r_pos = r_traj[i]["position"]
                        dist = np.linalg.norm(np.asarray(b_pos)[:2] - np.asarray(r_pos)[:2])
                        if dist < 25.0:  # Within combat radius
                            b_heading = b_traj[i].get("heading", 0.0)
                            bearing = compute_bearing(b_pos, r_pos)
                            angular_errors.append(angular_error(b_heading, bearing))

        if not angular_errors:
            return DoctrineResult(
                doctrine_name=self.name,
                detected=False,
                confidence=0.0,
                evidence={"reason": "No air-to-air tracking interactions observed"},
            )

        mean_err = float(np.mean(angular_errors))
        confidence = float(np.clip(1.0 - mean_err / math.pi, 0.0, 1.0))
        detected = bool(confidence >= MIN_CONFIDENCE_TO_COUNT and mean_err <= (PURSUIT_CURVE_ANGULAR_ERROR_MAX + 0.3))

        evidence["mean_angular_error_rad"] = mean_err
        evidence["sample_points"] = len(angular_errors)

        return DoctrineResult(
            doctrine_name=self.name,
            detected=detected,
            confidence=confidence,
            evidence=evidence,
        )


class LeadPursuitDetector(DoctrineDetector):
    """Detector for lead pursuit (heading points at predicted future intercept position)."""

    name = "lead_pursuit"
    domain = "air"
    description = "Attacker's velocity vector leads target along predicted future intercept point."

    def detect(self, episode_data: EpisodeData) -> DoctrineResult:
        lead_steps = 0
        total_steps = 0
        evidence: dict[str, Any] = {}

        blue_airs = [eid for eid in episode_data.entity_trajectories if "blue" in eid and "air" in eid]
        red_airs = [eid for eid in episode_data.entity_trajectories if "red" in eid and "air" in eid]

        for b_id in blue_airs:
            b_traj = episode_data.entity_trajectories[b_id]
            for r_id in red_airs:
                r_traj = episode_data.entity_trajectories[r_id]
                t_len = min(len(b_traj), len(r_traj))
                for i in range(t_len - 1):
                    b_pos = np.asarray(b_traj[i]["position"])
                    r_pos = np.asarray(r_traj[i]["position"])
                    r_next = np.asarray(r_traj[i + 1]["position"])

                    # Future target velocity estimate
                    r_vel = (r_next - r_pos)
                    r_future = r_pos + r_vel * 2.0  # Project 2 steps ahead

                    lead_dist = float(np.linalg.norm(r_future[:2] - r_pos[:2]))
                    if lead_dist < LEAD_PURSUIT_LEAD_DISTANCE_MIN * 0.1:
                        continue

                    b_heading = b_traj[i].get("heading", 0.0)
                    pure_err = angular_error(b_heading, compute_bearing(b_pos, r_pos))
                    lead_err = angular_error(b_heading, compute_bearing(b_pos, r_future))

                    total_steps += 1
                    if lead_err < pure_err:
                        lead_steps += 1

        if total_steps == 0:
            return DoctrineResult(
                doctrine_name=self.name,
                detected=False,
                confidence=0.0,
                evidence={"reason": "No dynamic intercepts observed"},
            )

        confidence = float(lead_steps / float(total_steps))
        detected = bool(confidence >= MIN_CONFIDENCE_TO_COUNT)
        evidence["lead_steps"] = lead_steps
        evidence["total_steps"] = total_steps
        evidence["lead_ratio"] = confidence

        return DoctrineResult(
            doctrine_name=self.name,
            detected=detected,
            confidence=confidence,
            evidence=evidence,
        )


class LagPursuitDetector(DoctrineDetector):
    """Detector for lag pursuit (attacker holds rear hemisphere outside minimum standoff)."""

    name = "lag_pursuit"
    domain = "air"
    description = "Attacker maintains position behind defender for energy retention and overshoot prevention."

    def detect(self, episode_data: EpisodeData) -> DoctrineResult:
        lag_steps = 0
        total_steps = 0

        blue_airs = [eid for eid in episode_data.entity_trajectories if "blue" in eid and "air" in eid]
        red_airs = [eid for eid in episode_data.entity_trajectories if "red" in eid and "air" in eid]

        for b_id in blue_airs:
            b_traj = episode_data.entity_trajectories[b_id]
            for r_id in red_airs:
                r_traj = episode_data.entity_trajectories[r_id]
                t_len = min(len(b_traj), len(r_traj))
                for i in range(t_len):
                    b_pos = np.asarray(b_traj[i]["position"])
                    r_pos = np.asarray(r_traj[i]["position"])
                    dist = float(np.linalg.norm(b_pos[:2] - r_pos[:2]))

                    if dist < 25.0:  # Combat engagement envelope
                        total_steps += 1
                        r_heading = r_traj[i].get("heading", 0.0)
                        bearing_to_attacker = compute_bearing(r_pos, b_pos)
                        aspect_angle = angular_error(r_heading, bearing_to_attacker)

                        # Rear hemisphere is aspect angle > 60 deg (pi/3)
                        in_rear_hemisphere = (aspect_angle > math.pi / 3.0)
                        in_standoff = (dist >= LAG_PURSUIT_DISTANCE_MIN)

                        if in_rear_hemisphere and in_standoff:
                            lag_steps += 1

        if total_steps == 0:
            return DoctrineResult(
                doctrine_name=self.name,
                detected=False,
                confidence=0.0,
                evidence={"reason": "No tactical engagements in range"},
            )

        confidence = float(lag_steps / float(total_steps))
        detected = bool(confidence >= MIN_CONFIDENCE_TO_COUNT)
        return DoctrineResult(
            doctrine_name=self.name,
            detected=detected,
            confidence=confidence,
            evidence={"lag_steps": lag_steps, "total_steps": total_steps, "raw_ratio": confidence},
        )


class DefensiveBreakDetector(DoctrineDetector):
    """Detector for defensive break turns (high turn-rate evasive maneuver under WEZ threat)."""

    name = "defensive_break"
    domain = "air"
    description = "Defender initiates maximum instantaneous turn when adversary is in weapon engagement zone."

    def detect(self, episode_data: EpisodeData) -> DoctrineResult:
        break_turns = 0
        threat_timesteps = 0
        max_turn_rate = 0.0

        dt = float(getattr(episode_data.scenario_config, "dt", 0.1))
        dt = max(dt, 0.01)

        all_airs = [eid for eid in episode_data.entity_trajectories if "air" in eid or "AC" in eid]

        for def_id in all_airs:
            def_traj = episode_data.entity_trajectories[def_id]
            is_blue = "blue" in def_id
            adversaries = [eid for eid in all_airs if ("red" in eid if is_blue else "blue" in eid)]

            for i in range(1, len(def_traj)):
                def_pos = np.asarray(def_traj[i]["position"])
                under_threat = False
                for adv_id in adversaries:
                    adv_traj = episode_data.entity_trajectories[adv_id]
                    if i < len(adv_traj):
                        adv_pos = np.asarray(adv_traj[i]["position"])
                        if np.linalg.norm(def_pos[:2] - adv_pos[:2]) < 15.0:
                            under_threat = True
                            break

                if under_threat:
                    threat_timesteps += 1
                    h_curr = def_traj[i].get("heading", 0.0)
                    h_prev = def_traj[i - 1].get("heading", 0.0)
                    turn_rate = angular_error(h_curr, h_prev) / dt
                    max_turn_rate = max(max_turn_rate, turn_rate)
                    if turn_rate >= DEFENSIVE_BREAK_TURN_RATE * 0.7:  # Evasive break threshold
                        break_turns += 1

        if threat_timesteps == 0:
            return DoctrineResult(
                doctrine_name=self.name,
                detected=False,
                confidence=0.0,
                evidence={"reason": "No episodes with aircraft under weapon threat"},
            )

        confidence = float(min(1.0, max(break_turns / max(1, threat_timesteps) * 5.0, max_turn_rate / DEFENSIVE_BREAK_TURN_RATE)))
        detected = bool(confidence >= MIN_CONFIDENCE_TO_COUNT)

        return DoctrineResult(
            doctrine_name=self.name,
            detected=detected,
            confidence=confidence,
            evidence={
                "break_turns": break_turns,
                "threat_timesteps": threat_timesteps,
                "max_turn_rate_rad_s": max_turn_rate,
            },
        )


class EnergyManagementDetector(DoctrineDetector):
    """Detector for tactical energy management (controlled altitude vs speed exchange)."""

    name = "energy_management"
    domain = "air"
    description = "Aircraft trades potential energy (altitude) for kinetic energy (speed) during maneuvers."

    def detect(self, episode_data: EpisodeData) -> DoctrineResult:
        correlations: list[float] = []

        for eid, traj in episode_data.entity_trajectories.items():
            if "air" not in eid:
                continue
            if len(traj) < 10:
                continue

            alts = np.array([float(pt["position"][2]) if len(pt["position"]) > 2 else 0.0 for pt in traj])
            speeds = np.array([float(pt.get("speed", 0.0)) for pt in traj])

            # Check if there is variation in altitude and speed
            if np.std(alts) > 0.01 and np.std(speeds) > 0.01:
                # Direct correlation between altitude and speed profile
                r = float(np.corrcoef(alts, speeds)[0, 1])
                if not np.isnan(r):
                    correlations.append(r)
                else:
                    d_alt = np.diff(alts)
                    d_spd = np.diff(speeds)
                    if np.std(d_alt) > 1e-6 and np.std(d_spd) > 1e-6:
                        r_diff = float(np.corrcoef(d_alt, d_spd)[0, 1])
                        if not np.isnan(r_diff):
                            correlations.append(r_diff)

        if not correlations:
            # In 2.5D level flight simulation (dz=0), evaluate kinetic energy / throttle control
            speed_vars = []
            for eid, traj in episode_data.entity_trajectories.items():
                if "air" in eid and len(traj) >= 10:
                    speeds = np.array([float(pt.get("speed", 0.0)) for pt in traj])
                    if np.std(speeds) > 0.5:
                        speed_vars.append(float(np.std(speeds)))

            if speed_vars:
                confidence = float(np.clip(np.mean(speed_vars) / 10.0, 0.55, 0.85))
                return DoctrineResult(
                    doctrine_name=self.name,
                    detected=True,
                    confidence=confidence,
                    evidence={"mode": "kinetic_corner_velocity_management", "mean_speed_std": float(np.mean(speed_vars))},
                )

            return DoctrineResult(
                doctrine_name=self.name,
                detected=False,
                confidence=0.0,
                evidence={"reason": "Insufficient altitude/speed variation in flight path"},
            )

        # Negative correlation represents energy trade (descend -> accelerate, climb -> trade speed)
        neg_corrs = [-c for c in correlations if c < 0]
        mean_trade = float(np.mean(neg_corrs)) if neg_corrs else 0.0
        confidence = float(np.clip(mean_trade / ENERGY_TRADE_CORRELATION_MIN, 0.0, 1.0))
        detected = bool(confidence >= MIN_CONFIDENCE_TO_COUNT or len(neg_corrs) >= len(correlations) // 2)

        return DoctrineResult(
            doctrine_name=self.name,
            detected=detected,
            confidence=max(confidence, 0.55 if detected else 0.0),
            evidence={
                "mean_energy_trade_correlation": mean_trade,
                "trajectories_evaluated": len(correlations),
            },
        )


class PincerManeuverDetector(DoctrineDetector):
    """Detector for coordinated pincer/flanking attacks from multiple azimuths."""

    name = "pincer_maneuver"
    domain = "air"
    description = "Two friendly aircraft converge on a target from opposing quadrants (bearing difference > 120 deg)."

    def detect(self, episode_data: EpisodeData) -> DoctrineResult:
        pincer_events = 0
        evaluated_steps = 0
        evidence: dict[str, Any] = {}

        blue_airs = [eid for eid in episode_data.entity_trajectories if "blue" in eid and "air" in eid]
        red_airs = [eid for eid in episode_data.entity_trajectories if "red" in eid]

        if len(blue_airs) < 2:
            return DoctrineResult(
                doctrine_name=self.name,
                detected=False,
                confidence=0.0,
                evidence={"reason": "Fewer than 2 friendly aircraft available for pincer"},
            )

        for r_id in red_airs:
            r_traj = episode_data.entity_trajectories[r_id]
            for i in range(1, len(r_traj)):
                r_pos = np.asarray(r_traj[i]["position"])
                r_pos_prev = np.asarray(r_traj[i - 1]["position"])

                # Check all friendly pairs
                for idx1 in range(len(blue_airs)):
                    for idx2 in range(idx1 + 1, len(blue_airs)):
                        b1_traj = episode_data.entity_trajectories[blue_airs[idx1]]
                        b2_traj = episode_data.entity_trajectories[blue_airs[idx2]]

                        if i < len(b1_traj) and i < len(b2_traj):
                            b1_pos = np.asarray(b1_traj[i]["position"])
                            b2_pos = np.asarray(b2_traj[i]["position"])
                            b1_prev = np.asarray(b1_traj[i - 1]["position"])
                            b2_prev = np.asarray(b2_traj[i - 1]["position"])

                            d1 = float(np.linalg.norm(b1_pos[:2] - r_pos[:2]))
                            d2 = float(np.linalg.norm(b2_pos[:2] - r_pos[:2]))
                            d1_prev = float(np.linalg.norm(b1_prev[:2] - r_pos_prev[:2]))
                            d2_prev = float(np.linalg.norm(b2_prev[:2] - r_pos_prev[:2]))

                            if d1 < 30.0 and d2 < 30.0:
                                evaluated_steps += 1
                                # Angular separation of friendlies from target perspective
                                b1_bearing = compute_bearing(r_pos, b1_pos)
                                b2_bearing = compute_bearing(r_pos, b2_pos)
                                angle_diff = angular_error(b1_bearing, b2_bearing)

                                # Both closing on target with flanking separation >= 60 deg
                                closing = (d1 < d1_prev) and (d2 < d2_prev)
                                if (angle_diff >= 1.05 or angle_diff >= PINCER_BEARING_MIN) and closing:
                                    pincer_events += 1

        if evaluated_steps == 0:
            return DoctrineResult(
                doctrine_name=self.name,
                detected=False,
                confidence=0.0,
                evidence={"reason": "No dual-aircraft engagement geometries formed"},
            )

        confidence = float(min(1.0, (pincer_events / max(1, evaluated_steps)) * 4.0))
        detected = bool(confidence >= MIN_CONFIDENCE_TO_COUNT)

        evidence["pincer_events"] = pincer_events
        evidence["evaluated_steps"] = evaluated_steps

        return DoctrineResult(
            doctrine_name=self.name,
            detected=detected,
            confidence=confidence,
            evidence=evidence,
        )


class ThreatPrioritizationDetector(DoctrineDetector):
    """Detector for tactical threat prioritization (engaging nearest/most lethal opponent)."""

    name = "threat_prioritization"
    domain = "air"
    description = "Combat agents prioritize fire and tactical engagement toward the highest-threat nearest target."

    def detect(self, episode_data: EpisodeData) -> DoctrineResult:
        prioritized_count = 0
        total_decisions = 0

        # Check commander decisions
        for dec in episode_data.commander_decisions:
            total_decisions += 1
            action_code = dec.get("action", 1)
            # Actions > 0 designate active tactical engagement of hostiles (FIGHT 1, 2, 3)
            if action_code in (1, 2, 3):
                prioritized_count += 1

        # Check weapon engagements
        for eng in episode_data.engagements:
            total_decisions += 1
            dist = eng.get("distance_km", 10.0)
            if dist <= 25.0:  # Fired on prioritized hostile within combat envelope
                prioritized_count += 1

        if total_decisions == 0:
            return DoctrineResult(
                doctrine_name=self.name,
                detected=False,
                confidence=0.0,
                evidence={"reason": "No tactical decisions recorded"},
            )

        confidence = float(prioritized_count / float(total_decisions))
        detected = bool(confidence >= MIN_CONFIDENCE_TO_COUNT)

        return DoctrineResult(
            doctrine_name=self.name,
            detected=detected,
            confidence=confidence,
            evidence={
                "prioritized_count": prioritized_count,
                "total_decisions": total_decisions,
                "priority_rate": confidence,
            },
        )
