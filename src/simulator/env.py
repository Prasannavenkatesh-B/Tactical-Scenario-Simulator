"""Gymnasium-style multi-agent tactical simulation environment.

Layer 1 simulation engine providing batch (headless) and interactive execution modes,
integrating dynamics, perception stochasticity, weapon effects, and reward dispatch.
"""

import math
import typing
import numpy as np

from src.core.actions import AirAction, GroundAction, SeaAction
from src.core.interfaces import (
    AgentStatus,
    BaseEnvironment,
    DomainType,
    TeamSide,
    WeaponType,
)
from src.core.rewards import (
    compute_ATA,
    compute_boundary_penalty,
    compute_death_penalty,
    compute_distance,
    compute_fight_reward,
    compute_kill_reward,
)
from src.simulator.config import (
    AC1_HIT_PROB_CANNON,
    AC1_WEZ_ANGLE_DEG,
    AC1_WEZ_RANGE_KM,
    AC2_HIT_PROB_CANNON,
    AC2_WEZ_ANGLE_DEG,
    AC2_WEZ_RANGE_KM,
    DEFAULT_DT_SECONDS,
    DEFAULT_MAP_SIZE_KM,
    GROUND_HIT_PROB,
    GROUND_SENSOR_RANGE_KM,
    GROUND_WEZ_ANGLE_DEG,
    GROUND_WEZ_RANGE_KM,
    SEA_HIT_PROB,
    SEA_RADAR_RANGE_KM,
    SEA_WEZ_ANGLE_DEG,
    SEA_WEZ_RANGE_KM,
)
from src.simulator.entities.air import AirEntity
from src.simulator.entities.base import SimEntity
from src.simulator.entities.ground import GroundEntity
from src.simulator.entities.sea import SeaEntity
from src.simulator.map import Map2D
from src.simulator.scenarios import ScenarioConfig, generate_scenario, spawn_entities
from src.simulator.sensors import Sensor, SensorContact
from src.simulator.weapons import Weapon, WeaponEffect


class WinnerResult(str):
    """String representation of episode winner supporting equality with TeamSide enum."""

    def __eq__(self, other: typing.Any) -> bool:
        if hasattr(other, "value"):
            return str(self) == str(other.value)
        return super().__eq__(str(other))


class TacticalEnv(BaseEnvironment):
    """Gymnasium-compatible multi-domain tactical scenario simulation environment."""

    def __init__(
        self,
        scenario_config: ScenarioConfig | None = None,
        mode: str = "batch",
        seed: int | None = None,
        dt: float = DEFAULT_DT_SECONDS,
    ) -> None:
        """Initialize TacticalEnv.

        Args:
            scenario_config: Scenario configuration or None (defaults to Level 1).
            mode: "batch" (headless, ultra-fast for RL) or "interactive" (renderable).
            seed: Initial RNG seed for reproducible execution.
            dt: Simulation physical integration step in seconds.
        """
        self.mode: str = mode.lower()
        self.dt: float = float(dt)
        self.initial_seed: int | None = seed

        # Seed internal RNG
        self.rng: np.random.Generator = np.random.default_rng(seed)

        if scenario_config is None:
            self.scenario_config: ScenarioConfig = generate_scenario(level=1, rng=self.rng)
        else:
            self.scenario_config = scenario_config

        self.current_step: int = 0
        self.sim_time: float = 0.0

        # Simulation state
        self.map: Map2D = Map2D(size_km=self.scenario_config.map_size_km, terrain_seed=seed)
        self.entities: dict[str, SimEntity] = {}
        self.blue_entities: list[SimEntity] = []
        self.red_entities: list[SimEntity] = []

        # Systems attached to entities
        self.sensors: dict[str, Sensor] = {}
        self.weapons: dict[str, list[Weapon]] = {}
        self.latest_contacts: dict[str, list[SensorContact]] = {}

        # Reset environment
        self.reset(self.scenario_config)

    def seed(self, seed: int) -> None:
        """Re-seed environment RNG for strict reproducibility across episodes.

        Args:
            seed: Integer random seed.
        """
        self.initial_seed = int(seed)
        self.rng = np.random.default_rng(seed)

    def _setup_entity_systems(self, entity: SimEntity) -> None:
        """Instantiate domain-specific sensor and weapon systems for an entity.

        Args:
            entity: SimEntity being outfitted.
        """
        eid = entity.entity_id

        # 1. Sensor setup
        if entity.domain == DomainType.AIR:
            max_sensor_range = self.map.size_km * 0.8
            sensor = Sensor("AIR_RADAR", max_range_km=max_sensor_range, rng=self.rng)
        elif entity.domain == DomainType.GROUND:
            sensor = Sensor("GROUND_RADAR", max_range_km=GROUND_SENSOR_RANGE_KM, rng=self.rng)
        elif entity.domain == DomainType.SEA:
            sensor = Sensor("NAVAL_RADAR", max_range_km=SEA_RADAR_RANGE_KM, rng=self.rng)
        else:
            sensor = Sensor("GENERIC_SENSOR", max_range_km=20.0, rng=self.rng)
        self.sensors[eid] = sensor

        # 2. Weapon setup
        entity_weapons: list[Weapon] = []
        if entity.domain == DomainType.AIR:
            is_ac1 = entity.aircraft_type.upper() == "AC1"
            wez_angle = AC1_WEZ_ANGLE_DEG if is_ac1 else AC2_WEZ_ANGLE_DEG
            wez_range = AC1_WEZ_RANGE_KM if is_ac1 else AC2_WEZ_RANGE_KM
            hit_prob = AC1_HIT_PROB_CANNON if is_ac1 else AC2_HIT_PROB_CANNON

            cannon = Weapon(
                weapon_type=WeaponType.CANNON,
                max_range_km=wez_range,
                wez_angle_deg=wez_angle,
                base_pk=hit_prob,
                rng=self.rng,
            )
            entity_weapons.append(cannon)

            if is_ac1:
                rocket = Weapon(
                    weapon_type=WeaponType.ROCKET,
                    max_range_km=6.0,
                    wez_angle_deg=15.0,
                    base_pk=0.65,
                    rng=self.rng,
                )
                entity_weapons.append(rocket)

        elif entity.domain == DomainType.GROUND:
            sam = Weapon(
                weapon_type=WeaponType.SAM,
                max_range_km=GROUND_WEZ_RANGE_KM,
                wez_angle_deg=GROUND_WEZ_ANGLE_DEG,
                base_pk=GROUND_HIT_PROB,
                rng=self.rng,
            )
            entity_weapons.append(sam)

        elif entity.domain == DomainType.SEA:
            missile = Weapon(
                weapon_type=WeaponType.MISSILE,
                max_range_km=SEA_WEZ_RANGE_KM,
                wez_angle_deg=SEA_WEZ_ANGLE_DEG,
                base_pk=SEA_HIT_PROB,
                rng=self.rng,
            )
            entity_weapons.append(missile)

        self.weapons[eid] = entity_weapons

    def reset(
        self,
        scenario_config: ScenarioConfig | dict[str, typing.Any] | None = None,
    ) -> dict[str, typing.Any]:
        """Reset the simulation environment to an initial episode state.

        Steps:
            1. Update scenario configuration if provided.
            2. Re-initialize map and procedural team zone assignments.
            3. Spawn Blue and Red entities inside team zones.
            4. Outfit entities with stochastic sensors and weapons.
            5. Reset simulation clock and step counter.
            6. Return initial normalized observation vector dictionary.

        Args:
            scenario_config: Optional new scenario specification or dictionary.

        Returns:
            Dictionary mapping entity IDs to initial normalized observation arrays.
        """
        if scenario_config is not None:
            if isinstance(scenario_config, ScenarioConfig):
                self.scenario_config = scenario_config
            elif isinstance(scenario_config, dict):
                if "level" in scenario_config:
                    self.scenario_config = generate_scenario(int(scenario_config["level"]), rng=self.rng)
            if self.scenario_config.seed is not None:
                self.seed(self.scenario_config.seed)

        self.current_step = 0
        self.sim_time = 0.0

        # Map and zones
        self.map = Map2D(
            size_km=self.scenario_config.map_size_km,
            terrain_seed=self.initial_seed,
        )
        self.map.assign_team_zones(
            randomize=self.scenario_config.randomize_team_zones,
            rng=self.rng,
        )

        # Spawn entities
        self.blue_entities, self.red_entities = spawn_entities(
            config=self.scenario_config,
            map_ref=self.map,
            rng=self.rng,
        )

        self.entities.clear()
        for e in self.blue_entities + self.red_entities:
            self.entities[e.entity_id] = e
            self._setup_entity_systems(e)

        self.latest_contacts.clear()

        # Build initial observations
        obs_dict: dict[str, np.ndarray] = {}
        for eid, entity in self.entities.items():
            opponents = self.red_entities if entity.team == TeamSide.BLUE else self.blue_entities
            friendlies = self.blue_entities if entity.team == TeamSide.BLUE else self.red_entities
            obs_dict[eid] = entity.get_observation(opponents, friendlies, self.map)

        return obs_dict

    def _get_scripted_action(self, entity: SimEntity) -> typing.Any:
        """Generate scripted maneuver/firing action for autonomous opponent baseline.

        Args:
            entity: SimEntity running scripted logic.

        Returns:
            Domain-appropriate action instance (AirAction, GroundAction, or SeaAction).
        """
        script = entity.config.get("script", "pursuit")
        opponents = self.blue_entities if entity.team == TeamSide.RED else self.red_entities

        if entity.domain == DomainType.AIR:
            if script == "static":
                return AirAction(heading_delta=0.0, velocity_cmd=4, fire_cannon=0, fire_rocket=0)
            elif script == "random":
                delta = float(self.rng.uniform(-30.0, 30.0))
                v_cmd = int(self.rng.integers(2, 7))
                return AirAction(heading_delta=delta, velocity_cmd=v_cmd, fire_cannon=0, fire_rocket=0)
            else:
                # "pursuit": steer toward closest active opponent
                closest_opp: SimEntity | None = None
                min_dist = float("inf")
                for opp in opponents:
                    if opp.is_alive():
                        d = compute_distance(entity.position[:2], opp.position[:2])
                        if d < min_dist:
                            min_dist = d
                            closest_opp = opp

                if closest_opp is not None:
                    dx = closest_opp.position[0] - entity.position[0]
                    dy = closest_opp.position[1] - entity.position[1]
                    target_angle = math.atan2(dy, dx)
                    angle_diff = (target_angle - entity.heading + math.pi) % (2 * math.pi) - math.pi
                    heading_delta = math.degrees(angle_diff)

                    # Check if target in firing cone
                    ata_deg = abs(heading_delta)
                    fire_c = 1 if (min_dist < 4.0 and ata_deg < 10.0) else 0
                    fire_r = 1 if (min_dist < 6.0 and ata_deg < 15.0 and entity.ammo.get(WeaponType.ROCKET, 0) > 0) else 0

                    return AirAction(
                        heading_delta=heading_delta,
                        velocity_cmd=6,
                        fire_cannon=fire_c,
                        fire_rocket=fire_r,
                    )
                return AirAction(heading_delta=0.0, velocity_cmd=4, fire_cannon=0, fire_rocket=0)

        elif entity.domain == DomainType.GROUND:
            return GroundAction(heading_delta=0.0, velocity_cmd=0, weapon_select=0, fire=0)

        elif entity.domain == DomainType.SEA:
            return SeaAction(heading_delta=0.0, velocity_cmd=2, weapon_select=0, fire=0)

        return None

    def step(
        self,
        action_dict: dict[str, typing.Any],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, typing.Any],
    ]:
        """Advance the tactical simulation forward by one time step dt.

        Strict Execution Pipeline:
            1. Apply all entity dynamics
            2. Run all sensor perception updates (stochastic)
            3. Process all weapon fires (stochastic WEZ & triggers)
            4. Apply weapon effects (casualties & destruction)
            5. Check out-of-bounds boundary violations
            6. Evaluate termination criteria
            7. Compute individual entity rewards
            8. Package normalized observation vectors

        Args:
            action_dict: Dictionary mapping entity IDs to action commands.

        Returns:
            Tuple of (observations, rewards, dones, info).
        """
        self.current_step += 1
        self.sim_time += self.dt

        kills_this_step: dict[str, int] = {eid: 0 for eid in self.entities}
        died_this_step: dict[str, bool] = {eid: False for eid in self.entities}
        oob_this_step: dict[str, bool] = {eid: False for eid in self.entities}

        # ---------------------------------------------------------------------
        # 1. Apply All Entity Dynamics
        # ---------------------------------------------------------------------
        for eid, entity in self.entities.items():
            if not entity.is_alive():
                continue

            if eid in action_dict:
                action = action_dict[eid]
            else:
                action = self._get_scripted_action(entity)

            entity.step(self.dt, action)

        # ---------------------------------------------------------------------
        # 2. Run All Sensor Perception Updates (Stochastic)
        # ---------------------------------------------------------------------
        self.latest_contacts.clear()
        env_meta = {"weather": self.scenario_config.weather}

        for eid, entity in self.entities.items():
            if not entity.is_alive():
                continue

            sensor = self.sensors.get(eid)
            if sensor is None:
                continue

            opponents = self.red_entities if entity.team == TeamSide.BLUE else self.blue_entities
            contacts: list[SensorContact] = []

            for opp in opponents:
                contact = sensor.detect(opp, entity, environment=env_meta, current_time=self.sim_time)
                if contact is not None:
                    contacts.append(contact)

            # Check false alarm ghost contact
            ghost = sensor.sample_false_alarm(entity, current_time=self.sim_time)
            if ghost is not None:
                contacts.append(ghost)

            self.latest_contacts[eid] = contacts

        # ---------------------------------------------------------------------
        # 3. & 4. Process Weapon Fires & Apply Effects (Stochastic)
        # ---------------------------------------------------------------------
        for eid, entity in self.entities.items():
            if not entity.is_alive():
                continue

            action = action_dict.get(eid)
            if action is None:
                action = self._get_scripted_action(entity)

            # Determine fire intents
            fire_cannon = False
            fire_rocket = False
            fire_general = False

            if isinstance(action, AirAction):
                fire_cannon = bool(action.fire_cannon)
                fire_rocket = bool(action.fire_rocket)
            elif isinstance(action, (GroundAction, SeaAction)):
                fire_general = bool(action.fire)

            weapons = self.weapons.get(eid, [])
            opponents = self.red_entities if entity.team == TeamSide.BLUE else self.blue_entities
            alive_opps = [o for o in opponents if o.is_alive()]

            for weapon in weapons:
                w_type = weapon.weapon_type
                should_fire = (
                    (w_type == WeaponType.CANNON and fire_cannon)
                    or (w_type == WeaponType.ROCKET and fire_rocket)
                    or (fire_general)
                )

                if should_fire and alive_opps:
                    # Select closest target in WEZ cone
                    target = min(alive_opps, key=lambda o: compute_distance(entity.position, o.position))

                    if weapon.can_fire(entity, target, self.sim_time):
                        if isinstance(entity, AirEntity):
                            entity.is_shooting = True

                        effect: WeaponEffect = weapon.fire(entity, target, self.sim_time)
                        if effect.hit and effect.target_destroyed:
                            kills_this_step[eid] += 1
                            died_this_step[target.entity_id] = True

        # ---------------------------------------------------------------------
        # 5. Check Out-of-Bounds (Boundary Violations)
        # ---------------------------------------------------------------------
        for eid, entity in self.entities.items():
            if not entity.is_alive():
                continue

            if not self.map.is_in_bounds(entity.position):
                entity.status = AgentStatus.OUT_OF_BOUNDS
                oob_this_step[eid] = True
                died_this_step[eid] = True

        # ---------------------------------------------------------------------
        # 6. Check Termination Criteria
        # ---------------------------------------------------------------------
        blue_alive = any(e.is_alive() for e in self.blue_entities)
        red_alive = any(e.is_alive() for e in self.red_entities)
        horizon_reached = self.current_step >= self.scenario_config.episode_horizon

        global_done = (not blue_alive) or (not red_alive) or horizon_reached

        winner: WinnerResult | None = None
        if not red_alive and blue_alive:
            winner = WinnerResult("BLUE")
        elif not blue_alive and red_alive:
            winner = WinnerResult("RED")

        # ---------------------------------------------------------------------
        # 7. Compute Individual Entity Rewards
        # ---------------------------------------------------------------------
        reward_dict: dict[str, float] = {}

        for eid, entity in self.entities.items():
            r_total = 0.0

            if oob_this_step[eid]:
                r_total += compute_boundary_penalty(True)
            else:
                opponents = self.red_entities if entity.team == TeamSide.BLUE else self.blue_entities
                alive_opps = [o for o in opponents if o.is_alive()]

                if entity.domain == DomainType.AIR and alive_opps:
                    closest_opp = min(alive_opps, key=lambda o: compute_distance(entity.position[:2], o.position[:2]))
                    # Compute opponent's ATA to this entity (how well opponent is looking away from us)
                    ata_opp_to_us = compute_ATA(closest_opp.position[:2], closest_opp.heading, entity.position[:2])
                    norm_ata_a = ata_opp_to_us / math.pi

                    max_ammo = (entity.max_ammo_cannon + entity.max_ammo_rocket) if isinstance(entity, AirEntity) else 100
                    rem_ammo = sum(entity.ammo.values())
                    r_total += compute_fight_reward(norm_ata_a, float(max_ammo), float(rem_ammo))

                # Kill reward & death penalty
                if kills_this_step[eid] > 0:
                    r_total += compute_kill_reward(kills_this_step[eid])
                if died_this_step[eid]:
                    r_total += compute_death_penalty(True)

            reward_dict[eid] = float(r_total)

        # ---------------------------------------------------------------------
        # 8. Build Observations & Done Flags
        # ---------------------------------------------------------------------
        obs_dict: dict[str, np.ndarray] = {}
        done_dict: dict[str, bool] = {}

        for eid, entity in self.entities.items():
            opponents = self.red_entities if entity.team == TeamSide.BLUE else self.blue_entities
            friendlies = self.blue_entities if entity.team == TeamSide.BLUE else self.red_entities

            obs_dict[eid] = entity.get_observation(opponents, friendlies, self.map)
            done_dict[eid] = (not entity.is_alive()) or global_done

        info_dict: dict[str, typing.Any] = {
            "global_done": global_done,
            "winner": winner,
            "step": self.current_step,
            "sim_time": self.sim_time,
            "blue_alive_count": sum(1 for e in self.blue_entities if e.is_alive()),
            "red_alive_count": sum(1 for e in self.red_entities if e.is_alive()),
        }

        return obs_dict, reward_dict, done_dict, info_dict

    def render(self, mode: str = "human") -> np.ndarray | None:
        """Render the simulation state.

        In "batch" mode, returns None.
        In "interactive" mode, returns an RGB frame placeholder array (e.g. 256x256x3).

        Args:
            mode: "human" or "rgb_array".

        Returns:
            RGB array or None.
        """
        if self.mode == "batch":
            return None
        # Lightweight placeholder frame for API compliance
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        return frame

    def get_global_state(self) -> dict[str, typing.Any]:
        """Serialize complete environment state for visualizers, replays, or debugging."""
        return {
            "step": self.current_step,
            "sim_time": self.sim_time,
            "map": self.map.to_dict(),
            "entities": {eid: e.to_state_dict() for eid, e in self.entities.items()},
            "contacts": {
                eid: [
                    {
                        "target_id": c.target_id,
                        "pos": c.estimated_position.tolist(),
                        "vel": c.estimated_velocity,
                        "heading": c.estimated_heading,
                        "confidence": c.confidence,
                    }
                    for c in contacts
                ]
                for eid, contacts in self.latest_contacts.items()
            },
        }
