"""UI state dataclass and simulation state manipulation methods."""

from dataclasses import dataclass, field
import typing
from typing import Any
import numpy as np

from src.core.actions import AirAction, GroundAction, SeaAction
from src.core.interfaces import AgentStatus, DomainType, TeamSide
from src.database.db import Database
from src.simulator.config import DEFAULT_EPISODE_HORIZON
from src.simulator.entities.air import AirEntity
from src.simulator.entities.base import SimEntity
from src.simulator.entities.ground import GroundEntity
from src.simulator.entities.sea import SeaEntity
from src.simulator.env import TacticalEnv
from src.simulator.scenarios import ScenarioConfig, generate_scenario
from src.ui.config import TRAIL_LENGTH
from src.ui.scenario_io import ScenarioIO


@dataclass
class UIState:
    """Manages reactive simulation state, entity selections, rendering flags, and UI modes."""

    env: TacticalEnv
    scenario_config: ScenarioConfig
    observations: dict[str, Any] = field(default_factory=dict)
    entities: dict[str, SimEntity] = field(default_factory=dict)
    selected_entity_id: str | None = None
    mode: str = "simulate"  # "edit" | "simulate" | "paused"
    current_step: int = 0
    max_steps: int = DEFAULT_EPISODE_HORIZON
    speed_multiplier: float = 1.0
    show_wez: bool = True
    show_sensor: bool = True
    show_trajectories: bool = True
    show_grid: bool = True
    show_labels: bool = True
    blue_zone_rect: tuple[float, float, float, float] | None = None
    red_zone_rect: tuple[float, float, float, float] | None = None
    pending_placement: dict[str, Any] | None = None  # e.g. {"domain": "air", "variant": "AC1", "team": TeamSide.BLUE}
    manual_control_entity: str | None = None
    manual_action: Any | None = None
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    db: Database | None = None
    trajectories: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    explosions: list[dict[str, Any]] = field(default_factory=list)
    scenario_io: ScenarioIO = field(init=False)
    ai_active: bool = True
    ai_policies: dict[str, Any] = field(default_factory=dict)
    ai_status_text: str = "H-MARL AI: Active (6,000 Iters)"

    def __post_init__(self) -> None:
        self.scenario_io = ScenarioIO(self.db)
        if not self.entities:
            self.entities = dict(self.env.entities)
        if not self.observations:
            # Generate initial observations
            for eid, e in self.entities.items():
                opps = self.env.red_entities if e.team == TeamSide.BLUE else self.env.blue_entities
                friends = self.env.blue_entities if e.team == TeamSide.BLUE else self.env.red_entities
                self.observations[eid] = e.get_observation(opps, friends, self.env.map)

        self.max_steps = self.scenario_config.episode_horizon
        self.blue_zone_rect = self.env.map.blue_zone
        self.red_zone_rect = self.env.map.red_zone

        # Seed initial trajectory points
        for eid, e in self.entities.items():
            self.trajectories[eid] = [(float(e.position[0]), float(e.position[1]))]

        # Initialize trained 6,000-iteration H-MARL policies
        self._init_ai_policies()

    def _init_ai_policies(self) -> None:
        """Load trained 6,000-iteration neural policy weights if available."""
        from pathlib import Path
        import torch
        from src.marl.policies.air_fight import AirFightPolicy
        from src.marl.policies.ground_engage import GroundEngagePolicy
        from src.marl.policies.sea_engage import SeaEngagePolicy

        ckpt_path = Path("checkpoints/final/checkpoint_final.pt")
        if not ckpt_path.exists():
            ckpt_path = Path("checkpoints/final/checkpoint_stage_b_iter_01000.pt")

        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                p_dict = ckpt.get("policies", {})

                air = AirFightPolicy()
                if "air_fight" in p_dict:
                    air.load_state_dict(p_dict["air_fight"])
                air.eval()

                ground = GroundEngagePolicy()
                if "ground_engage" in p_dict:
                    ground.load_state_dict(p_dict["ground_engage"])
                ground.eval()

                sea = SeaEngagePolicy()
                if "sea_engage" in p_dict:
                    sea.load_state_dict(p_dict["sea_engage"])
                sea.eval()

                self.ai_policies = {
                    "air_fight": air,
                    "ground_engage": ground,
                    "sea_engage": sea,
                }
                self.ai_status_text = "H-MARL AI: Active (6,000 Iters / Level 5)"
                print(f"[+] UIState: Successfully loaded 6,000-iteration trained policies from {ckpt_path}")
            except Exception as e:
                self.ai_status_text = "H-MARL AI: Heuristic Fallback"
                print(f"[!] UIState: Policy load fallback ({e})")
        else:
            self.ai_status_text = "H-MARL AI: Heuristic Fallback"

    def _get_ai_action(self, entity: SimEntity, obs: np.ndarray) -> Any:
        """Query trained 6,000-iteration H-MARL neural policy for an autonomous entity."""
        try:
            if entity.domain == DomainType.AIR:
                pol = self.ai_policies.get("air_fight")
                if pol is not None:
                    act_arr, _, _ = pol.act(obs[:13], deterministic=False)
                    sub = act_arr.astype(int)
                    h_step = int(sub[0]) - 6 if len(sub) > 0 else 0
                    v_cmd = int(sub[1]) if len(sub) > 1 else 4
                    f_c = int(sub[2]) if len(sub) > 2 else 0
                    f_r = int(sub[3]) if len(sub) > 3 else 0
                    return AirAction.from_discrete(h_step, v_cmd, f_c, f_r)

            elif entity.domain == DomainType.GROUND:
                pol = self.ai_policies.get("ground_engage")
                if pol is not None:
                    act_arr, _, _ = pol.act(obs[:9], deterministic=False)
                    return GroundAction(
                        heading_delta=float(act_arr[0]),
                        velocity_cmd=int(round(float(act_arr[1]))),
                        weapon_select=int(round(float(act_arr[2]))),
                        fire=int(round(float(act_arr[3]))),
                    )

            elif entity.domain == DomainType.SEA:
                pol = self.ai_policies.get("sea_engage")
                if pol is not None:
                    act_arr, _, _ = pol.act(obs[:9], deterministic=False)
                    return SeaAction(
                        heading_delta=float(act_arr[0]),
                        velocity_cmd=int(round(float(act_arr[1]))),
                        weapon_select=int(round(float(act_arr[2]))),
                        fire=int(round(float(act_arr[3]))),
                    )
        except Exception:
            pass
        return None

    def step_simulation(self) -> None:
        """Advance the environment forward by one physics/decision step."""
        action_dict: dict[str, Any] = {}

        # 1. Autonomous AI Policy Injection for Blue Team (6,000-Iteration H-MARL)
        if self.ai_active and self.ai_policies:
            for eid, e in self.entities.items():
                if e.team == TeamSide.BLUE and e.is_alive() and eid != self.manual_control_entity:
                    if eid in self.observations:
                        act = self._get_ai_action(e, self.observations[eid])
                        if act is not None:
                            action_dict[eid] = act

        # 2. Manual control injection (overrides AI for selected entity)
        if self.manual_control_entity is not None and self.manual_control_entity in self.entities:
            target_ent = self.entities[self.manual_control_entity]
            if target_ent.is_alive():
                if self.manual_action is not None:
                    action_dict[self.manual_control_entity] = self.manual_action
                else:
                    # Default neutral action if no key currently pressed
                    if target_ent.domain == DomainType.AIR:
                        action_dict[self.manual_control_entity] = AirAction(0.0, 4, 0, 0)
                    elif target_ent.domain == DomainType.GROUND:
                        action_dict[self.manual_control_entity] = GroundAction(0.0, 0, 0, 0)
                    elif target_ent.domain == DomainType.SEA:
                        action_dict[self.manual_control_entity] = SeaAction(0.0, 2, 0, 0)

        # Track previous alive status to detect casualties for explosions
        prev_alive = {eid: e.is_alive() for eid, e in self.entities.items()}

        obs, rewards, dones, info = self.env.step(action_dict)
        self.observations = obs
        self.current_step = self.env.current_step
        self.entities = dict(self.env.entities)

        # Check for newly destroyed entities to spawn explosions
        for eid, e in self.entities.items():
            if prev_alive.get(eid, False) and not e.is_alive():
                self.explosions.append({
                    "pos": (float(e.position[0]), float(e.position[1])),
                    "frames_left": 30,
                })

        # Update transient explosions
        active_explosions = []
        for exp in self.explosions:
            exp["frames_left"] -= 1
            if exp["frames_left"] > 0:
                active_explosions.append(exp)
        self.explosions = active_explosions

        # Update trajectory trails
        for eid, e in self.entities.items():
            if e.is_alive():
                trail = self.trajectories.setdefault(eid, [])
                trail.append((float(e.position[0]), float(e.position[1])))
                if len(trail) > TRAIL_LENGTH:
                    trail.pop(0)

        # Auto-pause if global done reached
        if info.get("global_done", False):
            self.mode = "paused"

    def reset_simulation(self) -> None:
        """Reset the simulation environment back to initial scenario state."""
        obs = self.env.reset(self.scenario_config)
        self.observations = obs
        self.current_step = 0
        self.entities = dict(self.env.entities)
        self.blue_zone_rect = self.env.map.blue_zone
        self.red_zone_rect = self.env.map.red_zone
        self.trajectories.clear()
        for eid, e in self.entities.items():
            self.trajectories[eid] = [(float(e.position[0]), float(e.position[1]))]
        self.explosions.clear()
        self.manual_action = None

    def place_entity(self, world_x: float, world_y: float) -> str | None:
        """Place pending entity profile at designated world coordinates (in km)."""
        if self.pending_placement is None:
            return None

        # Clamp inside map bounds
        map_size = self.env.map.size_km
        wx = float(np.clip(world_x, 0.5, map_size - 0.5))
        wy = float(np.clip(world_y, 0.5, map_size - 0.5))

        domain_str = str(self.pending_placement.get("domain", "air")).lower()
        variant = str(self.pending_placement.get("variant", "AC1")).upper()
        team = self.pending_placement.get("team", TeamSide.BLUE)
        if isinstance(team, str):
            team = TeamSide[team.upper()]

        prefix = "B" if team == TeamSide.BLUE else "R"
        num = len(self.entities) + 1
        new_eid = f"{prefix}_{domain_str.upper()}_{num}"

        entity: SimEntity
        if domain_str == "air":
            entity = AirEntity(
                entity_id=new_eid,
                team=team,
                aircraft_type=variant,
                position=np.array([wx, wy, 5.0]),
                heading=0.0,
                speed=500.0,
            )
        elif domain_str == "ground":
            entity = GroundEntity(
                entity_id=new_eid,
                team=team,
                position=np.array([wx, wy, 0.0]),
                heading=0.0,
                speed=30.0,
            )
        elif domain_str == "sea":
            entity = SeaEntity(
                entity_id=new_eid,
                team=team,
                position=np.array([wx, wy, 0.0]),
                heading=0.0,
                speed=20.0,
            )
        else:
            return None

        self.env.entities[new_eid] = entity
        if team == TeamSide.BLUE:
            self.env.blue_entities.append(entity)
            self.scenario_config.blue_entities.append({
                "domain": entity.domain,
                "type": variant,
                "count": 1,
            })
        else:
            self.env.red_entities.append(entity)
            self.scenario_config.red_entities.append({
                "domain": entity.domain,
                "type": variant,
                "count": 1,
            })

        self.env._setup_entity_systems(entity)
        self.entities[new_eid] = entity
        self.trajectories[new_eid] = [(wx, wy)]
        self.select_entity(new_eid)
        return new_eid

    def delete_entity(self, entity_id: str) -> None:
        """Remove entity from active environment and scenario configuration."""
        if entity_id not in self.entities:
            return

        ent = self.entities[entity_id]
        if ent in self.env.blue_entities:
            self.env.blue_entities.remove(ent)
        if ent in self.env.red_entities:
            self.env.red_entities.remove(ent)

        if entity_id in self.env.entities:
            del self.env.entities[entity_id]
        if entity_id in self.entities:
            del self.entities[entity_id]
        if entity_id in self.observations:
            del self.observations[entity_id]
        if entity_id in self.trajectories:
            del self.trajectories[entity_id]

        if self.selected_entity_id == entity_id:
            self.selected_entity_id = None
        if self.manual_control_entity == entity_id:
            self.manual_control_entity = None

    def select_entity(self, entity_id: str | None) -> None:
        """Update selected entity focus for inspection panel."""
        if entity_id is not None and entity_id in self.entities:
            self.selected_entity_id = entity_id
        else:
            self.selected_entity_id = None

    def save_scenario(self, name: str) -> int:
        """Persist current scenario and entity layout to database."""
        return self.scenario_io.save(self.scenario_config, name, self.entities)

    def load_scenario(self, scenario_id: int) -> None:
        """Load scenario from database and initialize simulation."""
        cfg, _ = self.scenario_io.load(scenario_id)
        self.scenario_config = cfg
        self.reset_simulation()
