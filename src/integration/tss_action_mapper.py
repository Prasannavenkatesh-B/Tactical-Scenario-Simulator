"""Action mapper converting policy decisions into DRDO TSS commands."""

from pathlib import Path
from typing import Any
import yaml

from src.core.actions import AirAction, CommanderAction, GroundAction, SeaAction
from src.integration.tss_observation_mapper import TSSMappingError


class TSSActionMapper:
    """Translates high-level MARL policy actions into vehicle command structures for TSS."""

    def __init__(self, protocol_path: str = "src/integration/protocol.yaml") -> None:
        self.protocol_path = Path(protocol_path)
        if not self.protocol_path.is_absolute():
            if not self.protocol_path.exists():
                alt_path = Path(__file__).parent / "protocol.yaml"
                if alt_path.exists():
                    self.protocol_path = alt_path

        if not self.protocol_path.exists():
            raise FileNotFoundError(
                f"TSS protocol specification not found at {protocol_path}."
            )

        with open(self.protocol_path, "r", encoding="utf-8") as f:
            self.protocol: dict[str, Any] = yaml.safe_load(f)

        self.action_fields: dict[str, Any] = self.protocol.get("action_fields", {})

    def from_air_action(
        self, action: AirAction, tss_agent_state: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Maps an AirAction dataclass into a TSS flight control command dictionary."""
        state = tss_agent_state or {}
        turn = float(action.heading_delta)

        # Validation: check against agent-specific turn rate limit if provided
        max_turn = float(state.get("max_turn_rate", 90.0))
        if abs(turn) > max_turn:
            raise TSSMappingError(
                f"Air turn command {turn:.1f}° exceeds vehicle maximum turn rate {max_turn:.1f}°. "
                f"Check aircraft maneuver limits in TSS."
            )

        # Speed command mapping (knots)
        min_speed = float(state.get("min_speed", 291.0))  # ~150 m/s
        max_speed = float(state.get("max_speed", 1166.0))  # ~600 m/s
        notch = max(0, min(8, int(action.velocity_cmd)))
        set_speed = min_speed + (notch / 8.0) * (max_speed - min_speed)

        return {
            "turn": turn,
            "set_speed": float(round(set_speed, 2)),
            "fire_cannon": bool(action.fire_cannon == 1),
            "fire_rocket": bool(action.fire_rocket == 1),
        }

    def from_ground_action(
        self, action: GroundAction, tss_agent_state: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Maps a GroundAction dataclass into a TSS vehicle guidance command dictionary."""
        state = tss_agent_state or {}
        turn = float(action.heading_delta)

        max_turn = float(state.get("max_turn_rate", 45.0))
        if abs(turn) > max_turn:
            raise TSSMappingError(
                f"Ground steering command {turn:.1f}° exceeds vehicle steering limit {max_turn:.1f}°."
            )

        min_speed = float(state.get("min_speed", 0.0))
        max_speed = float(state.get("max_speed", 58.3))  # ~30 m/s in knots
        notch = max(0, min(5, int(action.velocity_cmd)))
        set_speed = min_speed + (notch / 5.0) * (max_speed - min_speed)

        weapon_names = {0: "cannon", 1: "missile", 2: "sam"}
        weapon_sel = int(action.weapon_select)

        return {
            "turn": turn,
            "set_speed": float(round(set_speed, 2)),
            "weapon_select": weapon_names.get(weapon_sel, weapon_sel),
            "fire": bool(action.fire == 1),
        }

    def from_sea_action(
        self, action: SeaAction, tss_agent_state: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Maps a SeaAction dataclass into a TSS naval maneuvering command dictionary."""
        state = tss_agent_state or {}
        turn = float(action.heading_delta)

        max_turn = float(state.get("max_turn_rate", 30.0))
        if abs(turn) > max_turn:
            raise TSSMappingError(
                f"Naval rudder command {turn:.1f}° exceeds ship maximum rudder angle {max_turn:.1f}°."
            )

        min_speed = float(state.get("min_speed", 0.0))
        max_speed = float(state.get("max_speed", 48.6))  # ~25 m/s in knots
        notch = max(0, min(5, int(action.velocity_cmd)))
        set_speed = min_speed + (notch / 5.0) * (max_speed - min_speed)

        weapon_names = {0: "deck_gun", 1: "anti_ship_missile"}
        weapon_sel = int(action.weapon_select)

        return {
            "turn": turn,
            "set_speed": float(round(set_speed, 2)),
            "weapon_select": weapon_names.get(weapon_sel, weapon_sel),
            "fire": bool(action.fire == 1),
        }

    def from_commander_action(
        self, action: CommanderAction, tss_agent_state: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Maps a CommanderAction dataclass into a TSS tactical directive dictionary."""
        act_idx = int(action.action)
        if act_idx == 0:
            activate_policy = "escape"
            target_index = 0
        else:
            activate_policy = "fight"
            target_index = min(2, act_idx - 1)

        domain = str((tss_agent_state or {}).get("domain", "")).lower()
        if domain == "ground":
            activate_policy = "defend" if act_idx == 0 else "engage"
        elif domain == "sea":
            activate_policy = "defend" if act_idx == 0 else "engage"

        return {
            "activate_policy": activate_policy,
            "target_index": target_index,
        }
