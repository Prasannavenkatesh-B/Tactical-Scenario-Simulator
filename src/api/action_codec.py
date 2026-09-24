"""Encoders and decoders for serializing observations and actions across HTTP/JSON boundaries."""

import math
from typing import Any, Union
import numpy as np

from src.core.actions import AirAction, CommanderAction, GroundAction, SeaAction

Action = Union[AirAction, GroundAction, SeaAction, CommanderAction]

DOMAIN_OBS_DIMS: dict[str, int] = {
    "air": 13,
    "ground": 9,
    "sea": 9,
    "commander": 53,
}


class ObservationCodec:
    """Validates and formats raw client observation lists into normalized tensors."""

    @staticmethod
    def validate(domain: str, obs_vector: list[float]) -> None:
        """Validate observation vector dimensionality.

        Raises:
            ValueError: If obs_vector length does not match domain expectation.
        """
        dom = domain.lower().strip()
        expected = DOMAIN_OBS_DIMS.get(dom)
        if expected is not None and len(obs_vector) != expected:
            raise ValueError(
                f"Expected {expected} floats for domain '{domain}', got {len(obs_vector)}"
            )

    @staticmethod
    def encode(domain: str, variant: str, obs_vector: list[float]) -> np.ndarray:
        """Sanitize, pad/truncate if necessary, and shape observation into (1, obs_dim) float32 array."""
        dom = domain.lower().strip()
        target_dim = DOMAIN_OBS_DIMS.get(dom, len(obs_vector))

        # Sanitize NaNs and Infs to 0.0
        cleaned: list[float] = []
        for val in obs_vector:
            if math.isnan(val) or math.isinf(val):
                cleaned.append(0.0)
            else:
                cleaned.append(float(val))

        if len(cleaned) < target_dim:
            cleaned.extend([0.0] * (target_dim - len(cleaned)))
        elif len(cleaned) > target_dim:
            cleaned = cleaned[:target_dim]

        return np.array(cleaned, dtype=np.float32).reshape(1, target_dim)


class ActionCodec:
    """Serializes policy action structures into JSON-friendly lists and reconstructs Action dataclasses."""

    @staticmethod
    def encode(domain: str, action: Any) -> list[int] | dict[str, Any]:
        """Convert policy output or Action dataclass into JSON response format."""
        dom = domain.lower().strip()

        # Handle Action Dataclasses
        if isinstance(action, AirAction):
            return [
                int(action.discrete_heading_step),
                int(action.velocity_cmd),
                int(action.fire_cannon),
                int(action.fire_rocket),
            ]
        elif isinstance(action, GroundAction):
            return [
                int(round(action.heading_delta)),
                int(action.velocity_cmd),
                int(action.weapon_select),
                int(action.fire),
            ]
        elif isinstance(action, SeaAction):
            return [
                int(round(action.heading_delta)),
                int(action.velocity_cmd),
                int(action.weapon_select),
                int(action.fire),
            ]
        elif isinstance(action, CommanderAction):
            return [int(action.action)]

        # Handle Raw NumPy arrays or lists from policy inference
        if isinstance(action, (np.ndarray, list, tuple)):
            arr = list(action)
            if dom == "commander":
                return [int(arr[0])]
            return [int(round(float(x))) for x in arr]

        if isinstance(action, (int, np.integer)):
            return [int(action)]

        raise TypeError(f"Unsupported action type: {type(action)}")

    @staticmethod
    def decode(domain: str, variant: str, json_action: list[int] | dict[str, Any]) -> Action:
        """Reconstruct strongly-typed Action dataclass from JSON input."""
        dom = domain.lower().strip()

        if dom == "air":
            if isinstance(json_action, dict):
                return AirAction(
                    heading_delta=float(json_action.get("heading_delta", 0.0)),
                    velocity_cmd=int(json_action.get("velocity_cmd", 4)),
                    fire_cannon=int(json_action.get("fire_cannon", 0)),
                    fire_rocket=int(json_action.get("fire_rocket", 0)),
                )
            if len(json_action) == 3:
                h_step, vel, fc = json_action
                return AirAction.from_discrete(int(h_step), int(vel), int(fc), 0)
            elif len(json_action) >= 4:
                h_step, vel, fc, fr = json_action[:4]
                return AirAction.from_discrete(int(h_step), int(vel), int(fc), int(fr))
            raise ValueError(f"Invalid Air action list length: {len(json_action)}")

        elif dom == "ground":
            if isinstance(json_action, dict):
                return GroundAction(
                    heading_delta=float(json_action.get("heading_delta", 0.0)),
                    velocity_cmd=int(json_action.get("velocity_cmd", 0)),
                    weapon_select=int(json_action.get("weapon_select", 0)),
                    fire=int(json_action.get("fire", 0)),
                )
            if len(json_action) >= 4:
                return GroundAction(
                    heading_delta=float(json_action[0]),
                    velocity_cmd=int(json_action[1]),
                    weapon_select=int(json_action[2]),
                    fire=int(json_action[3]),
                )
            raise ValueError(f"Invalid Ground action list length: {len(json_action)}")

        elif dom == "sea":
            if isinstance(json_action, dict):
                return SeaAction(
                    heading_delta=float(json_action.get("heading_delta", 0.0)),
                    velocity_cmd=int(json_action.get("velocity_cmd", 2)),
                    weapon_select=int(json_action.get("weapon_select", 0)),
                    fire=int(json_action.get("fire", 0)),
                )
            if len(json_action) >= 4:
                return SeaAction(
                    heading_delta=float(json_action[0]),
                    velocity_cmd=int(json_action[1]),
                    weapon_select=int(json_action[2]),
                    fire=int(json_action[3]),
                )
            raise ValueError(f"Invalid Sea action list length: {len(json_action)}")

        elif dom == "commander":
            if isinstance(json_action, dict):
                return CommanderAction(action=int(json_action.get("action", 0)))
            return CommanderAction(action=int(json_action[0]))

        raise ValueError(f"Unknown domain: {domain}")
