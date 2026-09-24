"""Observation mapper translating DRDO TSS simulation telemetry into normalized policy vectors."""

import math
from pathlib import Path
from typing import Any
import warnings
import numpy as np
import yaml

from src.core.observations import AirObservation, CommanderObservation, GroundObservation, SeaObservation


class TSSMappingError(Exception):
    """Actionable exception raised when TSS telemetry fields fail validation or mapping."""
    pass


class TSSObservationMapper:
    """Maps TSS observation dictionaries to normalized [0, 1] numpy vectors.

    Implements unit conversions, range normalization, and strict validation
    against protocol.yaml specifications.
    """

    def __init__(self, protocol_path: str = "src/integration/protocol.yaml") -> None:
        self.protocol_path = Path(protocol_path)
        if not self.protocol_path.is_absolute():
            # Try current directory first, then relative to this file
            if not self.protocol_path.exists():
                alt_path = Path(__file__).parent / "protocol.yaml"
                if alt_path.exists():
                    self.protocol_path = alt_path

        if not self.protocol_path.exists():
            raise FileNotFoundError(
                f"TSS protocol specification not found at {protocol_path}. "
                f"Ensure protocol.yaml is placed in src/integration/."
            )

        with open(self.protocol_path, "r", encoding="utf-8") as f:
            self.protocol: dict[str, Any] = yaml.safe_load(f)

        self.obs_fields: dict[str, Any] = self.protocol.get("observation_fields", {})
        self.units: dict[str, str] = self.protocol.get("unit_conventions", {})

        # Precompile field extraction tuples: (fname, our_index, vmin, vmax, margin, units)
        self._compiled_fields: dict[str, list[tuple[str, int, float, float, float, str]]] = {}
        for dom in ["air", "ground", "sea"]:
            compiled: list[tuple[str, int, float, float, float, str]] = []
            for f in self.obs_fields.get(dom, []):
                fname = str(f["tss_field"])
                idx = int(f["our_index"])
                vmin = float(f["range"][0])
                vmax = float(f["range"][1])
                margin = 0.05 * (vmax - vmin)
                u = str(f.get("units", "standard"))
                compiled.append((fname, idx, vmin, vmax, margin, u))
            self._compiled_fields[dom] = compiled

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Unit Conversion Helpers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def convert_angle(value: float, from_unit: str, to_unit: str) -> float:
        """Convert angular values between degrees and radians."""
        f_u = from_unit.lower().strip()
        t_u = to_unit.lower().strip()
        if f_u == t_u:
            return float(value)

        if f_u in ("degrees", "deg", "degree") and t_u in ("radians", "rad", "radian"):
            return math.radians(value)
        elif f_u in ("radians", "rad", "radian") and t_u in ("degrees", "deg", "degree"):
            return math.degrees(value)

        raise TSSMappingError(
            f"Unsupported angular unit conversion from '{from_unit}' to '{to_unit}'. "
            f"Supported units: 'degrees', 'radians'."
        )

    @staticmethod
    def convert_distance(value: float, from_unit: str, to_unit: str) -> float:
        """Convert linear distances between km, m, and nautical miles (nm)."""
        f_u = from_unit.lower().strip()
        t_u = to_unit.lower().strip()
        if f_u == t_u:
            return float(value)

        # Base conversion to meters
        meters: float
        if f_u in ("m", "meters", "meter"):
            meters = float(value)
        elif f_u in ("km", "kilometers", "kilometer"):
            meters = value * 1000.0
        elif f_u in ("nm", "nautical_miles", "nautical_mile"):
            meters = value * 1852.0
        else:
            raise TSSMappingError(
                f"Unsupported distance unit '{from_unit}'. Supported: 'km', 'm', 'nm'."
            )

        # Meters to target
        if t_u in ("m", "meters", "meter"):
            return meters
        elif t_u in ("km", "kilometers", "kilometer"):
            return meters / 1000.0
        elif t_u in ("nm", "nautical_miles", "nautical_mile"):
            return meters / 1852.0

        raise TSSMappingError(
            f"Unsupported target distance unit '{to_unit}'. Supported: 'km', 'm', 'nm'."
        )

    @staticmethod
    def convert_speed(value: float, from_unit: str, to_unit: str) -> float:
        """Convert speed between knots, m/s, and km/h."""
        f_u = from_unit.lower().strip()
        t_u = to_unit.lower().strip()
        if f_u == t_u:
            return float(value)

        # Base conversion to m/s
        mps: float
        if f_u in ("m/s", "mps"):
            mps = float(value)
        elif f_u in ("knots", "knot", "kt", "kts"):
            mps = value * 0.5144444444444445
        elif f_u in ("km/h", "kmh", "kph"):
            mps = value / 3.6
        else:
            raise TSSMappingError(
                f"Unsupported speed unit '{from_unit}'. Supported: 'knots', 'm/s', 'km/h'."
            )

        # m/s to target
        if t_u in ("m/s", "mps"):
            return mps
        elif t_u in ("knots", "knot", "kt", "kts"):
            return mps / 0.5144444444444445
        elif t_u in ("km/h", "kmh", "kph"):
            return mps * 3.6

        raise TSSMappingError(
            f"Unsupported target speed unit '{to_unit}'. Supported: 'knots', 'm/s', 'km/h'."
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Range Normalization
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def normalize(value: float, vmin: float, vmax: float) -> float:
        """Normalize scalar value into [0.0, 1.0] and clamp strictly."""
        if math.isnan(value) or math.isinf(value):
            raise TSSMappingError(f"Cannot normalize non-finite value: {value}")
        denom = max(vmax - vmin, 1e-6)
        norm = (value - vmin) / denom
        return float(np.clip(norm, 0.0, 1.0))

    def _validate_and_extract_field(
        self,
        domain: str,
        field_meta: dict[str, Any],
        tss_obs: dict[str, Any],
    ) -> float:
        """Validate presence, finite value, and range bounds of a field, returning normalized scalar."""
        field_name = field_meta["tss_field"]
        if field_name not in tss_obs:
            raise TSSMappingError(
                f"Missing required TSS observation field '{field_name}' for domain '{domain}'. "
                f"Expected unit: '{field_meta.get('units', 'standard')}'. Description: {field_meta.get('description', '')}"
            )

        raw_val = tss_obs[field_name]
        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            raise TSSMappingError(
                f"Field '{field_name}' in domain '{domain}' has invalid non-numeric value: {raw_val}"
            )

        if math.isnan(val) or math.isinf(val):
            raise TSSMappingError(
                f"Field '{field_name}' in domain '{domain}' contains non-finite value: {val}. "
                f"Ensure valid sensor telemetry from TSS."
            )

        vmin, vmax = float(field_meta["range"][0]), float(field_meta["range"][1])
        span = vmax - vmin
        margin = 0.05 * span

        if val < (vmin - margin) or val > (vmax + margin):
            warnings.warn(
                f"TSS field '{field_name}' value {val} is outside expected range [{vmin}, {vmax}] "
                f"by > 5%. It will be clamped to [0.0, 1.0].",
                UserWarning,
                stacklevel=2,
            )

        return self.normalize(val, vmin, vmax)

    def _extract_domain_vector(
        self, domain: str, tss_obs: dict[str, Any], dim: int
    ) -> np.ndarray:
        """Fast extraction and normalization of domain observation vector using precompiled metadata."""
        compiled = self._compiled_fields.get(domain)
        if not compiled:
            raise TSSMappingError(f"Protocol definition missing '{domain}' observation fields.")

        vec = np.zeros(dim, dtype=np.float32)
        for fname, idx, vmin, vmax, margin, units in compiled:
            if fname not in tss_obs:
                raise TSSMappingError(
                    f"Missing required TSS observation field '{fname}' for domain '{domain}'. "
                    f"Expected unit: '{units}'."
                )

            raw = tss_obs[fname]
            try:
                val = float(raw)
            except (ValueError, TypeError):
                raise TSSMappingError(
                    f"Field '{fname}' in domain '{domain}' has invalid non-numeric value: {raw}"
                )

            if math.isnan(val) or math.isinf(val):
                raise TSSMappingError(
                    f"Field '{fname}' in domain '{domain}' contains non-finite value: {val}. "
                    f"Ensure valid sensor telemetry from TSS."
                )

            if val < (vmin - margin) or val > (vmax + margin):
                warnings.warn(
                    f"TSS field '{fname}' value {val} is outside expected range [{vmin}, {vmax}] "
                    f"by > 5%. It will be clamped to [0.0, 1.0].",
                    UserWarning,
                    stacklevel=2,
                )

            denom = vmax - vmin
            norm = (val - vmin) / (denom if denom > 1e-6 else 1e-6)
            if norm < 0.0:
                norm = 0.0
            elif norm > 1.0:
                norm = 1.0
            vec[idx] = norm

        return vec

    def to_air_observation(self, tss_obs: dict[str, Any]) -> np.ndarray:
        """Maps TSS air telemetry to normalized 13-dim float32 vector."""
        return self._extract_domain_vector("air", tss_obs, 13)

    def to_ground_observation(self, tss_obs: dict[str, Any]) -> np.ndarray:
        """Maps TSS ground telemetry to normalized 9-dim float32 vector."""
        return self._extract_domain_vector("ground", tss_obs, 9)

    def to_sea_observation(self, tss_obs: dict[str, Any]) -> np.ndarray:
        """Maps TSS naval telemetry to normalized 9-dim float32 vector."""
        return self._extract_domain_vector("sea", tss_obs, 9)

    def to_commander_observation(
        self, tss_obs: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Maps TSS commander telemetry to (own_state (5,), other_entities (N, 6)) tuple."""
        # 1. Own state (pos_x, pos_y, pos_z, speed, heading)
        own_dict = tss_obs.get("own_state", tss_obs)
        required_own = ["pos_x", "pos_y", "pos_z", "speed", "heading"]
        for k in required_own:
            if k not in own_dict:
                raise TSSMappingError(
                    f"Missing required commander own_state field '{k}'. "
                    f"Expected fields: {required_own}"
                )

        own_x = self.normalize(float(own_dict["pos_x"]), 0.0, 100.0)
        own_y = self.normalize(float(own_dict["pos_y"]), 0.0, 100.0)
        own_z = self.normalize(float(own_dict["pos_z"]), 0.0, 15.0)
        own_speed = self.normalize(float(own_dict["speed"]), 0.0, 1166.0)
        own_heading = self.normalize(float(own_dict["heading"]) % 360.0, 0.0, 360.0)
        own_state = np.array([own_x, own_y, own_z, own_speed, own_heading], dtype=np.float32)

        # 2. Other entities
        other_list: list[dict[str, Any]] = tss_obs.get("other_entities", [])
        if not other_list:
            # Check if separate opponents / friendlies keys provided
            opponents = tss_obs.get("opponents", [])
            friendlies = tss_obs.get("friendlies", [])
            other_list = list(opponents) + list(friendlies)

        domain_map = {"air": 0.0, "ground": 1.0, "sea": 2.0}

        entities_rows: list[list[float]] = []
        for ent in other_list:
            ex = self.normalize(float(ent.get("pos_x", 0.0)), 0.0, 100.0)
            ey = self.normalize(float(ent.get("pos_y", 0.0)), 0.0, 100.0)
            ez = self.normalize(float(ent.get("pos_z", 0.0)), 0.0, 15.0)
            ev = self.normalize(float(ent.get("speed", 0.0)), 0.0, 1166.0)
            eh = self.normalize(float(ent.get("heading", 0.0)) % 360.0, 0.0, 360.0)
            dom_str = str(ent.get("domain", "air")).lower().strip()
            dom_idx = domain_map.get(dom_str, 0.0)
            entities_rows.append([ex, ey, ez, ev, eh, dom_idx])

        if entities_rows:
            other_entities = np.array(entities_rows, dtype=np.float32)
        else:
            other_entities = np.zeros((0, 6), dtype=np.float32)

        return own_state, other_entities

    def to_commander_padded_vector(self, tss_obs: dict[str, Any]) -> np.ndarray:
        """Constructs the canonical 53-dimensional padded observation vector for Commander policy."""
        own_state, _ = self.to_commander_observation(tss_obs)

        # Extract opponents and friendlies
        opponents = tss_obs.get("opponents", [])
        friendlies = tss_obs.get("friendlies", [])

        # If not separated, partition other_entities
        if not opponents and not friendlies and "other_entities" in tss_obs:
            all_others = tss_obs["other_entities"]
            opponents = [e for e in all_others if e.get("team") == "red"][:3]
            friendlies = [e for e in all_others if e.get("team") == "blue"][:2]

        def _ent_to_5plus1(ent: dict[str, Any]) -> np.ndarray:
            ex = self.normalize(float(ent.get("pos_x", 0.0)), 0.0, 100.0)
            ey = self.normalize(float(ent.get("pos_y", 0.0)), 0.0, 100.0)
            ez = self.normalize(float(ent.get("pos_z", 0.0)), 0.0, 15.0)
            ev = self.normalize(float(ent.get("speed", 0.0)), 0.0, 1166.0)
            eh = self.normalize(float(ent.get("heading", 0.0)) % 360.0, 0.0, 360.0)
            dom_str = str(ent.get("domain", "air")).lower().strip()
            dom_val = {"air": 0.0, "ground": 0.5, "sea": 1.0}.get(dom_str, 0.0)
            return np.array([ex, ey, ez, ev, eh, dom_val], dtype=np.float32)

        opp_vecs = [_ent_to_5plus1(o) for o in opponents[:3]]
        fr_vecs = [_ent_to_5plus1(f) for f in friendlies[:2]]

        # Domain one-hot: 6 entities * 3 = 18
        domain_one_hots = np.zeros((6, 3), dtype=np.float32)
        # Self is commander / air
        domain_one_hots[0, 0] = 1.0
        for i, o in enumerate(opponents[:3]):
            d = {"air": 0, "ground": 1, "sea": 2}.get(str(o.get("domain", "air")).lower(), 0)
            domain_one_hots[1 + i, d] = 1.0
        for j, f in enumerate(friendlies[:2]):
            d = {"air": 0, "ground": 1, "sea": 2}.get(str(f.get("domain", "air")).lower(), 0)
            domain_one_hots[4 + j, d] = 1.0

        obs_dataclass = CommanderObservation(
            own_state=own_state,
            opponent_states=opp_vecs,
            friendly_states=fr_vecs,
            domain_ids=domain_one_hots,
        )
        return obs_dataclass.to_padded_array()
