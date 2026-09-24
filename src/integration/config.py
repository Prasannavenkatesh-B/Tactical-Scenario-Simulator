"""Integration configuration constants and environment variable bindings for DRDO TSS bridge."""

import os
from typing import Final

# Integration runtime modes
INTEGRATION_MODE: Final[str] = os.getenv("TSS_INTEGRATION_MODE", "in_process")  # "in_process" or "http"
HTTP_BASE_URL: Final[str] = os.getenv("TSS_HTTP_BASE_URL", "http://localhost:8000")
HTTP_TIMEOUT_S: Final[float] = float(os.getenv("TSS_HTTP_TIMEOUT_S", "0.5"))
HTTP_MAX_RETRIES: Final[int] = int(os.getenv("TSS_HTTP_MAX_RETRIES", "3"))

# Sim-to-Real domain randomization parameters (from Document 3.7)
SENSOR_RANGE_JITTER: Final[float] = 0.20
WEAPON_PK_JITTER: Final[float] = 0.15
AGENT_SPEED_JITTER: Final[float] = 0.10
MAP_SIZE_RANGE_KM: Final[tuple[float, float]] = (20.0, 100.0)
DT_RANGE_S: Final[tuple[float, float]] = (0.05, 0.20)

# Default units expected by DRDO TSS
TSS_ANGLE_UNIT: Final[str] = "degrees"       # "degrees" or "radians"
TSS_DISTANCE_UNIT: Final[str] = "km"         # "km", "m", or "nm"
TSS_SPEED_UNIT: Final[str] = "knots"         # "knots", "m/s", or "km/h"
