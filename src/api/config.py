"""API configuration constants, network defaults, and policy resolution mapping."""

from typing import Final

API_VERSION: Final[str] = "1.0.0"
DEFAULT_HOST: Final[str] = "0.0.0.0"
DEFAULT_PORT: Final[int] = 8000
DEFAULT_CHECKPOINT_DIR: Final[str] = "checkpoints/final"
LATENCY_TARGET_MS: Final[float] = 50.0
MAX_BATCH_SIZE: Final[int] = 256
LOG_LEVEL: Final[str] = "info"

CANONICAL_POLICY_NAMES: Final[list[str]] = [
    "air_fight_AC1",
    "air_fight_AC2",
    "air_escape_AC1",
    "air_escape_AC2",
    "ground_engage",
    "ground_defend",
    "sea_engage",
    "sea_defend",
    "commander",
]


def resolve_policy_name(domain: str, variant: str = "default") -> str:
    """Resolve a domain and variant specification to its canonical registered policy name."""
    dom = domain.lower().strip()
    var = (variant or "default").strip()

    # Direct canonical name match
    if var in CANONICAL_POLICY_NAMES:
        return var
    if dom in CANONICAL_POLICY_NAMES:
        return dom

    if dom == "commander":
        return "commander"

    if dom == "air":
        ac_type = "AC2" if "2" in var.upper() else "AC1"
        if "escape" in var.lower():
            return f"air_escape_{ac_type}"
        return f"air_fight_{ac_type}"

    if dom == "ground":
        if "defend" in var.lower():
            return "ground_defend"
        return "ground_engage"

    if dom == "sea":
        if "defend" in var.lower():
            return "sea_defend"
        return "sea_engage"

    raise ValueError(f"Cannot resolve policy for domain='{domain}', variant='{variant}'")
