"""Unit tests verifying protocol.yaml schema, completeness, and contiguous indices."""

from pathlib import Path
import yaml


def test_protocol_yaml_loads(protocol_path: str) -> None:
    """protocol.yaml must exist and load as valid YAML dictionary."""
    assert Path(protocol_path).exists()
    with open(protocol_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    assert "tss_protocol_version" in data
    assert data["tss_protocol_version"] == "1.0"


def test_protocol_yaml_required_sections(protocol_path: str) -> None:
    """All architectural sections specified by DRDO must be present."""
    with open(protocol_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    required = [
        "tss_protocol_version",
        "unit_conventions",
        "observation_fields",
        "action_fields",
        "agent_attrition_behavior",
        "fidelity_gaps",
    ]
    for sec in required:
        assert sec in data, f"Missing required protocol section: '{sec}'"


def test_observation_indices_contiguous(protocol_path: str) -> None:
    """Observation fields for air, ground, and sea must have contiguous 0-indexed indices."""
    with open(protocol_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    obs = data["observation_fields"]

    # Air: 13 fields (0..12)
    air_indices = [f["our_index"] for f in obs["air"]]
    assert len(air_indices) == 13
    assert air_indices == list(range(13))

    # Ground: 9 fields (0..8)
    ground_indices = [f["our_index"] for f in obs["ground"]]
    assert len(ground_indices) == 9
    assert ground_indices == list(range(9))

    # Sea: 9 fields (0..8)
    sea_indices = [f["our_index"] for f in obs["sea"]]
    assert len(sea_indices) == 9
    assert sea_indices == list(range(9))


def test_action_indices_contiguous(protocol_path: str) -> None:
    """Action fields for all domains must have contiguous 0-indexed indices."""
    with open(protocol_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    actions = data["action_fields"]

    # Air: 4 actions (0..3)
    air_act_idx = [a["our_index"] for a in actions["air"]]
    assert len(air_act_idx) == 4
    assert air_act_idx == list(range(4))

    # Ground: 4 actions (0..3)
    gnd_act_idx = [a["our_index"] for a in actions["ground"]]
    assert len(gnd_act_idx) == 4
    assert gnd_act_idx == list(range(4))

    # Sea: 4 actions (0..3)
    sea_act_idx = [a["our_index"] for a in actions["sea"]]
    assert len(sea_act_idx) == 4
    assert sea_act_idx == list(range(4))


def test_field_ranges_valid(protocol_path: str) -> None:
    """Every range specified in observation fields must have vmin < vmax."""
    with open(protocol_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    for domain in ["air", "ground", "sea"]:
        for fld in data["observation_fields"][domain]:
            rng = fld["range"]
            assert len(rng) == 2
            assert rng[0] < rng[1], f"Invalid range {rng} for field {fld['tss_field']} in {domain}"


def test_fidelity_gaps_documented(protocol_path: str) -> None:
    """Fidelity gaps section must contain actionable mitigations."""
    with open(protocol_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    gaps = data["fidelity_gaps"]
    assert len(gaps) >= 3
    for entry in gaps:
        assert "gap" in entry
        assert "mitigation" in entry
        assert len(entry["gap"]) > 10
        assert len(entry["mitigation"]) > 10
