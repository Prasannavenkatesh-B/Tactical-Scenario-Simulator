"""Generator creating human-readable DRDO integration documentation from protocol.yaml."""

from pathlib import Path
from typing import Any
import yaml


def generate_integration_md(protocol_path: str, output_path: str) -> None:
    """Read protocol.yaml and generate docs/INTEGRATION.md for DRDO engineers."""
    p_path = Path(protocol_path)
    if not p_path.is_absolute() and not p_path.exists():
        alt_path = Path(__file__).parent / "protocol.yaml"
        if alt_path.exists():
            p_path = alt_path

    with open(p_path, "r", encoding="utf-8") as f:
        proto: dict[str, Any] = yaml.safe_load(f)

    version = proto.get("tss_protocol_version", "1.0")
    units = proto.get("unit_conventions", {})
    obs_fields = proto.get("observation_fields", {})
    action_fields = proto.get("action_fields", {})
    attrition = proto.get("agent_attrition_behavior", {})
    gaps = proto.get("fidelity_gaps", [])

    lines: list[str] = [
        f"# DRDO Tactical Scenario Simulator (TSS) AI Integration Guide",
        f"",
        f"**Protocol Version**: `{version}`  ",
        f"**Target System**: Multi-Domain Hierarchical MARL Tactical Simulation System  ",
        f"**Contact**: DeepMind / DRDO Joint Taskforce  ",
        f"",
        f"---",
        f"",
        f"## Table of Contents",
        f"1. [Overview & Integration Modes](#1-overview--integration-modes)",
        f"2. [Unit Conventions](#2-unit-conventions)",
        f"3. [Observation Field Mappings](#3-observation-field-mappings)",
        f"   - [Air Domain (13 Dimensions)](#air-domain-13-dimensions)",
        f"   - [Ground Domain (9 Dimensions)](#ground-domain-9-dimensions)",
        f"   - [Sea Domain (9 Dimensions)](#sea-domain-9-dimensions)",
        f"   - [Commander Domain (53 Dimensions)](#commander-domain-53-dimensions)",
        f"4. [Action Field Mappings](#4-action-field-mappings)",
        f"5. [Agent Attrition Policy](#5-agent-attrition-policy)",
        f"6. [Sim-to-Real Fidelity Gaps & Mitigations](#6-sim-to-real-fidelity-gaps--mitigations)",
        f"7. [Integration Quickstart & Code Samples](#7-integration-quickstart--code-samples)",
        f"",
        f"---",
        f"",
        f"## 1. Overview & Integration Modes",
        f"",
        f"The AI inference engine can be linked with DRDO's Tactical Scenario Simulator (TSS) using either of two standardized operational modes:",
        f"",
        f"| Mode | Name | Transport | Latency | Language Support | Use Case |",
        f"|:---|:---|:---|:---|:---|:---|",
        f"| **Mode A** | **In-Process Wrapper** | Direct Python memory calls | **< 2.0 ms** | Python 3.11+ | High-rate real-time simulation running on the same host machine |",
        f"| **Mode B** | **HTTP REST API** | JSON over HTTP/REST (FastAPI) | **~3.2 ms** | Any (C++, C#, Java, Python) | Networked deployment, multi-machine clusters, foreign-language TSS |",
        f"",
        f"Both modes share an identical Python API surface (`get_action`, `get_actions_batch`, `get_commander_action`, `reset`), enabling seamless toggling via configuration without altering simulation code.",
        f"",
        f"---",
        f"",
        f"## 2. Unit Conventions",
        f"",
        f"Unless explicitly configured otherwise, TSS telemetry sent to the integration mappers must follow these standard operational units:",
        f"",
        f"| Physical Quantity | Standard Unit | Alternative Supported Units |",
        f"|:---|:---|:---|",
    ]

    for k, v in units.items():
        lines.append(f"| **{k.capitalize()}** | `{v}` | `radians`, `m`, `nm`, `m/s`, `km/h` |")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 3. Observation Field Mappings",
        f"",
        f"### Air Domain (13 Dimensions)",
        f"",
        f"| Index | TSS Field Name | Units | Expected Range | Description |",
        f"|:---|:---|:---|:---|:---|",
    ])

    for fld in obs_fields.get("air", []):
        lines.append(
            f"| `{fld['our_index']}` | `{fld['tss_field']}` | `{fld['units']}` | `[{fld['range'][0]}, {fld['range'][1]}]` | {fld.get('description', '')} |"
        )

    lines.extend([
        f"",
        f"### Ground Domain (9 Dimensions)",
        f"",
        f"| Index | TSS Field Name | Units | Expected Range | Description |",
        f"|:---|:---|:---|:---|:---|",
    ])

    for fld in obs_fields.get("ground", []):
        lines.append(
            f"| `{fld['our_index']}` | `{fld['tss_field']}` | `{fld['units']}` | `[{fld['range'][0]}, {fld['range'][1]}]` | {fld.get('description', '')} |"
        )

    lines.extend([
        f"",
        f"### Sea Domain (9 Dimensions)",
        f"",
        f"| Index | TSS Field Name | Units | Expected Range | Description |",
        f"|:---|:---|:---|:---|:---|",
    ])

    for fld in obs_fields.get("sea", []):
        lines.append(
            f"| `{fld['our_index']}` | `{fld['tss_field']}` | `{fld['units']}` | `[{fld['range'][0]}, {fld['range'][1]}]` | {fld.get('description', '')} |"
        )

    cmd_fields = obs_fields.get("commander", {})
    lines.extend([
        f"",
        f"### Commander Domain (53 Dimensions)",
        f"",
        f"Commander policies receive an aggregated tactical situation vector padded to 53 continuous values:",
        f"- **Own State (5 values)**: `pos_x`, `pos_y`, `pos_z`, `speed`, `heading`",
        f"- **Closest Hostile Entities (Up to 3, 18 values)**: 3 entities x (5 kinematics + 1 domain index)",
        f"- **Closest Friendly Entities (Up to 2, 12 values)**: 2 entities x (5 kinematics + 1 domain index)",
        f"- **One-Hot Domain Identifiers (18 values)**: 6 entity slots x 3 domain classes [Air, Ground, Sea]",
        f"",
        f"---",
        f"",
        f"## 4. Action Field Mappings",
        f"",
    ])

    for dom, actions in action_fields.items():
        lines.append(f"### {dom.capitalize()} Actions")
        lines.append(f"| Index | Action Name | Output Format | TSS Command | Range / Scale | Description |")
        lines.append(f"|:---|:---|:---|:---|:---|:---|")
        for a in actions:
            lines.append(
                f"| `{a['our_index']}` | `{a['name']}` | `{a['type']}` | `{a['tss_command']}` | `{a.get('range', '')}` | {a.get('description', '')} |"
            )
        lines.append("")

    lines.extend([
        f"---",
        f"",
        f"## 5. Agent Attrition Policy",
        f"",
        f"When an agent is eliminated or shot down during a TSS scenario:",
        f"- **Policy**: `{attrition.get('policy', 'zero_observation')}` — When an agent is destroyed, its observation slot is populated with zero-vectors.",
        f"- **Recurrent State Reset**: `{attrition.get('reset_hidden_state', True)}` — Internal GRU memory for destroyed units is immediately cleared to release memory and prevent stale state retention.",
        f"",
        f"---",
        f"",
        f"## 6. Sim-to-Real Fidelity Gaps & Mitigations",
        f"",
    ])

    for g in gaps:
        lines.append(f"- **Fidelity Gap**: {g['gap']}")
        lines.append(f"  - **Mitigation Strategy**: {g['mitigation']}")

    lines.extend([
        f"",
        f"---",
        f"",
        f"## 7. Integration Quickstart & Code Samples",
        f"",
        f"### Mode A: In-Process Wrapper (Python)",
        f"```python",
        f"from src.integration.in_process_wrapper import InProcessTSSWrapper",
        f"",
        f"# 1. Initialize wrapper (loads checkpoints automatically)",
        f"wrapper = InProcessTSSWrapper(checkpoint_dir='checkpoints/final')",
        f"",
        f"# 2. Reset at episode start",
        f"wrapper.reset()",
        f"",
        f"# 3. Execute step with TSS telemetry dict",
        f"tss_telemetry = {{",
        f"    'pos_x': 25.4, 'pos_y': 18.2, 'pos_z': 5.0,",
        f"    'speed': 450.0, 'heading': 180.0, 'heading_off': 15.0,",
        f"    'aspect_angle': 30.0, 'antenna_train_angle': 10.0,",
        f"    'distance_to_opponent': 12.5, 'cannon_ammo': 350.0,",
        f"    'rocket_ammo': 4.0, 'rocket_ready': 1.0, 'is_shooting': 0.0,",
        f"}}",
        f"command = wrapper.get_action(",
        f"    agent_id='blue_fighter_1',",
        f"    domain='air',",
        f"    variant='AC1',",
        f"    tss_observation=tss_telemetry,",
        f")",
        f"print('TSS Command:', command)",
        f"# Output: {{'turn': 0.0, 'set_speed': 728.5, 'fire_cannon': False, 'fire_rocket': False}}",
        f"```",
        f"",
        f"### Mode B: HTTP Client Wrapper (Decoupled / Multi-Language)",
        f"```python",
        f"from src.integration.http_wrapper import HTTPTSSWrapper",
        f"",
        f"client = HTTPTSSWrapper(base_url='http://localhost:8000')",
        f"client.reset()",
        f"command = client.get_action(",
        f"    agent_id='blue_fighter_1',",
        f"    domain='air',",
        f"    variant='AC1',",
        f"    tss_observation=tss_telemetry,",
        f")",
        f"```",
    ])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
