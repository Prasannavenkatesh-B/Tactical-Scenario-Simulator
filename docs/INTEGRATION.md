# DRDO Tactical Scenario Simulator (TSS) AI Integration Guide

**Protocol Version**: `1.0`  
**Target System**: Multi-Domain Hierarchical MARL Tactical Simulation System  
**Contact**: DeepMind / DRDO Joint Taskforce  

---

## Table of Contents
1. [Overview & Integration Modes](#1-overview--integration-modes)
2. [Unit Conventions](#2-unit-conventions)
3. [Observation Field Mappings](#3-observation-field-mappings)
   - [Air Domain (13 Dimensions)](#air-domain-13-dimensions)
   - [Ground Domain (9 Dimensions)](#ground-domain-9-dimensions)
   - [Sea Domain (9 Dimensions)](#sea-domain-9-dimensions)
   - [Commander Domain (53 Dimensions)](#commander-domain-53-dimensions)
4. [Action Field Mappings](#4-action-field-mappings)
5. [Agent Attrition Policy](#5-agent-attrition-policy)
6. [Sim-to-Real Fidelity Gaps & Mitigations](#6-sim-to-real-fidelity-gaps--mitigations)
7. [Integration Quickstart & Code Samples](#7-integration-quickstart--code-samples)

---

## 1. Overview & Integration Modes

The AI inference engine can be linked with DRDO's Tactical Scenario Simulator (TSS) using either of two standardized operational modes:

| Mode | Name | Transport | Latency | Language Support | Use Case |
|:---|:---|:---|:---|:---|:---|
| **Mode A** | **In-Process Wrapper** | Direct Python memory calls | **< 2.0 ms** | Python 3.11+ | High-rate real-time simulation running on the same host machine |
| **Mode B** | **HTTP REST API** | JSON over HTTP/REST (FastAPI) | **~3.2 ms** | Any (C++, C#, Java, Python) | Networked deployment, multi-machine clusters, foreign-language TSS |

Both modes share an identical Python API surface (`get_action`, `get_actions_batch`, `get_commander_action`, `reset`), enabling seamless toggling via configuration without altering simulation code.

---

## 2. Unit Conventions

Unless explicitly configured otherwise, TSS telemetry sent to the integration mappers must follow these standard operational units:

| Physical Quantity | Standard Unit | Alternative Supported Units |
|:---|:---|:---|
| **Angle** | `degrees` | `radians`, `m`, `nm`, `m/s`, `km/h` |
| **Distance** | `km` | `radians`, `m`, `nm`, `m/s`, `km/h` |
| **Speed** | `knots` | `radians`, `m`, `nm`, `m/s`, `km/h` |
| **Time** | `seconds` | `radians`, `m`, `nm`, `m/s`, `km/h` |
| **Altitude** | `km` | `radians`, `m`, `nm`, `m/s`, `km/h` |

---

## 3. Observation Field Mappings

### Air Domain (13 Dimensions)

| Index | TSS Field Name | Units | Expected Range | Description |
|:---|:---|:---|:---|:---|
| `0` | `pos_x` | `km` | `[0.0, 100.0]` | Entity continuous X coordinate |
| `1` | `pos_y` | `km` | `[0.0, 100.0]` | Entity continuous Y coordinate |
| `2` | `pos_z` | `km` | `[0.0, 15.0]` | Entity altitude / elevation Z coordinate |
| `3` | `speed` | `knots` | `[0.0, 1166.0]` | True airspeed in knots (max ~600 m/s / Mach 1.8) |
| `4` | `heading` | `degrees` | `[0.0, 360.0]` | Current vehicle compass heading in degrees |
| `5` | `heading_off` | `degrees` | `[0.0, 180.0]` | Absolute heading-off angle to closest detected adversary |
| `6` | `aspect_angle` | `degrees` | `[0.0, 180.0]` | Aspect angle to closest detected adversary |
| `7` | `antenna_train_angle` | `degrees` | `[0.0, 180.0]` | Antenna train angle (ATA) to primary contact |
| `8` | `distance_to_opponent` | `km` | `[0.0, 100.0]` | Line-of-sight distance to closest opponent |
| `9` | `cannon_ammo` | `count` | `[0.0, 500.0]` | Remaining internal gun/cannon ammunition rounds |
| `10` | `rocket_ammo` | `count` | `[0.0, 8.0]` | Remaining air-to-air missiles / rockets |
| `11` | `rocket_ready` | `flag` | `[0.0, 1.0]` | Missile weapon lock and guidance seeker ready flag |
| `12` | `is_shooting` | `flag` | `[0.0, 1.0]` | Active engagement trigger or weapon firing state |

### Ground Domain (9 Dimensions)

| Index | TSS Field Name | Units | Expected Range | Description |
|:---|:---|:---|:---|:---|
| `0` | `pos_x` | `km` | `[0.0, 100.0]` | Ground vehicle continuous X coordinate |
| `1` | `pos_y` | `km` | `[0.0, 100.0]` | Ground vehicle continuous Y coordinate |
| `2` | `speed` | `knots` | `[0.0, 58.3]` | Ground speed in knots (max ~30 m/s / 108 km/h) |
| `3` | `heading` | `degrees` | `[0.0, 360.0]` | Current vehicle chassis heading in degrees |
| `4` | `elevation` | `km` | `[0.0, 5.0]` | Local terrain surface elevation |
| `5` | `antenna_train_angle` | `degrees` | `[0.0, 180.0]` | Sensor/turret angle relative to closest target |
| `6` | `distance_to_opponent` | `km` | `[0.0, 100.0]` | Distance to closest detected adversary ground or air contact |
| `7` | `ammo` | `count` | `[0.0, 100.0]` | Remaining primary vehicle ammunition rounds |
| `8` | `sensor_range` | `km` | `[0.0, 50.0]` | Active electro-optical / radar sensor horizon |

### Sea Domain (9 Dimensions)

| Index | TSS Field Name | Units | Expected Range | Description |
|:---|:---|:---|:---|:---|
| `0` | `pos_x` | `km` | `[0.0, 100.0]` | Naval vessel continuous X coordinate |
| `1` | `pos_y` | `km` | `[0.0, 100.0]` | Naval vessel continuous Y coordinate |
| `2` | `speed` | `knots` | `[0.0, 48.6]` | Vessel velocity in knots (max ~25 m/s / 50 kt) |
| `3` | `heading` | `degrees` | `[0.0, 360.0]` | Vessel bow compass heading in degrees |
| `4` | `sea_state` | `beaufort` | `[0.0, 9.0]` | Local meteorological ocean wave condition (Beaufort scale) |
| `5` | `antenna_train_angle` | `degrees` | `[0.0, 180.0]` | Fire-control radar azimuth relative to target |
| `6` | `distance_to_opponent` | `km` | `[0.0, 100.0]` | Range to closest hostile naval or airborne contact |
| `7` | `ammo` | `count` | `[0.0, 200.0]` | Remaining magazine capacity for naval guns/missiles |
| `8` | `radar_range` | `km` | `[0.0, 100.0]` | Air and surface search radar instrumented range |

### Commander Domain (53 Dimensions)

Commander policies receive an aggregated tactical situation vector padded to 53 continuous values:
- **Own State (5 values)**: `pos_x`, `pos_y`, `pos_z`, `speed`, `heading`
- **Closest Hostile Entities (Up to 3, 18 values)**: 3 entities x (5 kinematics + 1 domain index)
- **Closest Friendly Entities (Up to 2, 12 values)**: 2 entities x (5 kinematics + 1 domain index)
- **One-Hot Domain Identifiers (18 values)**: 6 entity slots x 3 domain classes [Air, Ground, Sea]

---

## 4. Action Field Mappings

### Air Actions
| Index | Action Name | Output Format | TSS Command | Range / Scale | Description |
|:---|:---|:---|:---|:---|:---|
| `0` | `heading_delta` | `discrete_13` | `TURN` | `[-90.0, 90.0]` | Heading adjustment in discrete 15-degree increments (-6 to +6) |
| `1` | `velocity` | `discrete_9` | `SET_SPEED` | `[0, 8]` | Commanded throttle/airspeed notch (0=min cruise, 8=max afterburner) |
| `2` | `fire_cannon` | `binary` | `FIRE_CANNON` | `[0, 1]` | Burst trigger command for internal cannon |
| `3` | `fire_rocket` | `binary` | `FIRE_ROCKET` | `[0, 1]` | Missile launch rail release command |

### Ground Actions
| Index | Action Name | Output Format | TSS Command | Range / Scale | Description |
|:---|:---|:---|:---|:---|:---|
| `0` | `heading_delta` | `continuous` | `TURN` | `[-45.0, 45.0]` | Steering angle offset in degrees (-45 to +45 deg) |
| `1` | `velocity` | `discrete_6` | `SET_SPEED` | `[0, 5]` | Forward throttle speed gear (0=neutral/stop, 5=top road speed) |
| `2` | `weapon_select` | `discrete_3` | `SELECT_WEAPON` | `[0, 2]` | Weapon payload selector (0=cannon, 1=anti-tank missile, 2=SAM) |
| `3` | `fire` | `binary` | `FIRE` | `[0, 1]` | Main gun or missile launcher trigger command |

### Sea Actions
| Index | Action Name | Output Format | TSS Command | Range / Scale | Description |
|:---|:---|:---|:---|:---|:---|
| `0` | `heading_delta` | `continuous` | `TURN` | `[-30.0, 30.0]` | Rudder heading command in degrees (-30 to +30 deg) |
| `1` | `velocity` | `discrete_6` | `SET_SPEED` | `[0, 5]` | Engine order telegraph (0=all stop, 5=flank speed) |
| `2` | `weapon_select` | `discrete_2` | `SELECT_WEAPON` | `[0, 1]` | Naval weapon selector (0=deck gun, 1=anti-ship missile) |
| `3` | `fire` | `binary` | `FIRE` | `[0, 1]` | Weapon engagement fire command |

### Commander Actions
| Index | Action Name | Output Format | TSS Command | Range / Scale | Description |
|:---|:---|:---|:---|:---|:---|
| `0` | `directive` | `discrete_4` | `SET_DOCTRINE` | `[0, 3]` | High-level tactical directive (0=ESCAPE, 1=FIGHT_T1, 2=FIGHT_T2, 3=FIGHT_T3) |

---

## 5. Agent Attrition Policy

When an agent is eliminated or shot down during a TSS scenario:
- **Policy**: `zero_observation` — When an agent is destroyed, its observation slot is populated with zero-vectors.
- **Recurrent State Reset**: `True` — Internal GRU memory for destroyed units is immediately cleared to release memory and prevent stale state retention.

---

## 6. Sim-to-Real Fidelity Gaps & Mitigations

- **Fidelity Gap**: Our simulator uses simplified 2.5D kinematics with instant heading adjustments
  - **Mitigation Strategy**: Domain randomization over simulation dt and speed jitter forces policy robust to inertia lag
- **Fidelity Gap**: Our sensors utilize probabilistic line-of-sight and range-dependent Pd curves
  - **Mitigation Strategy**: TSS can supply raw detection matrices directly or override sensor probability parameters
- **Fidelity Gap**: Terrain elevation in our training simulator is represented via 2.5D synthetic heightmaps
  - **Mitigation Strategy**: TSS supplies normalized local elevation (0 to 5000m) directly into GroundObservation vectors
- **Fidelity Gap**: Sea state dynamics apply simplified Beaufort wave drag resistance
  - **Mitigation Strategy**: TSS passes calibrated sea state (0-9) to modulate naval vessel speed and accuracy envelopes

---

## 7. Integration Quickstart & Code Samples

### Mode A: In-Process Wrapper (Python)
```python
from src.integration.in_process_wrapper import InProcessTSSWrapper

# 1. Initialize wrapper (loads checkpoints automatically)
wrapper = InProcessTSSWrapper(checkpoint_dir='checkpoints/final')

# 2. Reset at episode start
wrapper.reset()

# 3. Execute step with TSS telemetry dict
tss_telemetry = {
    'pos_x': 25.4, 'pos_y': 18.2, 'pos_z': 5.0,
    'speed': 450.0, 'heading': 180.0, 'heading_off': 15.0,
    'aspect_angle': 30.0, 'antenna_train_angle': 10.0,
    'distance_to_opponent': 12.5, 'cannon_ammo': 350.0,
    'rocket_ammo': 4.0, 'rocket_ready': 1.0, 'is_shooting': 0.0,
}
command = wrapper.get_action(
    agent_id='blue_fighter_1',
    domain='air',
    variant='AC1',
    tss_observation=tss_telemetry,
)
print('TSS Command:', command)
# Output: {'turn': 0.0, 'set_speed': 728.5, 'fire_cannon': False, 'fire_rocket': False}
```

### Mode B: HTTP Client Wrapper (Decoupled / Multi-Language)
```python
from src.integration.http_wrapper import HTTPTSSWrapper

client = HTTPTSSWrapper(base_url='http://localhost:8000')
client.reset()
command = client.get_action(
    agent_id='blue_fighter_1',
    domain='air',
    variant='AC1',
    tss_observation=tss_telemetry,
)
```
