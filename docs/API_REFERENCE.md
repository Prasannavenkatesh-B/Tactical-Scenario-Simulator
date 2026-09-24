# Tactical MARL Inference API Reference

**Prepared for:** Defence Research & Development Organisation (DRDO)  
**API Version:** `1.0.0`  
**Default Host/Port:** `http://0.0.0.0:8000`  
**OpenAPI Title:** Tactical MARL Inference API (DRDO)

---

## Table of Contents
1. [Overview & Architecture](#1-overview--architecture)
2. [Authentication & Headers](#2-authentication--headers)
3. [Endpoints Overview](#3-endpoints-overview)
4. [Detailed Endpoint Specifications](#4-detailed-endpoint-specifications)
   - [GET /health](#get-health)
   - [POST /act](#post-act)
   - [POST /act/batch](#post-actbatch)
   - [POST /reset](#post-reset)
   - [GET /policies](#get-policies)
   - [POST /load_checkpoint](#post-load_checkpoint)
5. [Schema Data Models](#5-schema-data-models)
6. [Integration Code Examples](#6-integration-code-examples)

---

## 1. Overview & Architecture

The Tactical MARL Inference API exposes high-performance neural inference for hierarchical multi-agent 
reinforcement learning policies. It serves low-level policies for Air (AC1/AC2), Ground, and Naval forces, 
as well as the high-level Commander recurrent GRU network.

- **Mean Inference Latency:** `1.63 ms` (single), `< 4.5 ms` (HTTP overhead included)
- **Supported Data Formats:** JSON over HTTP/1.1 REST
- **CORS:** Enabled for local simulation environments and remote integration consoles
- **State Management:** Stateless for low-level policies; per-agent GRU hidden state preservation for Commander

---

## 2. Authentication & Headers

For local tactical simulation testbeds and embedded in-the-loop nodes, authentication is unconstrained. 
All POST endpoints require the standard content type header:
```http
Content-Type: application/json
```

---

## 3. Endpoints Overview

| Method | Path | Summary | Expected Latency | Response Model |
|:---|:---|:---|:---|:---|
| `GET` | `/health` | Service health status and loaded policies | < 1 ms | `HealthResponse` |
| `POST` | `/act` | Forward inference for single entity observation | < 2 ms | `ActResponse` |
| `POST` | `/act/batch` | Forward inference for array of entities | < 5 ms | `BatchActResponse` |
| `POST` | `/reset` | Clear hidden states across recurrent policies | < 1 ms | `ResetResponse` |
| `GET` | `/policies` | Metadata and parameters of registered models | < 1 ms | `PoliciesResponse` |
| `POST` | `/load_checkpoint` | Hot-swap policy weights from file system | < 15 ms | `LoadCheckpointResponse` |

---

## 4. Detailed Endpoint Specifications

### GET /health
Validates server readiness and lists all neural policies currently loaded into device memory.

**Example Request:**
```bash
curl -X GET http://localhost:8089/health
```

**Example Response (200 OK):**
```json
{
  "status": "ok",
  "loaded_policies": [
    "air_fight_AC1",
    "air_fight_AC2",
    "ground_engage",
    "sea_engage",
    "commander"
  ],
  "version": "1.0.0"
}
```

---

### POST /act
Executes forward neural network inference on an observed state vector for a specified entity domain and variant.

**Observation Vector Dimensions:**
- `air`: 13 elements (`pos_x`, `pos_y`, `pos_z`, `speed`, `heading`, `pitch`, `roll`, `health`, `cannon`, `rockets`, `target_x`, `target_y`, `target_z`)
- `ground`: 9 elements (`pos_x`, `pos_y`, `pos_z`, `speed`, `heading`, `health`, `ammo_cannon`, `target_dist`, `cover_ratio`)
- `sea`: 9 elements (`pos_x`, `pos_y`, `pos_z`, `speed`, `heading`, `health`, `missiles`, `target_dist`, `sea_state`)
- `commander`: 53 elements (aggregated operational theater picture)

**Example Request (Air AC1 Fighter):**
```json
{
  "agent_id": "blue_air_1",
  "domain": "air",
  "variant": "AC1",
  "observation": [
    50.0,
    50.0,
    8.5,
    350.0,
    90.0,
    0.0,
    0.0,
    1.0,
    100.0,
    4.0,
    60.0,
    55.0,
    8.0
  ],
  "deterministic": true
}
```

**Example Response (200 OK):**
```json
{
  "action": {
    "domain": "air",
    "heading_idx": 6,
    "speed_idx": 4,
    "fire_cannon": 0,
    "fire_rocket": 1
  },
  "log_prob": -0.052,
  "value": 1.45,
  "hidden_state": null,
  "latency_ms": 1.25
}
```

---

### POST /act/batch
Processes a batch of observation vectors for multiple entities concurrently, minimizing transport serialization overhead.

**Example Request:**
```json
{
  "requests": [
    {
      "agent_id": "blue_air_1",
      "domain": "air",
      "variant": "AC1",
      "observation": [
        50.0,
        50.0,
        8.5,
        350.0,
        90.0,
        0.0,
        0.0,
        1.0,
        100.0,
        4.0,
        60.0,
        55.0,
        8.0
      ],
      "deterministic": true
    },
    {
      "agent_id": "blue_ground_1",
      "domain": "ground",
      "variant": "SAM",
      "observation": [
        45.0,
        48.0,
        0.2,
        0.0,
        45.0,
        1.0,
        20.0,
        12.5,
        0.8
      ],
      "deterministic": true
    }
  ]
}
```

---

### POST /reset
Resets internal recurrent states (e.g. GRU memory in CommanderPolicy) across episode boundaries.

**Example Request (Reset All):**
```json
{
  "agent_ids": null
}
```

**Example Response (200 OK):**
```json
{
  "status": "reset",
  "cleared_agents": [
    "all"
  ]
}
```

---

### GET /policies
Returns complete inventory of active neural policies, architectures, and parameter sizes.

**Example Response:**
```json
{
  "policies": [
    {
      "name": "air_fight_AC1",
      "domain": "air",
      "action_type": "discrete",
      "param_count": 185420,
      "is_loaded": true
    },
    {
      "name": "commander",
      "domain": "commander",
      "action_type": "discrete",
      "param_count": 524100,
      "is_loaded": true
    }
  ]
}
```

---

### POST /load_checkpoint
Hot-swaps model weights for a specific policy from a checkpoint artifact file without restarting the service.

**Example Request:**
```json
{
  "policy_name": "air_fight_AC1",
  "checkpoint_path": "checkpoints/final/checkpoint_final_iter_00010.pt"
}
```

---

## 5. Schema Data Models

### ActRequest
| Field | Type | Required | Description |
|:---|:---|:---:|:---|
| `agent_id` | `str` | Yes | Unique simulation identifier (e.g. `'blue_air_1'`) |
| `domain` | `str` | Yes | Warfare domain: `'air'`, `'ground'`, `'sea'`, `'commander'` |
| `variant` | `str` | No | Vehicle classification: `'AC1'`, `'AC2'`, `'SAM'`, etc. |
| `observation` | `list[float]` | Yes | Continuous feature vector matching domain dimension |
| `hidden_state` | `list[float]` | No | Optional GRU hidden state vector (Commander only) |
| `deterministic` | `bool` | No | If true, selects argmax action; if false, samples stochastic policy |

### ActResponse
| Field | Type | Description |
|:---|:---|:---|
| `action` | `dict[str, Any]` | Domain-specific action parameters |
| `log_prob` | `float` | Log-likelihood of chosen action |
| `value` | `float` | Critic state-value estimation |
| `hidden_state` | `list[float]` | Updated recurrent memory vector (or null) |
| `latency_ms` | `float` | Server-side execution duration in milliseconds |

---

## 6. Integration Code Examples

### Python Client Example
```python
import requests

BASE_URL = 'http://localhost:8089'

# 1. Check health
resp = requests.get(f'{BASE_URL}/health')
print('Health:', resp.json())

# 2. Execute inference
payload = {
    'agent_id': 'f16_01',
    'domain': 'air',
    'variant': 'AC1',
    'observation': [45.2, 33.1, 9.0, 420.0, 180.0, 0.0, 0.0, 1.0, 120.0, 2.0, 50.0, 35.0, 9.0],
    'deterministic': True,
}
act_resp = requests.post(f'{BASE_URL}/act', json=payload)
print('Action:', act_resp.json()['action'])
```

---
*Document auto-generated for DRDO Tactical Scenario Simulator (TSS) Deliverable Package v1.0.0*
