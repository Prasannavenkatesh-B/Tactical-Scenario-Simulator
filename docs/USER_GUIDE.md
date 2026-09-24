# Tactical MARL System Operator & User Guide

**Prepared for:** Defence Research & Development Organisation (DRDO), Ministry of Defence, Government of India  
**Coordinating Laboratory:** Aeronautical Development Establishment (ADE), Bengaluru  
**Target Audience:** Simulation Engineers, Test Directors, Military Operators, and Software Developers  
**System Version:** `v1.0.0`

---

## Table of Contents
1. [System Prerequisites & Installation](#1-system-prerequisites--installation)
2. [Quick Verification & Automated Testing](#2-quick-verification--automated-testing)
3. [Interactive 2D Operational UI Console](#3-interactive-2d-operational-ui-console)
4. [High-Performance Inference API Microservice](#4-high-performance-inference-api-microservice)
5. [Training Autonomous MARL Policies](#5-training-autonomous-marl-policies)
6. [Statistical Non-Determinism Verification](#6-statistical-non-determinism-verification)
7. [Doctrinal Realism Validation](#7-doctrinal-realism-validation)
8. [TSS Integration Modes (In-Process vs HTTP)](#8-tss-integration-modes-in-process-vs-http)
9. [Database Administration & Inspection](#9-database-administration--inspection)

---

## 1. System Prerequisites & Installation

### Hardware Requirements
- **Processor:** 64-bit Intel Core i7 / AMD Ryzen 7 or higher (multi-core recommended for parallel rollouts).
- **System Memory:** Minimum 8 GB RAM (16 GB recommended for 100-episode validation).
- **Graphics:** OpenGL/Direct2D capable GPU for the 2D operational UI (optional for headless batch runs).
- **Disk Storage:** At least 2 GB free disk space for dependencies, SQLite logs, and model checkpoints.

### Software Prerequisites
- **Operating System:** Windows 10/11 64-bit, Ubuntu 22.04 LTS+, or RHEL 8+.
- **Python Version:** Python 3.11 or 3.12 64-bit.
- **Package Manager:** `uv` (recommended for ultra-fast dependency resolution) or standard `pip`.

### Step-by-Step Setup
```bash
# 1. Clone repository or extract deliverable package
cd C:\Users\bpras\Desktop\TSS

# 2. Synchronize virtual environment with uv
uv sync

# 3. Verify Python environment
uv run python --version
```

---

## 2. Quick Verification & Automated Testing

Before running missions, verify that all 10 architectural layers and 421 automated tests pass:

```bash
# Run complete test suite in quiet mode
uv run pytest tests/ -q

# Run with verbose output across evaluation detectors
uv run pytest tests/evaluation/ -v

# Run static type verification
uv run mypy --explicit-package-bases src/
```

Expected output:
```text
421 passed in 65.08s
Success: no issues found in source files
```

---

## 3. Interactive 2D Operational UI Console

The system includes a real-time Pygame tactical display for visualizing combat engagements, sensor coverage, line-of-sight terrain occlusion, and autonomous neural decisions.

```bash
# Launch Level 5 Joint Multi-Domain Scenario at 1.0x real-time speed
uv run python scripts/run_ui.py --scenario-level 5 --speed 1.0
```

### Command Line Arguments
- `--scenario-level <1-5>`: Scenario tier (1: 1v1 Air, 2: 2v2 Air, 3: Ground Ambush, 4: Littoral Standoff, 5: Joint Joint Multi-Domain).
- `--speed <float>`: Playback multiplier (e.g. `0.5`, `1.0`, `2.0`, `5.0`).
- `--checkpoint <path>`: Load custom weights (default: `checkpoints/final/checkpoint_final_iter_00010.pt`).
- `--headless`: Run simulation without rendering display window.

### Keyboard & Mouse Controls
| Key / Input | Action |
|:---|:---|
| **Spacebar** | Toggle Pause / Resume simulation |
| **Right Arrow** | Single-step forward (when paused) |
| **Up / Down Arrow** | Increase / Decrease playback speed ($0.5\times \leftrightarrow 5.0\times$) |
| **R** | Reset current scenario to initial state |
| **Left Click** | Select entity to inspect telemetry (speed, altitude, health, active intent) |
| **Esc / Q** | Terminate UI application |

---

## 4. High-Performance Inference API Microservice

The FastAPI inference microservice enables external simulators, C++ tactical runtimes, and remote client nodes to query trained neural policies over HTTP.

### Launching the Server
```bash
uv run python scripts/run_api.py --host 0.0.0.0 --port 8089 --checkpoint-dir checkpoints/final
```

### Health Check Request
```bash
curl -X GET http://localhost:8089/health
```

### Single Entity Inference Request (`/act`)
```bash
curl -X POST http://localhost:8089/act \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "blue_air_1",
    "domain": "air",
    "variant": "AC1",
    "observation": [50.0, 50.0, 8.5, 350.0, 90.0, 0.0, 0.0, 1.0, 100.0, 4.0, 60.0, 55.0, 8.0],
    "deterministic": true
  }'
```

Response:
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

## 5. Training Autonomous MARL Policies

The system includes a hierarchical curriculum training pipeline (`src/training/`).

### Training All Policies (Curriculum Pipeline)
```bash
uv run python scripts/train_all.py --iterations 100 --curriculum --save-dir checkpoints/custom_run
```

### Training Commander Policy Standalone
```bash
uv run python scripts/train_commander.py --iterations 50 --log-interval 5
```

### Training Low-Level Domain Policies Standalone
```bash
uv run python scripts/train_low_level.py --domain air --iterations 50
```

### Monitoring Training Progress
Training metrics are logged continuously to SQLite (`data/tactical_marl.db`) and TensorBoard:
```bash
uv run tensorboard --logdir runs/
```

---

## 6. Statistical Non-Determinism Verification

Verify that autonomous behaviors are non-deterministic, diverse, and reproducible:

```bash
uv run python scripts/verify_non_determinism.py --episodes 100 --output-dir reports/non_determinism
```

### Output Interpretation
- **Chi-Square Goodness-of-Fit ($p < 0.05$):** Confirms that engagement outcomes vary significantly across random seeds.
- **Levene's Variance Test ($p < 0.05$):** Verifies variance in casualty timelines.
- **Same-Seed Test ($L_\infty = 0.0, p = 1.0$):** Proves bit-identical replay capability.
- **K-Means Trajectory Clusters ($\ge 3$):** Proves that entities adopt distinct spatial maneuvers rather than converging to a single repetitive path.

Generated report files:
- `reports/non_determinism/report.json`
- `reports/non_determinism/report.md`
- `docs/NON_DETERMINISM_REPORT.md`

---

## 7. Doctrinal Realism Validation

Evaluate whether autonomous agents behave according to established military combat doctrine across 16 tactical patterns:

```bash
uv run python scripts/validate_realism.py --num-episodes 100 --scenario-level 5 --output-dir reports/realism
```

### Understanding the 16-Row Audit Table
Each doctrine is evaluated under a strict two-level standard:
1. **Level 1 (Per-Episode Presence):** Requires raw detector confidence $\ge 0.50$.
2. **Level 2 (Cross-Episode Acceptance):** Requires prevalence $\ge 20.0\%$ across all 100 episodes.

*Note on Dry-Run Checkpoints:* On the initial 5-iteration dry-run checkpoint, 8 of 16 doctrines are detected (50.0% detection rate). Complex collective tactics (such as synchronized pincer movements, mutual supporting fire, and screen formations) naturally require multi-thousand iteration training to cross the 20% prevalence threshold.

Generated report files:
- `reports/realism/report.json`
- `reports/realism/report.md`
- `docs/REALISM_VALIDATION_REPORT.md`

---

## 8. TSS Integration Modes (In-Process vs HTTP)

### Mode A: Zero-Copy In-Process Direct Adapter (Recommended for Maximum Speed)
Use Mode A when DRDO's TSS runs on the same host machine in Python:
```python
from src.integration.tss_wrapper import TSSInProcessWrapper

# Initialize wrapper
wrapper = TSSInProcessWrapper(checkpoint_path="checkpoints/final/checkpoint_final_iter_00010.pt")

# Step environment
actions = wrapper.get_actions_batch(observations_dict)
print("Inference executed in 0.58 ms:", actions)
```

### Mode B: HTTP REST Proxy Adapter (Multi-Machine / Foreign Language)
Use Mode B when TSS is deployed across a local network or implemented in C++/C#:
```python
from src.integration.tss_wrapper import TSSHttpWrapper

wrapper = TSSHttpWrapper(api_url="http://localhost:8089")
actions = wrapper.get_actions_batch(observations_dict)
```

Detailed field mappings and unit conversion tables are documented in [docs/INTEGRATION.md](INTEGRATION.md).

---

## 9. Database Administration & Inspection

The persistence layer stores scenario configurations, run histories, performance metrics, and model checkpoint registries in an embedded SQLite database (`data/tactical_marl.db`).

### Inspecting Database Tables
```bash
sqlite3 data/tactical_marl.db ".tables"
```

### Querying Scenarios
```bash
sqlite3 data/tactical_marl.db "SELECT scenario_id, name, level, map_size_km FROM scenarios;"
```

### Querying Model Checkpoints
```bash
sqlite3 data/tactical_marl.db "SELECT policy_name, stage, iteration, file_size_bytes FROM model_checkpoints;"
```

### Running Automated Database Smoke Tests
```bash
uv run python scripts/db_smoke_test.py
```

---
*Defence Research & Development Organisation (DRDO) - Operator & User Manual*
