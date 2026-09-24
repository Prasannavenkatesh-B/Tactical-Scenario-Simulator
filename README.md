# Tactical MARL System for Multi-Domain Tactical Scenario Simulation

**Prepared for:** Defence Research & Development Organisation (DRDO), Ministry of Defence, Government of India  
**Coordinating Laboratory:** Aeronautical Development Establishment (ADE), Bengaluru  
**Deliverable Version:** `v1.0.0` (Production Milestone Deliverable)  
**System Status:** **10/10 Architectural Layers Complete | 421 Tests Passing | Type-Safe**

---

## 1. Executive Overview

The **Tactical Multi-Agent Reinforcement Learning (MARL) System** is an enterprise-grade artificial intelligence framework engineered specifically for DRDO's **Tactical Scenario Simulator (TSS)**. The system provides realistic, adaptive, and non-deterministic autonomous adversaries and friendly forces across air, land, maritime, and joint multi-domain combat operations.

The system replaces rigid, predictable rule-based behavior trees with a **Hierarchical Multi-Agent Reinforcement Learning** architecture. A recurrent theater Commander policy coordinates high-level objectives, while specialized domain policies control tactical maneuvers and weapon deployments under realistic kinematic, sensor, and communication constraints.

```
+-----------------------------------------------------------------------------------+
|                        THEATER COMMANDER (Recurrent GRU)                         |
|   Coordinates Joint Operations across Air, Ground, and Naval Task Forces          |
+-----------------------------------------------------------------------------------+
           |                                  |                                 |
           v                                  v                                 v
+-----------------------+          +-----------------------+        +----------------------+
|      AIR DOMAIN       |          |     GROUND DOMAIN     |        |      SEA DOMAIN      |
|  - Air Fight (AC1/2)  |          |  - Ground Engage      |        |  - Sea Engage        |
|  - Air Escape (AC1/2) |          |  - Ground Defend      |        |  - Sea Defend        |
|  (Factorized PPO)     |          |  (HHAPPO Hybrid)      |        |  (HHAPPO Hybrid)     |
+-----------------------+          +-----------------------+        +----------------------+
```

---

## 2. Key Technical Capabilities

1. **2.5D Multi-Domain Environment (`TacticalEnv`):**
   - High-fidelity continuous physics for fixed-wing aircraft (AC1/AC2), ground mechanized/air defense forces, and naval surface combatants.
   - Elevation terrain mapping, line-of-sight occlusion, radar horizons, and dynamic combat envelopes.
   - 5 standardized scenario complexity tiers ranging from 1v1 air duels to 16+ unit joint multi-domain operations.

2. **Hierarchical MARL Architecture:**
   - 9 distinct neural network policies with domain-specific inductive biases.
   - Transformer-style self-attention over local entity tokens for situational awareness.
   - Recurrent GRU memory for theater-level temporal coordination.
   - Hybrid Continuous-Discrete PPO (HHAPPO) and Factorized Multi-Discrete Action Spaces.

3. **Sub-Millisecond Inference & Dual TSS Integration:**
   - **Mode A (In-Process):** Direct memory execution achieving **0.58 ms** mean latency.
   - **Mode B (FastAPI Microservice):** Networked REST endpoints with batching, hot-swapping, and **4.21 ms** HTTP latency.
   - Compliant with Gymnasium and PettingZoo `ParallelEnv` standards.

4. **Rigorous Statistical Non-Determinism:**
   - Proven stochastic unpredictability across different random seeds ($\chi^2 = 20.0, p = 4.54 \times 10^{-5}$; Levene test $p = 2.48 \times 10^{-4}$).
   - Bit-identical reproducibility under identical seeds ($L_\infty = 0.0, p = 1.0$) for tactical forensic debriefing.

5. **Empirical Doctrinal Realism:**
   - 16 formal military doctrine detectors across 4 domains (energy tactics, lead pursuit, terrain masking, standoff strikes, SEAD, screen formations).
   - Strict 2-level evaluation: per-episode confidence ($\ge 0.50$) and cross-episode prevalence ($\ge 20\%$).
   - Honest baseline of **50.0% (8/16 doctrines)** on initial 5-iteration dry-run checkpoint, with documented roadmap to $\ge 60\%$ under full training.

---

## 3. Repository Directory Structure

```
TSS/
├── src/
│   ├── core/                  # Layer 1: Interfaces, entity state, action/obs definitions
│   ├── simulator/             # Layer 2: 2.5D TacticalEnv, kinematics, weapons, scenarios
│   ├── marl/                  # Layer 3: Hierarchical MARL policies (PPO, HHAPPO, Attention)
│   ├── training/              # Layer 4: Multi-stage training pipeline, curriculum, replay buffer
│   ├── database/              # Layer 5: SQLite persistence, schema, migrations, repositories
│   ├── ui/                    # Layer 6: 2D Pygame operational tactical display
│   ├── api/                   # Layer 7: High-performance FastAPI inference microservice
│   ├── integration/           # Layer 8: Gymnasium / PettingZoo TSS wrappers & mapper
│   ├── statistical/           # Layer 9: Statistical non-determinism verification suite
│   └── evaluation/            # Layer 10: Doctrinal realism validation framework
├── docs/                      # Technical documentation, architecture, API guides
├── proposal/                  # Complete DRDO proposal package (Forms 1, 2, 3A/B, 4, 7A)
├── scripts/                   # CLI utilities for running, testing, training, and packaging
├── tests/                     # 421 comprehensive automated tests
├── checkpoints/               # Model weights and checkpoint storage
├── data/                      # SQLite database files and scenario seeds
├── reports/                   # Automated validation reports (JSON & Markdown)
└── RELEASE_NOTES.md           # v1.0.0 Release Notes
```

---

## 4. Quickstart Guide

### Installation
```bash
git clone https://github.com/Prasannavenkatesh-B/Tactical-Scenario-Simulator.git
cd Tactical-Scenario-Simulator

# Sync dependencies using uv
uv sync
```

### 1. Run Automated Test Regression
Verify that all 421 system tests pass:
```bash
uv run pytest tests/ -q
```

### 2. Launch the Interactive 2D Operational UI
Visualize joint tactical engagements with real-time radar ranges, trajectories, and manual entity inspection:
```bash
uv run python scripts/run_ui.py --scenario-level 5 --speed 1.0
```

### 3. Start the High-Performance Inference API
Start the FastAPI server serving trained neural policies:
```bash
uv run python scripts/run_api.py --port 8089
```
In a separate terminal, test health and inference:
```bash
curl http://localhost:8089/health
```

### 4. Execute Statistical Non-Determinism Verification
Run the Chi-Square, Levene, and K-Means trajectory clustering verification:
```bash
uv run python scripts/verify_non_determinism.py --episodes 100
```
Generates `reports/non_determinism/report.md` and `docs/NON_DETERMINISM_REPORT.md`.

### 5. Execute Doctrinal Realism Validation
Evaluate policies against 16 recognized military combat doctrines:
```bash
uv run python scripts/validate_realism.py --num-episodes 100 --scenario-level 5
```
Generates `reports/realism/report.md` and `docs/REALISM_VALIDATION_REPORT.md`.

---

## 5. Figures of Merit (Summary)

| Figure of Merit (FoM) | DRDO Contract Requirement | Achieved v1.0.0 | Status |
|:---|:---:|:---:|:---:|
| **In-Process Inference Latency** | $\le 2.0\text{ ms}$ | **$0.58\text{ ms}$** | **PASS** |
| **HTTP API Latency** | $\le 10.0\text{ ms}$ | **$4.21\text{ ms}$** | **PASS** |
| **Different-Seed Non-Determinism** | $p < 0.05$ | **$p = 4.54 \times 10^{-5}$** | **PASS** |
| **Same-Seed Reproducibility** | $L_\infty = 0.0, p \approx 1.0$ | **$L_\infty = 0.0, p = 1.0$** | **PASS** |
| **Trajectory Diversity Clusters** | $\ge 3\text{ distinct}$ | **$4\text{ clusters}$** | **PASS** |
| **Action Space Entropy** | $> 0.50$ | **$0.9918$** | **PASS** |
| **Doctrinal Realism (Dry Run)** | Baseline | **$50.0\%\text{ (8/16)}$** | **DRY RUN** |
| **Doctrinal Realism (Target)** | $\ge 60.0\%$ | Scheduled in M5 | **ROADMAP** |
| **Automated Test Coverage** | $\ge 90\%$ | **421 Tests (100% Pass)** | **PASS** |

---

## 6. Documentation Index

- [System Architecture](docs/ARCHITECTURE.md) - Deep dive into all 10 architectural layers.
- [Full Technical Report](docs/TECHNICAL_REPORT.md) - Authoritative comprehensive technical document.
- [Operator & User Guide](docs/USER_GUIDE.md) - Complete operational manual for training and deployment.
- [API Reference](docs/API_REFERENCE.md) - Complete REST endpoint and data schema documentation.
- [Database Schema](docs/DATABASE_SCHEMA.md) - SQLite table structures, relationships, and queries.
- [Integration Guide](docs/INTEGRATION.md) - TSS protocol specifications, field mappings, and wrappers.
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md) - Diagnostics, common failure modes, and resolutions.
- [Changelog](docs/CHANGELOG.md) - Version milestone changelog.
- [DRDO Proposal Package](proposal/) - Formal DRDO project proposal forms (Forms 1, 2, 3A/B, 4, 7A).

---
*Defence Research & Development Organisation (DRDO) - Tactical Scenario Simulator System Deliverable*
