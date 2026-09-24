# Tactical MARL Post-Delivery Training Roadmap & Scaling Strategy

**Document Reference:** DRDO-ADE-TSS-TRM-2026  
**Project Title:** Scaled Training Architecture for Hierarchical Multi-Agent Tactical Systems  
**Coordinating Laboratory:** Aeronautical Development Establishment (ADE), Bengaluru  
**Target Milestone:** Milestone M5 (Realism Acceptance $\ge 60.0\%$)  
**System Deliverable:** `v1.0.0` Production Training Specification  

---

## 1. Executive Context & Objective

The software deliverable v1.0.0 verifies the complete 10-layer software stack, achieving sub-millisecond inference (**0.58 ms**), rigorous statistical non-determinism ($\chi^2 = 20.0, p = 4.54 \times 10^{-5}$), and a baseline doctrinal realism rate of **50.0% (8/16 doctrines detected)** on the initial 5-iteration dry-run checkpoint (`checkpoint_final_iter_00010.pt`).

As established in the technical report, individual combat tactics (energy management, pursuit curves, standoff firing, terrain masking) emerge rapidly, while collective multi-agent doctrines (pincer maneuvers, mutual support, naval screens, and air-ground coordination) require multi-thousand iteration training to cross the strict $20\%$ prevalence threshold.

This document details the **Post-Delivery Training Roadmap** to execute **5,000+ distributed iterations** across $10^7$ environment steps, advancing the overall realism detection rate from **50.0% to $\ge 60.0\%$ (contractual requirement)** in Milestone M5.

---

## 2. Distributed Training Infrastructure Architecture

To transition from local workstation dry runs to production multi-agent reinforcement learning, the training pipeline will deploy across a distributed high-performance computing cluster at ADE:

```
+===================================================================================================+
|                        64-WORKER DISTRIBUTED RAY / PPO TRAINING CLUSTER                           |
+===================================================================================================+

    +-------------------+   +-------------------+       +-------------------+
    | Rollout Worker 01 |   | Rollout Worker 02 | ..... | Rollout Worker 64 |  (64 CPU Cores)
    | TacticalEnv Inst. |   | TacticalEnv Inst. |       | TacticalEnv Inst. |  (Simulates 100 km x 100 km)
    +-------------------+   +-------------------+       +-------------------+
              \                       |                       /
               \                      |                      / Trajectory Batches
                v                     v                     v (s_t, a_t, r_t, s_{t+1})
    +-----------------------------------------------------------------------+
    | DISTRIBUTED SHARED REPLAY BUFFER (Ray Plasma Store / GAE Worker Pool)  |
    | - Computes Generalized Advantage Estimation (gamma=0.99, lambda=0.95) |
    | - Normalizes Advantages & Calculates Action Probabilities             |
    +-----------------------------------------------------------------------+
                                      |
                                      | Minibatches (Batch Size = 4096)
                                      v
    +-----------------------------------------------------------------------+
    | LEARNER GPU NODE (8x NVIDIA A100 SXM4 80GB - PyTorch DDP)             |
    | - Parallel Backward Pass across 9 Domain Policies + Commander GRU     |
    | - PPO Clipped Surrogate Loss (epsilon = 0.20) + Gradient Clip (0.5)   |
    | - Ring AllReduce Parameter Synchronization across GPUs                |
    +-----------------------------------------------------------------------+
                                      |
                                      | Synchronized Model Weights Snapshot
                                      v
    +-----------------------------------------------------------------------+
    | MODEL REGISTRY & SQLITE PERSISTENCE (checkpoints/production/)         |
    | - Evaluates Realism Validation Pipeline every 250 Iterations (N=100)  |
    | - Logs Loss, Win Rate, and Doctrine Prevalence to SQLite & TensorBoard|
    +-----------------------------------------------------------------------+
```

### Cluster Resource Allocation
- **Rollout Sampling:** 64 CPU cores dedicated to parallel `TacticalEnv` simulation instances generating $\sim 65,000$ environment transitions per second.
- **Neural Backpropagation:** 8x NVIDIA A100 GPUs executing Distributed Data Parallel (DDP) gradient descent.
- **Estimated Wall-Clock Time:** 72 continuous hours for $5,000$ iterations ($10.24 \times 10^6$ total environment steps).

---

## 3. Phased Curriculum Training Schedule

```
+===================================================================================================+
|                                  THREE-STAGE CURRICULUM TIMELINE                                  |
+===================================================================================================+
Iteration:  0 -------- 1,000 ------------------- 3,000 ------------------------------ 5,000+
Stage:      [   STAGE 1: BASIC COMBAT   ] [  STAGE 2: DOMAIN TACTICS  ] [ STAGE 3: JOINT THEATER ]
Scenario:   Level 1 & Level 2 (Air 1v1/2v2) Levels 3 & 4 (Ground/Naval) Level 5 (Multi-Domain Joint)
Target:     Kinematic Control & BFM       Formations & Standoff Fire  Commander GRU & Cross-Domain
```

### Stage 1: Basic Platform Flight Dynamics & 1v1/2v2 Air Combat (Iterations 1–1,000)
- **Scenarios:** Scenario Level 1 (1v1 Duel) and Level 2 (2v2 Air Tactical Engagement).
- **Focus:** Basic Fighter Maneuvers (BFM), sustained vs instantaneous turn rates, energy retention, and gun/missile lead pursuit.
- **Expected Realism Progress:** Solidifies `pursuit_curve`, `lead_pursuit`, and `energy_management` to $> 95\%$ prevalence. Promotes `lag_pursuit` to $> 30\%$.

### Stage 2: Tactical Domain Combat & Formation Discipline (Iterations 1,001–3,000)
- **Scenarios:** Scenario Level 3 (Ground Convoy vs SAM Defense) and Level 4 (Naval Littoral Standoff).
- **Focus:** Terrain masking for ground mechanized armor, SAM radar line-of-sight defense, naval standoff missile exchanges, and defensive screens.
- **Reward Shaping:** Introduce spatial proximity rewards for mutual supporting ground fire and naval screen spacing around high-value units.
- **Expected Realism Progress:** Promotes `mutual_support` and `screen_formation` from $0\%$ to $> 35\%$ prevalence; raises `terrain_cover` to $> 65\%$.

### Stage 3: Joint Multi-Domain Theater Coordination (Iterations 3,001–5,000+)
- **Scenarios:** Scenario Level 5 (Full Joint Multi-Domain Theater with 16+ combined arms units).
- **Focus:** Coordinated recurrent Theater Commander GRU guidance. Synchronizes air penetrations with ground offensive thrusts and naval missile barrages.
- **Reward Shaping:** Cross-domain cooperative bonuses:
  - Air SEAD strikes disabling SAM radars receive bonus rewards proportional to subsequent ground unit survival.
  - Synchronized dual-aircraft pincer attacks (angular offset $> 60^\circ$ on common target) receive tactical convergence bonuses.
- **Expected Realism Progress:** Drives `pincer_maneuver` ($> 45\%$), `sead_support` ($> 60\%$), and `air_ground_coordination` ($> 40\%$).

---

## 4. Projected Doctrinal Realism Progression (Milestone M5 Target)

The following matrix projects the evolution of all 16 military combat doctrine patterns from the dry-run baseline to Milestone M5 completion:

```
========================================================================================================================
                               DOCTRINAL REALISM PROGRESSION (DRY RUN -> MILESTONE M5)
========================================================================================================================
Doctrine Name                   Domain      Current Dry Run (5 Iters)       Milestone M5 Target (5,000 Iters)
------------------------------------------------------------------------------------------------------------------------
pursuit_curve                   Air         97.0% (DETECTED)                98.0% (DETECTED - High Fidelity)
lead_pursuit                    Air         29.0% (DETECTED)                65.0% (DETECTED - Robust Tracking)
lag_pursuit                     Air          0.0% (NOT DETECTED)            32.0% (DETECTED - Emergent Overshoot Guard)
defensive_break                 Air          0.0% (NOT DETECTED)            38.0% (DETECTED - High-G Missile Break)
energy_management               Air        100.0% (DETECTED)               100.0% (DETECTED - Energy Conservation)
pincer_maneuver                 Air         13.0% (NOT DETECTED)            48.0% (DETECTED - Coordinated Bracket)
threat_prioritization           Air        100.0% (DETECTED)               100.0% (DETECTED - High Value Target Focus)
terrain_cover                   Ground      40.0% (DETECTED)                72.0% (DETECTED - Active Masking)
mutual_support                  Ground       0.0% (NOT DETECTED)            36.0% (DETECTED - Bounding Overwatch)
engagement_range_discipline     Ground       7.0% (NOT DETECTED)            42.0% (DETECTED - Max Weapon Envelope)
standoff_engagement             Maritime    76.0% (DETECTED)                85.0% (DETECTED - Over-the-Horizon Fire)
screen_formation                Maritime     0.0% (NOT DETECTED)            34.0% (DETECTED - Concentric Picket)
evasive_maneuver                Maritime     0.0% (NOT DETECTED)            38.0% (DETECTED - Serpentine Missile Evade)
air_ground_coordination         Joint        2.0% (NOT DETECTED)            44.0% (DETECTED - Joint Lasing & CAS)
sead_support                    Joint       31.0% (DETECTED)                68.0% (DETECTED - Integrated Air Suppression)
maritime_patrol                 Joint      100.0% (DETECTED)               100.0% (DETECTED - Chokepoint Dwell)
------------------------------------------------------------------------------------------------------------------------
OVERALL DETECTION RATE                      50.0% (8/16 Doctrines)          >= 60.0% (>= 10/16 Doctrines) -> CONTRACTUAL PASS
DRDO ACCEPTANCE VERDICT                     FAIL (Dry Run Baseline)         PASS (Meets >= 60.0% Standard)
========================================================================================================================
```

---

## 5. Verification & Acceptance Gates

1. **Automated Evaluation Checkpoints:** Every 250 training iterations, the pipeline automatically freezes model weights, executes 100 joint evaluation episodes across fixed test seeds, and runs `validate_realism.py`.
2. **Acceptance Gate Criteria (Milestone M5):**
   - Overall doctrine detection rate must be $\ge 60.0\%$ (at least 10 doctrines detected).
   - In-process forward inference latency must remain $\le 2.0\text{ ms}$.
   - Statistical non-determinism tests must maintain $p < 0.05$ across varying seeds.
   - Same-seed execution must maintain bit-identical reproducibility ($L_\infty = 0.0$).
3. **Model Artifact Handover:** Final production weights (`checkpoints/production/checkpoint_m5_final.pt`) will be transferred directly to ADE for hardware-in-the-loop TSS integration.

---
*DRDO Training Roadmap Deliverable - Restricted*
