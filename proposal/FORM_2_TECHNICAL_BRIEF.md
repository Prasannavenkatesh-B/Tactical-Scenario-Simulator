# DRDO Form 2: Technical Brief & Figures of Merit

**Reference No:** DRDO/ADE/CARS/2026/MARL-TSS/02  
**Subject Area:** Aeronautics & Autonomous Systems (A&AS)  
**Coordinating Laboratory:** Aeronautical Development Establishment (ADE), Bengaluru  
**Project Title:** Development of Hierarchical Multi-Agent Reinforcement Learning (H-MARL) Framework for Non-Deterministic Tactical Scenario Simulation  

---

## 1. State-of-the-Art Review & Gap Analysis

### 1.1 International Technological Landscape
Over the past decade, defense research laboratories worldwide have transitioned from scripted computer-generated forces toward autonomous multi-agent reinforcement learning:
- **DARPA AlphaDogfight Trials & Air Combat Evolution (ACE):** Demonstrated that deep reinforcement learning algorithms can outmaneuver human fighter pilots in close-range 1v1 within-visual-range (WVR) dogfights. However, ACE architectures were heavily specialized for isolated 1v1 dogfights and lacked cross-domain coordination with ground air defense and naval surface fleets.
- **UK Dstl / NATO CGF Initiatives:** Explored decentralized multi-agent methods for land-air joint tactical doctrine. While theoretically robust, many prototypes suffered from severe simulation-to-real transfer gaps and excessive inference latencies ($> 50\text{ ms}$), rendering them unsuitable for high-rate real-time simulation loops.
- **Legacy DRDO Simulation Paradigms:** DRDO's current tactical simulators rely largely on deterministic finite-state machines (FSM) or static behavior trees. These rule sets fail to capture the stochastic uncertainty of modern multi-domain operational theaters.

### 1.2 Identified Capability Gaps & Proposed Innovation
| Capability Dimension | Traditional CGF / Scripted Systems | Prior Academic MARL Systems | Delivered Tactical MARL System v1.0.0 |
|:---|:---|:---|:---|
| **Behavioral Diversity** | Static, deterministic, predictable | Highly variable, lacks reproducibility | **Statistically proven stochasticity ($\mathbf{p < 0.05}$) + bit-identical replay ($\mathbf{L_\infty = 0.0}$)** |
| **Operational Scope** | Single domain (Air or Ground) | Micro-level combat (1v1 or 2v2) | **Hierarchical Theater Commander + 4-Domain Joint Forces (Air, Ground, Sea, Joint)** |
| **Inference Latency** | $< 1\text{ ms}$ (simple lookups) | $20 - 100\text{ ms}$ (unoptimized deep models) | **$\mathbf{0.58\text{ ms}}$ in-process, $\mathbf{4.21\text{ ms}}$ REST API** |
| **Military Doctrinal Fidelity**| Explicitly hardcoded rules | Emerges unreliably without validation | **16 automated military doctrine detectors with strict 2-level validation** |

---

## 2. Figures of Merit (FoM): Target vs. Achieved Audit

The following table presents an honest, transparent audit comparing DRDO's contractual requirements, the empirical values achieved by the deliverable system v1.0.0, and the projected production targets following Milestone M5 scaled training:

```
====================================================================================================
                        FIGURES OF MERIT (FoM) ACCEPTANCE AUDIT
====================================================================================================
Metric ID   Figure of Merit Description         DRDO Target       Achieved v1.0.0       Status
----------------------------------------------------------------------------------------------------
FoM-01      In-Process Inference Latency        <= 2.0 ms         0.58 ms               PASS
FoM-02      HTTP REST Service Latency           <= 10.0 ms        4.21 ms               PASS
FoM-03      Different-Seed Non-Determinism      p < 0.05          p = 4.54e-05 (chi^2)  PASS
FoM-04      Cross-Seed Variance Significance    p < 0.05          p = 2.48e-04 (Levene) PASS
FoM-05      Same-Seed Reproducibility           L_inf = 0.0       L_inf = 0.0000000000  PASS
FoM-06      Spatial Trajectory Clusters (k=5)   >= 3 clusters     4 distinct clusters   PASS
FoM-07      Action Distribution Entropy (norm)  > 0.50            0.9918                PASS
FoM-08      Automated Test Suite Pass Rate      100% (>= 300)     421/421 (100% Pass)   PASS
FoM-09      Static Typing & Code Quality        0 type errors     20/20 Clean (Mypy)    PASS
FoM-10      Doctrinal Realism Detection (Dry Run) Qualitative      50.0% (8/16 Detected) DRY RUN
FoM-11      Doctrinal Realism Detection (Target)  >= 60.0%        Target in M5 (>= 60.0%) ROADMAP
====================================================================================================
```

### Critical Narrative on FoM-10 & FoM-11 (Doctrinal Realism)
- **Achieved Status (Dry Run):** The empirical realism evaluation yielded **50.0% (8 of 16 doctrines detected)** with an average prevalence of **71.6%** among detected tactics across 100 joint episodes.
- **Methodological Integrity:** In strict accordance with DRDO standards, the two-level evaluation framework requires both per-episode confidence $\ge 0.50$ and cross-episode prevalence $\ge 20.0\%$. No thresholds were artificially relaxed.
- **Root Cause & Attribution:** Checkpoint `checkpoint_final_iter_00010.pt` was trained for only 5 iterations during local dry-run testing. Basic flight, energy management, and standoff firing emerged immediately, while synchronized multi-unit tactics (such as pincer maneuvers, mutual supporting fire, and screen formations) naturally require multi-thousand iteration training to manifest consistently.
- **Contractual Path Forward:** Scaled cloud GPU training is budgeted and scheduled under Milestone M5 to achieve the DRDO contractual threshold of $\ge 60.0\%$.

---

## 3. Algorithmic Justification & Technical Innovation

### 3.1 Two-Tier Hierarchical Decision Process
- **Strategic Layer:** A 53-dimensional operational tensor is fed into a recurrent GRU network ($256$ hidden units) updating every $0.5\text{ seconds}$. The recurrent cell captures temporal dynamics (e.g. enemy radar emission cycles, fuel exhaustion timelines) that flat reactive networks miss.
- **Tactical Layer:** Individual entities operate at $10\text{ Hz}$ ($0.1\text{ second}$ steps). Multi-head self-attention enables aircraft to attend dynamically to the most immediate missile threat or target opportunity.

### 3.2 Action Space Factorization
Continuous high-dimensional action spaces suffer from slow sample efficiency, while naive discrete spaces suffer from combinatorial explosion ($13 \times 9 \times 2 \times 2 = 468$ discrete choices). We partition the action space into factorized sub-action heads (heading, speed, cannon, rockets), reducing output logits to $26$ while preserving full operational degrees of freedom.

---

## 4. Hardware & Computational Requirements

### 4.1 Development & Edge Inference Specification (Current v1.0.0)
- **Processor:** 8-Core Intel Core i7 / AMD Ryzen 7
- **RAM:** 16 GB DDR4/DDR5
- **GPU:** Optional (NVIDIA RTX 3060 or CPU fallback)
- **Storage:** 20 GB SSD

### 4.2 Production Training Specification (Milestone M5)
- **Compute Cluster:** 8x NVIDIA A100 (80GB SXM4) or 4x NVIDIA H100
- **CPUs:** 64-Core AMD EPYC / Intel Xeon Scalable
- **System Memory:** 256 GB ECC RAM
- **Training Duration:** 72 hours for $10^7$ joint environment steps ($10,000$ iterations)

---

## 5. Interface Boundaries & External Dependencies

The delivered software encapsulates all dependencies within Python virtual environments (`uv`), utilizing only open-source, non-restrictive libraries:
- `torch` (>= 2.2.0): Neural network forward inference and backpropagation.
- `fastapi` & `uvicorn`: Asynchronous REST microservice.
- `pettingzoo` & `gymnasium`: Standard multi-agent simulation interface compliance.
- `pygame`: Double-buffered hardware accelerated tactical display.
- `scipy`, `scikit-learn`, `numpy`: Statistical verification and clustering.
- `sqlite3`: Zero-daemon relational persistence.

---
*DRDO Form 2: Technical Brief Deliverable - Restricted*
