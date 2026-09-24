# DRDO Form 1: Project Synopsis & Executive Summary

**Reference No:** DRDO/ADE/CARS/2026/MARL-TSS/01  
**Scheme:** Extramural Research & Technology Development / Contract for Acquisition of Research Services (CARS)  
**Coordinating Laboratory:** Aeronautical Development Establishment (ADE), Bengaluru - 560075  
**Subject Technical Cluster:** Aeronautics & Autonomous Systems (A&AS)  

---

## 1. Project Identification

| Parameter | Specification |
|:---|:---|
| **Project Title** | Development of Hierarchical Multi-Agent Reinforcement Learning (H-MARL) Framework for Non-Deterministic Tactical Scenario Simulation |
| **Short Title** | Tactical MARL for DRDO TSS |
| **Project Category** | Applied Research & Prototype Technology Development |
| **Duration** | 12 Months (with phased milestone reviews) |
| **Principal Investigating Institution** | Advanced Agentic AI Laboratory / DRDO Joint Academic Consortium |
| **Lead DRDO Laboratory** | Aeronautical Development Establishment (ADE), New Thippasandra, Bengaluru |
| **Participating DRDO Laboratories** | CAIR (Bengaluru), DEAL (Dehradun), NPOL (Kochi) |
| **Total Proposed Outlay** | INR 8,000 (Direct Costs: Cloud GPU Colab Pro INR 6,000, Incidentals INR 2,000; all development workstations, lab space, and mentorship provided as institutional in-kind contributions) |

---

## 2. Project Synopsis

Modern tactical warfare simulations developed by the Defence Research & Development Organisation (DRDO), such as the **Tactical Scenario Simulator (TSS)**, require autonomous computer-generated forces (CGF) that emulate the fluid, reactive, and non-deterministic decision-making of authentic human combat operators. Existing commercial-off-the-shelf (COTS) and legacy military simulators rely predominantly on rule-based behavior trees or scripted state machines. These deterministic models exhibit severe tactical brittleness: combat trainees quickly detect predictable patterns, exploit algorithmic blind spots, and develop negative training habits that do not translate to contested operational theaters.

This project delivers a production-grade, hierarchical multi-agent reinforcement learning (H-MARL) system specifically engineered for seamless integration into DRDO's TSS architecture. The software deploys a two-tier cognitive hierarchy:
1. **Theater Strategic Commander:** A recurrent Gated Recurrent Unit (GRU) policy that processes the complete multi-domain tactical picture and coordinates joint operational posture.
2. **Tactical Domain Controllers:** Specialized neural policies operating across Air (AC1/AC2 fighters), Ground (mechanized units and air defense SAMs), and Maritime (naval surface combatants) domains. Policies leverage multi-head self-attention mechanisms and hybrid continuous-discrete action representations (HHAPPO).

The deliverable system operates at sub-millisecond execution speeds (**0.58 ms** direct in-process latency), guarantees bit-identical reproducibility under identical seeds ($L_\infty = 0.0, p = 1.0$), and statistically proves non-deterministic behavioral diversity under varying seeds ($\chi^2 = 20.0, p = 4.54 \times 10^{-5}$; Levene variance test $p = 2.48 \times 10^{-4}$). The initial v1.0.0 engineering deliverable establishes an empirical baseline of **50.0% (8/16 doctrines detected)** on dry-run checkpoints, with a funded 12-month compute and training plan to achieve $\ge 60\%$ realism in Milestone M5 and $\ge 80\%$ by project completion.

---

## 3. Key Objectives

1. **Deliver a Decoupled 10-Layer Software Architecture:** Build a modular tactical simulation and inference engine fully compliant with DRDO TSS protocols, supporting both Python in-process zero-copy execution and networked REST microservice operation.
2. **Prove Statistical Non-Determinism:** Eliminate scripted predictability by statistically validating behavioral and spatial trajectory diversity across scenario seeds.
3. **Establish Military Combat Doctrinal Realism:** Implement automated telemetry detectors for 16 recognized air, ground, naval, and joint doctrines based on Indian Air Force (IAF) and Indian Navy operational doctrine.
4. **Execute Scaled Cloud Multi-GPU Training:** Scale neural policy training from initial dry-run checkpoints to multi-thousand iteration production regimes on high-performance compute clusters, transitioning doctrinal realism from the 50.0% dry-run baseline to $\ge 60.0\%$ (Milestone M5) and $\ge 80.0\%$ (Final Acceptance).
5. **Ensure Real-Time Performance & Test Coverage:** Maintain $< 2.0\text{ ms}$ inference latencies, 100% automated test regression pass rates, and strict type safety across all system modules.

---

## 4. Tangible Deliverables

| Deliverable ID | Description | Delivery Timeline |
|:---:|:---|:---:|
| **D1** | Complete 10-Layer Source Codebase (Core, Simulator, MARL, Database, UI, API, Wrappers) | Month 0 (Achieved v1.0.0) |
| **D2** | Statistical Non-Determinism Verification Engine with Automated PDF/Markdown Reporting | Month 0 (Achieved v1.0.0) |
| **D3** | Doctrinal Realism Validation Framework (16 Detectors, Two-Level Evaluation System) | Month 0 (Achieved v1.0.0) |
| **D4** | High-Performance FastAPI Inference Microservice & Zero-Copy TSS Wrapper (0.58 ms) | Month 0 (Achieved v1.0.0) |
| **D5** | Scaled Multi-GPU Training Run & Validated Checkpoints ($\ge 60\%$ Doctrinal Realism) | Month 6 (Milestone M5) |
| **D6** | Full TSS Hardware-in-the-Loop (HIL) Integration & Joint User Acceptance Testing at ADE | Month 12 (Milestone M6) |

---

## 5. Strategic Relevance to DRDO & Indian Armed Forces

- **Enhanced Pilot & Operator Training Fidelity:** Exposes IAF fighter pilots and Indian Navy mission commanders to autonomous adversaries that employ dynamic pursuit curves, terrain masking, and evasive maneuvers rather than fixed scripts.
- **Indigenous Intellectual Property & Security:** Delivers an entirely indigenous AI software stack free from foreign proprietary licensing, vendor lock-in, or hidden telemetric backdoors.
- **Cross-Domain Joint Force Doctrine Formulation:** Provides military strategists with an empirical testbed to explore emergent cross-domain coordination (e.g. SEAD air sweeps coordinated with ground mechanized offensives and naval missile strikes).

---
*DRDO Form 1: Project Synopsis Deliverable - Restricted*
