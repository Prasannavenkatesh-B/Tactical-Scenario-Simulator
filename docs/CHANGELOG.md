# Changelog & Version History

All notable changes to the **Tactical Multi-Agent Reinforcement Learning (MARL) System** are documented in this file.

The project adheres strictly to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v1.0.0] - 2026-09-24 (Production Deliverable)
### Added
- Complete DRDO Proposal Package (`proposal/`):
  - `FORM_1_SUMMARY.md`: Project synopsis, PI details, coordinating lab (ADE Bengaluru).
  - `FORM_2_TECHNICAL_BRIEF.md`: State-of-the-art comparison, Figures of Merit (50% dry run achieved vs $\ge 60\%$ post-training target), data requirements.
  - `FORM_3A_LAB_RECOMMENDATION.md`: Recommendation template for Aeronautical Development Establishment.
  - `FORM_3B_COORDINATING_LAB.md`: Endorsement template for coordinating lab.
  - `FORM_4_EXTENDED_TECHNICAL.md`: Full multi-year research and execution plan, risk analysis, milestones M1–M6 (incorporating Milestone M5 for full cloud GPU training).
  - `FORM_7A_CERTIFICATE.md`: Standard compliance and intellectual property certification blocks.
  - `EXECUTIVE_SUMMARY.md`: Two-page executive brief framing dry-run lower bounds and scaling roadmap.
  - `FIGURES_OF_MERIT.md`: Complete FoM audit tracking all latency, statistical, and doctrinal realism metrics.
  - `TRAINING_ROADMAP.md`: Post-delivery training architecture (5,000+ iterations on 64-worker Ray/GPU cluster).
  - `BUDGET_AND_TIMELINE.md`: Comprehensive 12-month budget, personnel allocation, and equipment schedule.
- Comprehensive Technical Documentation Suite (`docs/`):
  - `ARCHITECTURE.md`: Detailed architectural design of all 10 system layers with dataflow diagrams.
  - `TECHNICAL_REPORT.md`: 5,000-word authoritative technical document detailing mathematics, dynamics, and benchmarks.
  - `USER_GUIDE.md`: Full operator and developer manual covering installation, training, evaluation, and UI control.
  - `API_REFERENCE.md`: Auto-generated FastAPI REST endpoint documentation.
  - `DATABASE_SCHEMA.md`: Complete SQLite relational schema, indices, ER diagram, and repository patterns.
  - `TROUBLESHOOTING.md`: Exhaustive failure diagnostics, UI fixes, port collisions, and concurrency guides.
  - `CHANGELOG.md`: Complete version history from inception to final release.
  - `LICENSE.txt`: DRDO-appropriate intellectual property placeholder.
- Automated Documentation & Packaging Scripts (`scripts/`):
  - `generate_api_docs.py`: Automated markdown generation from FastAPI OpenAPI introspections.
  - `generate_final_package.py`: Automated bundle creation into `deliverable.zip`.
- Full project test suite verified: **421 tests passing (100% success rate)**; static type checks clean across all source files.

---

## [v0.10.1] - 2026-09-24 (Doctrinal Realism Methodology Audit & Fix)
### Fixed
- Replaced permissive detector fallback clauses with strict two-level evaluation:
  - Level 1: Per-episode doctrine presence requires raw confidence $\ge 0.50$.
  - Level 2: Cross-episode acceptance requires prevalence $\ge 20.0\%$.
- Excised artificial heuristic defaults for missing units (e.g. mutual support returning $0.85$ when only 1 unit was present).
- Evaluated 100 episodes against dry-run checkpoint: honest baseline of 50.0% (8/16 doctrines detected) with average prevalence of 71.6% among detected tactics.

---

## [v0.10.0] - 2026-09-24 (Layer 10: Realism Validation Framework)
### Added
- 16 Recognized Military Combat Doctrine Detectors across Air, Ground, Maritime, and Joint Cross-Domain operational spheres.
- Automated JSON and Markdown report generation (`reports/realism/report.json`, `reports/realism/report.md`).
- Validation CLI utility `scripts/validate_realism.py`.

---

## [v0.9.0] - 2026-09-24 (Layer 9: Statistical Non-Determinism Engine)
### Added
- Statistical verification module (`src/statistical/`) replacing placeholder metrics.
- Chi-Square Goodness-of-Fit test ($p = 4.54 \times 10^{-5}$ rejecting uniform stagnation).
- Levene's test for variance homogeneity ($p = 2.48 \times 10^{-4}$).
- Normalized Shannon action space entropy ($H_{\text{norm}} = 0.9918$).
- K-Means spatial trajectory clustering ($k=5$, 4 distinct clusters, silhouette coefficient $0.385$).
- Same-seed bit-identical reproducibility verification ($L_\infty = 0.0, p = 1.0$).

---

## [v0.8.0] - 2026-09-24 (Layer 8: TSS Integration Wrapper)
### Added
- PettingZoo `ParallelEnv` compliant multi-agent wrapper (`src/integration/`).
- Gymnasium single-agent wrapper.
- Zero-copy direct in-process adapter achieving **0.58 ms** execution latency.
- Field alignment mapper converting foreign simulator units into normalized MARL tensors.
- Reference integration specification (`docs/INTEGRATION.md`).

---

## [v0.7.0] - 2026-09-24 (Layer 7: High-Performance Inference API)
### Added
- FastAPI asynchronous REST microservice (`src/api/`).
- Endpoints: `/health`, `/act`, `/act/batch`, `/reset`, `/policies`, `/load_checkpoint`.
- Sub-2ms forward neural pass; zero-downtime hot-swapping via `ModelRegistry`.
- Optimized `ActionCodec` and `ObservationCodec` for rapid serialization.

---

## [v0.6.0] - 2026-09-24 (Layer 6: 2D Tactical Operational UI)
### Added
- Interactive Pygame 2D tactical display (`src/ui/`).
- Real-time rendering of air, ground, and naval entities with radar coverage circles and weapon ranges.
- Interactive unit selection sidebar displaying live altitude, airspeed, health, and neural intent.
- Playback controls: play, pause, single-step, and speed throttling ($0.5\times \leftrightarrow 5.0\times$).

---

## [v0.5.0] - 2026-09-23 (Layer 5: Database Persistence Layer)
### Added
- SQLite persistence engine with Write-Ahead Logging (WAL) and foreign keys (`src/database/`).
- 6 relational tables: `schema_version`, `scenarios`, `agent_params`, `model_checkpoints`, `training_runs`, `metrics_history`.
- Repository classes decoupling SQL logic from simulation and training engines.

---

## [v0.4.0] - 2026-09-23 (Layer 4: Training Pipeline & Curriculum)
### Added
- Hierarchical multi-stage curriculum training manager (`src/training/`).
- Generalized Advantage Estimation ($\text{GAE}(\gamma=0.99, \lambda=0.95)$).
- Standalone and composite training scripts (`train_all.py`, `train_commander.py`, `train_low_level.py`).

---

## [v0.3.0] - 2026-09-23 (Layer 3: Hierarchical MARL Architectures)
### Added
- Recurrent Theater Commander policy (`CommanderPolicy`) with 256-unit GRU memory cell.
- Factorized Multi-Discrete PPO policies with `SelfAttentionBlock` for Air domain (`AirFightPolicy`, `AirEscapePolicy`).
- Hybrid Continuous-Discrete PPO (HHAPPO) policies for Ground and Naval forces.

---

## [v0.2.0] - 2026-09-23 (Layer 2: 2.5D Tactical Simulator)
### Added
- Continuous multi-domain simulation environment (`TacticalEnv`).
- 3D flight dynamics with energy-maneuverability, ground slope mechanics, and naval hydrodynamics.
- Continuous 2D digital elevation model ($100\text{ km} \times 100\text{ km}$) with line-of-sight terrain occlusion.
- 5-tier procedural scenario generator (from 1v1 duels up to joint multi-domain operations).

---

## [v0.1.0] - 2026-09-23 (Layer 1: Core Interfaces & Foundations)
### Added
- Base abstract classes: `BaseEnvironment`, `BaseEntity`, `BasePolicy` (`src/core/`).
- Strongly typed action and observation dataclasses for air, ground, maritime, and commander forces.
- Strict dimensionality invariants and unit definitions.
