# Tactical MARL System Diagnostics & Troubleshooting Guide

**Prepared for:** Defence Research & Development Organisation (DRDO), Ministry of Defence, Government of India  
**Coordinating Laboratory:** Aeronautical Development Establishment (ADE), Bengaluru  
**Target Systems:** Development Workstations, High-Performance Compute Clusters, Embedded Simulation Nodes  
**System Version:** `v1.0.0`

---

## Table of Contents
1. [Environment & Dependency Issues](#1-environment--dependency-issues)
2. [Runtime & Display / UI Errors](#2-runtime--display--ui-errors)
3. [API & Networking Diagnostics](#3-api--networking-diagnostics)
4. [Training & Checkpoint Failures](#4-training--checkpoint-failures)
5. [Database Locks & SQLite Concurrency](#5-database-locks--sqlite-concurrency)
6. [Statistical & Realism Validation Diagnostics](#6-statistical--realism-validation-diagnostics)
7. [TSS Integration Wrapper Mismatches](#7-tss-integration-wrapper-mismatches)

---

## 1. Environment & Dependency Issues

### Issue 1.1: `ModuleNotFoundError` when importing project packages
- **Symptom:** Python throws `ModuleNotFoundError: No module named 'src'` when executing scripts directly.
- **Cause:** The current working directory is not on `PYTHONPATH` or the script was run without the root directory in `sys.path`.
- **Resolution:** Always invoke scripts via `uv run` from the project root:
  ```bash
  cd C:\Users\bpras\Desktop\TSS
  uv run python scripts/run_api.py
  ```
  Alternatively, set `PYTHONPATH`:
  - *Windows (PowerShell):* `$env:PYTHONPATH = "C:\Users\bpras\Desktop\TSS"`
  - *Linux (Bash):* `export PYTHONPATH="/path/to/TSS"`

### Issue 1.2: PyTorch GPU / CPU mismatch
- **Symptom:** `AssertionError: Torch not compiled with CUDA enabled` or high CPU utilization when a dedicated GPU is present.
- **Resolution:** The system automatically falls back to CPU if CUDA is unavailable. To install GPU-accelerated PyTorch:
  ```bash
  uv pip install torch --index-url https://download.pytorch.org/whl/cu121
  ```

---

## 2. Runtime & Display / UI Errors

### Issue 2.1: `pygame.error: No available video device` (Headless Server)
- **Symptom:** Running `run_ui.py` on a remote headless Linux server or Docker container crashes immediately with SDL video initialization failure.
- **Cause:** No X11/Wayland display server is available.
- **Resolution:**
  1. Pass the `--headless` flag to run in batch simulation mode without rendering:
     ```bash
     uv run python scripts/run_ui.py --headless --scenario-level 5
     ```
  2. If visual rendering is required on Linux, launch using a virtual framebuffer:
     ```bash
     xvfb-run -a uv run python scripts/run_ui.py --scenario-level 5
     ```

### Issue 2.2: Stuttering UI frame rate or low simulation speed
- **Symptom:** Simulation clock runs significantly slower than $1.0\times$ real-time.
- **Resolution:** Disable heavy terminal logging. Lower the scenario tier to verify CPU load. Ensure your display driver has hardware acceleration enabled for Pygame.

---

## 3. API & Networking Diagnostics

### Issue 3.1: `OSError: [Errno 10048] error while attempting to bind on address ('0.0.0.0', 8089)`
- **Symptom:** FastAPI fails to start because port 8089 is already bound.
- **Resolution:**
  1. Identify and kill the lingering process:
     - *Windows:*
       ```powershell
       Get-Process -Id (Get-NetTCPConnection -LocalPort 8089).OwningProcess | Stop-Process -Force
       ```
     - *Linux:*
       ```bash
       fuser -k 8089/tcp
       ```
  2. Or assign a different port via the CLI flag:
     ```bash
     uv run python scripts/run_api.py --port 8095
     ```

### Issue 3.2: `HTTP 422 Unprocessable Entity` on `/act` endpoint
- **Symptom:** Client receives HTTP status 422 with message `Observation dimension mismatch`.
- **Cause:** The length of the sent `observation` array does not match the expected domain dimension.
- **Standard Dimensions:**
  - `air`: **13 elements**
  - `ground`: **9 elements**
  - `sea`: **9 elements**
  - `commander`: **53 elements**
- **Resolution:** Verify input telemetry using `ObservationCodec.validate(domain, obs)`. Refer to [API Reference](API_REFERENCE.md) for field definitions.

---

## 4. Training & Checkpoint Failures

### Issue 4.1: `CUDA out of memory` during training
- **Symptom:** PyTorch raises an OOM error during GAE computation or PPO batch updates.
- **Resolution:** Reduce the rollout buffer size or batch size in `src/marl/config.py`:
  - Decrease `ROLLOUT_HORIZON` from `2048` to `1024`.
  - Decrease `BATCH_SIZE` from `256` to `128` or `64`.

### Issue 4.2: Loss diverges to `NaN` during policy updates
- **Symptom:** PPO policy or value loss reports `nan` after several iterations.
- **Cause:** Exploding gradients or numerical underflow in continuous log-probability calculations.
- **Resolution:** Ensure gradient clipping is active (`max_norm = 0.5`). The system includes automatic epsilon guards ($10^{-8}$) inside all attention and Gaussian heads.

---

## 5. Database Locks & SQLite Concurrency

### Issue 5.1: `sqlite3.OperationalError: database is locked`
- **Symptom:** Multiple processes attempting simultaneous writes fail with a database lock exception.
- **Cause:** Traditional SQLite journal mode locks the entire file during writes.
- **Resolution:**
  1. Ensure WAL mode is active (configured automatically by `src/database/db.py`):
     ```sql
     PRAGMA journal_mode=WAL;
     PRAGMA busy_timeout=5000;
     ```
  2. Verify that long-running transactions are wrapped in short-lived context managers.

---

## 6. Statistical & Realism Validation Diagnostics

### Issue 6.1: Realism validation detection rate below 60%
- **Symptom:** `validate_realism.py` prints `OVERALL DOCTRINE DETECTION RATE: 50.0% -> FAIL`.
- **Explanation:** This is expected when testing against the initial 5-iteration dry-run checkpoint (`checkpoint_final_iter_00010.pt`). The two-level evaluation framework is strictly enforced: doctrines require $\ge 20\%$ cross-episode prevalence and $\ge 0.50$ confidence. Collective multi-agent tactics (such as synchronized pincer maneuvers or naval screen formations) require extended multi-GPU training ($5,000+$ iterations) to emerge consistently.
- **Resolution:** Refer to the post-delivery training plan in [proposal/TRAINING_ROADMAP.md](../proposal/TRAINING_ROADMAP.md) to execute production training in Milestone M5.

### Issue 6.2: `ZeroDivisionError` during statistical testing
- **Symptom:** Levene test or variance test crashes when all episodes have identical outcomes.
- **Resolution:** The statistical verification module incorporates zero-variance guards with Laplace smoothing. Ensure you evaluate at least 20 episodes (`--episodes 100` recommended).

---

## 7. TSS Integration Wrapper Mismatches

### Issue 7.1: Unit conversion errors (knots vs m/s, nautical miles vs km)
- **Symptom:** Aircraft maneuvers erratically or crashes immediately into terrain bounds.
- **Cause:** Raw telemetry fed into the wrapper in non-standard units (e.g. altitude in feet rather than kilometers).
- **Resolution:** Always route telemetry through `TSSFieldMapper` which automatically applies conversion matrices according to [docs/INTEGRATION.md](INTEGRATION.md).

---
*Defence Research & Development Organisation (DRDO) - Diagnostics & Troubleshooting Deliverable*
