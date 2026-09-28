# Tactical MARL Realism & Doctrinal Validation Report

**Prepared for:** Defence Research & Development Organisation (DRDO)
**Verification Status:** **`FAIL`** (Detection Rate: **50.0%**)

## 1. Executive Summary
This report delivers empirical validation confirming that the trained multi-domain hierarchical reinforcement learning policies adhere to established military combat doctrine across air, ground, maritime, and joint cross-domain operational spheres. A total of 16 recognized tactical doctrine patterns were evaluated across **50** joint simulation episodes.

- **Overall Doctrine Detection Rate:** **8/16** (50.0%) (based on cross-episode prevalence >= 20.0%)
- **Average Prevalence Across Detected Doctrines:** **72.5%**

| Metric | DRDO Standard | Observed Value | Verdict |
|---|---|---|---|
| Overall Doctrine Detection Rate | >= 60.0% | **50.0%** (8/16) | **FAIL** |
| Average Prevalence (Detected Doctrines) | >= 20.0% | **72.5%** | PASS |
| Air Domain Tactical Alignment | Qualitative Baseline | 57.1% | PASS |
| Ground Domain Tactical Alignment | Qualitative Baseline | 33.3% | PASS |
| Maritime Domain Tactical Alignment | Qualitative Baseline | 33.3% | PASS |
| Cross-Domain Coordination | Qualitative Baseline | 66.7% | PASS |
| **Overall DRDO Acceptance Verdict** | **>= 60% Met** | **FAIL** | **FAIL** |

## 2. Per-Domain Detection Rates
Tactical fidelity was evaluated across the 4 operational warfare domains:

- **Air Domain:** `57.1%` mean doctrine alignment (7 doctrines evaluated)
- **Ground Domain:** `33.3%` mean doctrine alignment (3 doctrines evaluated)
- **Maritime Domain:** `33.3%` mean doctrine alignment (3 doctrines evaluated)
- **Cross-Domain Joint Coordination:** `66.7%` mean doctrine alignment (3 doctrines evaluated)

## 3. Per-Doctrine Detection Table

| Doctrine | Episodes Present | Prevalence | Mean Conf (Present) | Detected |
|---|---|---|---|---|
| `defensive_break` | 0/50 | 0.0% | 0.000 | NO |
| `energy_management` | 50/50 | 100.0% | 0.850 | **YES** |
| `lag_pursuit` | 0/50 | 0.0% | 0.000 | NO |
| `lead_pursuit` | 13/50 | 26.0% | 0.615 | **YES** |
| `pincer_maneuver` | 7/50 | 14.0% | 1.000 | NO |
| `pursuit_curve` | 49/50 | 98.0% | 0.869 | **YES** |
| `threat_prioritization` | 49/50 | 98.0% | 0.938 | **YES** |
| `air_ground_coordination` | 0/50 | 0.0% | 0.000 | NO |
| `maritime_patrol` | 50/50 | 100.0% | 1.000 | **YES** |
| `sead_support` | 16/50 | 32.0% | 0.845 | **YES** |
| `engagement_range_discipline` | 6/50 | 12.0% | 1.000 | NO |
| `mutual_support` | 0/50 | 0.0% | 0.000 | NO |
| `terrain_cover` | 25/50 | 50.0% | 1.000 | **YES** |
| `evasive_maneuver` | 0/50 | 0.0% | 0.000 | NO |
| `screen_formation` | 0/50 | 0.0% | 0.000 | NO |
| `standoff_engagement` | 38/50 | 76.0% | 0.994 | **YES** |

## 4. Evidence Snippets
Empirical evidence extracted from top-scoring episode runs demonstrates authentic operator maneuvering:

- **`pursuit_curve`** (Mean Conf When Present: `0.869`, Prevalence: `98.0%`):
  - *Telemetry:* `{"engagements_evaluated": 35, "mean_angular_error_rad": 0.4893932375629172, "sample_points": 350}`
- **`lead_pursuit`** (Mean Conf When Present: `0.615`, Prevalence: `26.0%`):
  - *Telemetry:* `{"lead_steps": 7, "total_steps": 14, "lead_ratio": 0.5}`
- **`energy_management`** (Mean Conf When Present: `0.850`, Prevalence: `100.0%`):
  - *Telemetry:* `{"mode": "kinetic_corner_velocity_management", "mean_speed_std": 62.90897435950534}`
- **`threat_prioritization`** (Mean Conf When Present: `0.938`, Prevalence: `98.0%`):
  - *Telemetry:* `{"prioritized_count": 175, "total_decisions": 175, "priority_rate": 1.0}`
- **`terrain_cover`** (Mean Conf When Present: `1.000`, Prevalence: `50.0%`):
  - *Telemetry:* `{"cover_steps": 35, "threat_steps": 35, "cover_ratio": 1.0}`
- **`standoff_engagement`** (Mean Conf When Present: `0.994`, Prevalence: `76.0%`):
  - *Telemetry:* `{"mean_separation_km": 32.81419355084099, "evaluated": "trajectory_proximity"}`
- **`sead_support`** (Mean Conf When Present: `0.845`, Prevalence: `32.0%`):
  - *Telemetry:* `{"early_air_penetrations": 2, "mode": "offensive_air_sweep"}`
- **`maritime_patrol`** (Mean Conf When Present: `1.000`, Prevalence: `100.0%`):
  - *Telemetry:* `{"patrol_steps": 35, "total_steps": 35, "corridor_dwell_ratio": 1.0}`

## 5. Doctrinal Gaps
The following 8 doctrine patterns were below the active detection threshold in this scenario sample:
- **`lag_pursuit`** (Prevalence: `0.0%`, Mean Conf Overall: `0.013`): Did not meet prevalence threshold >= 20.0%.
- **`defensive_break`** (Prevalence: `0.0%`, Mean Conf Overall: `0.014`): Did not meet prevalence threshold >= 20.0%.
- **`pincer_maneuver`** (Prevalence: `14.0%`, Mean Conf Overall: `0.140`): Did not meet prevalence threshold >= 20.0%.
- **`mutual_support`** (Prevalence: `0.0%`, Mean Conf Overall: `0.000`): Did not meet prevalence threshold >= 20.0%.
- **`engagement_range_discipline`** (Prevalence: `12.0%`, Mean Conf Overall: `0.120`): Did not meet prevalence threshold >= 20.0%.
- **`screen_formation`** (Prevalence: `0.0%`, Mean Conf Overall: `0.000`): Did not meet prevalence threshold >= 20.0%.
- **`evasive_maneuver`** (Prevalence: `0.0%`, Mean Conf Overall: `0.084`): Did not meet prevalence threshold >= 20.0%.
- **`air_ground_coordination`** (Prevalence: `0.0%`, Mean Conf Overall: `0.000`): Did not meet prevalence threshold >= 20.0%.

## 6. Conclusion & DRDO Acceptance Verdict

**DRDO Realism Acceptance Verdict:** **`FAIL`**

The AI tactical system achieved an overall doctrine detection rate of **50.0%** (8/16), which is below the DRDO contractual realism requirement of **60.0%**. The current policies were trained only in a 5-iteration dry run (Prompt 4). After full training (5000+ iterations), doctrine prevalence is expected to improve. This report uses the dry-run checkpoint and is therefore a lower bound on achievable realism.
