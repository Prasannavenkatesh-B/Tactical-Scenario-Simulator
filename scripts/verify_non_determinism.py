"""CLI tool to verify tactical non-determinism, stochasticity, and reproducibility.

Executes the full DRDO acceptance suite:
- Validates bit-identical reproducibility under identical seeds
- Validates non-determinism and variance across distinct seeds (χ² and Levene p < 0.05)
- Evaluates spatial trajectory diversity (>= 3 distinct K-Means clusters)
- Measures action stochasticity (normalized Shannon entropy > 0.50)
- Generates JSON and Markdown acceptance reports and publication PNG figures

Usage:
    python scripts/verify_non_determinism.py --checkpoint-dir checkpoints/final \\
                                             --num-runs 100 \\
                                             --output-dir reports/non_determinism
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.plots import plot_all
from src.evaluation.verifier import NonDeterminismVerifier
from src.marl.policies.air_fight import AirFightPolicy
from src.marl.policies.ground_engage import GroundEngagePolicy
from src.marl.policies.sea_engage import SeaEngagePolicy


def load_policies(checkpoint_dir: str) -> dict[str, Any]:
    """Load trained policies from checkpoint folder or instantiate fresh policies."""
    policies: dict[str, Any] = {}
    cp_path = Path(checkpoint_dir)

    # 1. Air Policy
    air_policy = AirFightPolicy()
    air_file = cp_path / "air_fight.pt"
    if air_file.is_file():
        try:
            air_policy.load(str(air_file))
            print(f"[+] Loaded air policy from {air_file}")
        except Exception as e:
            print(f"[-] Could not load {air_file}: {e}")
    policies["air_fight"] = air_policy

    # 2. Ground Policy
    ground_policy = GroundEngagePolicy()
    ground_file = cp_path / "ground_engage.pt"
    if ground_file.is_file():
        try:
            ground_policy.load(str(ground_file))
            print(f"[+] Loaded ground policy from {ground_file}")
        except Exception as e:
            print(f"[-] Could not load {ground_file}: {e}")
    policies["ground_engage"] = ground_policy

    # 3. Sea Policy
    sea_policy = SeaEngagePolicy()
    sea_file = cp_path / "sea_engage.pt"
    if sea_file.is_file():
        try:
            sea_policy.load(str(sea_file))
            print(f"[+] Loaded sea policy from {sea_file}")
        except Exception as e:
            print(f"[-] Could not load {sea_file}: {e}")
    policies["sea_engage"] = sea_policy

    return policies


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify non-determinism, stochasticity, and reproducibility for DRDO acceptance."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/final",
        help="Directory containing trained policy checkpoint files.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of evaluation episodes per condition (default 100 for statistical validity).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/non_determinism",
        help="Directory to save generated reports and plots.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating graphical PNG plots.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Save only JSON report without Markdown or plots.",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("  TACTICAL MARL NON-DETERMINISM VERIFICATION MODULE")
    print("  Prepared for Defence Research & Development Organisation (DRDO)")
    print("=" * 80)
    print(f"Sample Size per condition: {args.num_runs}")
    print(f"Output Directory:          {args.output_dir}")
    print(f"Checkpoint Directory:      {args.checkpoint_dir}")
    print("-" * 80)

    # 1. Load Policies
    policies = load_policies(args.checkpoint_dir)

    # 2. Initialize Verifier
    verifier = NonDeterminismVerifier(
        policies=policies,
        output_dir=args.output_dir,
    )

    # 3. Run Full Verification
    print("\nExecuting comprehensive statistical verification suite...")
    results = verifier.run_full_verification(num_runs=args.num_runs)

    # 4. Save Reports
    formats = ["json"] if args.json_only else ["json", "markdown"]
    saved_paths = verifier.save_report(results, output_dir=args.output_dir, formats=formats)

    # 5. Generate Plots
    if not args.no_plots and not args.json_only:
        plots_dir = os.path.join(args.output_dir, "plots")
        plot_paths = plot_all(results, output_dir=plots_dir)
        print(f"\n[+] Generated {len(plot_paths)} visualization plots in {plots_dir}")

    # 6. Extract and Display Key DRDO Metrics
    diff_seeds = results.get("different_seeds", {})
    same_seed = results.get("same_seed", {})
    stoch = results.get("action_stochasticity", {})
    traj = diff_seeds.get("trajectory_diversity", {})
    stats = results.get("statistical_tests", {})

    diff_chi2_p = float(diff_seeds.get("chi2_p_value", 0.0))
    diff_levene_p = float(diff_seeds.get("levene_p_value", 0.0))
    same_chi2_p = float(same_seed.get("chi2_p_value", 1.0))
    distinct_clusters = int(traj.get("distinct_clusters", 0))
    norm_entropy = float(stoch.get("entropy", 0.0))
    verdict = str(results.get("verdict", "FAIL"))

    print("\n" + "=" * 80)
    print("  STATISTICAL ACCEPTANCE SUMMARY")
    print("=" * 80)
    print(f"1. Different-Seed Chi-Square p-value: {diff_chi2_p:.6e}  (Threshold: < 0.05)  -> {'PASS' if diff_chi2_p < 0.05 else 'FAIL'}")
    print(f"2. Different-Seed Levene p-value:     {diff_levene_p:.6e}  (Threshold: < 0.05)  -> {'PASS' if diff_levene_p < 0.05 else 'FAIL'}")
    print(f"3. Same-Seed Chi-Square p-value:      {same_chi2_p:.4f}        (Threshold: ~1.0)   -> {'PASS' if same_chi2_p >= 0.95 else 'FAIL'}")
    print(f"4. Trajectory Spatial Clusters:       {distinct_clusters} distinct modes  (Threshold: >= 3)    -> {'PASS' if distinct_clusters >= 3 else 'FAIL'}")
    print(f"5. Normalized Action Entropy:         {norm_entropy:.4f}         (Threshold: > 0.50)  -> {'PASS' if norm_entropy > 0.50 else 'FAIL'}")
    print("-" * 80)
    print(f"DRDO FINAL ACCEPTANCE VERDICT:      [{verdict}]")
    print("=" * 80)

    for fmt, p in saved_paths.items():
        print(f"Report saved ({fmt}): {p}")

    if verdict != "PASS":
        print("\nVerification failed to meet one or more statistical thresholds.")
        sys.exit(1)

    print("\nVerification passed successfully. System accepted.")
    sys.exit(0)


if __name__ == "__main__":
    main()
