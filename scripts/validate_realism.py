"""CLI tool to execute tactical realism validation across established combat doctrines.

Analyzes AI tactical behaviors against 16 recognized air, ground, sea, and joint doctrines:
- Air: Pursuit Curve, Lead Pursuit, Lag Pursuit, Defensive Break, Energy Mgmt, Pincer, Threat Priority
- Ground: Terrain Cover, Mutual Support, Engagement Range Discipline
- Sea: Standoff Engagement, Screen Formation, Evasive Maneuver
- Cross-Domain: Air-Ground CAS Coordination, SEAD Precedence, Maritime Patrol

Usage:
    python scripts/validate_realism.py --checkpoint-dir checkpoints/final \\
                                       --num-episodes 100 \\
                                       --scenario-level 5 \\
                                       --output-dir reports/realism
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.realism_scorer import RealismScorer
from src.marl.policies.air_fight import AirFightPolicy
from src.marl.commander import CommanderPolicy
from src.marl.policies.ground_engage import GroundEngagePolicy
from src.marl.policies.sea_engage import SeaEngagePolicy


def load_policies(checkpoint_dir: str) -> dict[str, Any]:
    """Load trained policy checkpoints or instantiate baseline models."""
    policies: dict[str, Any] = {}
    cp = Path(checkpoint_dir)

    air = AirFightPolicy()
    ground = GroundEngagePolicy()
    sea = SeaEngagePolicy()
    commander = CommanderPolicy()

    # Search for composite checkpoints or individual weights
    ckpt_candidates = list(cp.glob("*.pt")) + list(cp.glob("**/*.pt")) + list(Path("checkpoints").glob("**/*.pt"))
    for ckpt_path in ckpt_candidates:
        try:
            ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt_data, dict) and "policies" in ckpt_data:
                p_dict = ckpt_data["policies"]
                if "air_fight" in p_dict:
                    air.load_state_dict(p_dict["air_fight"])
                if "ground_engage" in p_dict:
                    ground.load_state_dict(p_dict["ground_engage"])
                if "sea_engage" in p_dict:
                    sea.load_state_dict(p_dict["sea_engage"])
                if "commander" in p_dict:
                    commander.load_state_dict(p_dict["commander"])
                print(f"[+] Loaded policies from composite checkpoint {ckpt_path}")
                break
        except Exception:
            pass

    policies["air_fight"] = air
    policies["ground_engage"] = ground
    policies["sea_engage"] = sea
    policies["commander"] = commander

    return policies


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate AI tactical realism against 16 military combat doctrine patterns."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/final",
        help="Directory containing trained policy checkpoints.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default 100 for statistical validity).",
    )
    parser.add_argument(
        "--scenario-level",
        type=int,
        default=5,
        help="Scenario complexity level (default 5 for multi-domain joint forces).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/realism",
        help="Destination directory for realism validation reports.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Skip generating human-readable Markdown report.",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("  TACTICAL DOCTRINE REALISM VALIDATION MODULE")
    print("  Prepared for Defence Research & Development Organisation (DRDO)")
    print("=" * 80)
    print(f"Sample Size (episodes):    {args.num_episodes}")
    print(f"Scenario Complexity Level: Level {args.scenario_level} (Joint Multi-Domain)")
    print(f"Output Directory:          {args.output_dir}")
    print(f"Checkpoint Directory:      {args.checkpoint_dir}")
    print("-" * 80)

    # 1. Load policies
    policies = load_policies(args.checkpoint_dir)

    # 2. Initialize Realism Scorer
    scorer = RealismScorer(
        policies=policies,
        output_dir=args.output_dir,
    )

    # 3. Execute validation
    print(f"\nSimulating {args.num_episodes} tactical episodes and analyzing telemetry against 16 doctrines...")
    results = scorer.run_full_validation(
        num_episodes=args.num_episodes,
        scenario_level=args.scenario_level,
    )

    # 4. Save reports
    formats = ["json"] if args.json_only else ["json", "markdown"]
    saved_paths = scorer.save_report(results, output_dir=args.output_dir, formats=formats)

    # 5. Extract metrics
    det_rate = float(results.get("detection_rate", 0.0))
    det_count = int(results.get("detected_count", 0))
    total_doc = int(results.get("total_doctrines", 16))
    verdict = str(results.get("acceptance_verdict", "FAIL"))
    by_domain = results.get("by_domain", {})
    per_doctrine = results.get("per_doctrine", {})

    print("\n" + "=" * 80)
    print("  DOCTRINAL REALISM ACCEPTANCE SUMMARY")
    print("=" * 80)
    print(f"Overall Detection Rate:      {det_rate * 100:.1f}% ({det_count}/{total_doc} doctrines detected) -> {verdict}")
    print(f"DRDO Contractual Threshold:  >= 60.0% (at least 10 doctrines required)")
    print("-" * 80)
    print("Domain Alignment Breakdown:")
    print(f"  - Air Domain:              {by_domain.get('air', 0.0) * 100:.1f}%")
    print(f"  - Ground Domain:           {by_domain.get('ground', 0.0) * 100:.1f}%")
    print(f"  - Maritime Domain:         {by_domain.get('sea', 0.0) * 100:.1f}%")
    print(f"  - Cross-Domain Joint:      {by_domain.get('cross', 0.0) * 100:.1f}%")
    print("-" * 80)
    print("Doctrine Breakdown (16 Patterns):")
    print(f"  {'Status':14} {'Doctrine':30} {'Episodes':10} {'Prevalence':12} {'Mean Conf (Pres)':16}")
    print("  " + "-" * 84)
    for name, info in sorted(per_doctrine.items()):
        status = "DETECTED" if info.get("detected", False) else "NOT DETECTED"
        ep_present = int(info.get("episodes_present", 0))
        tot_ep = int(info.get("total_episodes", args.num_episodes))
        prev = float(info.get("prevalence", 0.0)) * 100.0
        conf_pres = float(info.get("mean_confidence_when_present", 0.0))
        print(f"  [{status:12}] {name:30} {ep_present:>3}/{tot_ep:<3}   {prev:>5.1f}%       {conf_pres:.3f}")
    print("-" * 86)
    print(f"DRDO REALISM ACCEPTANCE VERDICT: [{verdict}]")
    print("=" * 80)

    for fmt, p in saved_paths.items():
        print(f"Report saved ({fmt}): {p}")

    if verdict != "PASS":
        print("\nRealism validation did not meet the 60.0% doctrine detection threshold.")
        sys.exit(1)

    print("\nRealism validation passed successfully. System accepted.")
    sys.exit(0)


if __name__ == "__main__":
    main()
