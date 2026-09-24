"""Command-line entry point for running the 2D Tactical Simulation System (TSS) interactive UI."""

import argparse
import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="2D Interactive Tactical Simulation UI (DRDO MARL)")
    parser.add_argument("--level", type=int, default=3, choices=[1, 2, 3, 4, 5], help="Curriculum level 1-5")
    parser.add_argument("--seed", type=int, default=0, help="Initial random seed")
    parser.add_argument("--scenario-id", type=int, default=None, help="Load scenario by ID from database")
    parser.add_argument("--empty", action="store_true", help="Start with an empty map in edit mode")
    parser.add_argument("--headless", action="store_true", help="Run without physical display (smoke/test mode)")
    parser.add_argument("--frames", type=int, default=None, help="Exit after N frames")
    parser.add_argument("--no-db", action="store_true", help="Disable database persistence")

    args = parser.parse_args()

    # Headless setup if requested
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

    # Database resolution
    db = None
    if not args.no_db:
        try:
            from src.database.config import DB_PATH
            from src.database.db import Database
            from src.database.migrations import SchemaManager
            from src.database.seed import seed_all

            db = Database(DB_PATH)
            SchemaManager(db).apply_migrations()
            seed_all(db)
        except Exception as exc:
            print(f"[!] Warning: Could not initialize database ({exc}). Running in no-db mode.")
            db = None

    # Scenario configuration resolution
    from src.simulator.scenarios import ScenarioConfig, generate_scenario
    from src.ui.app import TacticalUIApp
    from src.ui.scenario_io import ScenarioIO

    scenario_cfg: ScenarioConfig
    if args.scenario_id is not None:
        if db is None:
            print("[!] Error: Cannot load scenario from database when --no-db is active.", file=sys.stderr)
            sys.exit(1)
        io = ScenarioIO(db)
        scenario_cfg, _ = io.load(args.scenario_id)
        print(f"[*] Loaded scenario ID {args.scenario_id}: '{scenario_cfg.name}'")
    elif args.empty:
        scenario_cfg = ScenarioConfig(
            name="Empty_Sandbox_Map",
            map_size_km=30.0,
            episode_horizon=350,
            blue_entities=[],
            red_entities=[],
        )
        print("[*] Initialized empty sandbox scenario in Edit Mode.")
    else:
        scenario_cfg = generate_scenario(level=args.level, seed=args.seed)
        print(f"[*] Initialized Curriculum Level {args.level} scenario: '{scenario_cfg.name}' (Seed: {args.seed})")

    # Launch Application
    app = TacticalUIApp(scenario_config=scenario_cfg, db=db, seed=args.seed)
    if args.empty:
        app.state.mode = "edit"

    frames = args.frames
    if args.headless and frames is None:
        frames = 100

    print(f"[*] Launching TacticalUIApp (Headless={args.headless}, Max Frames={frames})...")
    app.run(max_frames=frames)
    print(f"[*] TacticalUIApp terminated cleanly at step {app.state.current_step}.")


if __name__ == "__main__":
    main()
