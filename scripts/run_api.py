"""CLI entry point for launching the Tactical Simulation Inference API server."""

import argparse
from pathlib import Path
import sys
import uvicorn

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.config import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_HOST,
    DEFAULT_PORT,
    LOG_LEVEL,
)
from src.api.model_registry import ModelRegistry
from src.api.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Tactical MARL Policy Inference API Server (DRDO)")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Host bind address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing trained policy *.pt weights",
    )
    parser.add_argument("--reload", action="store_true", help="Enable Uvicorn auto-reloading")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Skip checkpoint loading; start with randomly initialized policies",
    )

    args = parser.parse_args()

    registry = ModelRegistry(checkpoint_dir=args.checkpoint_dir)
    if args.no_checkpoints:
        print("[*] Starting with randomly initialized policy models (--no-checkpoints)...")
        registry.init_random_policies()
    else:
        loaded = registry.load_all_from_dir()
        if loaded:
            print(f"[*] Loaded {len(loaded)} checkpoint(s) from '{args.checkpoint_dir}':")
            for pol, ckpt_p in loaded.items():
                print(f"    - {pol}: {ckpt_p}")
        else:
            print(f"[!] No checkpoints found in '{args.checkpoint_dir}'. Falling back to baseline random initialization.")
            registry.init_random_policies()

    app = create_app(registry)

    print(f"[*] Launching Uvicorn server on http://{args.host}:{args.port} (Workers: {args.workers})...")
    uvicorn.run(
        app if not args.reload else "src.api.server:create_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level=LOG_LEVEL,
        factory=args.reload,
    )


if __name__ == "__main__":
    main()
