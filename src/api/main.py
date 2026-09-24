"""Main execution module running Uvicorn server for the Inference API."""

import os
import uvicorn

from src.api.config import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_HOST,
    DEFAULT_PORT,
    LOG_LEVEL,
)
from src.api.model_registry import ModelRegistry
from src.api.server import create_app


def main() -> None:
    """Read environment variables and launch Uvicorn server."""
    host = os.environ.get("API_HOST", DEFAULT_HOST)
    port = int(os.environ.get("API_PORT", str(DEFAULT_PORT)))
    ckpt_dir = os.environ.get("CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR)
    log_lvl = os.environ.get("LOG_LEVEL", LOG_LEVEL)

    registry = ModelRegistry(checkpoint_dir=ckpt_dir)
    app = create_app(registry)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_lvl,
    )


if __name__ == "__main__":
    main()
