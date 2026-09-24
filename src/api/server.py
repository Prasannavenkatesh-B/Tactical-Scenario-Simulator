"""FastAPI application factory, lifespan management, and middleware configuration."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import API_VERSION
from src.api.model_registry import ModelRegistry
from src.api.routes import router


def create_app(registry: ModelRegistry | None = None) -> FastAPI:
    """Factory creating and configuring the FastAPI inference application instance."""
    active_registry = registry if registry is not None else ModelRegistry()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # Startup: attempt loading checkpoints from configured directory
        reg: ModelRegistry = app.state.registry
        reg.load_all_from_dir()
        # Fall back to randomly-initialized baseline policies if no checkpoints exist
        if not reg.is_ready:
            reg.init_random_policies()
        yield
        # Shutdown: release resources

    app = FastAPI(
        title="Tactical MARL Inference API (DRDO)",
        description="High-performance HTTP inference server for hierarchical tactical scenario policies.",
        version=API_VERSION,
        lifespan=lifespan,
    )

    # Attach shared model registry to application state
    app.state.registry = active_registry

    # Enable CORS for local developer tools and TSS integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost", "http://127.0.0.1", "*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount endpoint routes
    app.include_router(router)

    return app
