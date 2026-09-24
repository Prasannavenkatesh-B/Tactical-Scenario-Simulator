"""Pytest fixtures for FastAPI inference server testing."""

from typing import Generator
from fastapi.testclient import TestClient
import pytest

from src.api.config import CANONICAL_POLICY_NAMES
from src.api.model_registry import ModelRegistry, build_policy
from src.api.server import create_app


@pytest.fixture
def registry() -> ModelRegistry:
    """Provide a ModelRegistry pre-populated with randomly initialized policies."""
    reg = ModelRegistry(checkpoint_dir="test_checkpoints_dummy")
    for name in CANONICAL_POLICY_NAMES:
        reg._policies[name] = build_policy(name)
    return reg


@pytest.fixture
def client(registry: ModelRegistry) -> Generator[TestClient, None, None]:
    """Provide a TestClient connected to the FastAPI application."""
    app = create_app(registry)
    with TestClient(app) as test_client:
        yield test_client
