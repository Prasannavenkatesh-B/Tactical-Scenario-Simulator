"""Pytest fixtures for TSS integration testing."""

from pathlib import Path
from typing import Generator
from fastapi.testclient import TestClient
import pytest

from src.api.config import CANONICAL_POLICY_NAMES
from src.api.model_registry import ModelRegistry, build_policy
from src.api.server import create_app
from src.integration.http_wrapper import HTTPTSSWrapper
from src.integration.in_process_wrapper import InProcessTSSWrapper


@pytest.fixture
def protocol_path() -> str:
    """Return absolute path to src/integration/protocol.yaml."""
    p = Path(__file__).resolve().parent.parent.parent / "src" / "integration" / "protocol.yaml"
    return str(p)


@pytest.fixture
def registry() -> ModelRegistry:
    """Provide a ModelRegistry pre-populated with all 9 canonical policies in eval mode."""
    reg = ModelRegistry(checkpoint_dir="dummy_checkpoint_dir")
    for name in CANONICAL_POLICY_NAMES:
        reg._policies[name] = build_policy(name)
    return reg


@pytest.fixture
def in_process_wrapper(registry: ModelRegistry, protocol_path: str) -> InProcessTSSWrapper:
    """Provide an InProcessTSSWrapper ready for immediate evaluation."""
    return InProcessTSSWrapper(registry=registry, protocol_path=protocol_path)


@pytest.fixture
def test_client(registry: ModelRegistry) -> Generator[TestClient, None, None]:
    """Provide a TestClient connected to the FastAPI application."""
    app = create_app(registry)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def http_wrapper(test_client: TestClient, protocol_path: str) -> HTTPTSSWrapper:
    """Provide an HTTPTSSWrapper using the in-memory FastAPI TestClient."""
    return HTTPTSSWrapper(
        base_url="http://testserver",
        client=test_client,
        protocol_path=protocol_path,
    )
