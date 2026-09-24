"""Pydantic v2 request and response data models for the inference API."""

from typing import Any, Literal
from pydantic import BaseModel, Field


class ActRequest(BaseModel):
    """Observation payload and configuration for model action inference."""

    agent_id: str = Field(..., description="Unique agent/entity identifier")
    domain: Literal["air", "ground", "sea", "commander"] = Field(
        ..., description="Operational domain"
    )
    variant: str = "default"
    observation: list[float] = Field(
        ..., description="Normalized agent observation vector"
    )
    hidden_state: list[float] | None = None
    deterministic: bool = False


class ActResponse(BaseModel):
    """Inference response containing selected action, log probability, and latency."""

    action: list[int] | list[float] | dict[str, Any] = Field(
        ..., description="Encoded action vector or action dictionary"
    )
    log_prob: float = Field(..., description="Log probability of selected action")
    value: float = Field(..., description="State-value estimate V(s)")
    hidden_state: list[float] | None = None
    latency_ms: float = Field(..., description="Inference compute latency in milliseconds")


class BatchActRequest(BaseModel):
    """High-throughput multi-agent batch inference request."""

    requests: list[ActRequest] = Field(
        ..., description="List of concurrent individual ActRequests"
    )


class BatchActResponse(BaseModel):
    """Batch inference response payload with total accumulated execution duration."""

    responses: list[ActResponse] = Field(
        ..., description="List of individual ActResponses in matching order"
    )
    total_latency_ms: float = Field(
        ..., description="Total batch processing latency in milliseconds"
    )


class ResetRequest(BaseModel):
    """Episode boundary notification to clear recurrent memory states."""

    episode_id: str | None = Field(
        None, description="Optional episode tracking identifier"
    )
    agent_ids: list[str] | None = Field(
        None, description="Optional subset of specific agent IDs to reset"
    )


class ResetResponse(BaseModel):
    """Reset execution status and list of cleared agents."""

    status: str = Field("reset", description="Operation status string")
    cleared_agents: list[str] = Field(
        default_factory=list, description="List of agent IDs whose hidden states were cleared"
    )


class HealthResponse(BaseModel):
    """Service health probe response."""

    status: str = Field("ok", description="Server health status")
    loaded_policies: list[str] = Field(
        default_factory=list, description="List of active canonical policy names loaded in memory"
    )
    version: str = Field(..., description="API semantic version")


class LoadCheckpointRequest(BaseModel):
    """Dynamic model weight hot-swap request."""

    policy_name: str = Field(..., description="Canonical policy name to update")
    checkpoint_path: str = Field(
        ..., description="Filesystem path to PyTorch checkpoint artifact (.pt)"
    )


class LoadCheckpointResponse(BaseModel):
    """Confirmation of dynamic checkpoint hot-swap."""

    status: str = Field("loaded", description="Loading status string")
    policy_name: str = Field(..., description="Canonical name of updated policy")
    param_count: int = Field(..., description="Total parameter count of updated model")


class PolicyInfo(BaseModel):
    """Summary metadata descriptor for an available policy model."""

    name: str = Field(..., description="Canonical policy name")
    variant: str = Field(..., description="Vehicle or model variant")
    param_count: int = Field(..., description="Total neural network parameter count")
    loaded: bool = Field(..., description="Whether weights are actively resident in memory")


class PoliciesResponse(BaseModel):
    """List of all registered policy models."""

    policies: list[PolicyInfo] = Field(
        default_factory=list, description="Array of registered policy descriptors"
    )
