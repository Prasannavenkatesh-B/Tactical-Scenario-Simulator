"""FastAPI route handlers for health, model inference, batching, reset, and hot-swapping."""

import time
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Request, status
import torch

from src.api.action_codec import ActionCodec, ObservationCodec
from src.api.config import API_VERSION, resolve_policy_name
from src.api.model_registry import ModelRegistry
from src.marl.commander import CommanderPolicy
from src.marl.policies.base_policy import BaseLowLevelPolicy
from src.api.schemas import (
    ActRequest,
    ActResponse,
    BatchActRequest,
    BatchActResponse,
    HealthResponse,
    LoadCheckpointRequest,
    LoadCheckpointResponse,
    PoliciesResponse,
    PolicyInfo,
    ResetRequest,
    ResetResponse,
)

router = APIRouter()


def get_registry(request: Request) -> ModelRegistry:
    """Dependency provider retrieving ModelRegistry from application state."""
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ModelRegistry is not configured on application state.",
        )
    return registry


@router.get("/health", response_model=HealthResponse)
async def health(registry: ModelRegistry = Depends(get_registry)) -> HealthResponse:
    """Service health check returning active loaded policy names and API version."""
    policies = registry.list_policies()
    loaded_names = [p["name"] for p in policies]
    return HealthResponse(
        status="ok",
        loaded_policies=loaded_names,
        version=API_VERSION,
    )


@router.post("/reset", response_model=ResetResponse)
async def reset(
    req: ResetRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> ResetResponse:
    """Clear recurrent hidden memory states at episode boundaries."""
    if req.agent_ids is None:
        registry.reset_commander_hiddens()
        cleared = ["all"]
    else:
        for aid in req.agent_ids:
            registry.reset_agent_hidden(aid)
        cleared = list(req.agent_ids)

    return ResetResponse(status="reset", cleared_agents=cleared)


@router.post("/act", response_model=ActResponse)
async def act(
    req: ActRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> ActResponse:
    """Execute forward neural inference for a single agent observation."""
    t0 = time.perf_counter()

    # 1. Dimensionality validation
    try:
        ObservationCodec.validate(req.domain, req.observation)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )

    # 2. Policy lookup
    try:
        policy_name = resolve_policy_name(req.domain, req.variant)
    except ValueError as res_err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Policy not found: {res_err}",
        )

    try:
        policy = registry.get(policy_name)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Policy '{policy_name}' is not loaded.",
        )

    # 3. Observation encoding
    obs_tensor = ObservationCodec.encode(req.domain, req.variant, req.observation)

    # 4. Neural inference
    try:
        with torch.no_grad():
            action_result: Any
            hidden_out: list[float] | None = None
            if req.domain == "commander":
                assert isinstance(policy, CommanderPolicy)
                hidden = (
                    torch.tensor(req.hidden_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    if req.hidden_state
                    else None
                )
                action_result, log_prob, value, new_hidden = policy.act(
                    obs_tensor,
                    agent_id=req.agent_id,
                    hidden_state=hidden,
                    deterministic=req.deterministic,
                )
                if new_hidden is not None:
                    hidden_out = new_hidden.flatten().tolist()
            else:
                assert isinstance(policy, BaseLowLevelPolicy)
                action_result, log_prob, value = policy.act(
                    obs_tensor,
                    deterministic=req.deterministic,
                )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failure: {str(exc)}",
        )

    # 5. Action encoding
    action_json = ActionCodec.encode(req.domain, action_result)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return ActResponse(
        action=action_json,
        log_prob=float(log_prob),
        value=float(value),
        hidden_state=hidden_out,
        latency_ms=latency_ms,
    )


@router.post("/act/batch", response_model=BatchActResponse)
async def act_batch(
    batch_req: BatchActRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> BatchActResponse:
    """Execute sequential batch inference across multiple agent requests."""
    t_start = time.perf_counter()
    responses: list[ActResponse] = []

    for req in batch_req.requests:
        resp = await act(req, registry=registry)
        responses.append(resp)

    total_latency_ms = (time.perf_counter() - t_start) * 1000.0
    return BatchActResponse(
        responses=responses,
        total_latency_ms=total_latency_ms,
    )


@router.post("/load_checkpoint", response_model=LoadCheckpointResponse)
async def load_checkpoint(
    req: LoadCheckpointRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> LoadCheckpointResponse:
    """Hot-swap model weights from disk checkpoint artifact."""
    try:
        param_count = registry.load_checkpoint(req.policy_name, req.checkpoint_path)
    except FileNotFoundError as fnf:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(fnf),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed loading checkpoint: {str(err)}",
        )

    return LoadCheckpointResponse(
        status="loaded",
        policy_name=req.policy_name,
        param_count=param_count,
    )


@router.get("/policies", response_model=PoliciesResponse)
async def get_policies(
    registry: ModelRegistry = Depends(get_registry),
) -> PoliciesResponse:
    """List summary metadata for all currently registered policies."""
    policies = registry.list_policies()
    return PoliciesResponse(
        policies=[PolicyInfo(**p) for p in policies]
    )
