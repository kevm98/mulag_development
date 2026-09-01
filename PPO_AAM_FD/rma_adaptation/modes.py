"""Explicit runtime arms and leakage guards."""

from __future__ import annotations

from enum import Enum

import torch

from .core import CONTEXT_DIM


class ContextMode(str, Enum):
    BASELINE = "BASELINE"
    PRIVILEGED = "PRIVILEGED"
    MEAN_CONTEXT = "MEAN_CONTEXT"
    SHUFFLED_CONTEXT = "SHUFFLED_CONTEXT"
    AAM_ESTIMATED = "AAM_ESTIMATED"
    AAM_FROZEN = "AAM_FROZEN"
    AAM_ESTIMATED_FINETUNED = "AAM_ESTIMATED_FINETUNED"


def _width(value: torch.Tensor, name: str) -> None:
    if value.ndim != 2 or value.shape[-1] != CONTEXT_DIM:
        raise ValueError(f"{name} must have shape (N, {CONTEXT_DIM})")


def assemble_policy_observation(
    base_obs: torch.Tensor,
    mode: ContextMode | str,
    *,
    privileged: torch.Tensor | None = None,
    mean_context: torch.Tensor | None = None,
    shuffled_context: torch.Tensor | None = None,
    estimated: torch.Tensor | None = None,
    frozen: torch.Tensor | None = None,
    estimated_finetuned: torch.Tensor | None = None,
) -> torch.Tensor:
    """Append exactly one declared context source to the base PPO observation."""

    base = torch.as_tensor(base_obs, dtype=torch.float32)
    if base.ndim != 2:
        raise ValueError("base_obs must be a 2-D tensor")
    try:
        arm = mode if isinstance(mode, ContextMode) else ContextMode(str(mode).upper())
    except ValueError as exc:
        raise ValueError(f"unknown context mode {mode!r}") from exc
    if arm is ContextMode.BASELINE:
        return base
    sources = {
        ContextMode.PRIVILEGED: privileged,
        ContextMode.MEAN_CONTEXT: mean_context,
        ContextMode.SHUFFLED_CONTEXT: shuffled_context,
        ContextMode.AAM_ESTIMATED: estimated,
        ContextMode.AAM_FROZEN: frozen,
        ContextMode.AAM_ESTIMATED_FINETUNED: estimated_finetuned,
    }
    context = sources[arm]
    if context is None:
        raise ValueError(f"{arm.value} requires its context tensor")
    context = torch.as_tensor(context, dtype=base.dtype, device=base.device)
    _width(context, arm.value)
    if context.shape[0] != base.shape[0]:
        raise ValueError("context and base observation batch sizes differ")
    return torch.cat((base, context), dim=-1)


def assert_no_context_leakage(feature_names: list[str] | tuple[str, ...]) -> None:
    """Reject privileged labels in the future AAM input schema."""

    forbidden = {
        "actuator_authority_alpha", "hydraulic_response_time_tau",
        "actuator_delay_steps", "mass_scale", "com_z", "privileged_context",
        "true_context", "plant_id",
    }
    leaked = sorted(set(feature_names).intersection(forbidden))
    if leaked:
        raise ValueError(f"AAM feature schema leaks privileged labels: {leaked}")
