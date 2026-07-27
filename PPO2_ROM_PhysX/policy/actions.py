"""Pure residual-action transform shared by the environment and contracts."""

from __future__ import annotations

import torch


def apply_residual_action(
    *,
    baseline_goal: torch.Tensor,
    previous_filtered_action: torch.Tensor,
    policy_action: torch.Tensor,
    residual_scale_m: float,
    filter_beta: float | torch.Tensor,
    learned_min: torch.Tensor,
    learned_max: torch.Tensor,
    residual_scale_by_action_m: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw/filtered action and corrected four-cylinder goal.

    Three actions preserve the accepted contract and leave cylinder 4 at its
    deterministic baseline. Four actions permit a separately scaled cylinder-4
    residual for the experimental policy.
    """
    action_dim = int(policy_action.shape[-1])
    if action_dim not in (3, 4):
        raise ValueError(f"Policy action dimension must be 3 or 4, got {action_dim}")
    raw = torch.clamp(
        torch.nan_to_num(policy_action, nan=0.0, posinf=0.0, neginf=0.0),
        -1.0,
        1.0,
    )
    beta = torch.as_tensor(
        filter_beta, dtype=raw.dtype, device=raw.device
    ).reshape(-1)
    if beta.numel() == 1:
        beta = beta.expand(action_dim)
    if beta.numel() != action_dim:
        raise ValueError(f"filter beta has {beta.numel()} values for {action_dim} actions")
    beta = torch.clamp(beta, 0.0, 1.0).reshape(1, action_dim)
    filtered = beta * previous_filtered_action + (1.0 - beta) * raw
    if residual_scale_by_action_m is None:
        scales = torch.full(
            (action_dim,), float(residual_scale_m), dtype=raw.dtype, device=raw.device
        )
    else:
        scales = torch.as_tensor(
            residual_scale_by_action_m, dtype=raw.dtype, device=raw.device
        ).reshape(-1)
        if scales.numel() != action_dim:
            raise ValueError(f"residual scales have {scales.numel()} values for {action_dim} actions")
    baseline_clamped = torch.clamp(baseline_goal, learned_min, learned_max)
    corrected = baseline_clamped.clone()
    corrected[:, :action_dim] += scales.reshape(1, action_dim) * filtered
    corrected = torch.clamp(corrected, learned_min, learned_max)
    if action_dim == 3:
        # Re-copy after clamping: accepted policy keeps cylinder 4 teacher-owned.
        corrected[:, 3] = baseline_clamped[:, 3]
    return raw, filtered, corrected
