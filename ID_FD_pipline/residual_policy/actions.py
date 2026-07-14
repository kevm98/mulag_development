"""Pure residual-action transform shared by the environment and contracts."""

from __future__ import annotations

import torch


def apply_residual_action(
    *,
    baseline_goal: torch.Tensor,
    previous_filtered_action: torch.Tensor,
    policy_action: torch.Tensor,
    residual_scale_m: float,
    filter_beta: float,
    learned_min: torch.Tensor,
    learned_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return clamped raw action, filtered action, and corrected 4-D goal."""
    raw = torch.clamp(
        torch.nan_to_num(policy_action, nan=0.0, posinf=0.0, neginf=0.0),
        -1.0,
        1.0,
    )
    beta = min(max(float(filter_beta), 0.0), 1.0)
    filtered = beta * previous_filtered_action + (1.0 - beta) * raw
    baseline_clamped = torch.clamp(baseline_goal, learned_min, learned_max)
    corrected = baseline_clamped.clone()
    corrected[:, :3] += float(residual_scale_m) * filtered
    corrected = torch.clamp(corrected, learned_min, learned_max)
    # Re-copy after clamping: cylinder 4 is owned solely by the teacher.
    corrected[:, 3] = baseline_clamped[:, 3]
    return raw, filtered, corrected
