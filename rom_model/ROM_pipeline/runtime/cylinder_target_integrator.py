from __future__ import annotations

import torch


class CylinderTargetIntegrator:
    def __init__(self, lower: torch.Tensor, upper: torch.Tensor, *, max_target_lead_m: float = 0.025) -> None:
        if lower.shape[-1] != 4 or upper.shape != lower.shape or not torch.all(lower < upper):
            raise ValueError("Cylinder limits must have matching [..., 4] shapes with lower < upper.")
        self.lower = lower.float()
        self.upper = upper.float()
        self.max_target_lead_m = float(max_target_lead_m)
        self.target: torch.Tensor | None = None

    def reset(self, measured_extension: torch.Tensor) -> torch.Tensor:
        self.target = measured_extension.float().clone()
        return self.target

    def step(self, delta_extension: torch.Tensor, measured_extension: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.target is None:
            self.reset(measured_extension)
        assert self.target is not None
        proposed = self.target + delta_extension
        lead_lower = measured_extension - self.max_target_lead_m
        lead_upper = measured_extension + self.max_target_lead_m
        lead_clamped = torch.maximum(torch.minimum(proposed, lead_upper), lead_lower)
        limited = torch.maximum(torch.minimum(lead_clamped, self.upper), self.lower)
        diagnostics = {
            "lead_clipped": proposed.ne(lead_clamped),
            "limit_clipped": lead_clamped.ne(limited),
            "target_lead": limited - measured_extension,
        }
        self.target = limited
        return limited.clone(), diagnostics
