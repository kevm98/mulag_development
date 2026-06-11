from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

try:
    from rom_data_contract import DP_SLICE, EXTENSION_SLICE, EXTENSION_VELOCITY_SLICE, JOINT_POS_SLICE, JOINT_VEL_SLICE
except ImportError:
    from kevin_integration.rom_cylinders.rom_data_contract import (
        DP_SLICE,
        EXTENSION_SLICE,
        EXTENSION_VELOCITY_SLICE,
        JOINT_POS_SLICE,
        JOINT_VEL_SLICE,
    )


FINAL_LINEAR_FEATURE_NAMES: tuple[tuple[str, ...], ...] = (
    ("dP_1_MPa", "Extension_1", "ExtensionVelocity_1", "JointPos_1", "JointVel_1"),
    ("dP_2_MPa", "Extension_2", "ExtensionVelocity_2", "JointPos_2", "JointVel_2"),
    ("dP_3_MPa", "Extension_3", "ExtensionVelocity_3", "JointPos_3", "JointVel_3"),
    ("dP_4_MPa", "Extension_4", "ExtensionVelocity_4", "JointPos_4", "JointVel_4"),
)


@dataclass(frozen=True)
class FinalCausalLinearConfig:
    dt: float = 0.05
    pressure_scale: float = 1.0e-6


class FinalCausalLinearCorrectionROM(nn.Module):
    """Minimal causal kinematic base plus bounded per-cylinder linear correction."""

    def __init__(
        self,
        *,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        weights: torch.Tensor,
        bias: torch.Tensor,
        correction_scale: torch.Tensor,
        config: FinalCausalLinearConfig = FinalCausalLinearConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("feature_mean", feature_mean.detach().clone().float().reshape(4, 5))
        self.register_buffer("feature_std", torch.clamp(feature_std.detach().clone().float().reshape(4, 5), min=1.0e-12))
        self.register_buffer("weights", weights.detach().clone().float().reshape(4, 5))
        self.register_buffer("bias", bias.detach().clone().float().reshape(4))
        self.register_buffer("correction_scale", torch.clamp(correction_scale.detach().clone().float().reshape(4), min=1.0e-12))

    def cylinder_features(self, x_raw: torch.Tensor, *, extension_override: torch.Tensor | None = None) -> torch.Tensor:
        current = x_raw[:, -1, :] if x_raw.ndim == 3 else x_raw
        if current.ndim != 2 or current.shape[-1] < 26:
            raise ValueError(f"x_raw must be [B,T,26] or [B,26], got {tuple(x_raw.shape)}")
        dP = current[:, DP_SLICE] * float(self.config.pressure_scale)
        extension = current[:, EXTENSION_SLICE] if extension_override is None else extension_override
        velocity = current[:, EXTENSION_VELOCITY_SLICE]
        joint_pos = current[:, JOINT_POS_SLICE][:, :4]
        joint_vel = current[:, JOINT_VEL_SLICE][:, :4]
        return torch.stack([dP, extension, velocity, joint_pos, joint_vel], dim=-1)

    def correction(self, x_raw: torch.Tensor, *, extension_override: torch.Tensor | None = None) -> torch.Tensor:
        features = self.cylinder_features(x_raw, extension_override=extension_override)
        normalized = (features - self.feature_mean.to(features.device)) / self.feature_std.to(features.device)
        raw = torch.sum(normalized * self.weights.to(features.device).reshape(1, 4, 5), dim=-1) + self.bias.to(features.device)
        scale = self.correction_scale.to(features.device).reshape(1, 4)
        return scale * torch.tanh(raw / scale)

    def forward(self, x_raw: torch.Tensor, *, extension_override: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        current = x_raw[:, -1, :] if x_raw.ndim == 3 else x_raw
        velocity = current[:, EXTENSION_VELOCITY_SLICE]
        base = velocity * float(self.config.dt)
        correction = self.correction(x_raw, extension_override=extension_override)
        final = base + correction
        return {
            "delta_extension_kinematic": base,
            "delta_extension_correction": correction,
            "delta_extension_final": final,
            "final_delta_extension": final,
            "finite_flags": torch.isfinite(final).all(dim=-1),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "dt": float(self.config.dt),
            "pressure_scale": float(self.config.pressure_scale),
            "feature_names_per_cylinder": [list(names) for names in FINAL_LINEAR_FEATURE_NAMES],
            "feature_mean": self.feature_mean.detach().cpu().tolist(),
            "feature_std": self.feature_std.detach().cpu().tolist(),
            "weights": self.weights.detach().cpu().tolist(),
            "bias": self.bias.detach().cpu().tolist(),
            "correction_scale": self.correction_scale.detach().cpu().tolist(),
            "coefficient_note": "Effective transition coefficients, not physical hydraulic parameters.",
        }


class FinalCausalROMInference:
    """Deployment-style wrapper around the final exported causal ROM package."""

    def __init__(self, export_dir: str | Path, *, device: str | torch.device = "cpu") -> None:
        self.export_dir = Path(export_dir)
        self.device = torch.device(device)
        payload = torch.load(self.export_dir / "best_causal_linear_rom.pth", map_location=self.device)
        scaler = torch.load(self.export_dir / "best_causal_linear_rom_scaler.pth", map_location=self.device)
        self.model = FinalCausalLinearCorrectionROM(
            feature_mean=scaler["feature_mean"],
            feature_std=scaler["feature_std"],
            weights=payload["weights"],
            bias=payload["bias"],
            correction_scale=payload["correction_scale"],
            config=FinalCausalLinearConfig(dt=float(payload["dt"]), pressure_scale=float(payload["pressure_scale"])),
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, x_raw: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(x_raw.to(self.device).float())
