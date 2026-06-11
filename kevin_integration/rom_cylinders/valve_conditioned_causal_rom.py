from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

try:
    from rom_data_contract import (
        CORE_INPUT_FEATURE_NAMES,
        DP_SLICE,
        EXTENSION_SLICE,
        EXTENSION_VELOCITY_SLICE,
        HISTORY_LENGTH,
        JOINT_POS_SLICE,
        JOINT_VEL_SLICE,
        VALVE_SLICE,
    )
except ImportError:
    from kevin_integration.rom_cylinders.rom_data_contract import (
        CORE_INPUT_FEATURE_NAMES,
        DP_SLICE,
        EXTENSION_SLICE,
        EXTENSION_VELOCITY_SLICE,
        HISTORY_LENGTH,
        JOINT_POS_SLICE,
        JOINT_VEL_SLICE,
        VALVE_SLICE,
    )


ROM_DIR = Path(__file__).resolve().parent
EXPORT_DIR = ROM_DIR / "exported_valve_conditioned_causal_rom"
DT = 0.05
PRESSURE_SCALE = 1.0e-6
MODEL_VERSION = "valve_conditioned_causal_rom_v1"


class ValveConditionedCausalCylinderROM(nn.Module):
    def __init__(
        self,
        *,
        coefficients: torch.Tensor,
        extension_mean: torch.Tensor,
        joint_pos_mean: torch.Tensor,
        correction_bound: torch.Tensor,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
        training_mean_dp: torch.Tensor | None = None,
        dt: float = DT,
    ) -> None:
        super().__init__()
        self.dt = float(dt)
        self.register_buffer("coefficients", coefficients.detach().clone().float().reshape(4, 17))
        self.register_buffer("extension_mean", extension_mean.detach().clone().float().reshape(4))
        self.register_buffer("joint_pos_mean", joint_pos_mean.detach().clone().float().reshape(5))
        self.register_buffer("correction_bound", torch.clamp(correction_bound.detach().clone().float().reshape(4), min=1.0e-8))
        if feature_mean is None:
            feature_mean = torch.zeros((4, 17), dtype=torch.float32)
        if feature_std is None:
            feature_std = torch.ones((4, 17), dtype=torch.float32)
        if training_mean_dp is None:
            training_mean_dp = torch.zeros((4,), dtype=torch.float32)
        self.register_buffer("feature_mean", feature_mean.detach().clone().float().reshape(4, 17))
        self.register_buffer("feature_std", torch.clamp(feature_std.detach().clone().float().reshape(4, 17), min=1.0e-12))
        self.register_buffer("training_mean_dp", training_mean_dp.detach().clone().float().reshape(4))

    def _features(self, current: torch.Tensor) -> torch.Tensor:
        u = current[:, VALVE_SLICE]
        dp = current[:, DP_SLICE] * PRESSURE_SCALE
        ext = current[:, EXTENSION_SLICE] - self.extension_mean.to(current.device).reshape(1, 4)
        vel = current[:, EXTENSION_VELOCITY_SLICE]
        q = current[:, JOINT_POS_SLICE] - self.joint_pos_mean.to(current.device).reshape(1, 5)
        qv = current[:, JOINT_VEL_SLICE]
        rows = []
        ones = torch.ones((current.shape[0],), dtype=current.dtype, device=current.device)
        for i in range(4):
            rows.append(torch.stack([ones, u[:, i], u[:, i] * torch.abs(u[:, i]), u[:, i] * dp[:, i], dp[:, i], vel[:, i], ext[:, i], *q.T, *qv.T], dim=-1))
        return torch.stack(rows, dim=1)

    def cylinder_features(self, x_raw: torch.Tensor, *, extension_override: torch.Tensor | None = None) -> torch.Tensor:
        current = x_raw[:, -1, :] if x_raw.ndim == 3 else x_raw
        if extension_override is not None:
            current = current.clone()
            current[:, EXTENSION_SLICE] = extension_override
        return self._features(current.float())

    def forward(self, x_raw: torch.Tensor, *, extension_override: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        current = x_raw[:, -1, :] if x_raw.ndim == 3 else x_raw
        if extension_override is not None:
            current = current.clone()
            current[:, EXTENSION_SLICE] = extension_override
        features = self._features(current.float())
        terms = features * self.coefficients.to(features.device).reshape(1, 4, 17)
        dynamic_unbounded = terms.sum(dim=-1)
        bound = self.correction_bound.to(features.device).reshape(1, 4)
        dynamic = bound * torch.tanh(dynamic_unbounded / bound)
        kinematic = current[:, EXTENSION_VELOCITY_SLICE].float() * self.dt
        final = kinematic + dynamic
        valve_span = 2.0 * self.coefficients[:, 1].to(features.device).reshape(1, 4)
        return {
            "delta_kinematic": kinematic,
            "delta_valve_linear": terms[:, :, 1],
            "delta_valve_quadratic": terms[:, :, 2],
            "delta_valve_pressure": terms[:, :, 3],
            "delta_pressure": terms[:, :, 4],
            "delta_velocity": terms[:, :, 5],
            "delta_load": terms[:, :, 7:17].sum(dim=-1),
            "delta_dynamic_correction": dynamic,
            "delta_extension_final": final,
            "delta_extension_kinematic": kinematic,
            "delta_extension_correction": dynamic,
            "final_delta_extension": final,
            "valve_authority_span": valve_span.expand_as(final),
            "finite": torch.isfinite(final).all(dim=-1),
            "finite_flags": torch.isfinite(final).all(dim=-1),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "model_version": MODEL_VERSION,
            "dt": self.dt,
            "pressure_scale": PRESSURE_SCALE,
            "coefficient_names": [
                "bias",
                "k_u",
                "k_uu",
                "k_udp",
                "k_dp",
                "k_v",
                "k_x",
                "Wq_1",
                "Wq_2",
                "Wq_3",
                "Wq_4",
                "Wq_5",
                "Wv_1",
                "Wv_2",
                "Wv_3",
                "Wv_4",
                "Wv_5",
            ],
            "coefficients": self.coefficients.detach().cpu().tolist(),
            "extension_mean": self.extension_mean.detach().cpu().tolist(),
            "joint_pos_mean": self.joint_pos_mean.detach().cpu().tolist(),
            "correction_bound": self.correction_bound.detach().cpu().tolist(),
            "feature_mean": self.feature_mean.detach().cpu().tolist(),
            "feature_std": self.feature_std.detach().cpu().tolist(),
            "training_mean_dp": self.training_mean_dp.detach().cpu().tolist(),
        }


class ValveConditionedROMInference:
    def __init__(self, export_dir: str | Path, *, device: str | torch.device = "cpu") -> None:
        self.export_dir = Path(export_dir)
        self.device = torch.device(device)
        payload = torch.load(self.export_dir / "best_valve_conditioned_causal_rom.pth", map_location=self.device)
        scaler = torch.load(self.export_dir / "best_valve_conditioned_causal_rom_scaler.pth", map_location=self.device)
        self.model = ValveConditionedCausalCylinderROM(
            coefficients=payload["coefficients"],
            extension_mean=scaler["extension_mean"],
            joint_pos_mean=scaler["joint_pos_mean"],
            correction_bound=payload["correction_bound"],
            feature_mean=scaler["feature_mean"],
            feature_std=scaler["feature_std"],
            training_mean_dp=scaler["training_mean_dp"],
            dt=float(payload["dt"]),
        ).to(self.device)
        self.model.eval()
        self.assert_valve_authority()

    @torch.no_grad()
    def predict(self, x_raw: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(x_raw.to(self.device).float())

    @torch.no_grad()
    def assert_valve_authority(self) -> None:
        x = torch.zeros((1, HISTORY_LENGTH, len(CORE_INPUT_FEATURE_NAMES)), dtype=torch.float32, device=self.device)
        x[:, :, EXTENSION_SLICE] = self.model.extension_mean.reshape(1, 1, 4)
        neg = x.clone()
        pos = x.clone()
        neg[:, -1, VALVE_SLICE] = -1.0
        pos[:, -1, VALVE_SLICE] = 1.0
        span = self.predict(pos)["final_delta_extension"] - self.predict(neg)["final_delta_extension"]
        if not torch.isfinite(span).all() or bool((span.abs() <= 1.0e-8).any()):
            raise RuntimeError(f"Valve-conditioned ROM startup authority assertion failed: span={span.detach().cpu().tolist()}")
