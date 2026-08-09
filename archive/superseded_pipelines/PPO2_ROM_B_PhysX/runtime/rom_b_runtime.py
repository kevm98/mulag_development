"""Per-episode ROM-B factors applied through explicit hydraulic channels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from PPO2_ROM_PhysX.runtime.command_rom_runtime import CommandCylinderROMRuntime


# ROM-B exposes exactly five factors through every public and private boundary.
# Flow is fixed by the fitted nominal ROM; it is not a randomized factor.
FACTOR_NAMES = ("e_comp", "d_seal_1", "d_seal_2", "d_seal_3", "d_seal_4")
SUPPORTED_FACTOR_NAMES = FACTOR_NAMES
ROMBMode = Literal["random", "nominal", "health"]


@dataclass(frozen=True)
class ROMBFactorConfig:
    """Audited uncertainty set and optional deterministic health-scale mode."""

    mode: ROMBMode = "random"
    e_comp_range: tuple[float, float] = (-0.125, 0.020)
    seal_range: tuple[float, float] = (0.0, 1.0)
    health_scale: float = 1.0
    randomization_scale: float = 1.0

    def validate(self) -> None:
        if self.mode not in ("random", "nominal", "health"):
            raise ValueError(f"unsupported ROM-B mode: {self.mode!r}")
        for name, bounds in (
            ("e_comp_range", self.e_comp_range),
            ("seal_range", self.seal_range),
        ):
            if len(bounds) != 2 or float(bounds[0]) > float(bounds[1]):
                raise ValueError(f"{name} must contain ordered low/high bounds")
        if not 0.4 <= float(self.health_scale) <= 1.2:
            raise ValueError("health_scale must be in [0.4, 1.2]")
        if not 0.0 <= float(self.randomization_scale) <= 1.0:
            raise ValueError("randomization_scale must be in [0, 1]")


def health_scale_to_factors(
    health_scale: float | torch.Tensor,
    *,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Map h in [0.4, 1.2] to the documented correlated five-factor path."""

    h = torch.as_tensor(health_scale, dtype=torch.float32, device=device)
    if torch.any((h < 0.4) | (h > 1.2)):
        raise ValueError("health_scale must be in [0.4, 1.2]")
    h = h.reshape(-1, 1)
    degraded = h <= 1.0
    degraded_severity = ((1.0 - h) / 0.6).clamp(0.0, 1.0)
    favorable_severity = ((h - 1.0) / 0.2).clamp(0.0, 1.0)
    e_comp = torch.where(
        degraded, -0.125 * degraded_severity, 0.020 * favorable_severity
    )
    seals = torch.where(
        degraded,
        degraded_severity.expand(-1, 4),
        torch.zeros(h.shape[0], 4, dtype=h.dtype, device=h.device),
    )
    return torch.cat((e_comp, seals), dim=-1)


def _validate_model_factors(factors: torch.Tensor) -> torch.Tensor:
    """Validate the exact five-factor tensor accepted by the ROM-B model."""
    if factors.ndim != 2 or factors.shape[1] != len(FACTOR_NAMES):
        raise ValueError(
            f"ROM-B factors must have shape (N, {len(FACTOR_NAMES)}), "
            f"got {tuple(factors.shape)}"
        )
    return factors


class ROMBCommandCylinderROMRuntime(CommandCylinderROMRuntime):
    """Command ROM with hidden factors sampled independently at reset."""

    def __init__(self, *, factor_cfg: ROMBFactorConfig | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.factor_cfg = factor_cfg or ROMBFactorConfig()
        self.factor_cfg.validate()
        self.true_factors = torch.zeros(
            self.num_envs, len(FACTOR_NAMES), dtype=torch.float32, device=self.device
        )
        self.randomization_scale = float(self.factor_cfg.randomization_scale)
        self._sample_factors(torch.arange(self.num_envs, device=self.device))

    def _sample_factors(self, env_ids: torch.Tensor) -> None:
        ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        if ids.numel() == 0:
            return
        cfg = self.factor_cfg
        if cfg.mode == "nominal":
            self.true_factors[ids] = 0.0
            return
        if cfg.mode == "health":
            factors = health_scale_to_factors(cfg.health_scale, device=self.device)
            self.true_factors[ids] = factors.expand(ids.numel(), -1)
            return
        scale = float(self.randomization_scale)
        unit = torch.rand(
            ids.numel(), len(FACTOR_NAMES), dtype=torch.float32, device=self.device
        )
        self.true_factors[ids, 0] = (
            scale * float(cfg.e_comp_range[0])
            + unit[:, 0] * scale * float(cfg.e_comp_range[1] - cfg.e_comp_range[0])
        )
        self.true_factors[ids, 1:5] = (
            scale * float(cfg.seal_range[0])
            + unit[:, 1:5] * scale * float(cfg.seal_range[1] - cfg.seal_range[0])
        )

    def set_randomization_scale(self, scale: float) -> None:
        """Set the curriculum envelope and resample all environments."""

        value = float(scale)
        if not 0.0 <= value <= 1.0:
            raise ValueError("randomization scale must be in [0, 1]")
        self.randomization_scale = value
        if self.factor_cfg.mode == "random":
            self._sample_factors(torch.arange(self.num_envs, device=self.device))

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        )
        self._sample_factors(ids)

    def _active_factors(self) -> torch.Tensor | None:
        return (
            None
            if self.factor_cfg.mode == "nominal"
            else _validate_model_factors(self.true_factors)
        )

    def predict_delta(self, **kwargs) -> torch.Tensor:
        if "system_factors" in kwargs or "rom_b_factors" in kwargs:
            raise ValueError("ROM-B factors are private runtime state")
        return super().predict_delta(
            **kwargs, rom_b_factors=self._active_factors()
        )

    def step(self, **kwargs):
        if "system_factors" in kwargs or "rom_b_factors" in kwargs:
            raise ValueError("ROM-B factors are private runtime state")
        return super().step(**kwargs, rom_b_factors=self._active_factors())

    def factor_log_scalars(self) -> dict[str, torch.Tensor]:
        logs = {
            f"rom_b/{name}_mean": self.true_factors[:, index].mean()
            for index, name in enumerate(FACTOR_NAMES)
        }
        logs.update({
            "rom_b/randomization_scale": torch.as_tensor(
                self.randomization_scale, dtype=torch.float32, device=self.device
            ),
            "rom_b/e_comp_min": self.true_factors[:, 0].min(),
            "rom_b/e_comp_max": self.true_factors[:, 0].max(),
            "rom_b/leakage_multiplier_mean": (
                1.0 + 0.25 * self.true_factors[:, 1:5]
            ).pow(3).mean(),
        })
        return logs
