"""Episode-level hydraulic randomization for ROM B.

The randomizer owns the hidden plant factors.  It never builds policy
observations; callers may log its tensors, but PPO should only observe the
unchanged task observation from PPO2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


ROMBMode = Literal["reference_rom", "rom_b_randomized"]

HIDDEN_FACTOR_NAMES = (
    "beta_eff",
    "beta_scale",
    "flow_scale",
    "pressure_scale",
    "pressure_limit_pa",
    "seal_factor",
    "leakage_scale",
    "valve_tau",
    "clearance_scale",
)

LOG_FACTOR_NAMES = (
    "beta_eff",
    "beta_scale",
    "flow_scale",
    "pressure_scale",
    "pressure_limit_pa",
    "seal_factor_1",
    "seal_factor_2",
    "seal_factor_3",
    "seal_factor_4",
    "leakage_scale_1",
    "leakage_scale_2",
    "leakage_scale_3",
    "leakage_scale_4",
    "valve_tau_1",
    "valve_tau_2",
    "valve_tau_3",
    "valve_tau_4",
)


@dataclass(frozen=True)
class ROMBRange:
    beta_eff_pa: tuple[float, float]
    flow_scale: tuple[float, float]
    pressure_scale: tuple[float, float]
    seal_factor: tuple[float, float]


NOMINAL_RANGE = ROMBRange(
    beta_eff_pa=(0.7e9, 0.7e9),
    flow_scale=(1.0, 1.0),
    pressure_scale=(1.0, 1.0),
    seal_factor=(0.0, 0.0),
)

CURRICULUM_STAGES: dict[int, ROMBRange] = {
    1: ROMBRange(
        beta_eff_pa=(0.6e9, 0.8e9),
        flow_scale=(0.95, 1.05),
        pressure_scale=(0.97, 1.03),
        seal_factor=(0.0, 0.25),
    ),
    2: ROMBRange(
        beta_eff_pa=(0.5e9, 0.9e9),
        flow_scale=(0.90, 1.10),
        pressure_scale=(0.95, 1.05),
        seal_factor=(0.0, 0.50),
    ),
    3: ROMBRange(
        beta_eff_pa=(0.4e9, 1.0e9),
        flow_scale=(0.85, 1.15),
        pressure_scale=(0.92, 1.08),
        seal_factor=(0.0, 0.75),
    ),
    4: ROMBRange(
        beta_eff_pa=(0.4e9, 1.2e9),
        flow_scale=(0.80, 1.20),
        pressure_scale=(0.90, 1.10),
        seal_factor=(0.0, 1.00),
    ),
}


@dataclass(frozen=True)
class ROMBRandomizationConfig:
    mode: ROMBMode = "rom_b_randomized"
    curriculum_stage: int = 4
    beta_ref_pa: float = 0.7e9
    pressure_ref_pa: float = 27.6e6
    nominal_valve_tau_s: float = 0.008
    nominal_leakage_decay_per_s: float = 0.05
    command_abs_limit: float = 0.0
    enable_pressure_saturation: bool = True
    preserve_nominal_equivalence: bool = True

    def resolved_range(self) -> ROMBRange:
        if self.mode == "reference_rom":
            return NOMINAL_RANGE
        if self.mode != "rom_b_randomized":
            raise ValueError(
                "ROM-B mode must be 'reference_rom' or 'rom_b_randomized', "
                f"got {self.mode!r}."
            )
        try:
            return CURRICULUM_STAGES[int(self.curriculum_stage)]
        except KeyError as exc:
            raise ValueError(
                f"ROM-B curriculum stage must be in [1, 4], got {self.curriculum_stage}."
            ) from exc


class ROMBRandomizer:
    """Sample and apply hidden ROM-B plant factors for vectorized envs."""

    def __init__(
        self,
        *,
        num_envs: int,
        device: str | torch.device,
        config: ROMBRandomizationConfig | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.config = config or ROMBRandomizationConfig()
        self.beta_eff = torch.empty(self.num_envs, device=self.device)
        self.beta_scale = torch.empty_like(self.beta_eff)
        self.flow_scale = torch.empty_like(self.beta_eff)
        self.pressure_scale = torch.empty_like(self.beta_eff)
        self.pressure_limit_pa = torch.empty_like(self.beta_eff)
        self.seal_factor = torch.empty(self.num_envs, 4, device=self.device)
        self.leakage_scale = torch.empty_like(self.seal_factor)
        self.valve_tau = torch.empty_like(self.seal_factor)
        self.clearance_scale = torch.empty_like(self.seal_factor)
        self.effective_valve = torch.zeros_like(self.seal_factor)
        self.effective_command = torch.zeros(self.num_envs, 8, device=self.device)
        self.last_pressure_saturation_mask = torch.zeros_like(self.seal_factor, dtype=torch.bool)
        self.last_invalid_pressure_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset()

    def _ids(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        ids = env_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        if ids.numel() and (int(ids.min().item()) < 0 or int(ids.max().item()) >= self.num_envs):
            raise IndexError(f"env_ids out of bounds for {self.num_envs} envs: {ids.detach().cpu().tolist()}")
        return ids

    def _uniform(self, shape: torch.Size | tuple[int, ...], low: float, high: float) -> torch.Tensor:
        if float(high) == float(low):
            return torch.full(shape, float(low), device=self.device, dtype=torch.float32)
        return torch.empty(shape, device=self.device, dtype=torch.float32).uniform_(float(low), float(high))

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = self._ids(env_ids)
        ranges = self.config.resolved_range()
        self.beta_eff[ids] = self._uniform((ids.numel(),), *ranges.beta_eff_pa)
        self.beta_scale[ids] = self.beta_eff[ids] / float(self.config.beta_ref_pa)
        self.flow_scale[ids] = self._uniform((ids.numel(),), *ranges.flow_scale)
        self.pressure_scale[ids] = self._uniform((ids.numel(),), *ranges.pressure_scale)
        self.pressure_limit_pa[ids] = float(self.config.pressure_ref_pa) * self.pressure_scale[ids]
        self.seal_factor[ids] = self._uniform((ids.numel(), 4), *ranges.seal_factor)
        self._derive_seal_quantities(ids)
        self.effective_valve[ids] = 0.0
        self.effective_command[ids] = 0.0
        self.last_pressure_saturation_mask[ids] = False
        self.last_invalid_pressure_mask[ids] = False

    def set_fixed(
        self,
        *,
        env_ids: torch.Tensor | None = None,
        beta_eff_pa: float = 0.7e9,
        flow_scale: float = 1.0,
        pressure_scale: float = 1.0,
        seal_factor: torch.Tensor | float = 0.0,
    ) -> None:
        ids = self._ids(env_ids)
        self.beta_eff[ids] = float(beta_eff_pa)
        self.beta_scale[ids] = self.beta_eff[ids] / float(self.config.beta_ref_pa)
        self.flow_scale[ids] = float(flow_scale)
        self.pressure_scale[ids] = float(pressure_scale)
        self.pressure_limit_pa[ids] = float(self.config.pressure_ref_pa) * self.pressure_scale[ids]
        seal = torch.as_tensor(seal_factor, device=self.device, dtype=torch.float32)
        if seal.ndim == 0:
            self.seal_factor[ids] = float(seal.item())
        else:
            seal = seal.reshape(-1, 4)
            if seal.shape[0] == 1:
                self.seal_factor[ids] = seal.expand(ids.numel(), -1)
            elif seal.shape[0] == ids.numel():
                self.seal_factor[ids] = seal
            else:
                raise ValueError(
                    f"seal_factor must have one row or {ids.numel()} rows, got {tuple(seal.shape)}"
                )
        self.seal_factor[ids] = self.seal_factor[ids].clamp(0.0, 1.0)
        self._derive_seal_quantities(ids)
        self.effective_valve[ids] = 0.0
        self.effective_command[ids] = 0.0
        self.last_pressure_saturation_mask[ids] = False
        self.last_invalid_pressure_mask[ids] = False

    def _derive_seal_quantities(self, ids: torch.Tensor) -> None:
        w = 0.25 * self.seal_factor[ids]
        self.leakage_scale[ids] = (1.0 + w).pow(3)
        self.valve_tau[ids] = float(self.config.nominal_valve_tau_s) * (1.0 + 2.5 * w)
        self.clearance_scale[ids] = 1.0 + w

    def valve_alpha(self, *, dt_model: float, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        ids = self._ids(env_ids)
        dt = torch.as_tensor(float(dt_model), device=self.device, dtype=torch.float32)
        tau = self.valve_tau[ids].clamp_min(1.0e-6)
        return 1.0 - torch.exp(-dt / tau)

    def randomization_active_mask(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        ids = self._ids(env_ids)
        if self.config.mode == "reference_rom":
            return torch.zeros(ids.numel(), dtype=torch.bool, device=self.device)
        if not self.config.preserve_nominal_equivalence:
            return torch.ones(ids.numel(), dtype=torch.bool, device=self.device)
        tol = 1.0e-6
        active = (
            (self.beta_scale[ids] - 1.0).abs() > tol
            | (self.flow_scale[ids] - 1.0).abs() > tol
            | (self.pressure_scale[ids] - 1.0).abs() > tol
            | (self.seal_factor[ids].abs().amax(dim=-1) > tol)
        )
        return active

    def transform_command(
        self,
        command: torch.Tensor,
        *,
        update_mask: torch.Tensor,
        dt_model: float,
    ) -> torch.Tensor:
        if command.shape != (self.num_envs, 8):
            raise ValueError(f"ROM-B command must have shape {(self.num_envs, 8)}, got {tuple(command.shape)}")
        command = command.to(device=self.device, dtype=torch.float32)
        if self.config.mode == "reference_rom":
            return command
        mask = update_mask.to(device=self.device, dtype=torch.bool).reshape(-1)
        if mask.shape != (self.num_envs,):
            raise ValueError(f"update_mask must have shape {(self.num_envs,)}, got {tuple(update_mask.shape)}")
        ids = torch.where(mask)[0]
        if ids.numel():
            alpha4 = self.valve_alpha(dt_model=dt_model, env_ids=ids).clamp(0.0, 1.0)
            # The learned command ROM already contains nominal valve dynamics.
            # The residual filter is therefore exact no-op for healthy seals.
            healthy = self.seal_factor[ids] <= 1.0e-8
            alpha4 = torch.where(healthy, torch.ones_like(alpha4), alpha4)
            alpha8 = torch.cat((alpha4, alpha4), dim=-1)
            self.effective_command[ids] = self.effective_command[ids] + alpha8 * (
                command[ids] - self.effective_command[ids]
            )
        scaled = self.effective_command * self.flow_scale.reshape(-1, 1)
        limit = float(self.config.command_abs_limit)
        if limit > 0.0:
            scaled = torch.clamp(scaled, -limit, limit)
        return torch.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    def apply_pressure_randomization(
        self,
        *,
        dP_current: torch.Tensor,
        delta_dP_ref: torch.Tensor,
        update_mask: torch.Tensor,
        dt_model: float,
    ) -> torch.Tensor:
        expected = (self.num_envs, 4)
        if dP_current.shape != expected or delta_dP_ref.shape != expected:
            raise ValueError(
                f"dP_current and delta_dP_ref must have shape {expected}, "
                f"got {tuple(dP_current.shape)} and {tuple(delta_dP_ref.shape)}"
            )
        corrected = delta_dP_ref.to(device=self.device, dtype=torch.float32).clone()
        if self.config.mode == "reference_rom":
            self.last_pressure_saturation_mask.zero_()
            self.last_invalid_pressure_mask.zero_()
            return torch.nan_to_num(corrected, nan=0.0, posinf=0.0, neginf=0.0)
        mask = update_mask.to(device=self.device, dtype=torch.bool).reshape(-1)
        ids = torch.where(mask & self.randomization_active_mask())[0]
        self.last_pressure_saturation_mask.zero_()
        self.last_invalid_pressure_mask.zero_()
        if ids.numel():
            dP = dP_current[ids].to(device=self.device, dtype=torch.float32)
            raw_next = dP + self.beta_scale[ids].reshape(-1, 1) * corrected[ids]
            leak_rate = max(float(self.config.nominal_leakage_decay_per_s), 0.0)
            if leak_rate > 0.0:
                extra_decay = leak_rate * (self.leakage_scale[ids] - 1.0) * dP * float(dt_model)
                raw_next = raw_next - extra_decay
            if self.config.enable_pressure_saturation:
                limit = self.pressure_limit_pa[ids].reshape(-1, 1).clamp_min(1.0)
                saturated_next = limit * torch.tanh(raw_next / limit)
                self.last_pressure_saturation_mask[ids] = (saturated_next - raw_next).abs() > 1.0
                raw_next = saturated_next
            invalid = ~torch.isfinite(raw_next).all(dim=-1)
            self.last_invalid_pressure_mask[ids] = invalid
            raw_next = torch.nan_to_num(raw_next, nan=0.0, posinf=0.0, neginf=0.0)
            corrected[ids] = raw_next - dP
        return torch.nan_to_num(corrected, nan=0.0, posinf=0.0, neginf=0.0)

    def update_effective_valve(self, predicted_valve: torch.Tensor, update_mask: torch.Tensor) -> None:
        if predicted_valve.shape != (self.num_envs, 4):
            raise ValueError(
                f"predicted_valve must have shape {(self.num_envs, 4)}, got {tuple(predicted_valve.shape)}"
            )
        ids = torch.where(update_mask.to(device=self.device, dtype=torch.bool).reshape(-1))[0]
        if ids.numel():
            self.effective_valve[ids] = predicted_valve[ids].detach()

    def factor_log_scalars(self) -> dict[str, torch.Tensor]:
        logs: dict[str, torch.Tensor] = {
            "rom_b/beta_eff_mean": torch.mean(self.beta_eff),
            "rom_b/beta_scale_mean": torch.mean(self.beta_scale),
            "rom_b/flow_scale_mean": torch.mean(self.flow_scale),
            "rom_b/pressure_scale_mean": torch.mean(self.pressure_scale),
            "rom_b/pressure_limit_pa_mean": torch.mean(self.pressure_limit_pa),
            "rom_b/pressure_saturation_rate": torch.mean(self.last_pressure_saturation_mask.float()),
            "rom_b/invalid_pressure_rate": torch.mean(self.last_invalid_pressure_mask.float()),
        }
        for idx in range(4):
            suffix = idx + 1
            logs[f"rom_b/seal_factor_{suffix}_mean"] = torch.mean(self.seal_factor[:, idx])
            logs[f"rom_b/leakage_scale_{suffix}_mean"] = torch.mean(self.leakage_scale[:, idx])
            logs[f"rom_b/valve_tau_{suffix}_mean"] = torch.mean(self.valve_tau[:, idx])
        return logs

    def factor_snapshot(self) -> dict[str, torch.Tensor]:
        return {
            "beta_eff": self.beta_eff.detach().clone(),
            "beta_scale": self.beta_scale.detach().clone(),
            "flow_scale": self.flow_scale.detach().clone(),
            "pressure_scale": self.pressure_scale.detach().clone(),
            "pressure_limit_pa": self.pressure_limit_pa.detach().clone(),
            "seal_factor": self.seal_factor.detach().clone(),
            "leakage_scale": self.leakage_scale.detach().clone(),
            "valve_tau": self.valve_tau.detach().clone(),
            "clearance_scale": self.clearance_scale.detach().clone(),
            "effective_valve": self.effective_valve.detach().clone(),
        }
