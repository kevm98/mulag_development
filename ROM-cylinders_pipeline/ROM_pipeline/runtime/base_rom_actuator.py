from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..data.base_rom_schema import BASE_ROM_SCHEMA, BaseROMSchema
from ..models.base_hydraulic_rom import BaseHydraulicROM
from .cylinder_target_integrator import CylinderTargetIntegrator
from .rom_history import ROMHistory


@dataclass(frozen=True)
class BaseRuntimeOutput:
    delta_pressure: torch.Tensor
    delta_velocity: torch.Tensor
    delta_extension: torch.Tensor
    predicted_force: torch.Tensor
    position_target: torch.Tensor
    diagnostics: dict[str, Any]


class BaseROMActuator:
    """Native-0.05 s base-ROM wrapper, independent of IsaacLab imports."""

    def __init__(
        self,
        model: BaseHydraulicROM,
        *,
        num_envs: int,
        cylinder_lower: torch.Tensor,
        cylinder_upper: torch.Tensor,
        max_target_lead_m: float = 0.025,
        schema: BaseROMSchema = BASE_ROM_SCHEMA,
    ) -> None:
        self.model = model.eval()
        self.schema = schema
        device = model.input_mean.device
        self.history = ROMHistory(num_envs, schema.history_length, schema.input_dim, device=device)
        self.targets = CylinderTargetIntegrator(
            cylinder_lower.to(device), cylinder_upper.to(device), max_target_lead_m=max_target_lead_m
        )

    @torch.no_grad()
    def reset(self, measured_extension: torch.Tensor, current_row: torch.Tensor | None = None) -> None:
        self.targets.reset(measured_extension)
        self.history.reset()
        if current_row is not None:
            self.history.prime(current_row)

    @torch.no_grad()
    def step(
        self,
        *,
        state_row: torch.Tensor,
        valve_command: torch.Tensor,
        measured_extension: torch.Tensor,
    ) -> BaseRuntimeOutput:
        if state_row.shape[-1] != 16:
            raise ValueError("Base ROM state_row must be [Extension, Velocity, dP, Force] (16D).")
        if valve_command.shape != (state_row.shape[0], 4):
            raise ValueError("valve_command must have shape [num_envs, 4].")
        row = torch.cat((valve_command, state_row), dim=-1)
        self.history.append(row)
        output = self.model(self.history.buffer, valve_command)
        target, target_diagnostics = self.targets.step(output.delta_extension, measured_extension)
        if not all(torch.isfinite(value).all() for value in (
            output.delta_pressure, output.delta_velocity, output.delta_extension, output.predicted_force, target
        )):
            raise FloatingPointError("Base ROM actuator produced NaN/Inf.")
        return BaseRuntimeOutput(
            output.delta_pressure, output.delta_velocity, output.delta_extension, output.predicted_force, target,
            {**target_diagnostics, "history_ready": self.history.ready.clone(),
             "effective_actuation": output.effective_actuation,
             "pressure_residual": output.pressure_residual,
             "velocity_residual": output.velocity_residual,
             "extension_residual": output.extension_residual},
        )
