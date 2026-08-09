from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..data.rom_schema import DEFAULT_SCHEMA, ROMSchema
from ..models.hydraulic_rom import HydraulicROM
from .cylinder_target_integrator import CylinderTargetIntegrator
from .rom_history import ROMHistory


@dataclass(frozen=True)
class RuntimeROMOutput:
    predicted_valve: torch.Tensor
    delta_pressure: torch.Tensor
    delta_velocity: torch.Tensor
    delta_extension: torch.Tensor
    position_target: torch.Tensor
    diagnostics: dict[str, Any]


class ROMActuator:
    """Deterministic runtime contract independent of IsaacLab imports."""

    def __init__(
        self,
        model: HydraulicROM,
        *,
        num_envs: int,
        cylinder_lower: torch.Tensor,
        cylinder_upper: torch.Tensor,
        max_target_lead_m: float = 0.025,
        schema: ROMSchema = DEFAULT_SCHEMA,
    ) -> None:
        self.model = model.eval()
        self.schema = schema
        device = model.input_mean.device
        self.history = ROMHistory(num_envs, schema.history_length, schema.input_dim, device=device)
        self.targets = CylinderTargetIntegrator(
            cylinder_lower.to(device), cylinder_upper.to(device), max_target_lead_m=max_target_lead_m
        )

    @torch.no_grad()
    def reset(self, measured_extension: torch.Tensor, state_command_row: torch.Tensor | None = None) -> None:
        self.targets.reset(measured_extension)
        self.history.reset()
        if state_command_row is not None:
            self.history.prime(state_command_row)

    @torch.no_grad()
    def step(
        self,
        *,
        state_row: torch.Tensor,
        desired_cylinder_command: torch.Tensor,
        measured_extension: torch.Tensor,
    ) -> RuntimeROMOutput:
        if state_row.shape[-1] != len(self.schema.state_feature_names):
            raise ValueError("state_row does not match the 16D cylinder-state contract.")
        row = torch.cat((state_row, desired_cylinder_command), dim=-1)
        self.history.append(row)
        output = self.model(self.history.buffer, desired_cylinder_command)
        target, target_diagnostics = self.targets.step(output.delta_extension, measured_extension)
        finite = all(torch.isfinite(value).all() for value in (
            output.predicted_valve, output.delta_pressure, output.delta_velocity, output.delta_extension, target
        ))
        if not finite:
            raise FloatingPointError("ROM actuator produced NaN/Inf.")
        return RuntimeROMOutput(
            predicted_valve=output.predicted_valve,
            delta_pressure=output.delta_pressure,
            delta_velocity=output.delta_velocity,
            delta_extension=output.delta_extension,
            position_target=target,
            diagnostics={
                **target_diagnostics,
                "history_ready": self.history.ready.clone(),
                "effective_actuation": output.effective_actuation,
                "pressure_residual": output.pressure_residual,
                "velocity_residual": output.velocity_residual,
                "extension_residual": output.extension_residual,
            },
        )
