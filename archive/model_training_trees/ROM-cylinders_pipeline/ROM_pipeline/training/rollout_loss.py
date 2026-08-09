from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..data.rom_schema import DEFAULT_SCHEMA, ROMSchema
from ..models.hydraulic_rom import HydraulicROM, ROMOutput


@dataclass(frozen=True)
class LossWeights:
    pressure: float = 1.0
    force: float = 1.0
    velocity: float = 2.0
    extension: float = 1.0
    rollout: float = 2.0
    command: float = 1.0
    residual: float = 0.1
    actuation: float = 0.5


def output_tensor(output: ROMOutput) -> torch.Tensor:
    return torch.cat((output.predicted_valve, output.delta_pressure, output.delta_velocity, output.delta_extension), dim=-1)


def compute_one_step_loss(
    output: ROMOutput,
    target: torch.Tensor,
    target_std: torch.Tensor,
    *,
    valve_label: torch.Tensor | None = None,
    weights: LossWeights = LossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    scale = target_std.reshape(1, -1).clamp_min(1.0e-8)
    prediction = output_tensor(output)
    error = (prediction - target) / scale
    terms = {
        "actuation": F.mse_loss(error[:, 0:4], torch.zeros_like(error[:, 0:4])),
        "pressure": F.mse_loss(error[:, 4:8], torch.zeros_like(error[:, 4:8])),
        "velocity": F.mse_loss(error[:, 8:12], torch.zeros_like(error[:, 8:12])),
        "extension": F.mse_loss(error[:, 12:16], torch.zeros_like(error[:, 12:16])),
        "residual": (
            output.pressure_residual.square().mean() / 1.0e12
            + output.velocity_residual.square().mean()
            + output.extension_residual.square().mean() / 1.0e-6
        ),
    }
    if valve_label is not None and not torch.allclose(target[:, 0:4], valve_label):
        raise ValueError("Valve auxiliary label does not match the ROM valve target.")
    total = (
        weights.pressure * terms["pressure"]
        + weights.velocity * terms["velocity"]
        + weights.extension * terms["extension"]
        + weights.residual * terms["residual"]
        + weights.actuation * terms["actuation"]
    )
    return total, terms


def advance_history(
    history: torch.Tensor,
    output: ROMOutput,
    next_command: torch.Tensor,
    *,
    schema: ROMSchema = DEFAULT_SCHEMA,
    teacher_next: torch.Tensor | None = None,
    teacher_probability: float = 0.0,
) -> torch.Tensor:
    next_row = history[:, -1].clone()
    next_row[:, schema.extension_slice] = output.next_extension
    next_row[:, schema.velocity_slice] = output.next_velocity
    next_row[:, schema.pressure_slice] = output.next_pressure
    next_row[:, schema.previous_valve_slice] = output.predicted_valve
    next_row[:, schema.command_slice] = next_command
    if teacher_next is not None and teacher_probability > 0.0:
        mask = torch.rand((len(history), 1), device=history.device) < teacher_probability
        next_row = torch.where(mask, teacher_next, next_row)
    return torch.cat((history[:, 1:], next_row.unsqueeze(1)), dim=1)


def rollout(
    model: HydraulicROM,
    initial_history: torch.Tensor,
    commands: torch.Tensor,
    *,
    teacher_rows: torch.Tensor | None = None,
    teacher_probability: float = 0.0,
) -> list[ROMOutput]:
    if commands.ndim != 3 or commands.shape[-1] != model.config.command_dim:
        raise ValueError("commands must have shape [batch, horizon, command_dim].")
    history = initial_history
    outputs: list[ROMOutput] = []
    for step in range(commands.shape[1]):
        output = model(history, commands[:, step])
        outputs.append(output)
        teacher = None if teacher_rows is None else teacher_rows[:, step]
        history = advance_history(
            history,
            output,
            commands[:, min(step + 1, commands.shape[1] - 1)],
            schema=model.schema,
            teacher_next=teacher,
            teacher_probability=teacher_probability,
        )
    return outputs
