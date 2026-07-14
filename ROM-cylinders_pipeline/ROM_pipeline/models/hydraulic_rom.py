from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from ..data.rom_schema import DEFAULT_SCHEMA, ROMSchema


@dataclass(frozen=True)
class HydraulicROMConfig:
    input_dim: int = 34
    command_dim: int = 8
    hidden_dim: int = 128
    recurrent_layers: int = 2
    dropout: float = 0.05
    sample_time_s: float = 0.01
    pressure_residual_fraction: float = 0.10
    velocity_residual_fraction: float = 0.10
    position_residual_bound_m: float = 2.0e-4
    max_delta_pressure_pa: float = 8.0e6
    max_delta_velocity_mps: float = 0.20
    max_force_residual_n: float = 2.0e4

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class ROMOutput(NamedTuple):
    delta_pressure: torch.Tensor
    delta_velocity: torch.Tensor
    delta_extension: torch.Tensor
    predicted_force: torch.Tensor
    effective_actuation: torch.Tensor
    pressure_residual: torch.Tensor
    velocity_residual: torch.Tensor
    extension_residual: torch.Tensor
    next_pressure: torch.Tensor
    next_velocity: torch.Tensor
    next_extension: torch.Tensor
    hidden: torch.Tensor


class SmoothDeadzone(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.raw_width = nn.Parameter(torch.full((channels,), -2.5))
        self.raw_positive_gain = nn.Parameter(torch.zeros(channels))
        self.raw_negative_gain = nn.Parameter(torch.zeros(channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        width = 0.25 * torch.sigmoid(self.raw_width)
        magnitude = F.softplus(12.0 * (value.abs() - width)) / 12.0
        direction = torch.tanh(20.0 * value)
        positive_gain = 0.5 + torch.sigmoid(self.raw_positive_gain)
        negative_gain = 0.5 + torch.sigmoid(self.raw_negative_gain)
        gain = torch.where(value >= 0.0, positive_gain, negative_gain)
        return torch.tanh(gain * direction * magnitude)


class HydraulicROM(nn.Module):
    """Coupled gray-box ROM with explicit pressure, force, velocity and position updates."""

    def __init__(
        self,
        config: HydraulicROMConfig = HydraulicROMConfig(),
        *,
        schema: ROMSchema = DEFAULT_SCHEMA,
        input_mean: torch.Tensor | None = None,
        input_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        schema.validate()
        if config.input_dim != schema.input_dim or config.command_dim != schema.command_dim:
            raise ValueError("Model configuration does not match the ROM schema.")
        self.config = config
        self.schema = schema
        self.register_buffer("input_mean", torch.zeros(config.input_dim) if input_mean is None else input_mean.float())
        self.register_buffer("input_std", torch.ones(config.input_dim) if input_std is None else input_std.float().clamp_min(1.0e-8))
        self.encoder = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.recurrent_layers,
            dropout=config.dropout if config.recurrent_layers > 1 else 0.0,
            batch_first=True,
        )
        self.actuation_head = nn.Sequential(nn.Linear(config.hidden_dim + config.command_dim, 64), nn.SiLU(), nn.Linear(64, 4))
        self.deadzone = SmoothDeadzone(4)
        self.pressure_residual_head = nn.Linear(config.hidden_dim, 4)
        self.force_residual_head = nn.Sequential(nn.Linear(config.hidden_dim + 4 + 5, 64), nn.SiLU(), nn.Linear(64, 4))
        self.load_head = nn.Sequential(nn.Linear(config.hidden_dim + 4 + 10, 64), nn.SiLU(), nn.Linear(64, 4))
        self.velocity_residual_head = nn.Linear(config.hidden_dim, 4)
        self.position_residual_head = nn.Linear(config.hidden_dim, 4)

        # Positive, trainable effective physical parameters. Values are conservative
        # initializations and must be replaced or bounded using the dataset audit.
        self.log_bulk_modulus = nn.Parameter(torch.log(torch.full((4,), 8.0e8)))
        self.log_nominal_volume = nn.Parameter(torch.log(torch.full((4,), 4.0e-3)))
        self.log_flow_gain = nn.Parameter(torch.log(torch.full((4,), 2.0e-6)))
        self.log_leakage = nn.Parameter(torch.log(torch.full((4,), 1.0e-12)))
        self.log_supply_pressure = nn.Parameter(torch.log(torch.full((4,), 2.0e7)))
        self.log_effective_area = nn.Parameter(torch.log(torch.full((4,), 8.0e-3)))
        self.log_effective_mass = nn.Parameter(torch.log(torch.full((4,), 2.0e4)))
        self.log_viscous_friction = nn.Parameter(torch.log(torch.full((4,), 5.0e4)))
        self.extension_reference = nn.Parameter(torch.zeros(4))

    @staticmethod
    def _bounded(value: torch.Tensor, bound: float) -> torch.Tensor:
        return float(bound) * torch.tanh(value / float(bound))

    def _replace_current_command(self, history: torch.Tensor, command: torch.Tensor | None) -> torch.Tensor:
        if command is None:
            return history
        if command.shape != (history.shape[0], self.config.command_dim):
            raise ValueError(f"command must have shape {(history.shape[0], self.config.command_dim)}, got {tuple(command.shape)}")
        history = history.clone()
        history[:, -1, self.schema.command_slice] = command
        return history

    def forward(
        self,
        history: torch.Tensor,
        desired_cylinder_command: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> ROMOutput:
        if history.ndim != 3 or history.shape[-1] != self.config.input_dim:
            raise ValueError(f"history must have shape [B, L, {self.config.input_dim}], got {tuple(history.shape)}")
        history = self._replace_current_command(history, desired_cylinder_command)
        normalized = (history - self.input_mean) / self.input_std
        encoded, next_hidden = self.encoder(normalized, hidden)
        latent = encoded[:, -1]
        current = history[:, -1]
        extension = current[:, self.schema.extension_slice]
        velocity = current[:, self.schema.velocity_slice]
        pressure = current[:, self.schema.pressure_slice]
        joint_position = current[:, self.schema.joint_position_slice]
        joint_velocity = current[:, self.schema.joint_velocity_slice]
        command = current[:, self.schema.command_slice]

        actuation_raw = self.actuation_head(torch.cat((latent, command), dim=-1))
        actuation = self.deadzone(actuation_raw)
        bulk = self.log_bulk_modulus.exp().clamp(1.0e7, 3.0e9)
        area = self.log_effective_area.exp().clamp(1.0e-4, 5.0e-2)
        volume0 = self.log_nominal_volume.exp().clamp(1.0e-4, 5.0e-2)
        volume = (volume0 + area * (extension - self.extension_reference)).clamp(1.0e-4, 0.10)
        supply = self.log_supply_pressure.exp().clamp(1.0e6, 5.0e7)
        flow_gain = self.log_flow_gain.exp().clamp(1.0e-9, 1.0e-3)
        leakage = self.log_leakage.exp().clamp(1.0e-15, 1.0e-7)
        smooth_sign = torch.tanh(20.0 * actuation)
        pressure_margin = F.softplus((supply - smooth_sign * pressure) / 1.0e6) * 1.0e6
        flow = actuation * flow_gain * torch.sqrt(0.5 * pressure_margin.clamp_min(1.0))
        pressure_rate = (4.0 * bulk / volume) * (flow - area * velocity - leakage * pressure)
        pressure_residual = self.config.pressure_residual_fraction * self.config.max_delta_pressure_pa * torch.tanh(
            self.pressure_residual_head(latent)
        )
        delta_pressure_physics = self.config.sample_time_s * pressure_rate
        delta_pressure = self._bounded(delta_pressure_physics + pressure_residual, self.config.max_delta_pressure_pa)
        next_pressure = pressure + delta_pressure

        force_residual = self.config.max_force_residual_n * torch.tanh(
            self.force_residual_head(torch.cat((latent, next_pressure / 1.0e7, joint_position), dim=-1))
        )
        predicted_force = next_pressure * area + force_residual
        friction = self.log_viscous_friction.exp().clamp(1.0, 1.0e7) * velocity
        load = self.config.max_force_residual_n * torch.tanh(
            self.load_head(torch.cat((latent, predicted_force / 1.0e5, joint_position, joint_velocity), dim=-1))
        )
        mass = self.log_effective_mass.exp().clamp(10.0, 1.0e7)
        acceleration = (predicted_force - friction - load) / mass
        velocity_residual = self.config.velocity_residual_fraction * self.config.max_delta_velocity_mps * torch.tanh(
            self.velocity_residual_head(latent)
        )
        delta_velocity = self._bounded(
            self.config.sample_time_s * acceleration + velocity_residual,
            self.config.max_delta_velocity_mps,
        )
        next_velocity = velocity + delta_velocity

        extension_residual = self.config.position_residual_bound_m * torch.tanh(self.position_residual_head(latent))
        delta_extension = 0.5 * (velocity + next_velocity) * self.config.sample_time_s + extension_residual
        next_extension = extension + delta_extension
        return ROMOutput(
            delta_pressure=delta_pressure,
            delta_velocity=delta_velocity,
            delta_extension=delta_extension,
            predicted_force=predicted_force,
            effective_actuation=actuation,
            pressure_residual=pressure_residual,
            velocity_residual=velocity_residual,
            extension_residual=extension_residual,
            next_pressure=next_pressure,
            next_velocity=next_velocity,
            next_extension=next_extension,
            hidden=next_hidden,
        )
