from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from ..data.rom_schema import DEFAULT_SCHEMA, ROMSchema


@dataclass(frozen=True)
class HydraulicROMConfig:
    input_dim: int = 24
    command_dim: int = 8
    hidden_dim: int = 128
    recurrent_layers: int = 2
    dropout: float = 0.05
    sample_time_s: float = 0.05
    pressure_residual_fraction: float = 0.10
    velocity_residual_fraction: float = 0.10
    position_residual_bound_m: float = 1.0e-2
    command_extension_correction_bound_m: float = 3.0e-3
    max_delta_pressure_pa: float = 1.2e8
    max_delta_velocity_mps: float = 0.65
    max_force_residual_n: float = 3.0e4

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class ROMOutput(NamedTuple):
    predicted_valve: torch.Tensor
    delta_pressure: torch.Tensor
    delta_velocity: torch.Tensor
    delta_extension: torch.Tensor
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
        self.force_residual_head = nn.Sequential(nn.Linear(config.hidden_dim + 8, 64), nn.SiLU(), nn.Linear(64, 4))
        self.load_head = nn.Sequential(nn.Linear(config.hidden_dim + 12, 64), nn.SiLU(), nn.Linear(64, 4))
        self.velocity_residual_head = nn.Linear(config.hidden_dim, 4)
        self.position_residual_head = nn.Linear(config.hidden_dim, 4)
        self.command_extension_correction_head = nn.Sequential(
            nn.Linear(config.hidden_dim + config.command_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 4),
        )
        # Backward-compatible exact no-op for existing checkpoints. Fine-tuning
        # may train this head without altering the zero-command equilibrium.
        nn.init.zeros_(self.command_extension_correction_head[-1].weight)
        nn.init.zeros_(self.command_extension_correction_head[-1].bias)

        # Positive, trainable effective physical parameters. Values are conservative
        # initializations and must be replaced or bounded using the dataset audit.
        self.log_bulk_modulus = nn.Parameter(torch.log(torch.full((4,), 1.5e9)))
        self.log_nominal_volume = nn.Parameter(torch.log(torch.full((4,), 1.0e-3)))
        self.log_flow_gain = nn.Parameter(torch.log(torch.full((4,), 1.0e-8)))
        self.log_leakage = nn.Parameter(torch.log(torch.full((4,), 1.0e-12)))
        self.log_supply_pressure = nn.Parameter(torch.log(torch.full((4,), 3.4e7)))
        self.log_effective_area = nn.Parameter(torch.log(torch.full((4,), 6.4e-3)))
        self.log_effective_mass = nn.Parameter(torch.log(torch.full((4,), 1.0e7)))
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
        system_factors: torch.Tensor | None = None,
        rom_b_factors: torch.Tensor | None = None,
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
        factor_response_gain = None
        if system_factors is not None and rom_b_factors is not None:
            raise ValueError("system_factors and rom_b_factors are mutually exclusive")
        if system_factors is not None:
            if system_factors.shape != (history.shape[0], 7):
                raise ValueError(
                    "system_factors must have shape "
                    f"{(history.shape[0], 7)}, got {tuple(system_factors.shape)}"
                )
            if not torch.isfinite(system_factors).all():
                raise FloatingPointError("system_factors contains NaN/Inf")
            factors = system_factors.to(history.device, history.dtype)
            compliance = factors[:, 0:1].clamp(-0.25, 0.10)
            flow_factor = factors[:, 1:2].clamp(-0.25, 0.25)
            pressure_factor = factors[:, 2:3].clamp(-0.15, 0.15)
            seal_factor = factors[:, 3:7].clamp(0.0, 1.0)

            # e_comp is log(Kp/Kp_nom), where Kp=beta/Vd.  The other
            # factors follow the standalone AAM's physical target contract.
            bulk = bulk.unsqueeze(0) * torch.exp(compliance)
            flow_gain = flow_gain.unsqueeze(0) * (1.0 + flow_factor)
            supply = supply.unsqueeze(0) * (1.0 + pressure_factor)
            leakage = leakage.unsqueeze(0) * (1.0 + 0.25 * seal_factor).pow(3)
            # w_seal also increases the injected valve time constant.  A
            # bounded attenuation represents that slower effective response.
            actuation = actuation / (1.0 + 0.625 * seal_factor)
            # The trained ROM has a learned velocity/position residual in
            # parallel with its physical pressure path.  Scale that complete
            # actuator response too; otherwise factors visibly alter dP but
            # have almost no effect on the displacement sent to PhysX.
            factor_response_gain = (
                torch.exp(0.5 * compliance)
                * (1.0 + flow_factor)
                * torch.sqrt((1.0 + pressure_factor).clamp_min(0.1))
                / (1.0 + 0.625 * seal_factor)
            ).clamp(0.35, 1.50)
        elif rom_b_factors is not None:
            if rom_b_factors.shape != (history.shape[0], 5):
                raise ValueError(
                    "rom_b_factors must have shape "
                    f"{(history.shape[0], 5)}, got {tuple(rom_b_factors.shape)}"
                )
            if not torch.isfinite(rom_b_factors).all():
                raise FloatingPointError("rom_b_factors contains NaN/Inf")
            factors = rom_b_factors.to(history.device, history.dtype)
            compliance = factors[:, 0:1].clamp(-0.125, 0.020)
            seal_factor = factors[:, 1:5].clamp(0.0, 1.0)

            # ROM-B varies only audited explicit hydraulic channels.  Motion is
            # recomputed through pressure and force; final outputs are not scaled.
            bulk = bulk.unsqueeze(0) * torch.exp(compliance)
            leakage = leakage.unsqueeze(0) * (1.0 + 0.25 * seal_factor).pow(3)
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
            self.force_residual_head(torch.cat((latent, next_pressure / 1.0e7, extension), dim=-1))
        )
        predicted_force = next_pressure * area + force_residual
        friction = self.log_viscous_friction.exp().clamp(1.0, 1.0e7) * velocity
        load = self.config.max_force_residual_n * torch.tanh(
            self.load_head(torch.cat((latent, predicted_force / 1.0e5, extension, velocity), dim=-1))
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
        if factor_response_gain is not None:
            delta_velocity = delta_velocity * factor_response_gain
        next_velocity = velocity + delta_velocity

        extension_residual = self.config.position_residual_bound_m * torch.tanh(self.position_residual_head(latent))
        command_gate = torch.tanh(200.0 * torch.linalg.norm(command, dim=-1, keepdim=True))
        command_extension_correction = (
            self.config.command_extension_correction_bound_m
            * command_gate
            * torch.tanh(
                self.command_extension_correction_head(torch.cat((latent, command), dim=-1))
            )
        )
        delta_extension = (
            0.5 * (velocity + next_velocity) * self.config.sample_time_s
            + extension_residual
            + command_extension_correction
        )
        if factor_response_gain is not None:
            delta_extension = delta_extension * factor_response_gain
        next_extension = extension + delta_extension
        return ROMOutput(
            predicted_valve=actuation,
            delta_pressure=delta_pressure,
            delta_velocity=delta_velocity,
            delta_extension=delta_extension,
            effective_actuation=actuation,
            pressure_residual=pressure_residual,
            velocity_residual=velocity_residual,
            extension_residual=extension_residual,
            next_pressure=next_pressure,
            next_velocity=next_velocity,
            next_extension=next_extension,
            hidden=next_hidden,
        )
