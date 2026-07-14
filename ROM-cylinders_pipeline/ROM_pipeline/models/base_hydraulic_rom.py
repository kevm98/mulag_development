from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..data.base_rom_schema import BASE_ROM_SCHEMA, BaseROMSchema
from .hydraulic_rom import ROMOutput, SmoothDeadzone


@dataclass(frozen=True)
class BaseHydraulicROMConfig:
    input_dim: int = 30
    command_dim: int = 4
    hidden_dim: int = 128
    recurrent_layers: int = 2
    dropout: float = 0.05
    sample_time_s: float = 0.05
    actuation_correction_bound: float = 0.25
    pressure_residual_fraction: float = 0.10
    velocity_residual_fraction: float = 0.10
    position_residual_bound_m: float = 1.0e-2
    max_delta_pressure_pa: float = 1.2e8
    max_delta_velocity_mps: float = 0.65
    max_force_residual_n: float = 3.0e4

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class BaseHydraulicROM(nn.Module):
    """Gray-box valve-conditioned hydraulic plant ROM.

    The measured valve is an explicit input. A bounded recurrent correction can
    represent spool delay and hysteresis, but it cannot replace the valve path.
    """

    def __init__(
        self,
        config: BaseHydraulicROMConfig = BaseHydraulicROMConfig(),
        *,
        schema: BaseROMSchema = BASE_ROM_SCHEMA,
        input_mean: torch.Tensor | None = None,
        input_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        schema.validate()
        if config.input_dim != schema.input_dim or config.command_dim != schema.command_dim:
            raise ValueError("Base ROM configuration does not match its schema.")
        self.config = config
        self.schema = schema
        self.register_buffer("input_mean", torch.zeros(config.input_dim) if input_mean is None else input_mean.float())
        self.register_buffer("input_std", torch.ones(config.input_dim) if input_std is None else input_std.float().clamp_min(1.0e-8))
        self.encoder = nn.GRU(
            config.input_dim,
            config.hidden_dim,
            num_layers=config.recurrent_layers,
            dropout=config.dropout if config.recurrent_layers > 1 else 0.0,
            batch_first=True,
        )
        self.actuation_memory_head = nn.Linear(config.hidden_dim, 4)
        self.deadzone = SmoothDeadzone(4)
        self.pressure_residual_head = nn.Linear(config.hidden_dim, 4)
        self.force_residual_head = nn.Sequential(nn.Linear(config.hidden_dim + 9, 64), nn.SiLU(), nn.Linear(64, 4))
        self.load_head = nn.Sequential(nn.Linear(config.hidden_dim + 14, 64), nn.SiLU(), nn.Linear(64, 4))
        self.velocity_residual_head = nn.Linear(config.hidden_dim, 4)
        self.position_residual_head = nn.Linear(config.hidden_dim, 4)

        self.log_bulk_modulus = nn.Parameter(torch.log(torch.full((4,), 1.5e9)))
        self.log_nominal_volume = nn.Parameter(torch.log(torch.full((4,), 1.0e-3)))
        # At the 50 ms reduced-order timestep, fast spool/line dynamics are
        # already collapsed. This is an effective discrete flow gain, not the
        # instantaneous orifice coefficient used by the 5 ms full model.
        self.log_flow_gain = nn.Parameter(torch.log(torch.full((4,), 1.0e-8)))
        self.log_leakage = nn.Parameter(torch.log(torch.full((4,), 1.0e-12)))
        self.log_supply_pressure = nn.Parameter(torch.log(torch.full((4,), 3.4e7)))
        self.log_effective_area = nn.Parameter(torch.log(torch.full((4,), 6.4e-3)))
        self.log_effective_mass = nn.Parameter(torch.log(torch.full((4,), 1.0e7)))
        self.log_viscous_friction = nn.Parameter(torch.log(torch.full((4,), 5.0e4)))
        self.extension_reference = nn.Parameter(torch.zeros(4))
        self._initialize_residual_heads()

    def _initialize_residual_heads(self) -> None:
        """Start from the physical backbone; let residuals grow only when data requires them."""
        for layer in (
            self.actuation_memory_head,
            self.pressure_residual_head,
            self.force_residual_head[-1],
            self.load_head[-1],
            self.velocity_residual_head,
            self.position_residual_head,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def _bounded(value: torch.Tensor, bound: float) -> torch.Tensor:
        return float(bound) * torch.tanh(value / float(bound))

    def _replace_current_valve(self, history: torch.Tensor, valve: torch.Tensor | None) -> torch.Tensor:
        if valve is None:
            return history
        if valve.shape != (history.shape[0], 4):
            raise ValueError(f"valve_command must have shape {(history.shape[0], 4)}, got {tuple(valve.shape)}")
        history = history.clone()
        history[:, -1, self.schema.valve_slice] = valve
        return history

    def forward(
        self,
        history: torch.Tensor,
        valve_command: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> ROMOutput:
        if history.ndim != 3 or history.shape[-1] != self.config.input_dim:
            raise ValueError(f"history must have shape [B, L, {self.config.input_dim}], got {tuple(history.shape)}")
        history = self._replace_current_valve(history, valve_command)
        encoded, next_hidden = self.encoder((history - self.input_mean) / self.input_std, hidden)
        latent = encoded[:, -1]
        current = history[:, -1]
        valve = current[:, self.schema.valve_slice]
        extension = current[:, self.schema.extension_slice]
        velocity = current[:, self.schema.velocity_slice]
        pressure = current[:, self.schema.pressure_slice]
        joint_position = current[:, self.schema.joint_position_slice]
        joint_velocity = current[:, self.schema.joint_velocity_slice]

        memory_correction = self.config.actuation_correction_bound * torch.tanh(self.actuation_memory_head(latent))
        actuation = self.deadzone((valve + memory_correction).clamp(-1.5, 1.5))
        bulk = self.log_bulk_modulus.exp().clamp(1.0e7, 3.0e9)
        area = self.log_effective_area.exp().clamp(1.0e-4, 2.0e-2)
        volume0 = self.log_nominal_volume.exp().clamp(1.0e-4, 5.0e-2)
        volume = (volume0 + area * (extension - self.extension_reference)).clamp(1.0e-4, 0.10)
        supply = self.log_supply_pressure.exp().clamp(1.0e6, 1.5e8)
        flow_gain = self.log_flow_gain.exp().clamp(1.0e-11, 1.0e-5)
        leakage = self.log_leakage.exp().clamp(1.0e-15, 1.0e-7)
        margin = F.softplus((supply - torch.tanh(20.0 * actuation) * pressure) / 1.0e6) * 1.0e6
        flow = actuation * flow_gain * torch.sqrt(0.5 * margin.clamp_min(1.0))
        pressure_rate = (4.0 * bulk / volume) * (flow - area * velocity - leakage * pressure)
        pressure_residual = self.config.pressure_residual_fraction * self.config.max_delta_pressure_pa * torch.tanh(
            self.pressure_residual_head(latent)
        )
        delta_pressure = self._bounded(
            self.config.sample_time_s * pressure_rate + pressure_residual, self.config.max_delta_pressure_pa
        )
        next_pressure = pressure + delta_pressure

        force_residual = self.config.max_force_residual_n * torch.tanh(
            self.force_residual_head(torch.cat((latent, next_pressure / 1.0e7, joint_position), -1))
        )
        predicted_force = next_pressure * area + force_residual
        friction = self.log_viscous_friction.exp().clamp(1.0, 1.0e7) * velocity
        load = self.config.max_force_residual_n * torch.tanh(
            self.load_head(torch.cat((latent, predicted_force / 1.0e5, joint_position, joint_velocity), -1))
        )
        mass = self.log_effective_mass.exp().clamp(10.0, 1.0e9)
        acceleration = (predicted_force - friction - load) / mass
        velocity_residual = self.config.velocity_residual_fraction * self.config.max_delta_velocity_mps * torch.tanh(
            self.velocity_residual_head(latent)
        )
        delta_velocity = self._bounded(
            self.config.sample_time_s * acceleration + velocity_residual, self.config.max_delta_velocity_mps
        )
        next_velocity = velocity + delta_velocity
        extension_residual = self.config.position_residual_bound_m * torch.tanh(self.position_residual_head(latent))
        delta_extension = 0.5 * (velocity + next_velocity) * self.config.sample_time_s + extension_residual
        return ROMOutput(
            delta_pressure, delta_velocity, delta_extension, predicted_force, actuation,
            pressure_residual, velocity_residual, extension_residual,
            next_pressure, next_velocity, extension + delta_extension, next_hidden,
        )
