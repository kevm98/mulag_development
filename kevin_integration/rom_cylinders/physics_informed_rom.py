from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

try:
    from rom_data_contract import (
        DP_SLICE,
        EXTENSION_SLICE,
        EXTENSION_VELOCITY_SLICE,
        JOINT_POS_SLICE,
        JOINT_VEL_SLICE,
        TARGET_DIM,
        VALVE_SLICE,
    )
except ImportError:
    from kevin_integration.rom_cylinders.rom_data_contract import (
        DP_SLICE,
        EXTENSION_SLICE,
        EXTENSION_VELOCITY_SLICE,
        JOINT_POS_SLICE,
        JOINT_VEL_SLICE,
        TARGET_DIM,
        VALVE_SLICE,
    )


class TrainableAsymmetricDeadZone(nn.Module):
    def __init__(self, num_cylinders: int = 4) -> None:
        super().__init__()
        self.raw_pos_threshold = nn.Parameter(torch.full((num_cylinders,), -4.0))
        self.raw_neg_threshold = nn.Parameter(torch.full((num_cylinders,), -4.0))
        self.raw_pos_gain = nn.Parameter(torch.zeros(num_cylinders))
        self.raw_neg_gain = nn.Parameter(torch.zeros(num_cylinders))

    def forward(self, valve: torch.Tensor) -> torch.Tensor:
        pos_threshold = F.softplus(self.raw_pos_threshold).reshape(1, -1)
        neg_threshold = F.softplus(self.raw_neg_threshold).reshape(1, -1)
        pos_gain = F.softplus(self.raw_pos_gain).reshape(1, -1)
        neg_gain = F.softplus(self.raw_neg_gain).reshape(1, -1)
        positive = pos_gain * torch.relu(valve - pos_threshold)
        negative = -neg_gain * torch.relu(-valve - neg_threshold)
        return torch.where(valve >= 0.0, positive, negative)


class _BoundedOutputAdapter(nn.Module):
    def __init__(
        self,
        *,
        output_scale: torch.Tensor | None,
        target_mean: torch.Tensor | None,
        target_std: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.enabled = output_scale is not None
        if self.enabled:
            assert output_scale is not None
            assert target_mean is not None
            assert target_std is not None
            self.register_buffer("output_scale", torch.clamp(output_scale.detach().clone().float().reshape(1, -1), min=1.0e-12))
            self.register_buffer("target_mean", target_mean.detach().clone().float().reshape(1, -1))
            self.register_buffer("target_std", torch.clamp(target_std.detach().clone().float().reshape(1, -1), min=1.0e-12))
        else:
            self.register_buffer("output_scale", torch.empty(0))
            self.register_buffer("target_mean", torch.empty(0))
            self.register_buffer("target_std", torch.empty(0))

    def forward(self, raw_head: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return raw_head
        raw_physical = self.output_scale.to(raw_head.device) * torch.tanh(raw_head)
        return (raw_physical - self.target_mean.to(raw_head.device)) / self.target_std.to(raw_head.device)


class PhysicsInformedCylinderROM(nn.Module):
    """Semi-parametric four-cylinder ROM in normalized dataset coordinates.

    The structured branches are intentionally modest: signed valve preprocessing,
    per-cylinder pressure transition, per-cylinder extension transition, and a
    small recurrent residual. Physical constants are not claimed to be known.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int = TARGET_DIM,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_cylinders: int = 4,
        bounded_output_scale: torch.Tensor | None = None,
        target_mean: torch.Tensor | None = None,
        target_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if output_dim not in (4, TARGET_DIM):
            raise ValueError(f"PhysicsInformedCylinderROM requires output_dim=4 or {TARGET_DIM}, got {output_dim}.")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_cylinders = int(num_cylinders)

        self.deadzone = TrainableAsymmetricDeadZone(num_cylinders=num_cylinders)
        self.raw_pressure_valve_gain = nn.Parameter(torch.zeros(num_cylinders))
        self.raw_pressure_leak = nn.Parameter(torch.full((num_cylinders,), -2.0))
        self.raw_pressure_velocity_gain = nn.Parameter(torch.zeros(num_cylinders))
        self.raw_extension_valve_gain = nn.Parameter(torch.zeros(num_cylinders))
        self.raw_extension_pressure_gain = nn.Parameter(torch.zeros(num_cylinders))
        self.raw_extension_damping = nn.Parameter(torch.full((num_cylinders,), -2.0))
        self.extension_bias = nn.Parameter(torch.zeros(num_cylinders))
        self.pressure_bias = nn.Parameter(torch.zeros(num_cylinders))

        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.residual_rnn = nn.GRU(
            self.input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        residual_input_dim = hidden_dim + 4 + 4 + 4 + 4 + 5 + 5
        self.residual_head = nn.Sequential(
            nn.LayerNorm(residual_input_dim),
            nn.Linear(residual_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, TARGET_DIM),
        )
        if output_dim == 4:
            self.residual_head[-1] = nn.Linear(hidden_dim, 4)
        self.output_adapter = _BoundedOutputAdapter(
            output_scale=bounded_output_scale,
            target_mean=target_mean,
            target_std=target_std,
        )

    def _positive(self, parameter: torch.Tensor) -> torch.Tensor:
        return F.softplus(parameter).reshape(1, -1)

    def structured_forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        current = x[:, -1, :]
        valve = current[:, VALVE_SLICE]
        dP = current[:, DP_SLICE]
        extension = current[:, EXTENSION_SLICE]
        extension_velocity = current[:, EXTENSION_VELOCITY_SLICE]
        joint_pos = current[:, JOINT_POS_SLICE]
        joint_vel = current[:, JOINT_VEL_SLICE]
        del extension, joint_pos, joint_vel

        valve_effective = self.deadzone(valve)
        pressure_structured = (
            self._positive(self.raw_pressure_valve_gain) * valve_effective
            - self._positive(self.raw_pressure_leak) * dP
            - self._positive(self.raw_pressure_velocity_gain) * extension_velocity
            + self.pressure_bias.reshape(1, -1)
        )
        extension_structured = (
            self._positive(self.raw_extension_valve_gain) * valve_effective
            + self._positive(self.raw_extension_pressure_gain) * dP
            - self._positive(self.raw_extension_damping) * extension_velocity
            + self.extension_bias.reshape(1, -1)
        )
        return {
            "valve_effective": valve_effective,
            "pressure_structured": pressure_structured,
            "extension_structured": extension_structured,
        }

    def forward(self, x: torch.Tensor, *, return_debug: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        structured = self.structured_forward(x)
        rnn_output, _ = self.residual_rnn(x)
        current = x[:, -1, :]
        residual_features = torch.cat(
            [
                rnn_output[:, -1, :],
                current[:, VALVE_SLICE],
                current[:, DP_SLICE],
                current[:, EXTENSION_SLICE],
                current[:, EXTENSION_VELOCITY_SLICE],
                current[:, JOINT_POS_SLICE],
                current[:, JOINT_VEL_SLICE],
            ],
            dim=-1,
        )
        residual = self.residual_head(residual_features)
        extension_residual = residual[:, :4]
        delta_extension = structured["extension_structured"] + extension_residual
        if self.output_dim == 4:
            output = self.output_adapter(delta_extension)
            if not return_debug:
                return output
            debug = {
                "valve_effective": structured["valve_effective"],
                "extension_structured": structured["extension_structured"],
                "extension_residual": extension_residual,
                "final_DeltaExtension": output,
            }
            return output, debug
        pressure_residual = residual[:, 4:8]
        delta_dp = structured["pressure_structured"] + pressure_residual
        output = self.output_adapter(torch.cat([delta_extension, delta_dp], dim=-1))
        if not return_debug:
            return output
        debug = {
            "valve_effective": structured["valve_effective"],
            "pressure_structured": structured["pressure_structured"],
            "pressure_residual": pressure_residual,
            "extension_structured": structured["extension_structured"],
            "extension_residual": extension_residual,
            "final_DeltaDP": delta_dp,
            "final_DeltaExtension": delta_extension,
        }
        return output, debug
