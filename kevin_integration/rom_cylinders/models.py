from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

try:
    from rom_data_contract import (
        DP_SLICE,
        EXTENSION_SLICE,
        TARGET_DIM,
    )
except ImportError:
    from kevin_integration.rom_cylinders.rom_data_contract import (
        DP_SLICE,
        EXTENSION_SLICE,
        TARGET_DIM,
    )


@dataclass(frozen=True)
class ModelConfig:
    model_type: str
    input_dim: int
    output_dim: int = TARGET_DIM
    target_mode: str = "extension_and_dp"
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    flatten_history: bool = True
    bounded_outputs: bool = False
    output_scale: tuple[float, ...] | None = None


class TargetScaledBaseline(nn.Module):
    def __init__(self, target_mean: torch.Tensor, target_std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("target_mean", target_mean.detach().clone().float().reshape(1, -1))
        self.register_buffer("target_std", torch.clamp(target_std.detach().clone().float().reshape(1, -1), min=1.0e-12))

    def normalize_target(self, raw_delta: torch.Tensor) -> torch.Tensor:
        return (raw_delta - self.target_mean.to(raw_delta.device)) / self.target_std.to(raw_delta.device)


class ZeroChangeBaseline(TargetScaledBaseline):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_delta = torch.zeros((x.shape[0], self.target_mean.shape[-1]), dtype=x.dtype, device=x.device)
        return self.normalize_target(raw_delta)


class PersistenceBaseline(TargetScaledBaseline):
    def __init__(
        self,
        *,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        output_dim: int,
    ) -> None:
        super().__init__(target_mean=target_mean, target_std=target_std)
        self.output_dim = int(output_dim)
        self.register_buffer("input_mean", input_mean.detach().clone().float().reshape(1, 1, -1))
        self.register_buffer("input_std", torch.clamp(input_std.detach().clone().float().reshape(1, 1, -1), min=1.0e-12))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_x = x * self.input_std.to(x.device) + self.input_mean.to(x.device)
        if raw_x.shape[1] < 2:
            raw_delta = torch.zeros((x.shape[0], self.output_dim), dtype=x.dtype, device=x.device)
            return self.normalize_target(raw_delta)
        extension_delta = raw_x[:, -1, EXTENSION_SLICE] - raw_x[:, -2, EXTENSION_SLICE]
        if self.output_dim == 4:
            return self.normalize_target(extension_delta)
        dp_delta = raw_x[:, -1, DP_SLICE] - raw_x[:, -2, DP_SLICE]
        raw_delta = torch.cat([extension_delta, dp_delta], dim=-1)
        return self.normalize_target(raw_delta)


class RidgeRegressionBaseline(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int = TARGET_DIM,
        history_length: int = 10,
        flatten_history: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.history_length = int(history_length)
        self.flatten_history = bool(flatten_history)
        feature_dim = self.input_dim * self.history_length if self.flatten_history else self.input_dim
        self.linear = nn.Linear(feature_dim, self.output_dim)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        if self.flatten_history:
            return x.reshape(x.shape[0], -1)
        return x[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self._features(x))

    @torch.no_grad()
    def fit_closed_form(self, x: torch.Tensor, y: torch.Tensor, *, ridge_alpha: float = 1.0) -> None:
        features = self._features(x).float()
        target = y.float()
        ones = torch.ones((features.shape[0], 1), dtype=features.dtype, device=features.device)
        design = torch.cat([features, ones], dim=-1)
        reg = float(ridge_alpha) * torch.eye(design.shape[1], dtype=features.dtype, device=features.device)
        reg[-1, -1] = 0.0
        lhs = design.T @ design + reg
        rhs = design.T @ target
        weights = torch.linalg.solve(lhs, rhs)
        self.linear.weight.copy_(weights[:-1].T)
        self.linear.bias.copy_(weights[-1])


class BoundedOutputAdapter(nn.Module):
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


class MLPBaseline(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int = TARGET_DIM,
        history_length: int = 10,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        flatten_history: bool = True,
        bounded_output_scale: torch.Tensor | None = None,
        target_mean: torch.Tensor | None = None,
        target_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.flatten_history = bool(flatten_history)
        feature_dim = input_dim * history_length if self.flatten_history else input_dim
        layers: list[nn.Module] = []
        current_dim = feature_dim
        for _ in range(max(1, int(num_layers))):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.output_adapter = BoundedOutputAdapter(
            output_scale=bounded_output_scale,
            target_mean=target_mean,
            target_std=target_std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = x.reshape(x.shape[0], -1) if self.flatten_history else x[:, -1, :]
        return self.output_adapter(self.net(features))


class RecurrentSequenceModel(nn.Module):
    def __init__(
        self,
        *,
        cell_type: str,
        input_dim: int,
        output_dim: int = TARGET_DIM,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bounded_output_scale: torch.Tensor | None = None,
        target_mean: torch.Tensor | None = None,
        target_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        if cell_type == "gru":
            self.rnn: nn.Module = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )
        elif cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )
        else:
            raise ValueError(f"Unsupported recurrent cell type {cell_type!r}.")
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))
        self.output_adapter = BoundedOutputAdapter(
            output_scale=bounded_output_scale,
            target_mean=target_mean,
            target_std=target_std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.output_adapter(self.head(output[:, -1, :]))


def _bounded_args(config: ModelConfig, normalization_stats: Any | None) -> dict[str, torch.Tensor | None]:
    if not config.bounded_outputs:
        return {"bounded_output_scale": None, "target_mean": None, "target_std": None}
    if config.output_scale is None:
        raise ValueError("bounded_outputs=True requires ModelConfig.output_scale.")
    if normalization_stats is None:
        raise ValueError("bounded_outputs=True requires normalization_stats.")
    return {
        "bounded_output_scale": torch.as_tensor(config.output_scale, dtype=torch.float32),
        "target_mean": normalization_stats.target_mean,
        "target_std": normalization_stats.target_std,
    }


def build_model(config: ModelConfig, *, normalization_stats: Any | None = None) -> nn.Module:
    model_type = config.model_type
    if model_type == "zero":
        if normalization_stats is None:
            raise ValueError("zero baseline requires normalization_stats.")
        return ZeroChangeBaseline(
            target_mean=normalization_stats.target_mean,
            target_std=normalization_stats.target_std,
        )
    if model_type == "persistence":
        if normalization_stats is None:
            raise ValueError("persistence baseline requires normalization_stats.")
        return PersistenceBaseline(
            input_mean=normalization_stats.input_mean,
            input_std=normalization_stats.input_std,
            target_mean=normalization_stats.target_mean,
            target_std=normalization_stats.target_std,
            output_dim=config.output_dim,
        )
    if model_type == "ridge":
        return RidgeRegressionBaseline(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            history_length=10,
            flatten_history=config.flatten_history,
        )
    if model_type == "mlp":
        return MLPBaseline(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            history_length=10,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            flatten_history=config.flatten_history,
            **_bounded_args(config, normalization_stats),
        )
    if model_type == "gru":
        return RecurrentSequenceModel(
            cell_type="gru",
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            **_bounded_args(config, normalization_stats),
        )
    if model_type == "lstm":
        return RecurrentSequenceModel(
            cell_type="lstm",
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            **_bounded_args(config, normalization_stats),
        )
    if model_type == "physics_informed":
        try:
            from physics_informed_rom import PhysicsInformedCylinderROM
        except ImportError:
            from kevin_integration.rom_cylinders.physics_informed_rom import PhysicsInformedCylinderROM

        return PhysicsInformedCylinderROM(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            **_bounded_args(config, normalization_stats),
        )
    raise ValueError(f"Unsupported model type {model_type!r}.")
