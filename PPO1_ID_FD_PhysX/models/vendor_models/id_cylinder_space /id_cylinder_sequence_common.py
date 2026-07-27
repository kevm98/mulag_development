#!/usr/bin/env python3
"""Shared CNN-LSTM utilities for cylinder-space inverse dynamics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from id_cylinder_common import (
    COMMAND_LIMITS,
    DATA_INPUT_DIM,
    DEFAULT_MODEL_INPUT_DIM,
    DEFAULT_MODEL_INPUT_FEATURE_NAMES,
    INPUT_FEATURE_NAMES,
    OUTPUT_DIM,
    IDData,
    activation_module,
    group_indices_by_run,
    load_checkpoint,
    parameter_summary,
    previous_valvecmd_features,
)


class IDCylinderCNNLSTM(torch.nn.Module):
    """Causal 1D-CNN feature extractor + LSTM temporal model + MLP head.

    The convolution is causal: at time k it only sees features from <= k.
    This keeps validation consistent with online IsaacLab use.
    """

    def __init__(
        self,
        *,
        input_dim: int = DEFAULT_MODEL_INPUT_DIM,
        output_dim: int = OUTPUT_DIM,
        cnn_channels: int = 96,
        cnn_kernel: int = 5,
        lstm_hidden: int = 192,
        head_hidden: int = 384,
        activation: str = "silu",
        dropout: float = 0.05,
        layer_norm: bool = True,
    ):
        super().__init__()
        if input_dim not in (DATA_INPUT_DIM, DEFAULT_MODEL_INPUT_DIM) or output_dim != OUTPUT_DIM:
            raise ValueError(
                f"Sequence ID expects {DATA_INPUT_DIM} or {DEFAULT_MODEL_INPUT_DIM} inputs and "
                f"{OUTPUT_DIM} outputs, got {input_dim}->{output_dim}"
            )
        if cnn_kernel < 1:
            raise ValueError("cnn_kernel must be >= 1")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.cnn_channels = int(cnn_channels)
        self.cnn_kernel = int(cnn_kernel)
        self.lstm_hidden = int(lstm_hidden)

        self.input_proj = torch.nn.Linear(input_dim, cnn_channels)
        self.input_norm = torch.nn.LayerNorm(cnn_channels) if layer_norm else torch.nn.Identity()
        self.input_act = activation_module(activation)
        self.input_drop = torch.nn.Dropout(float(dropout))

        self.causal_conv = torch.nn.Conv1d(cnn_channels, cnn_channels, kernel_size=cnn_kernel, padding=0)
        self.conv_norm = torch.nn.LayerNorm(cnn_channels) if layer_norm else torch.nn.Identity()
        self.conv_act = activation_module(activation)
        self.conv_drop = torch.nn.Dropout(float(dropout))

        self.lstm_cell = torch.nn.LSTMCell(cnn_channels, lstm_hidden)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(lstm_hidden) if layer_norm else torch.nn.Identity(),
            torch.nn.Linear(lstm_hidden, head_hidden),
            activation_module(activation),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(head_hidden, output_dim),
        )

    def initial_state(self, batch_size: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cnn_kernel > 1:
            buffer = torch.zeros((batch_size, self.cnn_channels, self.cnn_kernel - 1), dtype=torch.float32, device=device)
        else:
            buffer = torch.zeros((batch_size, self.cnn_channels, 0), dtype=torch.float32, device=device)
        h = torch.zeros((batch_size, self.lstm_hidden), dtype=torch.float32, device=device)
        c = torch.zeros((batch_size, self.lstm_hidden), dtype=torch.float32, device=device)
        return buffer, h, c

    def forward_step(
        self,
        x_t: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if state is None:
            state = self.initial_state(x_t.shape[0], device=x_t.device)
        buffer, h, c = state
        feat = self.input_proj(x_t)
        feat = self.input_act(self.input_norm(feat))
        feat = self.input_drop(feat)

        conv_in = torch.cat([buffer, feat.unsqueeze(-1)], dim=-1)
        conv_feat = self.causal_conv(conv_in).squeeze(-1)
        conv_feat = self.conv_act(self.conv_norm(conv_feat))
        conv_feat = self.conv_drop(conv_feat)
        if self.cnn_kernel > 1:
            new_buffer = conv_in[:, :, 1:].detach() if not self.training else conv_in[:, :, 1:]
        else:
            new_buffer = buffer
        h, c = self.lstm_cell(conv_feat, (h, c))
        return self.head(h), (new_buffer, h, c)

    def forward_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        state = self.initial_state(x_seq.shape[0], device=x_seq.device)
        outputs = []
        for step in range(x_seq.shape[1]):
            y_t, state = self.forward_step(x_seq[:, step, :], state)
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            y, _ = self.forward_step(x)
            return y
        if x.ndim == 3:
            return self.forward_sequence(x)
        raise ValueError(f"Expected 2-D or 3-D input, got {tuple(x.shape)}")


def sequence_model_config_from_args(args: Any) -> dict[str, Any]:
    return {
        "model_type": "IDCylinderCNNLSTM",
        "input_dim": DEFAULT_MODEL_INPUT_DIM if bool(getattr(args, "use_prev_valvecmd", False)) else DATA_INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "cnn_channels": int(args.cnn_channels),
        "cnn_kernel": int(args.cnn_kernel),
        "lstm_hidden": int(args.lstm_hidden),
        "head_hidden": int(args.head_hidden),
        "activation": str(args.activation),
        "dropout": float(args.dropout),
        "layer_norm": bool(args.layer_norm),
    }


def build_sequence_model(config: dict[str, Any]) -> IDCylinderCNNLSTM:
    model_type = str(config.get("model_type", "IDCylinderCNNLSTM"))
    if model_type != "IDCylinderCNNLSTM":
        raise ValueError(f"Unsupported sequence model type {model_type!r}")
    return IDCylinderCNNLSTM(
        input_dim=int(config.get("input_dim", DEFAULT_MODEL_INPUT_DIM)),
        output_dim=int(config.get("output_dim", OUTPUT_DIM)),
        cnn_channels=int(config.get("cnn_channels", 96)),
        cnn_kernel=int(config.get("cnn_kernel", 5)),
        lstm_hidden=int(config.get("lstm_hidden", 192)),
        head_hidden=int(config.get("head_hidden", 384)),
        activation=str(config.get("activation", "silu")),
        dropout=float(config.get("dropout", 0.05)),
        layer_norm=bool(config.get("layer_norm", True)),
    )


def initialize_sequence_model_weights(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Conv1d):
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity="linear")
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


def save_sequence_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    model_config: dict[str, Any],
    data: IDData,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    input_feature_names: list[str],
    target_mode: str,
    epoch: int,
    history: list[dict[str, Any]],
    metrics: dict[str, Any],
    args: Any,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": "id_cylinder_cnn_lstm_v1",
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "model_config": model_config,
            "target_mode": target_mode,
            "input_layout": (
                "[Extension(4), ExtensionVelocity(4), dP(4), Force(4), "
                "TargetDeltaExtension(4), TargetDeltaExtensionVelocity(4), optional PreviousValveCmd(4)]"
            ),
            "output_layout": "[ValveCmd(4)]" if target_mode == "normalized_valve_cmd" else "[DeltaValveCmd(4)]",
            "input_feature_names": input_feature_names,
            "output_feature_names": data.output_feature_names,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "command_limits": COMMAND_LIMITS,
            "epoch": int(epoch),
            "history": history,
            "metrics": metrics,
            "training_args": vars(args) if hasattr(args, "__dict__") else {},
            "data_path": str(data.path),
            "paper_reference": "Liu et al. 2025, DOI 10.1007/s12206-025-0445-5",
            "parameter_summary": parameter_summary(model),
        },
        output_path,
    )


def load_sequence_model_from_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[IDCylinderCNNLSTM, dict[str, Any], dict[str, torch.Tensor]]:
    checkpoint = load_checkpoint(path, map_location="cpu")
    model = build_sequence_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    normalizers: dict[str, Any] = {
        "x_mean": torch.as_tensor(checkpoint["x_mean"], dtype=torch.float32, device=device).reshape(1, -1),
        "x_std": torch.as_tensor(checkpoint["x_std"], dtype=torch.float32, device=device).reshape(1, -1),
        "y_mean": torch.as_tensor(checkpoint["y_mean"], dtype=torch.float32, device=device).reshape(1, -1),
        "y_std": torch.as_tensor(checkpoint["y_std"], dtype=torch.float32, device=device).reshape(1, -1),
        "target_mode": checkpoint.get("target_mode", "normalized_valve_cmd"),
    }
    return model, checkpoint, normalizers


def build_sequence_training_arrays(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    run_id: np.ndarray,
    time_index: np.ndarray,
    *,
    use_prev_valvecmd: bool,
    target_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    prev = previous_valvecmd_features(y_raw, run_id, time_index)
    if use_prev_valvecmd:
        x_model = np.concatenate([x_raw, prev], axis=1).astype(np.float32, copy=False)
        feature_names = list(DEFAULT_MODEL_INPUT_FEATURE_NAMES)
    else:
        x_model = x_raw.astype(np.float32, copy=False)
        feature_names = list(INPUT_FEATURE_NAMES)
    if target_mode == "delta":
        if not use_prev_valvecmd:
            raise ValueError("Delta target mode requires use_prev_valvecmd=True")
        target = (y_raw - prev).astype(np.float32, copy=False)
    elif target_mode == "absolute":
        target = y_raw.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported target_mode {target_mode!r}")
    return (
        np.ascontiguousarray(x_model),
        np.ascontiguousarray(target),
        np.ascontiguousarray(prev),
        feature_names,
    )


@torch.no_grad()
def predict_sequence_valvecmd(
    model: IDCylinderCNNLSTM,
    data: IDData,
    normalizers: dict[str, Any],
    *,
    device: torch.device,
    recursive: bool,
) -> np.ndarray:
    if data.x_val is None or data.y_val is None or data.val_run_id is None or data.val_time_index is None:
        raise RuntimeError("Validation tensors are missing")
    groups = group_indices_by_run(data.val_run_id, data.val_time_index)
    y_pred = np.zeros_like(data.y_val, dtype=np.float32)
    limits = COMMAND_LIMITS.reshape(1, -1)
    true_prev = previous_valvecmd_features(data.y_val, data.val_run_id, data.val_time_index)
    input_dim = int(getattr(model, "input_dim", DATA_INPUT_DIM))
    use_prev_valvecmd = input_dim == DEFAULT_MODEL_INPUT_DIM
    model.eval()

    for _, indices in groups.items():
        state = None
        prev_cmd = np.zeros((1, OUTPUT_DIM), dtype=np.float32)
        for sample_index in indices:
            if recursive:
                prev = prev_cmd
            else:
                prev = true_prev[int(sample_index) : int(sample_index) + 1]
            if use_prev_valvecmd:
                x_raw = np.concatenate([data.x_val[int(sample_index) : int(sample_index) + 1], prev], axis=1)
            else:
                x_raw = data.x_val[int(sample_index) : int(sample_index) + 1]
            batch = torch.as_tensor(x_raw, dtype=torch.float32, device=device)
            x_norm = (batch - normalizers["x_mean"]) / normalizers["x_std"]
            y_norm, state = model.forward_step(x_norm, state)
            target = y_norm * normalizers["y_std"] + normalizers["y_mean"]
            if checkpoint_target_mode(normalizers) == "normalized_delta_valve_cmd":
                y_cmd = torch.as_tensor(prev, dtype=torch.float32, device=device) + target
            else:
                y_cmd = target
            y_np = y_cmd.detach().cpu().numpy().astype(np.float32)
            y_np = np.clip(y_np, -limits, limits)
            y_pred[int(sample_index)] = y_np[0]
            prev_cmd = y_np
    return y_pred


def checkpoint_target_mode(normalizers: dict[str, Any]) -> str:
    value = normalizers.get("target_mode")
    if isinstance(value, str):
        return value
    return "normalized_valve_cmd"
