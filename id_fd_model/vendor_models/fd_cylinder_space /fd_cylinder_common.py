#!/usr/bin/env python3
"""Shared utilities for the cylinder-space forward-dynamics model.

This pipeline is a discrete-time nonlinear system-identification setup:

    x_k = [ValveCmd, Extension, ExtensionVelocity, dP, Force]
    y_k = [DeltaExtension, DeltaExtensionVelocity, DeltaDP]

Relevant background:
  - Schoukens and Ljung, "Nonlinear System Identification: A User-Oriented
    Roadmap", 2019.
  - Dai et al., "Deep Learning of Dynamic Systems using System Identification
    Toolbox(TM)", SYSID 2024.
  - Nisar and Munawar, "System Identification and Controller Design for
    Hydraulic Actuator", 2021. This is directly relevant because their
    hydraulic actuator model is validated against a nonlinear Simscape model.
  - Asadi et al., "Combating the Compounding-Error Problem with a Multi-step
    Model", 2019. This motivates rollout validation in addition to one-step MSE.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR / "FD_cylinder_data.mat"

INPUT_DIM = 20
OUTPUT_DIM = 12

INPUT_FEATURE_NAMES = [
    "ValveCmd_1",
    "ValveCmd_2",
    "ValveCmd_3",
    "ValveCmd_4",
    "Extension_1",
    "Extension_2",
    "Extension_3",
    "Extension_4",
    "ExtensionVelocity_1",
    "ExtensionVelocity_2",
    "ExtensionVelocity_3",
    "ExtensionVelocity_4",
    "dP_1",
    "dP_2",
    "dP_3",
    "dP_4",
    "Force_1",
    "Force_2",
    "Force_3",
    "Force_4",
]

OUTPUT_FEATURE_NAMES = [
    "DeltaExtension_next_1",
    "DeltaExtension_next_2",
    "DeltaExtension_next_3",
    "DeltaExtension_next_4",
    "DeltaExtensionVelocity_next_1",
    "DeltaExtensionVelocity_next_2",
    "DeltaExtensionVelocity_next_3",
    "DeltaExtensionVelocity_next_4",
    "DeltaDP_next_1",
    "DeltaDP_next_2",
    "DeltaDP_next_3",
    "DeltaDP_next_4",
]

TARGET_GROUPS = {
    "DeltaExtension": slice(0, 4),
    "DeltaExtensionVelocity": slice(4, 8),
    "DeltaDP": slice(8, 12),
}

STATE_GROUPS = {
    "Extension": slice(0, 4),
    "ExtensionVelocity": slice(4, 8),
    "dP": slice(8, 12),
}

LITERATURE_REFERENCES = [
    {
        "key": "schoukens_ljung_2019",
        "title": "Nonlinear System Identification: A User-Oriented Roadmap",
        "url": "https://arxiv.org/abs/1902.00683",
        "reason": "General nonlinear system-identification workflow and validation philosophy.",
    },
    {
        "key": "dai_etal_sysid_2024",
        "title": "Deep Learning of Dynamic Systems using System Identification Toolbox(TM)",
        "url": "https://arxiv.org/abs/2409.07642",
        "reason": "Recent MATLAB/System Identification Toolbox neural dynamic-model context.",
    },
    {
        "key": "nisar_munawar_2021",
        "title": "System Identification and Controller Design for Hydraulic Actuator",
        "url": "https://arxiv.org/abs/2108.11756",
        "reason": "Hydraulic actuator identification validated against a nonlinear Simscape model.",
    },
    {
        "key": "asadi_etal_2019",
        "title": "Combating the Compounding-Error Problem with a Multi-step Model",
        "url": "https://arxiv.org/abs/1905.13320",
        "reason": "Why one-step models must also be checked with autoregressive rollout metrics.",
    },
]


@dataclass
class FDData:
    path: Path
    input_feature_names: list[str]
    output_feature_names: list[str]
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    x_train: np.ndarray | None = None
    y_train: np.ndarray | None = None
    x_val: np.ndarray | None = None
    y_val: np.ndarray | None = None
    train_class: np.ndarray | None = None
    val_class: np.ndarray | None = None
    train_run_id: np.ndarray | None = None
    val_run_id: np.ndarray | None = None
    train_time_index: np.ndarray | None = None
    val_time_index: np.ndarray | None = None
    train_sample_time: np.ndarray | None = None
    val_sample_time: np.ndarray | None = None


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def read_matlab_cellstr(handle: h5py.File, name: str, fallback: list[str]) -> list[str]:
    if name not in handle:
        return list(fallback)
    dataset = handle[name]
    if dataset.dtype != object:
        return list(fallback)

    values: list[str] = []
    for ref in dataset[0]:
        raw = np.asarray(handle[ref][()])
        chars = [chr(int(v)) for v in raw.ravel(order="F") if int(v) != 0]
        values.append("".join(chars))
    return values or list(fallback)


def load_feature_matrix(handle: h5py.File, name: str, width: int) -> np.ndarray:
    if name not in handle:
        raise KeyError(f"Missing dataset {name!r} in {handle.filename}")
    raw = np.asarray(handle[name][()], dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {raw.shape}")
    if raw.shape[0] == width:
        raw = raw.T
    elif raw.shape[1] != width:
        raise ValueError(f"{name} must have feature width {width}; got shape {raw.shape}")
    return np.ascontiguousarray(raw, dtype=np.float32)


def load_vector(handle: h5py.File, name: str, dtype: np.dtype[Any] | type) -> np.ndarray:
    if name not in handle:
        raise KeyError(f"Missing dataset {name!r} in {handle.filename}")
    return np.asarray(handle[name][()]).reshape(-1).astype(dtype, copy=False)


def load_normalizer(handle: h5py.File, name: str, width: int) -> np.ndarray:
    vector = load_vector(handle, name, np.float32)
    if vector.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains NaN or Inf values")
    return np.ascontiguousarray(vector, dtype=np.float32)


def load_fd_data(path: str | Path, *, load_train: bool = True, load_val: bool = True) -> FDData:
    data_path = Path(path).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    with h5py.File(data_path, "r") as handle:
        input_names = read_matlab_cellstr(handle, "input_feature_names", INPUT_FEATURE_NAMES)
        output_names = read_matlab_cellstr(handle, "output_feature_names", OUTPUT_FEATURE_NAMES)
        x_mean = load_normalizer(handle, "x_mean_fd", INPUT_DIM)
        x_std = load_normalizer(handle, "x_std_fd", INPUT_DIM)
        y_mean = load_normalizer(handle, "y_mean_fd", OUTPUT_DIM)
        y_std = load_normalizer(handle, "y_std_fd", OUTPUT_DIM)

        data = FDData(
            path=data_path,
            input_feature_names=input_names,
            output_feature_names=output_names,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
        )

        if load_train:
            data.x_train = load_feature_matrix(handle, "X_train_fd", INPUT_DIM)
            data.y_train = load_feature_matrix(handle, "Y_train_fd", OUTPUT_DIM)
            data.train_class = load_vector(handle, "TrainClass", np.int64)
            data.train_run_id = load_vector(handle, "TrainRunID", np.int64)
            data.train_time_index = load_vector(handle, "TrainTimeIndex", np.int64)
            data.train_sample_time = load_vector(handle, "TrainSampleTime", np.float32)

        if load_val:
            data.x_val = load_feature_matrix(handle, "X_val_fd", INPUT_DIM)
            data.y_val = load_feature_matrix(handle, "Y_val_fd", OUTPUT_DIM)
            data.val_class = load_vector(handle, "ValClass", np.int64)
            data.val_run_id = load_vector(handle, "ValRunID", np.int64)
            data.val_time_index = load_vector(handle, "ValTimeIndex", np.int64)
            data.val_sample_time = load_vector(handle, "ValSampleTime", np.float32)

    validate_fd_data(data)
    return data


def validate_fd_data(data: FDData) -> None:
    if data.input_feature_names != INPUT_FEATURE_NAMES:
        raise ValueError(
            "Input feature names differ from the expected FD cylinder contract: "
            f"{data.input_feature_names}"
        )
    if data.output_feature_names != OUTPUT_FEATURE_NAMES:
        raise ValueError(
            "Output feature names differ from the expected FD cylinder contract: "
            f"{data.output_feature_names}"
        )
    if np.any(data.x_std <= 0.0) or np.any(data.y_std <= 0.0):
        raise ValueError("Normalizer standard deviations must be strictly positive")

    checks = {
        "x_train": data.x_train,
        "y_train": data.y_train,
        "x_val": data.x_val,
        "y_val": data.y_val,
    }
    expected_width = {"x_train": INPUT_DIM, "x_val": INPUT_DIM, "y_train": OUTPUT_DIM, "y_val": OUTPUT_DIM}
    for name, array in checks.items():
        if array is None:
            continue
        if array.ndim != 2 or array.shape[1] != expected_width[name]:
            raise ValueError(f"{name} has invalid shape {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf values")

    pairs = [
        ("train", data.x_train, data.y_train, data.train_class, data.train_run_id, data.train_time_index),
        ("val", data.x_val, data.y_val, data.val_class, data.val_run_id, data.val_time_index),
    ]
    for split, x, y, cls, run_id, time_index in pairs:
        if x is None:
            continue
        n = x.shape[0]
        if y is None or y.shape[0] != n:
            raise ValueError(f"{split} X/Y sample count mismatch")
        for array_name, array in (("class", cls), ("run_id", run_id), ("time_index", time_index)):
            if array is None or array.shape[0] != n:
                raise ValueError(f"{split} {array_name} sample count mismatch")


def normalize_x(x: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((x - x_mean.reshape(1, -1)) / x_std.reshape(1, -1), dtype=np.float32)


def normalize_y(y: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((y - y_mean.reshape(1, -1)) / y_std.reshape(1, -1), dtype=np.float32)


def denormalize_y(y_norm: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(y_norm * y_std.reshape(1, -1) + y_mean.reshape(1, -1), dtype=np.float32)


def torch_normalize_x(x: torch.Tensor, x_mean: torch.Tensor, x_std: torch.Tensor) -> torch.Tensor:
    return (x - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)


def torch_denormalize_y(y_norm: torch.Tensor, y_mean: torch.Tensor, y_std: torch.Tensor) -> torch.Tensor:
    return y_norm * y_std.reshape(1, -1) + y_mean.reshape(1, -1)


def activation_module(name: str) -> torch.nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return torch.nn.ReLU()
    if normalized == "gelu":
        return torch.nn.GELU()
    if normalized == "silu":
        return torch.nn.SiLU()
    if normalized == "tanh":
        return torch.nn.Tanh()
    if normalized == "elu":
        return torch.nn.ELU()
    raise ValueError(f"Unsupported activation {name!r}")


class ResidualMLPBlock(torch.nn.Module):
    def __init__(self, hidden_dim: int, *, activation: str, dropout: float, layer_norm: bool):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_dim) if layer_norm else torch.nn.Identity()
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.act = activation_module(activation)
        self.drop = torch.nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        return x + y


class FDCylinderMLP(torch.nn.Module):
    """MLP for normalized cylinder-space one-step dynamics.

    The module expects normalized inputs and returns normalized target deltas.
    Raw-unit normalization and denormalization are stored in the checkpoint.
    """

    def __init__(
        self,
        *,
        input_dim: int = INPUT_DIM,
        output_dim: int = OUTPUT_DIM,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        activation: str = "silu",
        dropout: float = 0.02,
        layer_norm: bool = True,
    ):
        super().__init__()
        if input_dim != INPUT_DIM:
            raise ValueError(f"FD cylinder input_dim must be {INPUT_DIM}, got {input_dim}")
        if output_dim != OUTPUT_DIM:
            raise ValueError(f"FD cylinder output_dim must be {OUTPUT_DIM}, got {output_dim}")
        if hidden_dim < 16:
            raise ValueError("hidden_dim should be at least 16")
        if num_blocks < 1:
            raise ValueError("num_blocks should be at least 1")

        self.input = torch.nn.Linear(input_dim, hidden_dim)
        self.input_norm = torch.nn.LayerNorm(hidden_dim) if layer_norm else torch.nn.Identity()
        self.input_act = activation_module(activation)
        self.blocks = torch.nn.ModuleList(
            [
                ResidualMLPBlock(
                    hidden_dim,
                    activation=activation,
                    dropout=dropout,
                    layer_norm=layer_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.output_norm = torch.nn.LayerNorm(hidden_dim) if layer_norm else torch.nn.Identity()
        self.output_act = activation_module(activation)
        self.output = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_act(self.input_norm(self.input(x)))
        for block in self.blocks:
            h = block(h)
        h = self.output_act(self.output_norm(h))
        return self.output(h)


def model_config_from_args(args: Any) -> dict[str, Any]:
    return {
        "model_type": "FDCylinderMLP",
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "hidden_dim": int(args.hidden_dim),
        "num_blocks": int(args.num_blocks),
        "activation": str(args.activation),
        "dropout": float(args.dropout),
        "layer_norm": bool(args.layer_norm),
    }


def build_model(config: dict[str, Any]) -> FDCylinderMLP:
    return FDCylinderMLP(
        input_dim=int(config.get("input_dim", INPUT_DIM)),
        output_dim=int(config.get("output_dim", OUTPUT_DIM)),
        hidden_dim=int(config.get("hidden_dim", 512)),
        num_blocks=int(config.get("num_blocks", 4)),
        activation=str(config.get("activation", "silu")),
        dropout=float(config.get("dropout", 0.02)),
        layer_norm=bool(config.get("layer_norm", True)),
    )


def initialize_model_weights(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def assert_finite_model(model: torch.nn.Module, *, label: str) -> None:
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            raise RuntimeError(f"{label}: model parameter {name} contains NaN or Inf")


def assert_finite_gradients(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            raise RuntimeError(f"Gradient for {name} contains NaN or Inf")


def parameter_summary(model: torch.nn.Module) -> dict[str, float | int]:
    values = []
    for param in model.parameters():
        if param.requires_grad:
            values.append(param.detach().float().reshape(-1).cpu())
    if not values:
        return {"num_parameters": 0}
    flat = torch.cat(values)
    return {
        "num_parameters": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "max_abs": float(flat.abs().max().item()),
        "l2_norm": float(torch.linalg.norm(flat).item()),
    }


def target_weight_vector(group_weights: dict[str, float]) -> np.ndarray:
    weights = np.ones((OUTPUT_DIM,), dtype=np.float32)
    for group_name, group_slice in TARGET_GROUPS.items():
        weights[group_slice] = float(group_weights.get(group_name, 1.0))
    return weights


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> dict[str, Any]:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Metric shape mismatch: {y_true.shape} vs {y_pred.shape}")
    error = y_pred - y_true
    per_target = []
    for idx, name in enumerate(names):
        target = y_true[:, idx]
        pred = y_pred[:, idx]
        residual = pred - target
        mse = float(np.mean(residual**2))
        mae = float(np.mean(np.abs(residual)))
        rmse = float(math.sqrt(max(mse, 0.0)))
        denom = float(np.sum((target - np.mean(target)) ** 2))
        r2 = float(1.0 - np.sum(residual**2) / denom) if denom > 0.0 else float("nan")
        per_target.append({"name": name, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2})

    grouped: dict[str, dict[str, float]] = {}
    for group_name, group_slice in TARGET_GROUPS.items():
        group_error = error[:, group_slice]
        group_true = y_true[:, group_slice]
        mse = float(np.mean(group_error**2))
        mae = float(np.mean(np.abs(group_error)))
        rmse = float(math.sqrt(max(mse, 0.0)))
        denom = float(np.sum((group_true - np.mean(group_true, axis=0, keepdims=True)) ** 2))
        r2 = float(1.0 - np.sum(group_error**2) / denom) if denom > 0.0 else float("nan")
        grouped[group_name] = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    return {"per_target": per_target, "grouped": grouped}


def metrics_by_regime(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    names: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    labels = {1: "low_command", 2: "mid_command", 3: "high_command"}
    for class_value, label in labels.items():
        mask = classes.reshape(-1) == class_value
        if not np.any(mask):
            continue
        result[label] = {
            "num_samples": int(np.sum(mask)),
            "metrics": compute_regression_metrics(y_true[mask], y_pred[mask], names),
        }
    return result


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(to_jsonable(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    model_config: dict[str, Any],
    data: FDData,
    epoch: int,
    history: list[dict[str, Any]],
    metrics: dict[str, Any],
    args: Any,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
    checkpoint = {
        "checkpoint_version": "fd_cylinder_mlp_v1",
        "model_state_dict": state_dict,
        "model_config": model_config,
        "target_mode": "normalized_delta_cylinder_state",
        "input_layout": "[ValveCmd(4), Extension(4), ExtensionVelocity(4), dP(4), Force(4)]",
        "output_layout": "[DeltaExtension(4), DeltaExtensionVelocity(4), DeltaDP(4)]",
        "input_feature_names": data.input_feature_names,
        "output_feature_names": data.output_feature_names,
        "x_mean": data.x_mean,
        "x_std": data.x_std,
        "y_mean": data.y_mean,
        "y_std": data.y_std,
        "sample_time_s": 0.05,
        "epoch": int(epoch),
        "history": history,
        "metrics": metrics,
        "parameter_summary": parameter_summary(model),
        "training_args": vars(args) if hasattr(args, "__dict__") else {},
        "data_path": str(data.path),
        "literature_references": LITERATURE_REFERENCES,
    }
    torch.save(checkpoint, output_path)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location=map_location)


def load_model_from_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[FDCylinderMLP, dict[str, Any], dict[str, torch.Tensor]]:
    checkpoint = load_checkpoint(path, map_location="cpu")
    if "model_state_dict" not in checkpoint or "model_config" not in checkpoint:
        raise KeyError(f"{path} does not look like an FD cylinder checkpoint")
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    normalizers = {
        "x_mean": torch.as_tensor(checkpoint["x_mean"], dtype=torch.float32, device=device).reshape(-1),
        "x_std": torch.as_tensor(checkpoint["x_std"], dtype=torch.float32, device=device).reshape(-1),
        "y_mean": torch.as_tensor(checkpoint["y_mean"], dtype=torch.float32, device=device).reshape(-1),
        "y_std": torch.as_tensor(checkpoint["y_std"], dtype=torch.float32, device=device).reshape(-1),
    }
    if normalizers["x_mean"].numel() != INPUT_DIM or normalizers["y_mean"].numel() != OUTPUT_DIM:
        raise ValueError("Checkpoint normalizer dimensions do not match the FD cylinder contract")
    return model, checkpoint, normalizers


@torch.no_grad()
def predict_raw_batches(
    model: torch.nn.Module,
    x_raw: np.ndarray,
    *,
    normalizers: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    for start in range(0, x_raw.shape[0], batch_size):
        batch = torch.as_tensor(x_raw[start : start + batch_size], dtype=torch.float32, device=device)
        x_norm = torch_normalize_x(batch, normalizers["x_mean"], normalizers["x_std"])
        y_norm = model(x_norm)
        y_raw = torch_denormalize_y(y_norm, normalizers["y_mean"], normalizers["y_std"])
        preds.append(y_raw.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(preds, axis=0)


def group_indices_by_run(run_id: np.ndarray, time_index: np.ndarray) -> dict[int, np.ndarray]:
    groups: dict[int, list[int]] = {}
    for sample_index, rid in enumerate(run_id.reshape(-1)):
        groups.setdefault(int(rid), []).append(sample_index)

    sorted_groups: dict[int, np.ndarray] = {}
    for rid, indices in groups.items():
        idx = np.asarray(indices, dtype=np.int64)
        order = np.argsort(time_index[idx], kind="stable")
        sorted_groups[rid] = idx[order]
    return dict(sorted(sorted_groups.items()))


def parse_group_weights(values: Iterable[float]) -> dict[str, float]:
    weights = list(values)
    if len(weights) != 3:
        raise ValueError("Expected exactly three target group weights")
    return {
        "DeltaExtension": float(weights[0]),
        "DeltaExtensionVelocity": float(weights[1]),
        "DeltaDP": float(weights[2]),
    }
