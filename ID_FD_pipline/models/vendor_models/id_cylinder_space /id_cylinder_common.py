#!/usr/bin/env python3
"""Shared utilities for cylinder-space inverse dynamics.

ID learns:

    [Extension, ExtensionVelocity, dP, Force,
     TargetDeltaExtension, TargetDeltaExtensionVelocity] -> ValveCmd

The target deltas are achieved transitions from Simscape, used as a proxy for
desired cylinder-space motion because no independent cylinder-space reference
was logged.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR / "ID_cylinder_data.mat"

DATA_INPUT_DIM = 24
DEFAULT_MODEL_INPUT_DIM = 28
OUTPUT_DIM = 4

INPUT_FEATURE_NAMES = [
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
    "TargetDeltaExtension_1",
    "TargetDeltaExtension_2",
    "TargetDeltaExtension_3",
    "TargetDeltaExtension_4",
    "TargetDeltaExtensionVelocity_1",
    "TargetDeltaExtensionVelocity_2",
    "TargetDeltaExtensionVelocity_3",
    "TargetDeltaExtensionVelocity_4",
]

OUTPUT_FEATURE_NAMES = ["ValveCmd_1", "ValveCmd_2", "ValveCmd_3", "ValveCmd_4"]
PREVIOUS_VALVECMD_FEATURE_NAMES = [
    "PreviousValveCmd_1",
    "PreviousValveCmd_2",
    "PreviousValveCmd_3",
    "PreviousValveCmd_4",
]
DEFAULT_MODEL_INPUT_FEATURE_NAMES = INPUT_FEATURE_NAMES + PREVIOUS_VALVECMD_FEATURE_NAMES

COMMAND_LIMITS = np.asarray([0.95, 0.95, 0.80, 0.80], dtype=np.float32)


@dataclass
class IDData:
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
    if name not in handle or handle[name].dtype != object:
        return list(fallback)
    out: list[str] = []
    for ref in handle[name][0]:
        raw = np.asarray(handle[ref][()])
        out.append("".join(chr(int(v)) for v in raw.ravel(order="F") if int(v) != 0))
    return out or list(fallback)


def load_feature_matrix(handle: h5py.File, name: str, width: int) -> np.ndarray:
    if name not in handle:
        raise KeyError(f"Missing dataset {name!r} in {handle.filename}")
    raw = np.asarray(handle[name][()], dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got {raw.shape}")
    if raw.shape[0] == width:
        raw = raw.T
    elif raw.shape[1] != width:
        raise ValueError(f"{name} must have feature width {width}; got {raw.shape}")
    return np.ascontiguousarray(raw, dtype=np.float32)


def load_vector(handle: h5py.File, name: str, dtype: np.dtype[Any] | type) -> np.ndarray:
    if name not in handle:
        raise KeyError(f"Missing dataset {name!r} in {handle.filename}")
    return np.asarray(handle[name][()]).reshape(-1).astype(dtype, copy=False)


def load_normalizer(handle: h5py.File, name: str, width: int, *, require_positive: bool = False) -> np.ndarray:
    vec = load_vector(handle, name, np.float32)
    if vec.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {vec.shape}")
    if not np.isfinite(vec).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if require_positive and np.any(vec <= 0.0):
        raise ValueError(f"{name} contains non-positive std values")
    return np.ascontiguousarray(vec, dtype=np.float32)


def load_id_data(path: str | Path, *, load_train: bool = True, load_val: bool = True) -> IDData:
    data_path = Path(path).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    with h5py.File(data_path, "r") as handle:
        data = IDData(
            path=data_path,
            input_feature_names=read_matlab_cellstr(handle, "input_feature_names", INPUT_FEATURE_NAMES),
            output_feature_names=read_matlab_cellstr(handle, "output_feature_names", OUTPUT_FEATURE_NAMES),
            x_mean=load_normalizer(handle, "x_mean_id", DATA_INPUT_DIM),
            x_std=load_normalizer(handle, "x_std_id", DATA_INPUT_DIM, require_positive=True),
            y_mean=load_normalizer(handle, "y_mean_id", OUTPUT_DIM),
            y_std=load_normalizer(handle, "y_std_id", OUTPUT_DIM, require_positive=True),
        )
        if load_train:
            data.x_train = load_feature_matrix(handle, "X_train_id", DATA_INPUT_DIM)
            data.y_train = load_feature_matrix(handle, "Y_train_id", OUTPUT_DIM)
            data.train_class = load_vector(handle, "TrainClass", np.int64)
            data.train_run_id = load_vector(handle, "TrainRunID", np.int64)
            data.train_time_index = load_vector(handle, "TrainTimeIndex", np.int64)
            data.train_sample_time = load_vector(handle, "TrainSampleTime", np.float32)
        if load_val:
            data.x_val = load_feature_matrix(handle, "X_val_id", DATA_INPUT_DIM)
            data.y_val = load_feature_matrix(handle, "Y_val_id", OUTPUT_DIM)
            data.val_class = load_vector(handle, "ValClass", np.int64)
            data.val_run_id = load_vector(handle, "ValRunID", np.int64)
            data.val_time_index = load_vector(handle, "ValTimeIndex", np.int64)
            data.val_sample_time = load_vector(handle, "ValSampleTime", np.float32)
    validate_id_data(data)
    return data


def validate_id_data(data: IDData) -> None:
    if data.input_feature_names != INPUT_FEATURE_NAMES:
        raise ValueError(f"Unexpected input features: {data.input_feature_names}")
    if data.output_feature_names != OUTPUT_FEATURE_NAMES:
        raise ValueError(f"Unexpected output features: {data.output_feature_names}")
    for name in ("x_train", "y_train", "x_val", "y_val"):
        arr = getattr(data, name)
        if arr is None:
            continue
        expected = DATA_INPUT_DIM if name.startswith("x") else OUTPUT_DIM
        if arr.ndim != 2 or arr.shape[1] != expected:
            raise ValueError(f"{name} has invalid shape {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains NaN or Inf")


def normalize_x(x: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((x - x_mean.reshape(1, -1)) / x_std.reshape(1, -1), dtype=np.float32)


def normalize_y(y: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray((y - y_mean.reshape(1, -1)) / y_std.reshape(1, -1), dtype=np.float32)


def denormalize_y(y_norm: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(y_norm * y_std.reshape(1, -1) + y_mean.reshape(1, -1), dtype=np.float32)


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


class IDCylinderMLP(torch.nn.Module):
    """MLP that expects normalized ID inputs and returns normalized ValveCmd."""

    def __init__(
        self,
        *,
        input_dim: int = DEFAULT_MODEL_INPUT_DIM,
        output_dim: int = OUTPUT_DIM,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        activation: str = "silu",
        dropout: float = 0.02,
        layer_norm: bool = True,
    ):
        super().__init__()
        if input_dim not in (DATA_INPUT_DIM, DEFAULT_MODEL_INPUT_DIM) or output_dim != OUTPUT_DIM:
            raise ValueError(
                f"ID expects {DATA_INPUT_DIM} or {DEFAULT_MODEL_INPUT_DIM} inputs and {OUTPUT_DIM} outputs, "
                f"got {input_dim}->{output_dim}"
            )
        self.input = torch.nn.Linear(input_dim, hidden_dim)
        self.input_norm = torch.nn.LayerNorm(hidden_dim) if layer_norm else torch.nn.Identity()
        self.input_act = activation_module(activation)
        self.blocks = torch.nn.ModuleList(
            [
                ResidualMLPBlock(hidden_dim, activation=activation, dropout=dropout, layer_norm=layer_norm)
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
        "model_type": "IDCylinderMLP",
        "input_dim": DEFAULT_MODEL_INPUT_DIM if bool(getattr(args, "use_prev_valvecmd", True)) else DATA_INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "hidden_dim": int(args.hidden_dim),
        "num_blocks": int(args.num_blocks),
        "activation": str(args.activation),
        "dropout": float(args.dropout),
        "layer_norm": bool(args.layer_norm),
    }


def build_model(config: dict[str, Any]) -> IDCylinderMLP:
    return IDCylinderMLP(
        input_dim=int(config.get("input_dim", DEFAULT_MODEL_INPUT_DIM)),
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


def assert_finite_model(model: torch.nn.Module, *, label: str) -> None:
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            raise RuntimeError(f"{label}: parameter {name} contains NaN or Inf")


def assert_finite_gradients(model: torch.nn.Module) -> None:
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            raise RuntimeError(f"Gradient for {name} contains NaN or Inf")


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def parameter_summary(model: torch.nn.Module) -> dict[str, float | int]:
    chunks = [p.detach().float().reshape(-1).cpu() for p in model.parameters() if p.requires_grad]
    if not chunks:
        return {"num_parameters": 0}
    flat = torch.cat(chunks)
    return {
        "num_parameters": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "max_abs": float(flat.abs().max().item()),
        "l2_norm": float(torch.linalg.norm(flat).item()),
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]) -> dict[str, Any]:
    err = y_pred - y_true
    per_target = []
    for idx, name in enumerate(names):
        target = y_true[:, idx]
        residual = err[:, idx]
        mse = float(np.mean(residual**2))
        denom = float(np.sum((target - np.mean(target)) ** 2))
        r2 = float(1.0 - np.sum(residual**2) / denom) if denom > 0.0 else float("nan")
        per_target.append(
            {
                "name": name,
                "mse": mse,
                "rmse": float(math.sqrt(max(mse, 0.0))),
                "mae": float(np.mean(np.abs(residual))),
                "r2": r2,
            }
        )
    return {
        "per_target": per_target,
        "mean_mse": float(np.mean(err**2)),
        "mean_rmse": float(math.sqrt(float(np.mean(err**2)))),
        "mean_mae": float(np.mean(np.abs(err))),
    }


def metrics_by_regime(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray, names: list[str]) -> dict[str, Any]:
    labels = {1: "low_command", 2: "mid_command", 3: "high_command"}
    out: dict[str, Any] = {}
    for class_value, label in labels.items():
        mask = classes.reshape(-1) == class_value
        if np.any(mask):
            out[label] = {
                "num_samples": int(np.sum(mask)),
                "metrics": compute_regression_metrics(y_true[mask], y_pred[mask], names),
            }
    return out


def command_stress_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    pred_clipped = np.clip(y_pred, -COMMAND_LIMITS.reshape(1, -1), COMMAND_LIMITS.reshape(1, -1))
    abs_true = np.abs(y_true)
    abs_pred = np.abs(y_pred)
    sign_mask = abs_true > 0.05
    sat_mask = abs_true >= (0.95 * COMMAND_LIMITS.reshape(1, -1))
    return {
        "raw_prediction_out_of_limit_fraction": float(
            np.mean(abs_pred > COMMAND_LIMITS.reshape(1, -1))
        ),
        "clip_delta_mae": float(np.mean(np.abs(pred_clipped - y_pred))),
        "sign_accuracy_abs_true_gt_0p05": float(np.mean(np.sign(y_true[sign_mask]) == np.sign(y_pred[sign_mask])))
        if np.any(sign_mask)
        else float("nan"),
        "saturation_mae": float(np.mean(np.abs(y_pred[sat_mask] - y_true[sat_mask]))) if np.any(sat_mask) else float("nan"),
        "non_saturation_mae": float(np.mean(np.abs(y_pred[~sat_mask] - y_true[~sat_mask]))) if np.any(~sat_mask) else float("nan"),
    }


def group_indices_by_run(run_id: np.ndarray, time_index: np.ndarray) -> dict[int, np.ndarray]:
    groups: dict[int, list[int]] = {}
    for sample_index, rid in enumerate(run_id.reshape(-1)):
        groups.setdefault(int(rid), []).append(sample_index)
    out: dict[int, np.ndarray] = {}
    for rid, indices in groups.items():
        idx = np.asarray(indices, dtype=np.int64)
        out[rid] = idx[np.argsort(time_index[idx], kind="stable")]
    return dict(sorted(out.items()))


def previous_valvecmd_features(
    y_raw: np.ndarray,
    run_id: np.ndarray,
    time_index: np.ndarray,
) -> np.ndarray:
    prev = np.zeros_like(y_raw, dtype=np.float32)
    for indices in group_indices_by_run(run_id, time_index).values():
        if len(indices) < 2:
            continue
        prev[indices[1:]] = y_raw[indices[:-1]]
    return np.ascontiguousarray(prev, dtype=np.float32)


def append_previous_valvecmd(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    run_id: np.ndarray,
    time_index: np.ndarray,
) -> np.ndarray:
    prev = previous_valvecmd_features(y_raw, run_id, time_index)
    return np.ascontiguousarray(np.concatenate([x_raw, prev], axis=1), dtype=np.float32)


def model_input_names_for_dim(input_dim: int) -> list[str]:
    if input_dim == DATA_INPUT_DIM:
        return list(INPUT_FEATURE_NAMES)
    if input_dim == DEFAULT_MODEL_INPUT_DIM:
        return list(DEFAULT_MODEL_INPUT_FEATURE_NAMES)
    raise ValueError(f"Unsupported ID input dimension {input_dim}")


def prepare_features_for_checkpoint(data: IDData, checkpoint: dict[str, Any], *, split: str) -> np.ndarray:
    input_names = checkpoint.get("input_feature_names", INPUT_FEATURE_NAMES)
    if split == "train":
        x_raw, y_raw, run_id, time_index = data.x_train, data.y_train, data.train_run_id, data.train_time_index
    elif split == "val":
        x_raw, y_raw, run_id, time_index = data.x_val, data.y_val, data.val_run_id, data.val_time_index
    else:
        raise ValueError("split must be 'train' or 'val'")
    if x_raw is None or y_raw is None or run_id is None or time_index is None:
        raise RuntimeError(f"Missing {split} tensors for feature preparation")
    if input_names == INPUT_FEATURE_NAMES:
        return x_raw
    if input_names == DEFAULT_MODEL_INPUT_FEATURE_NAMES:
        return append_previous_valvecmd(x_raw, y_raw, run_id, time_index)
    raise ValueError(f"Unsupported checkpoint input_feature_names: {input_names}")


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
    data: IDData,
    input_feature_names: list[str],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
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
            "checkpoint_version": "id_cylinder_mlp_v3_recursive",
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "model_config": model_config,
            "target_mode": target_mode,
            "input_layout": (
                "[Extension(4), ExtensionVelocity(4), dP(4), Force(4), "
                "TargetDeltaExtension(4), TargetDeltaExtensionVelocity(4), optional PreviousValveCmd(4)]"
            ),
            "output_layout": "[ValveCmd(4)]",
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
            "parameter_summary": parameter_summary(model),
        },
        output_path,
    )


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location=map_location)


def load_model_from_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[IDCylinderMLP, dict[str, Any], dict[str, torch.Tensor]]:
    checkpoint = load_checkpoint(path, map_location="cpu")
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    normalizers = {
        "x_mean": torch.as_tensor(checkpoint["x_mean"], dtype=torch.float32, device=device).reshape(-1),
        "x_std": torch.as_tensor(checkpoint["x_std"], dtype=torch.float32, device=device).reshape(-1),
        "y_mean": torch.as_tensor(checkpoint["y_mean"], dtype=torch.float32, device=device).reshape(-1),
        "y_std": torch.as_tensor(checkpoint["y_std"], dtype=torch.float32, device=device).reshape(-1),
        "target_mode": checkpoint.get("target_mode", "normalized_valve_cmd"),
    }
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
    preds: list[np.ndarray] = []
    model.eval()
    for start in range(0, x_raw.shape[0], batch_size):
        batch = torch.as_tensor(x_raw[start : start + batch_size], dtype=torch.float32, device=device)
        x_norm = (batch - normalizers["x_mean"].reshape(1, -1)) / normalizers["x_std"].reshape(1, -1)
        y_norm = model(x_norm)
        y_raw = y_norm * normalizers["y_std"].reshape(1, -1) + normalizers["y_mean"].reshape(1, -1)
        if normalizers.get("target_mode") == "normalized_delta_valve_cmd":
            if batch.shape[1] < DEFAULT_MODEL_INPUT_DIM:
                raise RuntimeError("Delta ValveCmd checkpoint requires PreviousValveCmd features")
            y_raw = batch[:, -OUTPUT_DIM:] + y_raw
        preds.append(y_raw.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(preds, axis=0)
