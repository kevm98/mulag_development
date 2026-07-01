#!/usr/bin/env python3
"""Shared utilities for coupled cylinder sequence FD/ID models."""

from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
MODELS_ROOT = Path(__file__).resolve().parent / "Models_cylinder "
OUT_DIR = ROOT / "outputs" / "cylinder_diagnostics"

CYLINDER_NAMES = (
    "Drehzapfen_joint",
    "Ausleger_I_joint",
    "Ausleger_II_joint",
    "Messerkopf_Schwenk_joint",
)


def _ensure_model_paths() -> None:
    for path in (MODELS_ROOT / "fd_cylinder_space ", MODELS_ROOT / "id_cylinder_space "):
        text = str(path.resolve())
        if text not in sys.path:
            sys.path.insert(0, text)


def import_fd_common():
    _ensure_model_paths()
    return importlib.import_module("fd_cylinder_common")


def import_id_common():
    _ensure_model_paths()
    return importlib.import_module("id_cylinder_common")


COMMAND_LIMITS = np.asarray([0.95, 0.95, 0.80, 0.80], dtype=np.float32)
SAMPLE_TIME_S = 0.05

FD_SEQ_INPUT_NAMES = [
    "ValveCmd_1", "ValveCmd_2", "ValveCmd_3", "ValveCmd_4",
    "Extension_1", "Extension_2", "Extension_3", "Extension_4",
    "ExtensionVelocity_1", "ExtensionVelocity_2", "ExtensionVelocity_3", "ExtensionVelocity_4",
]
FD_SEQ_TARGET_NAMES = [
    "DeltaExtension_next_1", "DeltaExtension_next_2", "DeltaExtension_next_3", "DeltaExtension_next_4",
]
ID_SEQ_STEP_NAMES = [
    "Extension_1", "Extension_2", "Extension_3", "Extension_4",
    "ExtensionVelocity_1", "ExtensionVelocity_2", "ExtensionVelocity_3", "ExtensionVelocity_4",
    "ValveCmd_1", "ValveCmd_2", "ValveCmd_3", "ValveCmd_4",
]
ID_SEQ_CURRENT_NAMES = [
    "TargetDeltaExtension_1", "TargetDeltaExtension_2", "TargetDeltaExtension_3", "TargetDeltaExtension_4",
    "TargetDeltaExtensionVelocity_1", "TargetDeltaExtensionVelocity_2", "TargetDeltaExtensionVelocity_3", "TargetDeltaExtensionVelocity_4",
    "PreviousValveCmd_1", "PreviousValveCmd_2", "PreviousValveCmd_3", "PreviousValveCmd_4",
]
ID_SEQ_TARGET_NAMES = [
    "DeltaValveCmd_1", "DeltaValveCmd_2", "DeltaValveCmd_3", "DeltaValveCmd_4",
]


class HistoryMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        layers: list[torch.nn.Module] = []
        dim = input_dim
        for _ in range(num_layers):
            layers += [torch.nn.Linear(dim, hidden_dim), torch.nn.SiLU()]
            dim = hidden_dim
        layers.append(torch.nn.Linear(dim, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        return self.net(x)


class HistoryGRU(torch.nn.Module):
    def __init__(self, step_dim: int, current_dim: int, output_dim: int, hidden_dim: int = 192, num_layers: int = 1):
        super().__init__()
        self.gru = torch.nn.GRU(step_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim + current_dim),
            torch.nn.Linear(hidden_dim + current_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
        self.current_dim = int(current_dim)

    def forward(self, seq: torch.Tensor, current: torch.Tensor | None = None) -> torch.Tensor:
        _, h = self.gru(seq)
        last = h[-1]
        if current is None:
            current = torch.zeros((seq.shape[0], self.current_dim), dtype=seq.dtype, device=seq.device)
        return self.head(torch.cat([last, current], dim=1))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_json(path: Path, payload: Any) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def group_indices_by_run(run_id: np.ndarray, time_index: np.ndarray) -> dict[int, np.ndarray]:
    groups: dict[int, list[int]] = {}
    for i, rid in enumerate(run_id.reshape(-1)):
        groups.setdefault(int(rid), []).append(i)
    out: dict[int, np.ndarray] = {}
    for rid, values in groups.items():
        idx = np.asarray(values, dtype=np.int64)
        out[rid] = idx[np.argsort(time_index[idx], kind="stable")]
    return dict(sorted(out.items()))


def previous_valvecmd(y: np.ndarray, run_id: np.ndarray, time_index: np.ndarray) -> np.ndarray:
    prev = np.zeros_like(y, dtype=np.float32)
    for idx in group_indices_by_run(run_id, time_index).values():
        if len(idx) > 1:
            prev[idx[1:]] = y[idx[:-1]]
    return prev


def normalizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0).astype(np.float32)
    std = np.std(x, axis=0).astype(np.float32)
    std[std < 1.0e-8] = 1.0
    return mean, std


def make_fd_step_features(x20: np.ndarray) -> np.ndarray:
    return np.concatenate([x20[:, 0:4], x20[:, 4:8], x20[:, 8:12]], axis=1).astype(np.float32, copy=False)


def make_id_step_features(id_x24: np.ndarray, y_valve: np.ndarray) -> np.ndarray:
    return np.concatenate([id_x24[:, 0:4], id_x24[:, 4:8], y_valve], axis=1).astype(np.float32, copy=False)


def make_id_current_features(id_x24: np.ndarray, prev: np.ndarray) -> np.ndarray:
    return np.concatenate([id_x24[:, 16:20], id_x24[:, 20:24], prev], axis=1).astype(np.float32, copy=False)


def build_windows(
    step_features: np.ndarray,
    current_features: np.ndarray | None,
    targets: np.ndarray,
    run_id: np.ndarray,
    time_index: np.ndarray,
    *,
    history: int,
    stride: int,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seqs: list[np.ndarray] = []
    currents: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    sample_indices: list[int] = []
    for indices in group_indices_by_run(run_id, time_index).values():
        if len(indices) < history:
            continue
        for local_end in range(history - 1, len(indices), stride):
            sample_idx = int(indices[local_end])
            window = indices[local_end - history + 1 : local_end + 1]
            if mask is not None and (not bool(mask[sample_idx]) or not bool(np.all(mask[window]))):
                continue
            seqs.append(step_features[window])
            if current_features is None:
                currents.append(np.zeros((0,), dtype=np.float32))
            else:
                currents.append(current_features[sample_idx])
            ys.append(targets[sample_idx])
            sample_indices.append(sample_idx)
    if not seqs:
        raise RuntimeError("No sequence windows were built")
    return (
        np.ascontiguousarray(np.stack(seqs), dtype=np.float32),
        np.ascontiguousarray(np.stack(currents), dtype=np.float32),
        np.ascontiguousarray(np.stack(ys), dtype=np.float32),
        np.asarray(sample_indices, dtype=np.int64),
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    err = y_pred - y_true
    denom = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - np.sum(err * err, axis=0) / np.maximum(denom, 1.0e-12)
    sign_mask = np.abs(y_true) > 1.0e-8
    sign = np.sum((np.sign(y_true) == np.sign(y_pred)) & sign_mask, axis=0) / np.maximum(np.sum(sign_mask, axis=0), 1)
    out: dict[str, Any] = {
        "mae_mean": float(np.mean(np.abs(err))),
        "rmse_mean": float(np.sqrt(np.mean(err * err))),
        "r2_mean": float(np.mean(r2)),
        "sign_accuracy_mean": float(np.mean(sign)),
    }
    for i, name in enumerate(CYLINDER_NAMES):
        out[f"{name}_mae"] = float(np.mean(np.abs(err[:, i])))
        out[f"{name}_rmse"] = float(np.sqrt(np.mean(err[:, i] * err[:, i])))
        out[f"{name}_r2"] = float(r2[i])
        out[f"{name}_sign_accuracy"] = float(sign[i])
    return out


@torch.no_grad()
def predict_sequence_checkpoint(
    checkpoint_path: Path,
    seq_raw: np.ndarray,
    current_raw: np.ndarray | None,
    *,
    device: torch.device,
    batch_size: int = 2048,
) -> tuple[np.ndarray, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    model_type = cfg["model_type"]
    if model_type == "HistoryMLP":
        model = HistoryMLP(int(cfg["input_dim"]), int(cfg["output_dim"]), int(cfg["hidden_dim"]), int(cfg["num_layers"]))
    elif model_type == "HistoryGRU":
        model = HistoryGRU(
            int(cfg["step_dim"]),
            int(cfg.get("current_dim", 0)),
            int(cfg["output_dim"]),
            int(cfg["hidden_dim"]),
            int(cfg.get("num_layers", 1)),
        )
    else:
        raise ValueError(f"Unsupported sequence model type {model_type}")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    seq_mean = np.asarray(ckpt["seq_mean"], dtype=np.float32)
    seq_std = np.asarray(ckpt["seq_std"], dtype=np.float32)
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std = np.asarray(ckpt["y_std"], dtype=np.float32)
    current_mean = np.asarray(ckpt.get("current_mean", []), dtype=np.float32)
    current_std = np.asarray(ckpt.get("current_std", []), dtype=np.float32)
    preds: list[np.ndarray] = []
    for start in range(0, seq_raw.shape[0], batch_size):
        seq = (seq_raw[start : start + batch_size] - seq_mean.reshape(1, 1, -1)) / seq_std.reshape(1, 1, -1)
        seq_t = torch.as_tensor(seq, dtype=torch.float32, device=device)
        if model_type == "HistoryMLP":
            y_norm = model(seq_t)
        else:
            if current_raw is None or current_raw.shape[1] == 0:
                current_t = None
            else:
                cur = (current_raw[start : start + batch_size] - current_mean.reshape(1, -1)) / current_std.reshape(1, -1)
                current_t = torch.as_tensor(cur, dtype=torch.float32, device=device)
            y_norm = model(seq_t, current_t)
        y = y_norm.detach().cpu().numpy() * y_std.reshape(1, -1) + y_mean.reshape(1, -1)
        preds.append(y.astype(np.float32))
    return np.concatenate(preds, axis=0), ckpt


def load_sequence_predictor(checkpoint_path: Path, *, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    model_type = cfg["model_type"]
    if model_type == "HistoryMLP":
        model = HistoryMLP(int(cfg["input_dim"]), int(cfg["output_dim"]), int(cfg["hidden_dim"]), int(cfg["num_layers"]))
    elif model_type == "HistoryGRU":
        model = HistoryGRU(
            int(cfg["step_dim"]),
            int(cfg.get("current_dim", 0)),
            int(cfg["output_dim"]),
            int(cfg["hidden_dim"]),
            int(cfg.get("num_layers", 1)),
        )
    else:
        raise ValueError(f"Unsupported sequence model type {model_type}")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def predict_loaded_sequence(
    model: torch.nn.Module,
    ckpt: dict[str, Any],
    seq_raw: np.ndarray,
    current_raw: np.ndarray | None,
    *,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    model_type = ckpt["model_config"]["model_type"]
    seq_mean = np.asarray(ckpt["seq_mean"], dtype=np.float32)
    seq_std = np.asarray(ckpt["seq_std"], dtype=np.float32)
    y_mean = np.asarray(ckpt["y_mean"], dtype=np.float32)
    y_std = np.asarray(ckpt["y_std"], dtype=np.float32)
    current_mean = np.asarray(ckpt.get("current_mean", []), dtype=np.float32)
    current_std = np.asarray(ckpt.get("current_std", []), dtype=np.float32)
    preds: list[np.ndarray] = []
    for start in range(0, seq_raw.shape[0], batch_size):
        seq = (seq_raw[start : start + batch_size] - seq_mean.reshape(1, 1, -1)) / seq_std.reshape(1, 1, -1)
        seq_t = torch.as_tensor(seq, dtype=torch.float32, device=device)
        if model_type == "HistoryMLP":
            y_norm = model(seq_t)
        else:
            if current_raw is None or current_raw.shape[1] == 0:
                current_t = None
            else:
                cur = (current_raw[start : start + batch_size] - current_mean.reshape(1, -1)) / current_std.reshape(1, -1)
                current_t = torch.as_tensor(cur, dtype=torch.float32, device=device)
            y_norm = model(seq_t, current_t)
        y = y_norm.detach().cpu().numpy() * y_std.reshape(1, -1) + y_mean.reshape(1, -1)
        preds.append(y.astype(np.float32))
    return np.concatenate(preds, axis=0)


def save_sequence_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    model_config: dict[str, Any],
    seq_mean: np.ndarray,
    seq_std: np.ndarray,
    current_mean: np.ndarray | None,
    current_std: np.ndarray | None,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    payload = {
        "checkpoint_version": "coupled_cylinder_sequence_v1",
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": model_config,
        "seq_mean": seq_mean,
        "seq_std": seq_std,
        "current_mean": np.zeros((0,), dtype=np.float32) if current_mean is None else current_mean,
        "current_std": np.ones((0,), dtype=np.float32) if current_std is None else current_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "sample_time_s": SAMPLE_TIME_S,
        **metadata,
    }
    torch.save(payload, path)
