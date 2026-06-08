from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

try:
    from models import ModelConfig, RidgeRegressionBaseline, build_model
    from rom_data_contract import DP_SLICE, EXTENSION_SLICE, EXTENSION_VELOCITY_SLICE, TARGET_DELTA_DP_SLICE, TARGET_DELTA_EXTENSION_SLICE
    from rom_dataset import DEFAULT_DATASET_PATH, ROMSequenceDataset, find_default_dataset, load_metadata
except ImportError:
    from kevin_integration.rom_cylinders.models import ModelConfig, RidgeRegressionBaseline, build_model
    from kevin_integration.rom_cylinders.rom_data_contract import (
        DP_SLICE,
        EXTENSION_SLICE,
        EXTENSION_VELOCITY_SLICE,
        TARGET_DELTA_DP_SLICE,
        TARGET_DELTA_EXTENSION_SLICE,
    )
    from kevin_integration.rom_cylinders.rom_dataset import (
        DEFAULT_DATASET_PATH,
        ROMSequenceDataset,
        find_default_dataset,
        load_metadata,
    )


ROM_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROM_DIR / "checkpoints"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def collate_xy(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    return torch.stack(list(xs), dim=0), torch.stack(list(ys), dim=0)


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    extension_loss_weight: float,
    pressure_loss_weight: float,
    sign_loss_weight: float,
    sign_deadband: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ext_pred = pred[:, TARGET_DELTA_EXTENSION_SLICE]
    ext_target = target[:, TARGET_DELTA_EXTENSION_SLICE]
    ext_loss = torch.mean((ext_pred - ext_target) ** 2)
    if pred.shape[-1] >= 8 and target.shape[-1] >= 8:
        dp_pred = pred[:, TARGET_DELTA_DP_SLICE]
        dp_target = target[:, TARGET_DELTA_DP_SLICE]
        dp_loss = torch.mean((dp_pred - dp_target) ** 2)
    else:
        dp_loss = pred.new_tensor(0.0)
    sign_loss = pred.new_tensor(0.0)
    if sign_loss_weight > 0.0:
        mask = torch.abs(ext_target) > sign_deadband
        wrong_sign = torch.relu(-(ext_pred * ext_target))
        if mask.any():
            sign_loss = torch.mean(wrong_sign[mask])
    total = extension_loss_weight * ext_loss + pressure_loss_weight * dp_loss + sign_loss_weight * sign_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "extension_mse": float(ext_loss.detach().cpu()),
        "pressure_mse": float(dp_loss.detach().cpu()),
        "sign_loss": float(sign_loss.detach().cpu()),
    }


def _target_slice_for_mode(target_mode: str) -> slice:
    if target_mode == "extension_only":
        return TARGET_DELTA_EXTENSION_SLICE
    if target_mode == "extension_and_dp":
        return slice(0, 8)
    raise ValueError("target_mode must be 'extension_only' or 'extension_and_dp'.")


def _source_split_for_dataset(dataset: ROMSequenceDataset) -> tuple[str, np.ndarray]:
    if getattr(dataset, "_sample_indices", None) is not None:
        return dataset._source_split, dataset._sample_indices
    return dataset.split, np.arange(len(dataset), dtype=np.int64)


def compute_output_scales_hdf5(
    dataset: ROMSequenceDataset,
    *,
    target_mode: str,
    percentile: float,
    chunk_samples: int = 32768,
) -> tuple[float, ...]:
    source_split, sample_indices = _source_split_for_dataset(dataset)
    y_key = f"Y_{source_split}_rom"
    target_slice = _target_slice_for_mode(target_mode)
    values: list[np.ndarray] = []
    with h5py.File(dataset.dataset_path, "r") as handle:
        for start in range(0, sample_indices.size, chunk_samples):
            idx = np.sort(sample_indices[start : start + chunk_samples])
            # h5py fancy indexing requires increasing indices.
            y = np.asarray(handle[y_key][:, idx], dtype=np.float32).T[:, target_slice]
            values.append(np.abs(y))
    all_values = np.concatenate(values, axis=0)
    scales = np.percentile(all_values, float(percentile), axis=0)
    scales = np.maximum(scales, 1.0e-12)
    return tuple(float(value) for value in scales.tolist())


def build_rollout_start_indices(dataset: ROMSequenceDataset, *, horizon: int) -> np.ndarray:
    if horizon <= 0:
        return np.asarray([], dtype=np.int64)
    run_ids = dataset.run_ids
    time_indices = dataset.time_indices
    valid: list[int] = []
    for index in range(0, max(0, len(dataset) - horizon)):
        stop = index + horizon
        if not np.all(run_ids[index : stop + 1] == run_ids[index]):
            continue
        if not np.all(np.diff(time_indices[index : stop + 1]) == 1):
            continue
        valid.append(index)
    return np.asarray(valid, dtype=np.int64)


def _normalize_model_input(dataset: ROMSequenceDataset, history_raw: torch.Tensor) -> torch.Tensor:
    stats = dataset.normalization_stats
    mean = stats.input_mean.to(device=history_raw.device, dtype=history_raw.dtype)
    std = torch.clamp(stats.input_std.to(device=history_raw.device, dtype=history_raw.dtype), min=1.0e-12)
    return (history_raw - mean) / std


def _denormalize_model_output(dataset: ROMSequenceDataset, pred_norm: torch.Tensor) -> torch.Tensor:
    stats = dataset.normalization_stats
    mean = stats.target_mean.to(device=pred_norm.device, dtype=pred_norm.dtype)
    std = torch.clamp(stats.target_std.to(device=pred_norm.device, dtype=pred_norm.dtype), min=1.0e-12)
    return pred_norm * std + mean


def compute_short_rollout_loss(
    *,
    model: nn.Module,
    raw_dataset: ROMSequenceDataset,
    start_indices: np.ndarray,
    device: torch.device,
    horizon: int,
    batch_size: int,
    scheduled_sampling_probability: float,
) -> torch.Tensor:
    if horizon <= 0 or batch_size <= 0 or start_indices.size == 0:
        return torch.zeros((), dtype=torch.float32, device=device)
    count = min(int(batch_size), int(start_indices.size))
    chosen = np.random.choice(start_indices, size=count, replace=False)
    stats = raw_dataset.normalization_stats
    extension_state_std = torch.clamp(
        stats.input_std[EXTENSION_SLICE].to(device=device, dtype=torch.float32),
        min=1.0e-12,
    )
    total = torch.zeros((), dtype=torch.float32, device=device)
    terms = 0
    for start_index in chosen:
        _, _, first_info = raw_dataset[int(start_index)]
        history_raw = first_info["raw_input"].to(device=device, dtype=torch.float32)
        dt = max(float(first_info["sample_time"]), 1.0e-12)
        for offset in range(1, horizon + 1):
            _, _, next_info = raw_dataset[int(start_index + offset)]
            next_raw_true = next_info["raw_input"].to(device=device, dtype=torch.float32)
            x_norm = _normalize_model_input(raw_dataset, history_raw).unsqueeze(0)
            pred_norm = model(x_norm)
            pred_raw = _denormalize_model_output(raw_dataset, pred_norm).squeeze(0)
            delta_ext = pred_raw[TARGET_DELTA_EXTENSION_SLICE]
            next_ext_pred = history_raw[-1, EXTENSION_SLICE] + delta_ext
            true_ext = next_raw_true[-1, EXTENSION_SLICE]
            total = total + torch.mean(((next_ext_pred - true_ext) / extension_state_std) ** 2)
            terms += 1

            use_predicted_pressure = (
                pred_raw.shape[-1] >= 8
                and random.random() < float(scheduled_sampling_probability)
            )
            if use_predicted_pressure:
                delta_dp = pred_raw[TARGET_DELTA_DP_SLICE]
                next_dp = history_raw[-1, DP_SLICE] + delta_dp
                dp_history = torch.cat([history_raw[1:, DP_SLICE], next_dp.unsqueeze(0)], dim=0)
            else:
                dp_history = next_raw_true[:, DP_SLICE]
            ext_history = torch.cat([history_raw[1:, EXTENSION_SLICE], next_ext_pred.unsqueeze(0)], dim=0)
            vel_history = torch.cat(
                [history_raw[1:, EXTENSION_VELOCITY_SLICE], (delta_ext / dt).unsqueeze(0)],
                dim=0,
            )
            base = next_raw_true
            history_raw = torch.cat(
                [
                    base[:, : DP_SLICE.start],
                    dp_history,
                    ext_history,
                    vel_history,
                    base[:, EXTENSION_VELOCITY_SLICE.stop :],
                ],
                dim=-1,
            )
    return total / max(terms, 1)


@torch.no_grad()
def evaluate_one_step(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    extension_loss_weight: float,
    pressure_loss_weight: float,
    sign_loss_weight: float,
    sign_deadband: float,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {"loss": 0.0, "extension_mse": 0.0, "pressure_mse": 0.0, "sign_loss": 0.0}
    samples = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        _, metrics = compute_loss(
            pred,
            y,
            extension_loss_weight=extension_loss_weight,
            pressure_loss_weight=pressure_loss_weight,
            sign_loss_weight=sign_loss_weight,
            sign_deadband=sign_deadband,
        )
        batch = x.shape[0]
        for key, value in metrics.items():
            totals[key] += value * batch
        samples += batch
    return {key: value / max(samples, 1) for key, value in totals.items()}


def fit_ridge_streaming(
    model: RidgeRegressionBaseline,
    loader: DataLoader,
    device: torch.device,
    *,
    ridge_alpha: float,
) -> None:
    model.to(device)
    feature_dim = model.linear.in_features
    lhs = torch.zeros((feature_dim + 1, feature_dim + 1), dtype=torch.float64, device=device)
    rhs = torch.zeros((feature_dim + 1, model.output_dim), dtype=torch.float64, device=device)
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        features = model._features(x).double()
        design = torch.cat([features, torch.ones((features.shape[0], 1), dtype=torch.float64, device=device)], dim=-1)
        lhs += design.T @ design
        rhs += design.T @ y.double()
    reg = float(ridge_alpha) * torch.eye(feature_dim + 1, dtype=torch.float64, device=device)
    reg[-1, -1] = 0.0
    weights = torch.linalg.solve(lhs + reg, rhs).float()
    model.linear.weight.data.copy_(weights[:-1].T)
    model.linear.bias.data.copy_(weights[-1])


def fit_ridge_hdf5_streaming(
    model: RidgeRegressionBaseline,
    dataset: ROMSequenceDataset,
    device: torch.device,
    *,
    ridge_alpha: float,
    max_samples: int | None = None,
    chunk_samples: int = 16384,
) -> None:
    model.to(device)
    feature_dim = model.linear.in_features
    lhs = torch.zeros((feature_dim + 1, feature_dim + 1), dtype=torch.float64, device=device)
    rhs = torch.zeros((feature_dim + 1, model.output_dim), dtype=torch.float64, device=device)
    stats = dataset.normalization_stats
    input_mean = stats.input_mean.to(device=device).float().reshape(1, 1, -1)
    input_std = torch.clamp(stats.input_std.to(device=device).float().reshape(1, 1, -1), min=1.0e-12)
    target_mean = stats.target_mean.to(device=device).float().reshape(1, -1)
    target_std = torch.clamp(stats.target_std.to(device=device).float().reshape(1, -1), min=1.0e-12)
    sample_count = min(len(dataset), max_samples) if max_samples is not None else len(dataset)

    source_split, sample_indices = _source_split_for_dataset(dataset)
    if max_samples is not None:
        sample_indices = sample_indices[:max_samples]
    with h5py.File(dataset.dataset_path, "r") as handle:
        x_key = f"X_{source_split}_rom"
        y_key = f"Y_{source_split}_rom"
        force_key = f"Force_{source_split}_hist"
        for start in range(0, sample_count, chunk_samples):
            stop = min(start + chunk_samples, sample_count)
            idx = np.sort(sample_indices[start:stop])
            x_np = np.asarray(handle[x_key][:, :, idx], dtype=np.float32).transpose(2, 1, 0)
            if dataset.use_force_input:
                force_np = np.asarray(handle[force_key][:, :, idx], dtype=np.float32).transpose(2, 1, 0)
                x_np = np.concatenate([x_np, force_np], axis=-1)
            y_np = np.asarray(handle[y_key][:, idx], dtype=np.float32).T[:, _target_slice_for_mode(dataset.target_mode)]
            x = torch.from_numpy(x_np).to(device=device)
            y = torch.from_numpy(y_np).to(device=device)
            x = (x - input_mean) / input_std
            y = (y - target_mean) / target_std
            features = model._features(x).double()
            design = torch.cat(
                [features, torch.ones((features.shape[0], 1), dtype=torch.float64, device=device)],
                dim=-1,
            )
            lhs += design.T @ design
            rhs += design.T @ y.double()

    reg = float(ridge_alpha) * torch.eye(feature_dim + 1, dtype=torch.float64, device=device)
    reg[-1, -1] = 0.0
    weights = torch.linalg.solve(lhs + reg, rhs).float()
    model.linear.weight.data.copy_(weights[:-1].T)
    model.linear.bias.data.copy_(weights[-1])


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    config: dict[str, Any],
    metadata: Any,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "dataset_metadata": metadata.__dict__,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "epoch": epoch,
            "saved_at_unix": time.time(),
        },
        path,
    )


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hydraulic cylinder ROM candidates.")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--model", choices=["linear", "ridge", "mlp", "gru", "lstm", "physics_informed"], default="gru")
    parser.add_argument("--target_mode", choices=["extension_only", "extension_and_dp"], default="extension_only")
    parser.add_argument("--split_scheme", choices=["original", "locked"], default="original")
    parser.add_argument("--locked_manifest", type=Path, default=ROM_DIR / "rom_split_manifest.json")
    parser.add_argument("--use_force_input", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--rollout_loss_weight", type=float, default=0.0)
    parser.add_argument("--pressure_loss_weight", type=float, default=1.0)
    parser.add_argument("--extension_loss_weight", type=float, default=1.0)
    parser.add_argument("--sign_loss_weight", type=float, default=0.0)
    parser.add_argument("--smoothness_loss_weight", type=float, default=0.0)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--sign_deadband", type=float, default=1.0e-4)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--bounded_outputs", action="store_true")
    parser.add_argument("--output_scale_percentile", type=float, default=99.9)
    parser.add_argument("--lr_scheduler", choices=["none", "plateau"], default="plateau")
    parser.add_argument("--rollout_horizon", type=int, default=5)
    parser.add_argument("--rollout_warmup_epochs", type=int, default=10)
    parser.add_argument("--scheduled_sampling_probability", type=float, default=0.0)
    parser.add_argument("--rollout_batch_size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoothness_loss_weight > 0.0:
        print("smoothness_loss_weight is accepted for experiment tracking; batch-smoothness loss is not enabled by default.")

    set_seed(args.seed)
    dataset_path = args.dataset or (DEFAULT_DATASET_PATH if DEFAULT_DATASET_PATH.exists() else find_default_dataset())
    device = torch.device(args.device)
    model_type = "ridge" if args.model == "linear" else args.model
    run_name = f"{model_type}_{'force' if args.use_force_input else 'core'}_seed{args.seed}"
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = ROMSequenceDataset(
        dataset_path,
        split="train",
        use_force_input=args.use_force_input,
        split_scheme=args.split_scheme,
        locked_manifest=args.locked_manifest,
        target_mode=args.target_mode,
    )
    val_dataset = ROMSequenceDataset(
        dataset_path,
        split="val",
        use_force_input=args.use_force_input,
        split_scheme=args.split_scheme,
        locked_manifest=args.locked_manifest,
        target_mode=args.target_mode,
    )
    rollout_dataset = None
    rollout_start_indices = np.asarray([], dtype=np.int64)
    if args.rollout_loss_weight > 0.0 and model_type not in {"ridge"}:
        rollout_dataset = ROMSequenceDataset(
            dataset_path,
            split="train",
            use_force_input=args.use_force_input,
            split_scheme=args.split_scheme,
            locked_manifest=args.locked_manifest,
            target_mode=args.target_mode,
            normalize_inputs=False,
            normalize_targets=False,
            return_metadata=True,
        )
        rollout_start_indices = build_rollout_start_indices(rollout_dataset, horizon=args.rollout_horizon)
        if args.max_train_samples is not None:
            rollout_start_indices = rollout_start_indices[rollout_start_indices < max(0, args.max_train_samples - args.rollout_horizon)]
        if rollout_start_indices.size == 0:
            print("rollout_loss_weight requested, but no same-run rollout windows were available; rollout loss is disabled.")
        else:
            print(
                f"Using short rollout loss after epoch {args.rollout_warmup_epochs}: "
                f"horizon={args.rollout_horizon}, windows={rollout_start_indices.size}, "
                f"batch_size={args.rollout_batch_size}."
            )
    train_source = (
        Subset(train_dataset, range(min(args.max_train_samples, len(train_dataset))))
        if args.max_train_samples is not None
        else train_dataset
    )
    val_source = (
        Subset(val_dataset, range(min(args.max_val_samples, len(val_dataset))))
        if args.max_val_samples is not None
        else val_dataset
    )
    metadata = load_metadata(dataset_path, compute_hash=False)
    train_loader = DataLoader(
        train_source,
        batch_size=args.batch_size,
        shuffle=model_type != "ridge",
        num_workers=args.num_workers,
        collate_fn=collate_xy,
    )
    val_loader = DataLoader(
        val_source,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_xy,
    )

    output_scale = (
        compute_output_scales_hdf5(
            train_dataset,
            target_mode=args.target_mode,
            percentile=args.output_scale_percentile,
        )
        if args.bounded_outputs and model_type not in {"ridge"}
        else None
    )
    config = ModelConfig(
        model_type=model_type,
        input_dim=train_dataset.input_dim,
        output_dim=train_dataset.output_dim,
        target_mode=args.target_mode,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bounded_outputs=bool(args.bounded_outputs and model_type not in {"ridge"}),
        output_scale=output_scale,
    )
    model = build_model(config, normalization_stats=train_dataset.normalization_stats).to(device)
    config_dict = {
        "model_type": model_type,
        "input_dim": train_dataset.input_dim,
        "output_dim": train_dataset.output_dim,
        "target_mode": args.target_mode,
        "split_scheme": args.split_scheme,
        "locked_manifest": str(args.locked_manifest),
        "history_length": train_dataset.history_length,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "use_force_input": args.use_force_input,
        "normalize_inputs": True,
        "normalize_targets": True,
        "dataset_path": str(dataset_path),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "ridge_alpha": args.ridge_alpha,
        "bounded_outputs": bool(args.bounded_outputs and model_type not in {"ridge"}),
        "output_scale_percentile": args.output_scale_percentile,
        "output_scale": output_scale,
        "rollout_loss_weight": args.rollout_loss_weight,
        "rollout_horizon": args.rollout_horizon,
        "rollout_warmup_epochs": args.rollout_warmup_epochs,
        "scheduled_sampling_probability": args.scheduled_sampling_probability,
        "rollout_batch_size": args.rollout_batch_size,
        "rollout_loss_space": "normalized_extension_state",
    }

    best_path = output_dir / "best_checkpoint.pth"
    last_path = output_dir / "last_checkpoint.pth"
    history: list[dict[str, Any]] = []

    if model_type == "ridge":
        print("Fitting ridge baseline with chunked HDF5 streaming normal equations.")
        assert isinstance(model, RidgeRegressionBaseline)
        fit_ridge_hdf5_streaming(
            model,
            train_dataset,
            device,
            ridge_alpha=args.ridge_alpha,
            max_samples=args.max_train_samples,
        )
        train_metrics = evaluate_one_step(
            model,
            train_loader,
            device,
            extension_loss_weight=args.extension_loss_weight,
            pressure_loss_weight=args.pressure_loss_weight,
            sign_loss_weight=args.sign_loss_weight,
            sign_deadband=args.sign_deadband,
        )
        val_metrics = evaluate_one_step(
            model,
            val_loader,
            device,
            extension_loss_weight=args.extension_loss_weight,
            pressure_loss_weight=args.pressure_loss_weight,
            sign_loss_weight=args.sign_loss_weight,
            sign_deadband=args.sign_deadband,
        )
        save_checkpoint(
            best_path,
            model=model,
            config=config_dict,
            metadata=metadata,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            epoch=0,
        )
        history.append({"epoch": 0, "train": train_metrics, "val": val_metrics})
        write_history(output_dir / "training_history.json", history)
        print(f"Saved ridge checkpoint: {best_path}")
        print(f"Validation metrics: {val_metrics}")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        if args.lr_scheduler == "plateau"
        else None
    )
    best_val = float("inf")
    best_epoch = -1
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {
            "loss": 0.0,
            "one_step_loss": 0.0,
            "extension_mse": 0.0,
            "pressure_mse": 0.0,
            "sign_loss": 0.0,
            "rollout_loss": 0.0,
        }
        samples = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss, metrics = compute_loss(
                pred,
                y,
                extension_loss_weight=args.extension_loss_weight,
                pressure_loss_weight=args.pressure_loss_weight,
                sign_loss_weight=args.sign_loss_weight,
                sign_deadband=args.sign_deadband,
            )
            one_step_loss = loss
            rollout_loss = x.new_tensor(0.0)
            rollout_active = (
                rollout_dataset is not None
                and rollout_start_indices.size > 0
                and args.rollout_loss_weight > 0.0
                and epoch > args.rollout_warmup_epochs
            )
            if rollout_active:
                rollout_loss = compute_short_rollout_loss(
                    model=model,
                    raw_dataset=rollout_dataset,
                    start_indices=rollout_start_indices,
                    device=device,
                    horizon=args.rollout_horizon,
                    batch_size=args.rollout_batch_size,
                    scheduled_sampling_probability=args.scheduled_sampling_probability,
                )
                loss = loss + float(args.rollout_loss_weight) * rollout_loss
            metrics["one_step_loss"] = float(one_step_loss.detach().cpu())
            metrics["rollout_loss"] = float(rollout_loss.detach().cpu())
            metrics["loss"] = float(loss.detach().cpu())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}.")
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            batch = x.shape[0]
            for key, value in metrics.items():
                totals[key] += value * batch
            samples += batch
        train_metrics = {key: value / max(samples, 1) for key, value in totals.items()}
        val_metrics = evaluate_one_step(
            model,
            val_loader,
            device,
            extension_loss_weight=args.extension_loss_weight,
            pressure_loss_weight=args.pressure_loss_weight,
            sign_loss_weight=args.sign_loss_weight,
            sign_deadband=args.sign_deadband,
        )
        if scheduler is not None:
            scheduler.step(val_metrics["loss"])
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(f"epoch={epoch} train_loss={train_metrics['loss']:.6g} val_loss={val_metrics['loss']:.6g}")

        save_checkpoint(
            last_path,
            model=model,
            config=config_dict,
            metadata=metadata,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            epoch=epoch,
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            patience = 0
            save_checkpoint(
                best_path,
                model=model,
                config=config_dict,
                metadata=metadata,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                epoch=epoch,
            )
        else:
            patience += 1
            if patience >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break
        write_history(output_dir / "training_history.json", history)

    print(f"Saved best checkpoint: {best_path}")
    print(f"Best validation loss: {best_val:.6g} at epoch {best_epoch}")
    if rollout_dataset is not None:
        rollout_dataset.close()


if __name__ == "__main__":
    main()
