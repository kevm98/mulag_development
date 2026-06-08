from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from models import ModelConfig, build_model
    from rom_data_contract import DP_SLICE, EXTENSION_VELOCITY_SLICE, TARGET_DELTA_DP_SLICE, TARGET_DELTA_EXTENSION_SLICE, VALVE_SLICE
    from rom_dataset import DEFAULT_DATASET_PATH, ROMSequenceDataset, find_default_dataset
except ImportError:
    from kevin_integration.rom_cylinders.models import ModelConfig, build_model
    from kevin_integration.rom_cylinders.rom_data_contract import (
        DP_SLICE,
        EXTENSION_VELOCITY_SLICE,
        TARGET_DELTA_DP_SLICE,
        TARGET_DELTA_EXTENSION_SLICE,
        VALVE_SLICE,
    )
    from kevin_integration.rom_cylinders.rom_dataset import DEFAULT_DATASET_PATH, ROMSequenceDataset, find_default_dataset


ROM_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROM_DIR / "results"


def collate_xy(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    return torch.stack(list(xs), dim=0), torch.stack(list(ys), dim=0)


def metric_dict(pred: np.ndarray, true: np.ndarray, *, sign_deadband: float | None = None) -> dict[str, float]:
    err = pred - true
    abs_err = np.abs(err)
    rmse = float(np.sqrt(np.mean(err**2))) if err.size else float("nan")
    mae = float(np.mean(abs_err)) if err.size else float("nan")
    true_std = float(np.std(true)) if true.size else float("nan")
    nrmse = float(rmse / max(true_std, 1.0e-12)) if np.isfinite(rmse) else float("nan")
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2)) if true.size else float("nan")
    r2 = float(1.0 - ss_res / max(ss_tot, 1.0e-12)) if true.size else float("nan")
    if pred.size > 1 and np.std(pred) > 1.0e-12 and np.std(true) > 1.0e-12:
        corr = float(np.corrcoef(pred, true)[0, 1])
    else:
        corr = float("nan")
    out = {
        "mae": mae,
        "rmse": rmse,
        "normalized_rmse": nrmse,
        "r_squared": r2,
        "correlation": corr,
        "bias": float(np.mean(err)) if err.size else float("nan"),
        "p95_abs_error": float(np.percentile(abs_err, 95)) if err.size else float("nan"),
        "count": int(pred.size),
        "near_zero_target_fraction": float(np.mean(np.abs(true) <= (sign_deadband or 1.0e-12))) if true.size else float("nan"),
        "predicted_magnitude_ratio": float(np.mean(np.abs(pred)) / max(float(np.mean(np.abs(true))), 1.0e-12)) if true.size else float("nan"),
        "underprediction_fraction": float(np.mean(np.abs(pred) < np.abs(true))) if true.size else float("nan"),
        "overprediction_fraction": float(np.mean(np.abs(pred) > np.abs(true))) if true.size else float("nan"),
    }
    if sign_deadband is not None:
        mask = np.abs(true) > sign_deadband
        if np.any(mask):
            out["sign_accuracy"] = float(np.mean(np.sign(pred[mask]) == np.sign(true[mask])))
            out["sign_count"] = int(np.sum(mask))
        else:
            out["sign_accuracy"] = float("nan")
            out["sign_count"] = 0
    return out


def rows_from_metrics(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, values in metrics.items():
        if not isinstance(values, dict):
            continue
        for name, metric in values.items():
            if isinstance(metric, dict) and "rmse" in metric:
                row = {"group": group, "name": name}
                row.update(metric)
                rows.append(row)
    return rows


def evaluate_model(
    model: nn.Module,
    dataset: ROMSequenceDataset,
    *,
    batch_size: int = 512,
    device: torch.device | str = "cpu",
    sign_deadband: float = 1.0e-5,
    valve_deadband: float = 0.05,
) -> dict[str, Any]:
    device = torch.device(device)
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_xy)
    pred_chunks: list[np.ndarray] = []
    true_chunks: list[np.ndarray] = []
    valve_chunks: list[np.ndarray] = []
    dP_chunks: list[np.ndarray] = []
    velocity_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred_norm = model(x)
            pred_raw = dataset.denormalize_y(pred_norm.detach().cpu()).numpy()
            true_raw = dataset.denormalize_y(y.detach().cpu()).numpy()
            raw_x = dataset.denormalize_x(x.detach().cpu()).numpy()
            pred_chunks.append(pred_raw)
            true_chunks.append(true_raw)
            valve_chunks.append(raw_x[:, -1, VALVE_SLICE])
            dP_chunks.append(raw_x[:, -1, DP_SLICE])
            velocity_chunks.append(raw_x[:, -1, EXTENSION_VELOCITY_SLICE])
    pred = np.concatenate(pred_chunks, axis=0)
    true = np.concatenate(true_chunks, axis=0)
    valve = np.concatenate(valve_chunks, axis=0)
    dP = np.concatenate(dP_chunks, axis=0)
    velocity = np.concatenate(velocity_chunks, axis=0)
    has_pressure_output = pred.shape[-1] >= 8 and true.shape[-1] >= 8

    metrics: dict[str, Any] = {
        "split": dataset.split,
        "split_scheme": dataset.split_scheme,
        "target_mode": dataset.target_mode,
        "sample_count": int(pred.shape[0]),
        "run_count": int(dataset.number_of_unique_runs),
        "delta_extension": {},
        "delta_dp": {},
        "regimes": {},
    }
    for cyl in range(4):
        metrics["delta_extension"][f"cylinder_{cyl + 1}"] = metric_dict(
            pred[:, cyl],
            true[:, cyl],
            sign_deadband=sign_deadband,
        )
        if has_pressure_output:
            metrics["delta_dp"][f"cylinder_{cyl + 1}"] = metric_dict(pred[:, 4 + cyl], true[:, 4 + cyl])
    metrics["delta_extension"]["aggregate"] = metric_dict(
        pred[:, TARGET_DELTA_EXTENSION_SLICE].reshape(-1),
        true[:, TARGET_DELTA_EXTENSION_SLICE].reshape(-1),
        sign_deadband=sign_deadband,
    )
    if has_pressure_output:
        metrics["delta_dp"]["aggregate"] = metric_dict(
            pred[:, TARGET_DELTA_DP_SLICE].reshape(-1),
            true[:, TARGET_DELTA_DP_SLICE].reshape(-1),
        )

    valve_abs = np.max(np.abs(valve), axis=1)
    dP_abs = np.max(np.abs(dP), axis=1)
    speed_abs = np.max(np.abs(velocity), axis=1)
    pressure_quantiles = np.quantile(dP_abs, [0.33, 0.66])
    speed_quantiles = np.quantile(speed_abs, [0.5])
    ext_abs = np.max(np.abs(true[:, TARGET_DELTA_EXTENSION_SLICE]), axis=1)
    ext_bins = np.quantile(ext_abs, [0.25, 0.5, 0.75])
    regimes = {
        "near_zero_valve": valve_abs <= valve_deadband,
        "positive_valve": np.max(valve, axis=1) > valve_deadband,
        "negative_valve": np.min(valve, axis=1) < -valve_deadband,
        "low_pressure": dP_abs <= pressure_quantiles[0],
        "medium_pressure": (dP_abs > pressure_quantiles[0]) & (dP_abs <= pressure_quantiles[1]),
        "high_pressure": dP_abs > pressure_quantiles[1],
        "low_speed": speed_abs <= speed_quantiles[0],
        "high_speed": speed_abs > speed_quantiles[0],
        "very_small_target": ext_abs <= ext_bins[0],
        "small_target": (ext_abs > ext_bins[0]) & (ext_abs <= ext_bins[1]),
        "medium_target": (ext_abs > ext_bins[1]) & (ext_abs <= ext_bins[2]),
        "large_target": ext_abs > ext_bins[2],
    }
    for name, mask in regimes.items():
        if not np.any(mask):
            continue
        regime_metrics = {
            "delta_extension_aggregate": metric_dict(
                pred[mask, TARGET_DELTA_EXTENSION_SLICE].reshape(-1),
                true[mask, TARGET_DELTA_EXTENSION_SLICE].reshape(-1),
                sign_deadband=sign_deadband,
            )
        }
        if has_pressure_output:
            regime_metrics["delta_dp_aggregate"] = metric_dict(
                pred[mask, TARGET_DELTA_DP_SLICE].reshape(-1),
                true[mask, TARGET_DELTA_DP_SLICE].reshape(-1),
            )
        metrics["regimes"][name] = regime_metrics
    return metrics


def load_checkpoint_model(checkpoint: Path, dataset: ROMSequenceDataset, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device)
    config = payload["config"]
    model_config = ModelConfig(
        model_type=config["model_type"],
        input_dim=int(config["input_dim"]),
        output_dim=int(config.get("output_dim", 8)),
        target_mode=str(config.get("target_mode", "extension_and_dp")),
        hidden_dim=int(config.get("hidden_dim", 64)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.1)),
        bounded_outputs=bool(config.get("bounded_outputs", False)),
        output_scale=tuple(config["output_scale"]) if config.get("output_scale") is not None else None,
    )
    model = build_model(model_config, normalization_stats=dataset.normalization_stats)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    return model, payload


def write_reports(metrics: dict[str, Any], output_dir: Path, *, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}_metrics.json"
    csv_path = output_dir / f"{stem}_metrics.csv"
    md_path = output_dir / f"{stem}_metrics.md"
    json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    rows = rows_from_metrics(metrics)
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# ROM Evaluation Metrics", "", f"- Split: `{metrics['split']}`", f"- Samples: `{metrics['sample_count']}`", f"- Runs: `{metrics['run_count']}`", ""]
    for section in ("delta_extension", "delta_dp"):
        lines.append(f"## {section}")
        lines.append("")
        lines.append("| Name | RMSE | MAE | NRMSE | R2 | Corr | Bias | P95 Abs | Sign Acc | Count |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name, item in metrics[section].items():
            lines.append(
                f"| {name} | {item.get('rmse')} | {item.get('mae')} | {item.get('normalized_rmse')} | "
                f"{item.get('r_squared')} | {item.get('correlation')} | {item.get('bias')} | "
                f"{item.get('p95_abs_error')} | {item.get('sign_accuracy', '')} | {item.get('count')} |"
            )
        lines.append("")
    lines.append("## Regimes")
    lines.append("")
    for name, item in metrics["regimes"].items():
        ext = item["delta_extension_aggregate"]
        dp = item.get("delta_dp_aggregate")
        dp_text = f", DeltaDP RMSE `{dp['rmse']}`" if dp is not None else ""
        lines.append(f"- `{name}`: DeltaExtension RMSE `{ext['rmse']}`{dp_text}, count `{ext['count']}`")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained ROM checkpoint on a held-out split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--split", choices=["val", "train", "test"], default="val")
    parser.add_argument("--split_scheme", choices=["original", "locked"], default=None)
    parser.add_argument("--locked_manifest", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--sign_deadband", type=float, default=1.0e-5)
    parser.add_argument("--valve_deadband", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset or (DEFAULT_DATASET_PATH if DEFAULT_DATASET_PATH.exists() else find_default_dataset())
    payload = torch.load(args.checkpoint, map_location="cpu")
    use_force = bool(payload["config"].get("use_force_input", False))
    target_mode = str(payload["config"].get("target_mode", "extension_and_dp"))
    split_scheme = args.split_scheme or str(payload["config"].get("split_scheme", "original"))
    locked_manifest = args.locked_manifest or Path(payload["config"].get("locked_manifest", ROM_DIR / "rom_split_manifest.json"))
    dataset = ROMSequenceDataset(
        dataset_path,
        split=args.split,
        use_force_input=use_force,
        split_scheme=split_scheme,
        locked_manifest=locked_manifest,
        target_mode=target_mode,
    )
    device = torch.device(args.device)
    model, _ = load_checkpoint_model(args.checkpoint, dataset, device)
    metrics = evaluate_model(
        model,
        dataset,
        batch_size=args.batch_size,
        device=device,
        sign_deadband=args.sign_deadband,
        valve_deadband=args.valve_deadband,
    )
    stem = args.checkpoint.parent.name + f"_{args.split}"
    write_reports(metrics, args.output_dir, stem=stem)
    dataset.close()


if __name__ == "__main__":
    main()
