from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

try:
    from evaluate_rom import load_checkpoint_model
    from rom_data_contract import DP_SLICE, EXTENSION_SLICE, EXTENSION_VELOCITY_SLICE, TARGET_DELTA_DP_SLICE, TARGET_DELTA_EXTENSION_SLICE
    from rom_dataset import DEFAULT_DATASET_PATH, LOCKED_SPLIT_MANIFEST_PATH, ROMSequenceDataset, find_default_dataset
except ImportError:
    from kevin_integration.rom_cylinders.evaluate_rom import load_checkpoint_model
    from kevin_integration.rom_cylinders.rom_data_contract import (
        DP_SLICE,
        EXTENSION_SLICE,
        EXTENSION_VELOCITY_SLICE,
        TARGET_DELTA_DP_SLICE,
        TARGET_DELTA_EXTENSION_SLICE,
    )
    from kevin_integration.rom_cylinders.rom_dataset import (
        DEFAULT_DATASET_PATH,
        LOCKED_SPLIT_MANIFEST_PATH,
        ROMSequenceDataset,
        find_default_dataset,
    )


ROM_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROM_DIR / "results"


def _rmse(values: list[np.ndarray]) -> float:
    if not values:
        return float("nan")
    arr = np.concatenate([value.reshape(-1) for value in values])
    return float(np.sqrt(np.mean(arr**2)))


def _max_abs(values: list[np.ndarray]) -> float:
    if not values:
        return float("nan")
    arr = np.concatenate([np.abs(value.reshape(-1)) for value in values])
    return float(np.max(arr))


def _state_limit_violated(
    extension: torch.Tensor,
    dP: torch.Tensor,
    *,
    extension_limit_abs: float,
    dp_limit_abs: float,
    state_limits: dict[str, torch.Tensor] | None = None,
) -> int:
    violations = 0
    if state_limits is not None:
        ext_lower = state_limits["extension_lower"].to(device=extension.device, dtype=extension.dtype)
        ext_upper = state_limits["extension_upper"].to(device=extension.device, dtype=extension.dtype)
        dp_lower = state_limits["dp_lower"].to(device=dP.device, dtype=dP.dtype)
        dp_upper = state_limits["dp_upper"].to(device=dP.device, dtype=dP.dtype)
        if bool(torch.any((extension < ext_lower) | (extension > ext_upper))):
            violations += 1
        if bool(torch.any((dP < dp_lower) | (dP > dp_upper))):
            violations += 1
    else:
        if torch.max(torch.abs(extension)).item() > extension_limit_abs:
            violations += 1
        if torch.max(torch.abs(dP)).item() > dp_limit_abs:
            violations += 1
    return violations


def _normalize_x(dataset: ROMSequenceDataset, raw_x: torch.Tensor) -> torch.Tensor:
    mean = dataset.normalization_stats.input_mean.to(dtype=raw_x.dtype, device=raw_x.device)
    std = dataset.normalization_stats.input_std.to(dtype=raw_x.dtype, device=raw_x.device)
    return (raw_x - mean) / std


def compute_training_state_limits(
    *,
    dataset_path: Path,
    split_scheme: str,
    locked_manifest: Path,
    target_mode: str,
    use_force_input: bool,
    margin_fraction: float,
    chunk_samples: int = 32768,
) -> dict[str, torch.Tensor]:
    train_dataset = ROMSequenceDataset(
        dataset_path,
        split="train",
        use_force_input=use_force_input,
        split_scheme=split_scheme,
        locked_manifest=locked_manifest,
        target_mode=target_mode,
        normalize_inputs=False,
        normalize_targets=False,
    )
    source_split = getattr(train_dataset, "_source_split")
    sample_indices = getattr(train_dataset, "_sample_indices")
    if sample_indices is None:
        sample_indices = np.arange(len(train_dataset), dtype=np.int64)
    else:
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
    train_dataset.close()

    ext_min = np.full((4,), np.inf, dtype=np.float64)
    ext_max = np.full((4,), -np.inf, dtype=np.float64)
    dp_min = np.full((4,), np.inf, dtype=np.float64)
    dp_max = np.full((4,), -np.inf, dtype=np.float64)
    with h5py.File(dataset_path, "r") as handle:
        x_key = f"X_{source_split}_rom"
        for start in range(0, sample_indices.size, chunk_samples):
            idx = np.sort(sample_indices[start : start + chunk_samples])
            x = np.asarray(handle[x_key][:, :, idx], dtype=np.float32)
            ext = x[EXTENSION_SLICE, :, :].reshape(4, -1)
            dp = x[DP_SLICE, :, :].reshape(4, -1)
            ext_min = np.minimum(ext_min, np.min(ext, axis=1))
            ext_max = np.maximum(ext_max, np.max(ext, axis=1))
            dp_min = np.minimum(dp_min, np.min(dp, axis=1))
            dp_max = np.maximum(dp_max, np.max(dp, axis=1))
    margin = max(float(margin_fraction), 0.0)
    ext_pad = np.maximum((ext_max - ext_min) * margin, 1.0e-6)
    dp_pad = np.maximum((dp_max - dp_min) * margin, 1.0)
    return {
        "extension_lower": torch.from_numpy((ext_min - ext_pad).astype(np.float32)),
        "extension_upper": torch.from_numpy((ext_max + ext_pad).astype(np.float32)),
        "dp_lower": torch.from_numpy((dp_min - dp_pad).astype(np.float32)),
        "dp_upper": torch.from_numpy((dp_max + dp_pad).astype(np.float32)),
    }


def rollout_single_run(
    *,
    model: torch.nn.Module,
    dataset: ROMSequenceDataset,
    indices: np.ndarray,
    device: torch.device,
    extension_limit_abs: float,
    dp_limit_abs: float,
    rollout_mode: str,
    state_limits: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    if rollout_mode not in {"teacher_state", "teacher_pressure", "predicted_pressure"}:
        raise ValueError(f"Unsupported rollout_mode {rollout_mode!r}.")
    model.eval()
    first = int(indices[0])
    _, _, info = dataset[first]
    history_raw = info["raw_input"].clone().float()
    dt = float(info["sample_time"])
    extension_errors: list[np.ndarray] = []
    dp_errors: list[np.ndarray] = []
    unstable = False
    nan_inf_count = 0
    limit_violation_count = 0
    predicted_extension: list[list[float]] = []
    true_extension: list[list[float]] = []
    predicted_dp: list[list[float]] = []
    true_dp_preview: list[list[float]] = []
    extension_error_series: list[list[float]] = []
    dp_error_series: list[list[float]] = []
    limit_violation_series: list[int] = []

    with torch.no_grad():
        for next_index in indices[1:]:
            assert history_raw.ndim == 2 and history_raw.shape[0] == 10 and history_raw.shape[1] >= 26
            assert EXTENSION_SLICE == slice(8, 12)
            assert DP_SLICE == slice(4, 8)
            x_norm = _normalize_x(dataset, history_raw).unsqueeze(0).to(device)
            pred_norm = model(x_norm).detach().cpu()
            pred_raw = dataset.denormalize_y(pred_norm).squeeze(0)
            delta_ext = pred_raw[TARGET_DELTA_EXTENSION_SLICE]
            has_dp_output = pred_raw.shape[-1] >= 8
            delta_dp = pred_raw[TARGET_DELTA_DP_SLICE] if has_dp_output else torch.zeros(4, dtype=pred_raw.dtype)
            current_ext = history_raw[-1, EXTENSION_SLICE]
            current_dp = history_raw[-1, DP_SLICE]
            next_ext_pred = current_ext + delta_ext
            next_dp_pred = current_dp + delta_dp
            if rollout_mode == "predicted_pressure" and not has_dp_output:
                raise ValueError("rollout_mode='predicted_pressure' requires an extension_and_dp checkpoint.")
            if not torch.isfinite(pred_norm).all() or not torch.isfinite(pred_raw).all():
                nan_inf_count += 1
                unstable = True
                break
            if not torch.isfinite(next_ext_pred).all() or not torch.isfinite(next_dp_pred).all():
                nan_inf_count += 1
                unstable = True
                break

            _, _, next_info = dataset[int(next_index)]
            next_raw_true = next_info["raw_input"].clone().float()
            true_ext = next_raw_true[-1, EXTENSION_SLICE]
            true_dp_tensor = next_raw_true[-1, DP_SLICE]
            if rollout_mode == "teacher_state":
                eval_ext = current_ext + delta_ext
                eval_dp = current_dp + delta_dp if has_dp_output else true_dp_tensor
                next_history = next_raw_true.clone()
            elif rollout_mode == "teacher_pressure":
                eval_ext = next_ext_pred
                eval_dp = true_dp_tensor
                next_history = next_raw_true.clone()
                next_history[:-1, EXTENSION_SLICE] = history_raw[1:, EXTENSION_SLICE]
                next_history[:-1, EXTENSION_VELOCITY_SLICE] = history_raw[1:, EXTENSION_VELOCITY_SLICE]
                next_history[-1, EXTENSION_SLICE] = next_ext_pred
                next_history[-1, EXTENSION_VELOCITY_SLICE] = delta_ext / max(dt, 1.0e-12)
            else:
                eval_ext = next_ext_pred
                eval_dp = next_dp_pred
                next_history = next_raw_true.clone()
                next_history[:-1, EXTENSION_SLICE] = history_raw[1:, EXTENSION_SLICE]
                next_history[:-1, DP_SLICE] = history_raw[1:, DP_SLICE]
                next_history[:-1, EXTENSION_VELOCITY_SLICE] = history_raw[1:, EXTENSION_VELOCITY_SLICE]
                next_history[-1, EXTENSION_SLICE] = next_ext_pred
                next_history[-1, DP_SLICE] = next_dp_pred
                next_history[-1, EXTENSION_VELOCITY_SLICE] = delta_ext / max(dt, 1.0e-12)

            step_limit_violations = _state_limit_violated(
                eval_ext,
                eval_dp,
                extension_limit_abs=extension_limit_abs,
                dp_limit_abs=dp_limit_abs,
                state_limits=state_limits,
            )
            limit_violation_count += step_limit_violations
            ext_error = (eval_ext - true_ext).numpy()
            dp_error = (eval_dp - true_dp_tensor).numpy()
            extension_errors.append(ext_error)
            dp_errors.append(dp_error)
            extension_error_series.append(ext_error.tolist())
            dp_error_series.append(dp_error.tolist())
            limit_violation_series.append(int(step_limit_violations))
            predicted_extension.append(eval_ext.tolist())
            true_extension.append(true_ext.tolist())
            predicted_dp.append(eval_dp.tolist())
            true_dp_preview.append(true_dp_tensor.tolist())
            history_raw = next_history

    steps = len(extension_errors)
    drift_rate = float(np.linalg.norm(extension_errors[-1]) / max(steps * dt, 1.0e-12)) if extension_errors else float("nan")
    return {
        "steps": steps,
        "duration_seconds": steps * dt,
        "extension_rmse": _rmse(extension_errors),
        "dp_rmse": _rmse(dp_errors),
        "extension_max_abs_error": _max_abs(extension_errors),
        "dp_max_abs_error": _max_abs(dp_errors),
        "drift_rate": drift_rate,
        "unstable": unstable,
        "nan_inf_count": nan_inf_count,
        "limit_violation_count": limit_violation_count,
        "extension_error_series": extension_error_series,
        "dp_error_series": dp_error_series,
        "limit_violation_series": limit_violation_series,
        "predicted_extension_preview": predicted_extension[:5],
        "true_extension_preview": true_extension[:5],
        "predicted_dp_preview": predicted_dp[:5],
        "true_dp_preview": true_dp_preview[:5],
    }


def summarize_horizon(run_results: list[dict[str, Any]], horizon_steps: int | None) -> dict[str, float | int]:
    ext_error_arrays: list[np.ndarray] = []
    dp_error_arrays: list[np.ndarray] = []
    unstable = 0
    nan_inf = 0
    limit = 0
    counts = 0
    for result in run_results:
        if horizon_steps is None or result["steps"] >= horizon_steps:
            use_steps = result["steps"] if horizon_steps is None else min(int(horizon_steps), int(result["steps"]))
            ext_series = np.asarray(result.get("extension_error_series", []), dtype=np.float64)[:use_steps]
            dp_series = np.asarray(result.get("dp_error_series", []), dtype=np.float64)[:use_steps]
            if ext_series.size:
                ext_error_arrays.append(ext_series)
            if dp_series.size:
                dp_error_arrays.append(dp_series)
            counts += 1
            unstable += int(result["unstable"])
            nan_inf += int(result["nan_inf_count"])
            if "limit_violation_series" in result:
                limit += int(np.sum(np.asarray(result["limit_violation_series"], dtype=np.int64)[:use_steps]))
            else:
                limit += int(result["limit_violation_count"])
    ext_flat = np.concatenate([value.reshape(-1) for value in ext_error_arrays]) if ext_error_arrays else np.asarray([])
    dp_flat = np.concatenate([value.reshape(-1) for value in dp_error_arrays]) if dp_error_arrays else np.asarray([])
    return {
        "run_count": counts,
        "extension_trajectory_rmse_mean": float(np.sqrt(np.mean(ext_flat**2))) if ext_flat.size else float("nan"),
        "dp_trajectory_rmse_mean": float(np.sqrt(np.mean(dp_flat**2))) if dp_flat.size else float("nan"),
        "extension_max_abs_error_mean": float(np.max(np.abs(ext_flat))) if ext_flat.size else float("nan"),
        "fraction_unstable_rollouts": float(unstable / max(counts, 1)),
        "nan_inf_count": nan_inf,
        "limit_violation_count": limit,
    }


def run_rollouts(
    model: torch.nn.Module,
    dataset: ROMSequenceDataset,
    *,
    device: torch.device,
    max_runs: int | None,
    extension_limit_abs: float,
    dp_limit_abs: float,
    rollout_mode: str = "teacher_pressure",
    state_limits: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    run_ids = dataset.run_ids
    unique_runs = np.unique(run_ids)
    if max_runs is not None:
        unique_runs = unique_runs[:max_runs]
    run_results: dict[str, Any] = {}
    for run_id in unique_runs:
        indices = np.flatnonzero(run_ids == run_id)
        indices = indices[np.argsort(dataset.time_indices[indices])]
        if len(indices) < 2:
            continue
        run_results[str(int(run_id))] = rollout_single_run(
            model=model,
            dataset=dataset,
            indices=indices,
            device=device,
            extension_limit_abs=extension_limit_abs,
            dp_limit_abs=dp_limit_abs,
            rollout_mode=rollout_mode,
            state_limits=state_limits,
        )

    dt = float(dataset.metadata.dt_seconds)
    results_list = list(run_results.values())
    horizons = {
        "0.1s": max(1, int(round(0.1 / dt))),
        "0.5s": max(1, int(round(0.5 / dt))),
        "1.0s": max(1, int(round(1.0 / dt))),
        "2.0s": max(1, int(round(2.0 / dt))),
        "full_run": None,
    }
    return {
        "split": dataset.split,
        "rollout_mode": rollout_mode,
        "state_limits": {
            key: value.detach().cpu().tolist()
            for key, value in state_limits.items()
        } if state_limits is not None else None,
        "run_count": len(run_results),
        "horizons": {name: summarize_horizon(results_list, steps) for name, steps in horizons.items()},
        "runs": run_results,
    }


def write_rollout_report(results: dict[str, Any], output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}_rollout.json"
    md_path = output_dir / f"{stem}_rollout.md"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# ROM Rollout Evaluation",
        "",
        f"- Split: `{results['split']}`",
        f"- Rollout mode: `{results['rollout_mode']}`",
        f"- State limits: `{'training_range' if results.get('state_limits') is not None else 'absolute_thresholds'}`",
        f"- Runs: `{results['run_count']}`",
        "",
        "## Horizons",
        "",
    ]
    lines.append("| Horizon | Runs | Extension RMSE | dP RMSE | Max Ext Error | Unstable Fraction | NaN/Inf | Limit Violations |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, item in results["horizons"].items():
        lines.append(
            f"| {name} | {item['run_count']} | {item['extension_trajectory_rmse_mean']} | "
            f"{item['dp_trajectory_rmse_mean']} | {item['extension_max_abs_error_mean']} | "
            f"{item['fraction_unstable_rollouts']} | {item['nan_inf_count']} | {item['limit_violation_count']} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoregressively rollout a trained cylinder ROM on complete held-out runs.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--split", choices=["val", "train", "test"], default="val")
    parser.add_argument("--split_scheme", choices=["original", "locked"], default=None)
    parser.add_argument("--locked_manifest", type=Path, default=None)
    parser.add_argument("--rollout_mode", choices=["teacher_state", "teacher_pressure", "predicted_pressure"], default="teacher_pressure")
    parser.add_argument("--state_limit_source", choices=["training", "absolute", "none"], default="training")
    parser.add_argument("--state_limit_margin_fraction", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--extension_limit_abs", type=float, default=10.0)
    parser.add_argument("--dp_limit_abs", type=float, default=2.0e8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    use_force = bool(payload["config"].get("use_force_input", False))
    target_mode = str(payload["config"].get("target_mode", "extension_and_dp"))
    split_scheme = args.split_scheme or str(payload["config"].get("split_scheme", "original"))
    locked_manifest = args.locked_manifest or Path(payload["config"].get("locked_manifest", LOCKED_SPLIT_MANIFEST_PATH))
    dataset_path = args.dataset or (DEFAULT_DATASET_PATH if DEFAULT_DATASET_PATH.exists() else find_default_dataset())
    dataset = ROMSequenceDataset(
        dataset_path,
        split=args.split,
        use_force_input=use_force,
        split_scheme=split_scheme,
        locked_manifest=locked_manifest,
        target_mode=target_mode,
        normalize_inputs=False,
        normalize_targets=False,
        return_metadata=True,
    )
    device = torch.device(args.device)
    model, _ = load_checkpoint_model(args.checkpoint, dataset, device)
    state_limits = None
    if args.state_limit_source == "training":
        state_limits = compute_training_state_limits(
            dataset_path=dataset_path,
            split_scheme=split_scheme,
            locked_manifest=locked_manifest,
            target_mode=target_mode,
            use_force_input=use_force,
            margin_fraction=args.state_limit_margin_fraction,
        )
    elif args.state_limit_source == "none":
        args.extension_limit_abs = float("inf")
        args.dp_limit_abs = float("inf")
    results = run_rollouts(
        model,
        dataset,
        device=device,
        max_runs=args.max_runs,
        extension_limit_abs=args.extension_limit_abs,
        dp_limit_abs=args.dp_limit_abs,
        rollout_mode=args.rollout_mode,
        state_limits=state_limits,
    )
    stem = args.checkpoint.parent.name + f"_{args.split}_{args.rollout_mode}"
    write_rollout_report(results, args.output_dir, stem)
    dataset.close()


if __name__ == "__main__":
    main()
